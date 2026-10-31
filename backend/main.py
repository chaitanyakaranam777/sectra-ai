"""SENTRA AI API: model, training, evaluation and dataset interfaces for the dashboard.

Run:  uvicorn backend.main:app --reload
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import time
from typing import Literal, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.database import HistoryStore
from backend.services.artifacts import ArtifactStore, NotFound
from backend.services.inference import InferenceError, InferenceService, decode_upload
from backend.services.training_jobs import TrainingJobs
from backend.settings import Settings
from ml.architectures import MODEL_REGISTRY, build_model
from ml.training.run import load_config

ModelName = Literal["cnn", "lstm", "bert", "cnn_lstm", "cnn_bert"]


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    run_id: Optional[str] = Field(None, pattern=r"^[A-Za-z0-9_.\-]+$")


class TrainRequest(BaseModel):
    model: ModelName
    dataset: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.\-]+$")
    epochs: Optional[int] = Field(None, ge=1, le=500)
    batch_size: Optional[int] = Field(None, ge=1, le=1024)
    lr: Optional[float] = Field(None, gt=0, lt=1)
    optimizer: Optional[Literal["adam", "adamw"]] = None
    device: Optional[Literal["auto", "cpu", "cuda"]] = None


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()
    store, jobs = ArtifactStore(settings), TrainingJobs(settings)
    history_db, infer = HistoryStore(settings.database_url), None
    infer = InferenceService(settings, store)
    app = FastAPI(title="SENTRA AI API", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    @app.exception_handler(NotFound)
    async def _nf(_: Request, exc: NotFound) -> JSONResponse:
        return _error(404, "not_found", str(exc))

    @app.exception_handler(InferenceError)
    async def _ie(_: Request, exc: InferenceError) -> JSONResponse:
        return _error(exc.status, "inference_error", str(exc))

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "time": time.time(), "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": "cuda" if torch.cuda.is_available() else "cpu", "database": history_db.ping()}

    @app.get("/api/models")
    def models() -> list[dict]:
        runs = store.list_runs()
        return [{"name": n, "modality": s.cls.modality, "description": s.description, "required_inputs": list(s.cls.required_inputs),
                 "runs": [r for r in runs if r["model"] == n]} for n, s in MODEL_REGISTRY.items()]

    @app.get("/api/models/{name}/architecture")
    def architecture(name: ModelName, run_id: Optional[str] = None) -> dict:
        """Layer/parameter breakdown: from a trained run when ``run_id`` is given, else from the default config."""
        if run_id:
            return {"source": "run", **store.run_architecture(run_id)}
        cfg = dict(load_config(settings.config_path)["models"][name])
        if MODEL_REGISTRY[name].text_encoder == "word":
            cfg["vocab_size"] = load_config(settings.config_path)["data"]["vocab_max_size"]
        if "pretrained_name" in cfg:
            cfg["pretrained_name"] = None   # structure only: no download, no weights
        with torch.device("meta"):
            model = build_model(name, **cfg)
        return {"source": "default_config", "note": "structure of the configured architecture; untrained", **model.summary()}

    @app.post("/api/models/train", status_code=202)
    def train(req: TrainRequest) -> dict:
        samples = store.dataset_dir(req.dataset) / "samples.csv"
        job_id = jobs.start(req.model, req.dataset, samples, req.model_dump(exclude={"model", "dataset"}))
        return {"job_id": job_id, "status_url": f"/api/training/status/{job_id}"}

    @app.get("/api/training/runs")
    def runs() -> list[dict]:
        return store.list_runs()

    @app.get("/api/training/status/{job_id}")
    def status(job_id: str) -> dict:
        return jobs.reconcile(job_id, store.run_status(job_id))

    @app.get("/api/training/history/{run_id}")
    def history(run_id: str) -> dict:
        return store.run_history(run_id)

    @app.get("/api/evaluation/comparison")
    def comparison() -> dict:
        return store.comparison()

    @app.get("/api/evaluation/{run_id}")
    def evaluation(run_id: str) -> dict:
        return store.run_evaluation(run_id)

    @app.get("/api/datasets")
    def datasets() -> list[dict]:
        return store.list_datasets()

    @app.get("/api/datasets/statistics")
    def dataset_statistics(dataset: str) -> dict:
        return store.dataset_statistics(dataset)

    @app.get("/api/datasets/processing-log")
    def processing_log(dataset: str) -> list[dict]:
        return store.dataset_log(dataset)

    async def _read_image(file: UploadFile):
        data = await file.read(settings.max_upload_bytes + 1)
        return decode_upload(data, file.content_type, settings.max_upload_bytes)

    def _save(mode: str, res: dict, text: Optional[str], image_name: Optional[str]) -> dict:
        row = history_db.add(mode=mode, run_id=res["run_id"], model=res["model"], label=res["label"],
                             confidence=res["confidence"], probabilities=res["probabilities"], text=text, image_name=image_name)
        return {**res, "prediction_id": row["id"]}

    @app.post("/api/predict/text")
    def predict_text(req: TextRequest) -> dict:
        return _save("text", infer.predict("text", req.text, None, req.run_id), req.text, None)

    @app.post("/api/predict/image")
    async def predict_image(file: UploadFile = File(...), run_id: Optional[str] = Form(None)) -> dict:
        img = await _read_image(file)
        return _save("image", infer.predict("image", None, img, run_id), None, (file.filename or "upload")[:255])

    @app.post("/api/predict/multimodal")
    async def predict_multimodal(text: str = Form(..., min_length=1, max_length=2000), file: UploadFile = File(...),
                                 run_id: Optional[str] = Form(None)) -> dict:
        img = await _read_image(file)
        return _save("multimodal", infer.predict("multimodal", text, img, run_id), text, (file.filename or "upload")[:255])

    @app.post("/api/explain/{mode}")
    async def explain(mode: Literal["text", "image", "multimodal"], text: Optional[str] = Form(None),
                      file: Optional[UploadFile] = File(None), run_id: Optional[str] = Form(None)) -> dict:
        img = await _read_image(file) if file is not None else None
        return infer.explain(mode, text, img, run_id)

    @app.get("/api/predictions/history")
    def prediction_history(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)) -> dict:
        return history_db.list(limit, offset)

    return app


app = create_app()
