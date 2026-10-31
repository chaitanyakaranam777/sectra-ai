import json
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.settings import Settings
from ml.evaluation.model_comparison import evaluate_and_compare


@pytest.fixture()
def env(tmp_path, processed, runs):
    runs_dir, proc_dir = tmp_path / "runs", tmp_path / "processed"
    runs_dir.mkdir(); (proc_dir / "synthetic").mkdir(parents=True)
    for f in ("samples.csv", "report.json", "processing_log.jsonl"):
        shutil.copy(processed[0].parent / f, proc_dir / "synthetic" / f)
    for name, d in runs.items():
        shutil.copytree(d, runs_dir / d.name)
    s = Settings(runs_dir=runs_dir, processed_dir=proc_dir, comparison_path=tmp_path / "comparison.json",
                 database_url=f"sqlite:///{tmp_path / 't.db'}", max_upload_bytes=200_000)
    return s, TestClient(create_app(s))


def test_health(env):
    r = env[1].get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["device"] in {"cpu", "cuda"}


def test_models_listing_includes_real_runs(env):
    data = env[1].get("/api/models").json()
    assert [m["name"] for m in data] == ["cnn", "lstm", "bert", "cnn_lstm", "cnn_bert"]
    assert all(len(m["runs"]) == 1 for m in data)


@pytest.mark.parametrize("name", ["cnn", "lstm", "bert", "cnn_lstm", "cnn_bert"])
def test_default_architecture_via_meta_device(env, name):
    r = env[1].get(f"/api/models/{name}/architecture")
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "default_config" and j["parameters_total"] > 0 and j["components"]


def test_run_architecture_matches_real_model(env):
    rid = next(r["run_id"] for r in env[1].get("/api/training/runs").json() if r["model"] == "cnn_bert")
    j = env[1].get(f"/api/models/cnn_bert/architecture?run_id={rid}").json()
    real = json.loads((env[0].runs_dir / rid / "architecture.json").read_text())
    assert j["parameters_total"] == real["parameters_total"] and j["source"] == "run"


def test_training_status_and_history_come_from_files(env):
    c = env[1]
    run = c.get("/api/training/runs").json()[0]
    st = c.get(f"/api/training/status/{run['run_id']}").json()
    assert st["state"] == "completed"
    h = c.get(f"/api/training/history/{run['run_id']}").json()
    assert len(h["epochs"]) == 6 and "val_loss" in h["epochs"][0]


def test_evaluation_requires_real_evaluation_first(env):
    s, c = env
    rid = c.get("/api/training/runs").json()[0]["run_id"]
    assert c.get(f"/api/evaluation/{rid}").status_code == 404
    assert c.get("/api/evaluation/comparison").status_code == 404
    evaluate_and_compare([s.runs_dir / r["run_id"] for r in c.get("/api/training/runs").json()],
                         s.processed_dir / "synthetic" / "samples.csv", s.comparison_path)
    ev = c.get(f"/api/evaluation/{rid}").json()
    assert ev["metrics"]["confusion_matrix"]["labels"] == ["negative", "neutral", "positive"]
    assert len(c.get("/api/evaluation/comparison").json()["ranking"]) == 5


def test_dataset_endpoints(env):
    c = env[1]
    assert c.get("/api/datasets").json()[0]["name"] == "synthetic"
    st = c.get("/api/datasets/statistics?dataset=synthetic").json()
    assert st["final_samples"] == 242 and set(st["splits"]) == {"train", "val", "test"}
    assert c.get("/api/datasets/processing-log?dataset=synthetic").json()[0]["stage"] == "ingestion"


def test_structured_errors_and_traversal_blocked(env):
    c = env[1]
    r = c.get("/api/training/status/does_not_exist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert c.get("/api/datasets/statistics?dataset=nope").status_code == 404
    assert c.get("/api/training/history/..%2F..%2Fetc").status_code == 404
    assert c.get("/api/datasets/statistics?dataset=..").status_code == 404


def test_train_request_validation(env):
    c = env[1]
    assert c.post("/api/models/train", json={"model": "resnet", "dataset": "synthetic"}).status_code == 422
    assert c.post("/api/models/train", json={"model": "cnn", "dataset": "synthetic", "epochs": 0}).status_code == 422
    assert c.post("/api/models/train", json={"model": "cnn", "dataset": "missing"}).status_code == 404


def test_background_training_job_runs_for_real(env):
    """POST launches a real subprocess; poll until it reports completion from its own status.json."""
    c = env[1]
    r = c.post("/api/models/train", json={"model": "lstm", "dataset": "synthetic", "epochs": 2, "batch_size": 16,
                                           "device": "cpu"})
    assert r.status_code == 202
    job = r.json()["job_id"]
    for _ in range(100):
        st = c.get(f"/api/training/status/{job}").json()
        if st["state"] in {"completed", "failed"}:
            break
        time.sleep(1)
    assert st["state"] == "completed", st
    assert len(c.get(f"/api/training/history/{job}").json()["epochs"]) == 2
