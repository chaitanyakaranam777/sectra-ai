import base64
import io

import pandas as pd
import pytest
from PIL import Image

from tests.backend.test_api import env  # noqa: F401  (fixture)


@pytest.fixture()
def sample(processed):
    r = pd.read_csv(processed[0]).query("split == 'test'").iloc[0]
    return r.text_raw, open(r.image_path, "rb").read()


def img_bytes(fmt="PNG", size=(64, 64)):
    b = io.BytesIO(); Image.new("RGB", size, (200, 60, 40)).save(b, fmt); return b.getvalue()


def test_text_prediction_is_real_and_saved(env, sample):
    c = env[1]
    r = c.post("/api/predict/text", json={"text": sample[0]})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["label"] in {"negative", "neutral", "positive"} and sum(j["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert j["model"] in {"lstm", "bert"} and j["prediction_id"] == 1


def test_image_and_multimodal_prediction(env, sample):
    c = env[1]
    r = c.post("/api/predict/image", files={"file": ("a.jpg", sample[1], "image/jpeg")})
    assert r.status_code == 200 and r.json()["model"] == "cnn"
    r = c.post("/api/predict/multimodal", data={"text": sample[0]}, files={"file": ("a.jpg", sample[1], "image/jpeg")})
    assert r.status_code == 200 and r.json()["model"] in {"cnn_lstm", "cnn_bert"}


def test_best_validation_run_is_selected_and_explicit_run_honoured(env, sample):
    c = env[1]
    runs = {r["run_id"]: r for r in c.get("/api/training/runs").json() if r["model"] in {"lstm", "bert"}}
    best = max(runs.values(), key=lambda r: r["best_metric"])["run_id"]
    assert c.post("/api/predict/text", json={"text": sample[0]}).json()["run_id"] == best
    other = next(k for k in runs if k != best)
    assert c.post("/api/predict/text", json={"text": sample[0], "run_id": other}).json()["run_id"] == other
    assert c.post("/api/predict/text", json={"text": "x", "run_id": "cnn_synthetic_v1"}).status_code == 404  # wrong modality


def test_history_lists_newest_first_with_paging(env, sample):
    c = env[1]
    for _ in range(3):
        c.post("/api/predict/text", json={"text": sample[0]})
    h = c.get("/api/predictions/history?limit=2").json()
    assert h["total"] == 3 and [i["id"] for i in h["items"]] == [3, 2]
    assert c.get("/api/predictions/history?limit=0").status_code == 422


@pytest.mark.parametrize("payload,ctype,status", [
    (b"not an image", "image/png", 422),
    (img_bytes("PNG"), "image/jpeg", 415),            # content does not match declared type
    (img_bytes("PNG"), "image/gif", 415),
    (img_bytes("PNG", (8, 8)), "image/png", 422),      # too small
    (b"\x89PNG" + b"0" * 300_000, "image/png", 413),   # over size limit
])
def test_upload_validation(env, payload, ctype, status):
    r = env[1].post("/api/predict/image", files={"file": ("x", payload, ctype)})
    assert r.status_code == status and r.json()["error"]["code"] == "inference_error"


def test_input_validation(env, sample):
    c = env[1]
    assert c.post("/api/predict/text", json={"text": ""}).status_code == 422
    assert c.post("/api/predict/text", json={"text": "a" * 2001}).status_code == 422
    assert c.post("/api/predict/multimodal", data={"text": "hi"}).status_code == 422          # image missing
    assert c.post("/api/predict/multimodal", data={"text": " "}, files={"file": ("a.png", img_bytes(), "image/png")}).status_code == 422


def test_no_trained_model_gives_409(tmp_path, processed):
    from fastapi.testclient import TestClient
    from backend.main import create_app
    from backend.settings import Settings
    s = Settings(runs_dir=tmp_path / "r", processed_dir=tmp_path / "p", comparison_path=tmp_path / "c.json",
                 database_url=f"sqlite:///{tmp_path / 'd.db'}")
    r = TestClient(create_app(s)).post("/api/predict/text", json={"text": "hello"})
    assert r.status_code == 409 and "Train one first" in r.json()["error"]["message"]


def test_explain_multimodal_returns_separate_evidence(env, sample):
    r = env[1].post("/api/explain/multimodal", data={"text": sample[0]}, files={"file": ("a.jpg", sample[1], "image/jpeg")})
    assert r.status_code == 200, r.text
    j = r.json()
    png = Image.open(io.BytesIO(base64.b64decode(j["image_explanation"]["overlay_png_base64"])))
    assert png.size == (32, 32) or png.size[0] > 0
    toks = j["text_explanation"]["tokens"]
    assert toks and toks[0]["token"] and all(0 <= t["weight"] <= 1 for t in toks)
    assert len(j["limitations"]) >= 3


def test_explain_text_only_has_no_image_part(env, sample):
    j = env[1].post("/api/explain/text", data={"text": sample[0]}).json()
    assert "text_explanation" in j and "image_explanation" not in j
