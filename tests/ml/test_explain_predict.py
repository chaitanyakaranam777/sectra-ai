import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from ml.explainability.gradcam import GradCAM, overlay_heatmap
from ml.explainability.text_attribution import attribute_tokens
from ml.inference.predictor import Predictor, PredictorCache
from ml.training.checkpoint import load_model_from_checkpoint


@pytest.fixture(scope="module")
def sample(processed):
    return pd.read_csv(processed[0]).query("split == 'test'").iloc[0]


@pytest.mark.parametrize("name", ["cnn", "cnn_lstm", "cnn_bert"])
def test_gradcam_shape_range_and_hooks_removed(runs, name):
    model, _ = load_model_from_checkpoint(runs[name] / "best.pt")
    n_hooks = len(model.gradcam_layer()._forward_hooks)
    cam = GradCAM(model)
    assert len(model.gradcam_layer()._forward_hooks) == n_hooks + 1
    inputs = {"image": torch.randn(1, 3, 32, 32), "input_ids": torch.randint(5, 20, (1, 16)),
              "attention_mask": torch.ones(1, 16, dtype=torch.long)}
    out = cam({k: v for k, v in inputs.items() if k in model.required_inputs})
    assert out["cam"].shape == (32, 32) and 0 <= out["cam"].min() and out["cam"].max() <= 1
    assert out["probs"].sum() == pytest.approx(1.0, abs=1e-5)
    cam.close()
    assert len(model.gradcam_layer()._forward_hooks) == n_hooks


def test_gradcam_unavailable_on_text_model(runs):
    model, _ = load_model_from_checkpoint(runs["lstm"] / "best.pt")
    with pytest.raises(NotImplementedError):
        GradCAM(model)


def test_gradcam_highlights_are_not_constant_and_overlay_works(runs):
    model, _ = load_model_from_checkpoint(runs["cnn"] / "best.pt")
    cam = GradCAM(model)(  {"image": torch.randn(1, 3, 32, 32)})["cam"]
    assert cam.std() > 0
    img = np.full((32, 32, 3), 100, np.uint8)
    out = overlay_heatmap(img, cam)
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.array_equal(out, img)
    with pytest.raises(ValueError):
        overlay_heatmap(img, np.zeros((8, 8)))


@pytest.mark.parametrize("name", ["lstm", "bert", "cnn_lstm", "cnn_bert"])
def test_token_attribution(runs, name):
    pred = Predictor(runs[name] / "best.pt")
    inputs = pred.prepare("i love this great day", Image.new("RGB", (40, 40), (200, 60, 40)) if pred.modality == "multimodal" else None)
    out = attribute_tokens(pred.model, inputs)
    mask = inputs["attention_mask"][0].numpy()
    assert out["scores"].shape == mask.shape
    assert np.all(out["scores"][mask == 0] == 0) and out["normalized"].max() == pytest.approx(1.0)
    assert np.isfinite(out["scores"]).all()


def test_token_attribution_unavailable_on_image_model(runs):
    model, _ = load_model_from_checkpoint(runs["cnn"] / "best.pt")
    with pytest.raises(NotImplementedError):
        attribute_tokens(model, {"image": torch.randn(1, 3, 32, 32)})


@pytest.mark.parametrize("name,uses_text,uses_image", [("cnn", False, True), ("lstm", True, False), ("bert", True, False),
                                                      ("cnn_lstm", True, True), ("cnn_bert", True, True)])
def test_predictor_modes(runs, sample, name, uses_text, uses_image):
    p = Predictor(runs[name] / "best.pt")
    kw = {}
    if uses_text:
        kw["text"] = sample.text_raw
    if uses_image:
        kw["image"] = sample.image_path
    r = p.predict(**kw)
    assert r["label"] in {"negative", "neutral", "positive"} and 0 <= r["confidence"] <= 1
    assert sum(r["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert r["confidence"] == max(r["probabilities"].values())


def test_predictor_input_validation(runs, sample):
    multi = Predictor(runs["cnn_lstm"] / "best.pt")
    with pytest.raises(ValueError, match="text"):
        multi.predict(image=sample.image_path)
    with pytest.raises(ValueError, match="image"):
        multi.predict(text="hello")
    with pytest.raises(ValueError):
        Predictor(runs["lstm"] / "best.pt").predict(text="   ")


def test_predictor_cache_loads_once(runs):
    cache = PredictorCache()
    assert cache.get(runs["cnn"] / "best.pt") is cache.get(runs["cnn"] / "best.pt")


def test_predictor_accepts_pil_and_matches_path(runs, sample):
    p = Predictor(runs["cnn"] / "best.pt")
    a = p.predict(image=sample.image_path)["probabilities"]
    b = p.predict(image=Image.open(sample.image_path))["probabilities"]
    assert a == pytest.approx(b)
