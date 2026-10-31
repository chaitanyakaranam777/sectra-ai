import json

import numpy as np
import pandas as pd
import pytest
import torch

from ml.evaluation.metrics import compute_metrics
from ml.evaluation.model_comparison import compare_results, evaluate_and_compare, evaluate_run
from ml.training.callbacks import EarlyStopping
from ml.training.losses import build_loss, compute_class_weights
from ml.training.optimizer import build_optimizer, build_scheduler, select_device
from ml.architectures import build_model
from tests.conftest import train_tiny

NAMES = ["cnn", "lstm", "bert", "cnn_lstm", "cnn_bert"]


@pytest.mark.parametrize("name", NAMES)
def test_every_architecture_trains_and_writes_artifacts(runs, name):
    d = runs[name]
    for f in ("best.pt", "last.pt", "history.json", "status.json", "summary.json", "config.json", "architecture.json"):
        assert (d / f).is_file(), f
    hist = json.loads((d / "history.json").read_text())
    assert len(hist["epochs"]) == 6 and 1 <= hist["best_epoch"] <= 6
    e = hist["epochs"][0]
    assert {"train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "duration_s"} <= set(e)
    assert json.loads((d / "status.json").read_text())["state"] == "completed"


@pytest.mark.parametrize("name", NAMES)
def test_training_actually_learns(runs, name):
    """Loss must drop and validation accuracy must beat chance (1/3) on the separable synthetic fixture."""
    ep = json.loads((runs[name] / "history.json").read_text())["epochs"]
    assert ep[-1]["train_loss"] < ep[0]["train_loss"]
    assert max(e["val_acc"] for e in ep) > 0.5


def test_unimodal_signals_are_learned_by_matching_modality(runs):
    best = lambda n: max(e["val_acc"] for e in json.loads((runs[n] / "history.json").read_text())["epochs"])
    assert best("cnn") > 0.8 and best("lstm") > 0.8 and best("cnn_lstm") > 0.8


def test_same_seed_reproduces_training(processed, tiny, tmp_path):
    a = train_tiny("cnn_lstm", tiny, processed[0], tmp_path / "a", epochs=2)
    b = train_tiny("cnn_lstm", tiny, processed[0], tmp_path / "b", epochs=2)
    la = [e["train_loss"] for e in json.loads((a / "history.json").read_text())["epochs"]]
    lb = [e["train_loss"] for e in json.loads((b / "history.json").read_text())["epochs"]]
    assert la == lb


def test_test_split_never_used_in_training(runs, processed):
    cfg = json.loads((runs["cnn"] / "config.json").read_text())
    df = pd.read_csv(processed[0])
    assert cfg["n_train"] == (df.split == "train").sum() and cfg["n_val"] == (df.split == "val").sum()


def test_best_checkpoint_matches_best_epoch(runs):
    from ml.training.checkpoint import load_checkpoint
    hist = json.loads((runs["cnn"] / "history.json").read_text())
    assert load_checkpoint(runs["cnn"] / "best.pt")["epoch"] == hist["best_epoch"]


def test_early_stopping_stops_and_reports_best_epoch(processed, tiny, tmp_path):
    from ml.training.run import load_config, run_training
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs/default.yaml")
    d = run_training("lstm", processed[0], "s", tmp_path, cfg, tiny["models"]["lstm"],
                     dict(epochs=30, batch_size=16, lr=0.0, device="cpu", early_stopping_patience=2, scheduler="none"),
                     dict(max_text_len=16, vocab_min_freq=1))
    s = json.loads((d / "summary.json").read_text())
    assert s["stopped_early"] and s["epochs_run"] < 30


def test_early_stopping_logic():
    es = EarlyStopping(patience=2, mode="max")
    assert es.update(0.5, 1) and not es.update(0.4, 2) and not es.should_stop
    assert not es.update(0.4, 3) and es.should_stop and es.best_epoch == 1


def test_class_weights_balanced_and_loss_weighting():
    w = compute_class_weights([0, 0, 0, 0, 1, 2])
    assert torch.allclose(w, torch.tensor([6 / (3 * 4), 6 / 3, 6 / 3]))
    logits, y = torch.tensor([[2.0, 0, 0], [2.0, 0, 0]]), torch.tensor([0, 1])
    assert build_loss(w)(logits, y) != build_loss(None)(logits, y)
    with pytest.raises(ValueError):
        compute_class_weights([])


def test_optimizer_groups_and_validation():
    m = build_model("cnn", channels=[4, 8], hidden_dim=8)
    for name, cls in (("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)):
        assert isinstance(build_optimizer(m, name), cls)
    opt = build_optimizer(m, "adamw", weight_decay=0.1)
    assert {g["weight_decay"] for g in opt.param_groups} == {0.0, 0.1}
    with pytest.raises(ValueError):
        build_optimizer(m, "sgd")
    for s in ("none", "cosine", "step", "plateau"):
        build_scheduler(opt, s)
    with pytest.raises(ValueError):
        build_scheduler(opt, "magic")


def test_bert_pretrained_lr_scale_and_freezing():
    cfg = dict(vocab_size=30, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    m = build_model("bert", pretrained_name=None, bert_config=cfg)
    lrs = {g["lr"] for g in build_optimizer(m, lr=1e-3, pretrained_lr_scale=0.1).param_groups}
    assert lrs == {1e-3, 1e-4}
    frozen = build_model("bert", pretrained_name=None, bert_config=cfg, freeze_encoder=True)
    assert 0 < frozen.count_parameters(True) < frozen.count_parameters()


def test_device_selection_cpu_and_cuda_error():
    assert select_device("cpu").type == "cpu" and select_device("auto").type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            select_device("cuda")


def test_metrics_match_hand_computation():
    y = np.array([0, 0, 1, 1, 2, 2]); p = np.array([0, 1, 1, 1, 2, 0])
    m = compute_metrics(y, p)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["negative"]["precision"] == pytest.approx(1 / 2)
    assert m["per_class"]["negative"]["recall"] == pytest.approx(1 / 2)
    assert m["per_class"]["neutral"]["precision"] == pytest.approx(2 / 3)
    assert m["f1_macro"] == pytest.approx(np.mean([0.5, 0.8, 2 / 3]))
    assert m["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["roc_auc_ovr_macro"] is None and "precision" in m["classification_report"]


def test_roc_auc_only_when_defined():
    y = np.array([0, 1, 2, 0, 1, 2]); probs = np.eye(3)[y] * 0.7 + 0.1  # rows sum to 1
    assert compute_metrics(y, probs.argmax(1), probs)["roc_auc_ovr_macro"] == pytest.approx(1.0)
    assert compute_metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), np.full((3, 3), 1 / 3))["roc_auc_ovr_macro"] is None


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        compute_metrics([], [])
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0])


def test_evaluate_all_models_on_same_test_set(runs, processed, tmp_path):
    out = tmp_path / "cmp.json"
    cmp = evaluate_and_compare(list(runs.values()), processed[0], out, "test", "cpu")
    assert [r["model"] for r in cmp["ranking"]].__len__() == 5
    assert cmp["n_samples"] == (pd.read_csv(processed[0]).split == "test").sum()
    f1s = [r["f1_macro"] for r in cmp["ranking"]]
    assert f1s == sorted(f1s, reverse=True)
    assert out.is_file() and (runs["cnn"] / "test_metrics.json").is_file()


def test_comparison_refuses_different_eval_sets(runs, processed):
    df = pd.read_csv(processed[0])
    a = evaluate_run(runs["cnn"], df, "test")
    b = evaluate_run(runs["lstm"], df, "val")
    with pytest.raises(ValueError, match="different evaluation sets"):
        compare_results([a, b])
