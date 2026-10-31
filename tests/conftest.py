"""Shared fixtures.

IMPORTANT: everything generated here is a SYNTHETIC TEST FIXTURE laid out like MVSA so the parser and
pipeline can be exercised offline. It is not MVSA data and no result obtained from it is a dataset result.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.training.run import load_config, run_training  # noqa: E402
from pipelines.run_pipeline import run_pipeline  # noqa: E402

CLASS_WORDS = {
    "positive": ["love", "happy", "great", "wonderful", "amazing"],
    "negative": ["hate", "sad", "terrible", "awful", "angry"],
    "neutral": ["today", "meeting", "table", "report", "weather"],
}
FILLER = ["just", "really", "so", "very", "here", "now"]
COLORS = {"negative": (40, 40, 200), "neutral": (120, 120, 120), "positive": (200, 60, 40)}
N_PER_CLASS = 80


def _write_image(path: Path, label: str, rng: np.random.Generator) -> None:
    arr = rng.normal(COLORS[label], 25, size=(48, 48, 3)).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture(scope="session")
def mvsa_like_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic MVSA-Single-layout folder with deliberate defects (see comments)."""
    root = tmp_path_factory.mktemp("synthetic_mvsa_single")
    data = root / "data"
    data.mkdir()
    rng, py = np.random.default_rng(0), random.Random(0)
    lines = ["ID\ttext,image"]
    i = 0
    for label, words in CLASS_WORDS.items():
        for _ in range(N_PER_CLASS):
            i += 1
            text = " ".join(py.sample(words, 3) + py.sample(FILLER, 2) + [f"post{i}"])
            (data / f"{i}.txt").write_text(text, encoding="utf-8")
            _write_image(data / f"{i}.jpg", label, rng)
            lines.append(f"{i}\t{label},{label}")
    n_ok = i
    # defects, each with a known outcome
    i += 1; (data / f"{i}.txt").write_text("no image for me"); lines.append(f"{i}\tpositive,positive")   # missing image
    i += 1; (data / f"{i}.txt").write_text("   "); _write_image(data / f"{i}.jpg", "positive", rng); lines.append(f"{i}\tpositive,positive")  # empty text
    i += 1; (data / f"{i}.txt").write_text("conflicting labels"); _write_image(data / f"{i}.jpg", "positive", rng); lines.append(f"{i}\tpositive,negative")  # conflict
    i += 1; (data / f"{i}.txt").write_text("broken image"); (data / f"{i}.jpg").write_bytes(b"not an image"); lines.append(f"{i}\tneutral,neutral")  # corrupt
    i += 1; (data / f"{i}.txt").write_text("text neutral image positive love"); _write_image(data / f"{i}.jpg", "positive", rng); lines.append(f"{i}\tneutral,positive")  # resolves to positive
    # exact retweet duplicate of sample 1 (same text, same image bytes) -> must stay in the same split
    i += 1
    (data / f"{i}.txt").write_text((data / "1.txt").read_text()); (data / f"{i}.jpg").write_bytes((data / "1.jpg").read_bytes())
    lines.append(f"{i}\tnegative,negative")
    (root / "labelResultAll.txt").write_text("\n".join(lines) + "\n")
    (root / "N_OK").write_text(str(n_ok))
    return root


@pytest.fixture(scope="session")
def processed(mvsa_like_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("processed")
    report = run_pipeline(mvsa_like_dir, "single", out, seed=42)
    return out / "samples.csv", report


def write_tiny_vocab(path: Path) -> int:
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "url", "user", "text", "image", "no", "me", "for", "broken"]
    words += [w for ws in CLASS_WORDS.values() for w in ws] + FILLER
    path.write_text("\n".join(words) + "\n")
    return len(words)


@pytest.fixture(scope="session")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> dict:
    vocab = tmp_path_factory.mktemp("bert_vocab") / "vocab.txt"
    vs = write_tiny_vocab(vocab)
    bert_cfg = dict(vocab_size=vs, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=64, max_position_embeddings=32)
    ch = [8, 16, 32, 64]
    models = {
        "cnn": dict(channels=ch, hidden_dim=32),
        "lstm": dict(embed_dim=32, hidden_dim=32, head_hidden_dim=32),
        "bert": dict(pretrained_name=None, bert_config=bert_cfg, head_hidden_dim=32),
        "cnn_lstm": dict(image_channels=ch, embed_dim=32, lstm_hidden_dim=32, proj_dim=32, fusion_hidden=[32, 16]),
        "cnn_bert": dict(image_channels=ch, pretrained_name=None, bert_config=bert_cfg, proj_dim=32, fusion_hidden=[32, 16]),
    }
    return {"vocab": str(vocab), "bert_config": bert_cfg, "models": models}


def train_tiny(name: str, tiny: dict, samples: Path, runs: Path, epochs: int = 6, seed_cfg: int = 42) -> Path:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    cfg["seed"] = seed_cfg
    return run_training(
        name, samples, "synthetic", runs, cfg, model_overrides=tiny["models"][name],
        train_overrides=dict(epochs=epochs, batch_size=16, lr=0.003, pretrained_lr_scale=1.0, weight_decay=0.0001,
                             device="cpu", early_stopping_patience=epochs, scheduler="none"),
        data_overrides=dict(image_size=32, max_text_len=16, vocab_min_freq=1, bert_tokenizer=tiny["vocab"]))


@pytest.fixture(scope="session")
def runs(processed, tiny, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """All five architectures trained for real (tiny sizes) on the synthetic fixture."""
    root = tmp_path_factory.mktemp("runs")
    return {n: train_tiny(n, tiny, processed[0], root) for n in tiny["models"]}
