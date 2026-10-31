"""High-level entry point: config + processed samples -> trained, versioned run directory."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from ml.architectures import MODEL_REGISTRY, build_model
from ml.data.dataset import SentimentDataset, make_loader
from ml.data.text_processing import BertTokenizerEncoder, WordVocabEncoder
from ml.training.callbacks import write_json_atomic
from ml.training.checkpoint import next_run_dir
from ml.training.losses import compute_class_weights
from ml.training.trainer import TrainConfig, Trainer, set_seed

_TRAIN_FIELDS = set(TrainConfig.__dataclass_fields__)


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def run_training(model_name: str, samples_csv: str | Path, dataset: str, runs_root: str | Path,
                 config: dict[str, Any], model_overrides: Optional[dict[str, Any]] = None,
                 train_overrides: Optional[dict[str, Any]] = None, data_overrides: Optional[dict[str, Any]] = None,
                 run_dir: Optional[str | Path] = None) -> Path:
    """Train ``model_name`` on the train/val splits of ``samples_csv`` and return the run directory.

    ``run_dir`` may be pre-created by the caller (the API does this so the job id is known up front).
    The test split is never read. Vocabulary (LSTM models) is built from training text only.
    """
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]
    cfg = copy.deepcopy(config)
    data_cfg = {**cfg["data"], **(data_overrides or {})}
    train_cfg = {**cfg["training"], **cfg.get("training_overrides", {}).get(model_name, {}), **(train_overrides or {})}
    model_params = {**cfg["models"][model_name], **(model_overrides or {})}
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    df = pd.read_csv(samples_csv)
    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    if train_df.empty or val_df.empty:
        raise ValueError("samples file needs non-empty 'train' and 'val' splits")

    encoder = None
    if spec.text_encoder == "word":
        encoder = WordVocabEncoder.build(train_df["text_clean"].fillna(""), data_cfg["max_text_len"],
                                         data_cfg["vocab_max_size"], data_cfg["vocab_min_freq"])
        model_params["vocab_size"] = encoder.vocab_size
    elif spec.text_encoder == "bert":
        encoder = BertTokenizerEncoder(data_cfg["bert_tokenizer"], data_cfg["max_text_len"])
    model = build_model(model_name, **model_params)

    bs = int(train_cfg.pop("batch_size"))
    use_weights = bool(train_cfg.pop("use_class_weights", True))
    tkw = {k: v for k, v in train_cfg.items() if k in _TRAIN_FIELDS}
    tkw.setdefault("seed", seed)
    tcfg = TrainConfig(**tkw)

    size = int(data_cfg["image_size"])
    train_ds = SentimentDataset(train_df, model.modality, encoder, size, train=True)
    val_ds = SentimentDataset(val_df, model.modality, encoder, size)
    weights = compute_class_weights(train_ds.labels) if use_weights else None

    run_dir = Path(run_dir) if run_dir else next_run_dir(runs_root, model_name, dataset)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "config.json", {"model": model_name, "dataset": dataset, "data": data_cfg,
                                                "training": train_cfg | {"batch_size": bs, "use_class_weights": use_weights},
                                                "model_params": model.config, "samples_csv": str(samples_csv),
                                                "n_train": len(train_ds), "n_val": len(val_ds)})
    write_json_atomic(run_dir / "architecture.json", model.summary())
    extra = {"text_encoder": encoder.state() if encoder else None, "image_size": size, "dataset": dataset}
    trainer = Trainer(model, make_loader(train_ds, bs, True, seed=seed), make_loader(val_ds, bs),
                      tcfg, run_dir, weights, extra)
    trainer.fit()
    return run_dir
