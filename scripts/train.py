"""Train one architecture on a processed dataset.

    python scripts/train.py --model cnn_lstm --samples data/processed/mvsa_single/samples.csv --dataset mvsa_single
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.architectures import MODEL_REGISTRY  # noqa: E402
from ml.training.run import load_config, run_training  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODEL_REGISTRY), required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", default="artifacts/runs")
    ap.add_argument("--run-dir", default=None, help="use this directory instead of a new versioned one")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--optimizer", choices=["adam", "adamw"])
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()
    overrides = {k: v for k, v in {"epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
                                   "optimizer": a.optimizer, "device": a.device}.items() if v is not None}
    run_dir = run_training(a.model, a.samples, a.dataset, a.runs, load_config(a.config), train_overrides=overrides,
                          run_dir=a.run_dir)
    print(f"Run saved to {run_dir}")


if __name__ == "__main__":
    main()
