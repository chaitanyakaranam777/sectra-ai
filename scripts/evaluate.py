"""Score runs on the held-out split and write a comparison.

    python scripts/evaluate.py --samples data/processed/mvsa_single/samples.csv --runs artifacts/runs/cnn_mvsa_single_v1 ...
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.evaluation.model_comparison import evaluate_and_compare  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/comparison.json")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    print(json.dumps(evaluate_and_compare(a.runs, a.samples, a.out, a.split, a.device)["ranking"], indent=2))


if __name__ == "__main__":
    main()
