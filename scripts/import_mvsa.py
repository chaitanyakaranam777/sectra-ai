"""Validate and preprocess a manually downloaded MVSA dataset.

    python scripts/import_mvsa.py --variant single   --raw data/raw/MVSA_Single
    python scripts/import_mvsa.py --variant multiple --raw data/raw/MVSA_Multiple
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.run_pipeline import run_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["single", "multiple"], required=True)
    ap.add_argument("--raw", required=True, help="folder holding labelResultAll.txt and data/")
    ap.add_argument("--out", default=None, help="default: data/processed/mvsa_<variant>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-image-check", action="store_true")
    a = ap.parse_args()
    out = a.out or f"data/processed/mvsa_{a.variant}"
    report = run_pipeline(a.raw, a.variant, out, a.seed, check_images=not a.skip_image_check)
    print(json.dumps({k: report[k] for k in ("dataset", "final_samples", "class_distribution", "splits")}, indent=2))


if __name__ == "__main__":
    main()
