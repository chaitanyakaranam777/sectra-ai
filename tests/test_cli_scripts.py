"""The documented command-line workflow, executed for real on the synthetic fixture."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=240)


def test_import_train_evaluate_workflow(mvsa_like_dir, tmp_path):
    out = tmp_path / "mvsa_single"
    r = run("scripts/import_mvsa.py", "--variant", "single", "--raw", str(mvsa_like_dir), "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["final_samples"] == 242
    r = run("scripts/train.py", "--model", "lstm", "--samples", str(out / "samples.csv"), "--dataset", "cli",
            "--runs", str(tmp_path / "runs"), "--epochs", "2", "--batch-size", "16", "--device", "cpu")
    assert r.returncode == 0, r.stderr
    run_dir = tmp_path / "runs" / "lstm_cli_v1"
    assert (run_dir / "best.pt").is_file()
    r = run("scripts/evaluate.py", "--samples", str(out / "samples.csv"), "--runs", str(run_dir),
            "--out", str(tmp_path / "cmp.json"))
    assert r.returncode == 0, r.stderr
    assert 0 <= json.loads(r.stdout)[0]["accuracy"] <= 1


def test_import_without_dataset_fails_with_instructions(tmp_path):
    r = run("scripts/import_mvsa.py", "--variant", "single", "--raw", str(tmp_path / "absent"))
    assert r.returncode != 0 and "DATASET_IMPORT" in r.stderr
