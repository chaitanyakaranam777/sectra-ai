"""Start training as a separate OS process and report its state from the files it writes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from backend.settings import ROOT, Settings
from ml.training.checkpoint import next_run_dir


class TrainingJobs:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._procs: dict[str, subprocess.Popen] = {}

    def start(self, model: str, dataset: str, samples_csv: Path, overrides: dict[str, Any]) -> str:
        """Launch ``scripts/train.py`` detached from the API process; returns the job id (= run id)."""
        run_dir = next_run_dir(self.s.runs_dir, model, dataset)
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--model", model, "--samples", str(samples_csv),
               "--dataset", dataset, "--config", str(self.s.config_path), "--runs", str(self.s.runs_dir),
               "--run-dir", str(run_dir)]
        flags = {"epochs": "--epochs", "batch_size": "--batch-size", "lr": "--lr", "optimizer": "--optimizer",
                 "device": "--device"}
        for k, flag in flags.items():
            if overrides.get(k) is not None:
                cmd += [flag, str(overrides[k])]
        log = open(run_dir / "train.log", "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(ROOT)})
        (run_dir / "job.json").write_text(json.dumps({"pid": proc.pid, "command": cmd}))
        self._procs[run_dir.name] = proc
        return run_dir.name

    def reconcile(self, run_id: str, status: dict[str, Any]) -> dict[str, Any]:
        """Mark a 'running'/'queued' job as failed if its process has exited without finishing."""
        proc: Optional[subprocess.Popen] = self._procs.get(run_id)
        if proc is not None and proc.poll() is not None and status.get("state") in {"queued", "running"}:
            log = (self.s.runs_dir / run_id / "train.log")
            tail = log.read_text()[-600:] if log.is_file() else ""
            return {**status, "state": "failed", "error": status.get("error") or f"process exited ({proc.returncode}): {tail}"}
        return status
