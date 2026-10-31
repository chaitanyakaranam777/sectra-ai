"""Read-only access to everything the dashboard shows: runs, histories, metrics, dataset reports.

Every value returned is read from files written by the pipeline/trainer; nothing is computed from constants.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.settings import Settings

_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class NotFound(Exception):
    """Requested artifact does not exist (maps to HTTP 404)."""


def _safe_child(root: Path, name: str) -> Path:
    if not _SAFE.match(name) or name in {".", ".."}:
        raise NotFound(f"Invalid identifier '{name}'")
    path = (root / name).resolve()
    if root.resolve() not in path.parents:
        raise NotFound(f"Invalid identifier '{name}'")
    return path


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise NotFound(f"{what} not available yet")
    return json.loads(path.read_text())


class ArtifactStore:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    # ---- runs -------------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        d = _safe_child(self.s.runs_dir, run_id)
        if not d.is_dir():
            raise NotFound(f"Run '{run_id}' not found")
        return d

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.s.runs_dir.is_dir():
            return []
        out = []
        for d in sorted(p for p in self.s.runs_dir.iterdir() if p.is_dir()):
            st = json.loads((d / "status.json").read_text()) if (d / "status.json").is_file() else {}
            cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").is_file() else {}
            out.append({"run_id": d.name, "model": cfg.get("model"), "dataset": cfg.get("dataset"),
                        "state": st.get("state", "queued"), "epoch": st.get("epoch"), "total_epochs": st.get("total_epochs"),
                        "best_epoch": st.get("best_epoch"), "best_metric": st.get("best_metric"),
                        "duration_s": st.get("duration_s"), "evaluated": (d / "test_metrics.json").is_file()})
        return out

    def run_status(self, run_id: str) -> dict[str, Any]:
        d = self.run_dir(run_id)
        st = json.loads((d / "status.json").read_text()) if (d / "status.json").is_file() else {"state": "queued"}
        return {"run_id": run_id, **st}

    def run_history(self, run_id: str) -> dict[str, Any]:
        return _read_json(self.run_dir(run_id) / "history.json", "Training history")

    def run_architecture(self, run_id: str) -> dict[str, Any]:
        return _read_json(self.run_dir(run_id) / "architecture.json", "Architecture summary")

    def run_evaluation(self, run_id: str) -> dict[str, Any]:
        return _read_json(self.run_dir(run_id) / "test_metrics.json", "Test evaluation (run scripts/evaluate.py first)")

    def comparison(self) -> dict[str, Any]:
        return _read_json(self.s.comparison_path, "Model comparison")

    # ---- datasets ---------------------------------------------------------------------
    def list_datasets(self) -> list[dict[str, Any]]:
        if not self.s.processed_dir.is_dir():
            return []
        return [{"name": d.name, "samples_csv": str(d / "samples.csv"),
                 "final_samples": json.loads((d / "report.json").read_text()).get("final_samples")}
                for d in sorted(self.s.processed_dir.iterdir()) if (d / "report.json").is_file()]

    def dataset_dir(self, name: str) -> Path:
        d = _safe_child(self.s.processed_dir, name)
        if not (d / "samples.csv").is_file():
            raise NotFound(f"Dataset '{name}' has not been imported (run scripts/import_mvsa.py)")
        return d

    def dataset_statistics(self, name: str) -> dict[str, Any]:
        return _read_json(self.dataset_dir(name) / "report.json", "Dataset report")

    def dataset_log(self, name: str) -> list[dict[str, Any]]:
        p = self.dataset_dir(name) / "processing_log.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
