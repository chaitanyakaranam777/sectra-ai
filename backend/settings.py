"""Environment-driven settings (no secrets are hard-coded)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    runs_dir: Path = field(default_factory=lambda: Path(os.getenv("SENTRA_RUNS_DIR", ROOT / "artifacts" / "runs")))
    processed_dir: Path = field(default_factory=lambda: Path(os.getenv("SENTRA_PROCESSED_DIR", ROOT / "data" / "processed")))
    comparison_path: Path = field(default_factory=lambda: Path(os.getenv("SENTRA_COMPARISON", ROOT / "artifacts" / "comparison.json")))
    config_path: Path = field(default_factory=lambda: Path(os.getenv("SENTRA_CONFIG", ROOT / "configs" / "default.yaml")))
    database_url: str = field(default_factory=lambda: os.getenv("SENTRA_DATABASE_URL", f"sqlite:///{ROOT / 'artifacts' / 'sentra.db'}"))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("SENTRA_MAX_UPLOAD_BYTES", 5 * 1024 * 1024)))
    max_text_chars: int = 2000
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        os.getenv("SENTRA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")))
