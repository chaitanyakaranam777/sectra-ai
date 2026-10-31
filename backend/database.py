"""Prediction history storage (SQLAlchemy). Any SQLAlchemy URL works, e.g. postgresql+psycopg://user:pw@host/db."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    mode: Mapped[str] = mapped_column(String(16))
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    model: Mapped[str] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict] = mapped_column(JSON)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    __table_args__ = (Index("ix_predictions_mode_label", "mode", "label"),)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "created_at": self.created_at.isoformat(), "mode": self.mode, "run_id": self.run_id,
                "model": self.model, "label": self.label, "confidence": self.confidence,
                "probabilities": self.probabilities, "text": self.text, "image_name": self.image_name}


class HistoryStore:
    def __init__(self, url: str) -> None:
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_size": 5, "pool_pre_ping": True}
        self.engine = create_engine(url, **kw)
        Base.metadata.create_all(self.engine)
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def add(self, **fields: Any) -> dict[str, Any]:
        with self._sessions.begin() as s:
            row = Prediction(**fields)
            s.add(row)
            s.flush()
            return row.to_dict()

    def list(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        with Session(self.engine) as s:
            rows = s.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset)).all()
            total = s.query(Prediction).count()
            return {"total": total, "items": [r.to_dict() for r in rows]}

    def ping(self) -> bool:
        with self.engine.connect() as c:
            c.exec_driver_sql("SELECT 1")
        return True
