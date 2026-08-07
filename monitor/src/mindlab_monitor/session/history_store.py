"""Local session history: SQLite for summaries (enables trend queries later),
a JSONL sidecar per session for the full per-tick time series (only ever
read back for charts/debugging, never queried relationally)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from mindlab_monitor.session.models import SessionSample, SessionSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    eye_score REAL,
    movement_score REAL,
    breath_score REAL,
    stability_score REAL NOT NULL,
    avg_breath_bpm REAL,
    eyes_deviation_ratio REAL,
    eyes_transitions_per_min REAL
);
"""

_COLUMNS = [
    "session_id",
    "started_at",
    "duration_seconds",
    "sample_count",
    "eye_score",
    "movement_score",
    "breath_score",
    "stability_score",
    "avg_breath_bpm",
    "eyes_deviation_ratio",
    "eyes_transitions_per_min",
]


class HistoryStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._sessions_dir = data_dir / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = data_dir / "sessions.db"
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def save_session(self, summary: SessionSummary, samples: list[SessionSample]) -> None:
        sidecar_path = self._sessions_dir / f"{summary.session_id}.jsonl"
        with sidecar_path.open("w") as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample)) + "\n")

        row = asdict(summary)
        placeholders = ", ".join(f":{c}" for c in _COLUMNS)
        columns = ", ".join(_COLUMNS)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO sessions ({columns}) VALUES ({placeholders})",
                row,
            )

    def list_summaries(self, limit: int = 50) -> list[SessionSummary]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [SessionSummary(**{c: row[c] for c in _COLUMNS}) for row in rows]

    def load_samples(self, session_id: str) -> list[SessionSample]:
        sidecar_path = self._sessions_dir / f"{session_id}.jsonl"
        if not sidecar_path.exists():
            return []
        samples = []
        with sidecar_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(SessionSample(**json.loads(line)))
        return samples
