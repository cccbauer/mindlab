from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionConfig:
    session_id: str
    planned_duration_seconds: float
    target_eye_state: str  # "closed" | "open" | "none" — snapshot of settings at session start


@dataclass(frozen=True)
class SessionSample:
    """One analysis tick's worth of raw signal, kept for the summary chart /
    debug tooling. Not queried relationally — stored in a JSONL sidecar."""

    timestamp: float
    ear: float | None
    eyes_open: bool | None
    stillness_score: float | None
    breath_bpm: float | None


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    started_at: float  # epoch seconds
    duration_seconds: float
    sample_count: int
    eye_score: float | None
    movement_score: float | None
    breath_score: float | None
    stability_score: float
    avg_breath_bpm: float | None
    eyes_deviation_ratio: float | None
    eyes_transitions_per_min: float | None
