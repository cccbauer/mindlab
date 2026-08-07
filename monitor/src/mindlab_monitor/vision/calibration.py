"""Silhouette-alignment gating: is the meditator positioned well enough to
start? Reuses the same FaceResult the live session will use to anchor the
chest ROI, so calibration and tracking share one code path."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from mindlab_monitor.config import CalibrationSettings
from mindlab_monitor.vision.types import FaceResult

CalibrationStatus = Literal["no_face", "off_center", "too_far", "too_close", "not_frontal", "ready"]


def _frontal_ok(face: FaceResult, tolerance: float = 0.35) -> bool:
    def mean_x(points: tuple[tuple[float, float], ...]) -> float:
        return sum(p[0] for p in points) / len(points)

    center = face.bbox.center_x
    d_left = abs(mean_x(face.eye_points.left) - center)
    d_right = abs(mean_x(face.eye_points.right) - center)
    total = d_left + d_right
    if total == 0:
        return True
    return abs(d_left - d_right) / total <= tolerance


def evaluate_calibration(face: FaceResult | None, settings: CalibrationSettings) -> CalibrationStatus:
    if face is None or face.confidence < settings.min_confidence:
        return "no_face"
    if abs(face.bbox.center_x - 0.5) > settings.center_tolerance or abs(face.bbox.center_y - 0.5) > settings.center_tolerance:
        return "off_center"
    lo, hi = settings.target_face_width_range
    if face.bbox.width < lo:
        return "too_far"
    if face.bbox.width > hi:
        return "too_close"
    if not _frontal_ok(face):
        return "not_frontal"
    return "ready"


@dataclass(frozen=True)
class CalibrationState:
    status: CalibrationStatus
    can_start: bool  # True once "ready" has held, OR the manual override window has elapsed
    override_available: bool  # True once override_after_seconds has elapsed, regardless of status
    face_width_fraction: float | None  # measured value, for on-screen diagnostics while tuning thresholds


class CalibrationTracker:
    """Stateful debounce: `status` must hold continuously for `hold_seconds`
    before `can_start` flips true, and a manual override becomes available
    after `override_after_seconds` regardless of status (so bad lighting /
    glasses glare / unusual seating can't permanently block Start)."""

    def __init__(self, settings: CalibrationSettings) -> None:
        self._settings = settings
        self._first_seen_ts: float | None = None
        self._ready_since: float | None = None

    def update(self, face: FaceResult | None) -> CalibrationState:
        now = time.time()
        if self._first_seen_ts is None:
            self._first_seen_ts = now

        status = evaluate_calibration(face, self._settings)

        if status == "ready":
            if self._ready_since is None:
                self._ready_since = now
        else:
            self._ready_since = None

        held_ready = self._ready_since is not None and (now - self._ready_since) >= self._settings.hold_seconds
        override_available = (now - self._first_seen_ts) >= self._settings.override_after_seconds

        return CalibrationState(
            status=status,
            can_start=held_ready or override_available,
            override_available=override_available,
            face_width_fraction=face.bbox.width if face is not None else None,
        )

    def reset(self) -> None:
        self._first_seen_ts = None
        self._ready_since = None
