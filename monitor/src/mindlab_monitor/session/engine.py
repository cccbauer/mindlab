"""Orchestrates one meditation session: subscribes to the shared
CameraPipeline, runs EAR + shared movement/breath analysis per tick,
accumulates samples, and on completion computes the stability score and
persists the session to history.

Runs entirely on the pipeline's background thread — `on_tick`/`on_complete`
callbacks fire from that thread, so UI code must marshal them onto the Flet
page thread itself (e.g. via `page.pubsub`), not call `control.update()`
directly here.
"""

from __future__ import annotations

import statistics
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable

import numpy as np

from mindlab_monitor.config import Settings
from mindlab_monitor.session.history_store import HistoryStore
from mindlab_monitor.session.models import SessionConfig, SessionSample, SessionSummary
from mindlab_monitor.session.stability import breath_regularity_score, compute_stability_score
from mindlab_monitor.vision.eyes import EyeSample, combined_ear, compute_eye_stability, eyes_open_from_ear
from mindlab_monitor.vision.movement_breath import MotionTracker
from mindlab_monitor.vision.pipeline import CameraPipeline
from mindlab_monitor.vision.types import FaceResult


@dataclass(frozen=True)
class EngineTick:
    elapsed_seconds: float
    remaining_seconds: float
    face_detected: bool
    ear: float | None
    eyes_open: bool | None
    stillness_score: float | None
    breath_bpm: float | None
    live_stability_score: float


class SessionEngine:
    def __init__(self, pipeline: CameraPipeline, settings: Settings, history_store: HistoryStore) -> None:
        self._pipeline = pipeline
        self._settings = settings
        self._history_store = history_store
        self._lock = threading.Lock()

        self._unsubscribe: Callable[[], None] | None = None
        self._session_id: str | None = None
        self._start_ts: float = 0.0
        self._planned_duration: float = 0.0
        self._samples: list[SessionSample] = []
        self._breath_estimates: list[float] = []
        self._motion_tracker: MotionTracker | None = None
        self._on_tick: Callable[[EngineTick], None] | None = None
        self._on_complete: Callable[[SessionSummary], None] | None = None
        self._finished = True

    def start_session(
        self,
        planned_duration_seconds: float,
        on_tick: Callable[[EngineTick], None] | None = None,
        on_complete: Callable[[SessionSummary], None] | None = None,
    ) -> SessionConfig:
        with self._lock:
            self._session_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
            self._start_ts = time.time()
            self._planned_duration = planned_duration_seconds
            self._samples = []
            self._breath_estimates = []
            self._motion_tracker = MotionTracker(self._settings.motion)
            self._on_tick = on_tick
            self._on_complete = on_complete
            self._finished = False

        self._unsubscribe = self._pipeline.subscribe(self._handle_tick)
        return SessionConfig(
            session_id=self._session_id,
            planned_duration_seconds=planned_duration_seconds,
            target_eye_state=self._settings.eyes.target_state,
        )

    def stop_session(self) -> SessionSummary:
        return self._finalize()

    def _handle_tick(self, frame: np.ndarray, face_result: FaceResult | None, ts: float) -> None:
        with self._lock:
            if self._finished:
                return
            assert self._motion_tracker is not None

            elapsed = ts - self._start_ts
            ear = None
            eyes_open = None
            if face_result is not None:
                ear = combined_ear(face_result.eye_points)
                eyes_open = eyes_open_from_ear(ear, self._settings.eyes.ear_closed_threshold)

            motion_tick = self._motion_tracker.update(
                frame, face_result.bbox if face_result is not None else None, ts
            )

            self._samples.append(
                SessionSample(
                    timestamp=ts,
                    ear=ear,
                    eyes_open=eyes_open,
                    stillness_score=motion_tick.stillness_score,
                    breath_bpm=motion_tick.breath_bpm,
                )
            )
            if motion_tick.breath_bpm is not None and (
                not self._breath_estimates or self._breath_estimates[-1] != motion_tick.breath_bpm
            ):
                self._breath_estimates.append(motion_tick.breath_bpm)

            eye_score = self._current_eye_score()
            movement_score = self._current_movement_score()
            breath_score = breath_regularity_score(self._breath_estimates)
            live_stability = compute_stability_score(
                eye_score, movement_score, breath_score, self._settings.stability_weights
            )

            remaining = max(0.0, self._planned_duration - elapsed)
            on_tick = self._on_tick

        if on_tick is not None:
            on_tick(
                EngineTick(
                    elapsed_seconds=elapsed,
                    remaining_seconds=remaining,
                    face_detected=face_result is not None,
                    ear=ear,
                    eyes_open=eyes_open,
                    stillness_score=motion_tick.stillness_score,
                    breath_bpm=motion_tick.breath_bpm,
                    live_stability_score=live_stability,
                )
            )

        if remaining <= 0.0:
            self._finalize()

    def _current_eye_score(self) -> float | None:
        eye_samples = [
            EyeSample(timestamp=s.timestamp, eyes_open=s.eyes_open)
            for s in self._samples
            if s.eyes_open is not None
        ]
        result = compute_eye_stability(eye_samples, self._settings.eyes)
        return result.score if result is not None else None

    def _current_movement_score(self) -> float | None:
        scores = [s.stillness_score for s in self._samples if s.stillness_score is not None]
        return statistics.mean(scores) if scores else None

    def _finalize(self) -> SessionSummary:
        with self._lock:
            if self._finished:
                # Already finalized (e.g. duration elapsed and stop_session()
                # called right after) — nothing to redo.
                assert self._session_id is not None
                existing = self._history_store.list_summaries(limit=1)
                if existing and existing[0].session_id == self._session_id:
                    return existing[0]
            self._finished = True
            session_id = self._session_id
            started_at = self._start_ts
            samples = list(self._samples)
            breath_estimates = list(self._breath_estimates)
            on_complete = self._on_complete

        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

        assert session_id is not None
        eye_score = self._current_eye_score()
        movement_score = self._current_movement_score()
        breath_score = breath_regularity_score(breath_estimates)
        stability_score = compute_stability_score(
            eye_score, movement_score, breath_score, self._settings.stability_weights
        )

        eye_samples = [
            EyeSample(timestamp=s.timestamp, eyes_open=s.eyes_open) for s in samples if s.eyes_open is not None
        ]
        eye_result = compute_eye_stability(eye_samples, self._settings.eyes)

        duration_seconds = (samples[-1].timestamp - started_at) if samples else 0.0
        summary = SessionSummary(
            session_id=session_id,
            started_at=started_at,
            duration_seconds=duration_seconds,
            sample_count=len(samples),
            eye_score=eye_score,
            movement_score=movement_score,
            breath_score=breath_score,
            stability_score=stability_score,
            avg_breath_bpm=statistics.mean(breath_estimates) if breath_estimates else None,
            eyes_deviation_ratio=eye_result.deviation_ratio if eye_result else None,
            eyes_transitions_per_min=eye_result.transitions_per_min if eye_result else None,
        )
        self._history_store.save_session(summary, samples)

        if on_complete is not None:
            on_complete(summary)
        return summary
