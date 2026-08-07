"""Eye-aspect-ratio (EAR) and eye-state-stability scoring.

Eyes-closed is the *expected* state during most meditation practice, so the
score below doesn't reward "closed" directly — it penalizes sustained
deviation from whatever the configured target state is, and penalizes
restlessness (frequent sustained open/closed transitions), while ignoring
ordinary blinks via a minimum-dwell-time debounce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mindlab_monitor.config import EyeSettings
from mindlab_monitor.vision.types import EyePoints


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(points: tuple[tuple[float, float], ...]) -> float:
    """Classic 6-point EAR: points ordered [outer, top1, top2, inner, bottom1, bottom2]."""
    p1, p2, p3, p4, p5, p6 = points
    return (_dist(p2, p6) + _dist(p3, p5)) / (2 * _dist(p1, p4) + 1e-9)


def combined_ear(eye_points: EyePoints) -> float:
    return (eye_aspect_ratio(eye_points.left) + eye_aspect_ratio(eye_points.right)) / 2


def eyes_open_from_ear(ear: float, closed_threshold: float) -> bool:
    return ear >= closed_threshold


@dataclass(frozen=True)
class EyeSample:
    timestamp: float
    eyes_open: bool


@dataclass(frozen=True)
class EyeStabilityResult:
    scored_seconds: float
    deviation_seconds: float
    deviation_ratio: float
    transition_count: int
    transitions_per_min: float
    score: float | None  # None when target_state == "none"


def _sustained_runs(
    samples: list[EyeSample], min_dwell_seconds: float
) -> list[tuple[bool, float, float]]:
    """Collapse raw per-frame samples into (state, start_ts, end_ts) runs,
    absorbing any run shorter than `min_dwell_seconds` into its neighbor
    (this is what filters out ordinary blinks)."""
    if not samples:
        return []

    raw_runs: list[tuple[bool, float, float]] = []
    state = samples[0].eyes_open
    start = samples[0].timestamp
    for s in samples[1:]:
        if s.eyes_open != state:
            raw_runs.append((state, start, s.timestamp))
            state = s.eyes_open
            start = s.timestamp
    raw_runs.append((state, start, samples[-1].timestamp))

    sustained: list[tuple[bool, float, float]] = []
    cur_state, cur_start, cur_end = raw_runs[0]
    for state, start, end in raw_runs[1:]:
        duration = end - start
        if duration >= min_dwell_seconds and state != cur_state:
            sustained.append((cur_state, cur_start, cur_end))
            cur_state, cur_start, cur_end = state, start, end
        else:
            cur_end = end
    sustained.append((cur_state, cur_start, cur_end))
    return sustained


def compute_eye_stability(
    samples: list[EyeSample], settings: EyeSettings
) -> EyeStabilityResult | None:
    if not samples:
        return None
    if settings.target_state == "none":
        return None

    target_open = settings.target_state == "open"
    session_start = samples[0].timestamp
    settle_end = session_start + settings.settle_seconds
    last_ts = samples[-1].timestamp

    scored_seconds = max(0.0, last_ts - settle_end)
    if scored_seconds == 0.0:
        return EyeStabilityResult(0.0, 0.0, 0.0, 0, 0.0, None)

    runs = _sustained_runs(samples, settings.min_dwell_seconds)

    deviation_seconds = 0.0
    transition_count = 0
    for i, (state, start, end) in enumerate(runs):
        overlap = max(0.0, min(end, last_ts) - max(start, settle_end))
        if overlap <= 0:
            continue
        if state != target_open:
            deviation_seconds += overlap
        if i > 0 and start >= settle_end:
            transition_count += 1

    deviation_ratio = deviation_seconds / scored_seconds
    transitions_per_min = transition_count / (scored_seconds / 60.0)

    deviation_penalty = min(1.0, deviation_ratio) * settings.deviation_penalty_cap
    transition_penalty = min(1.0, transitions_per_min / settings.reference_transitions_per_min) * (
        settings.transition_penalty_cap
    )
    score = max(0.0, 100.0 - deviation_penalty - transition_penalty)

    return EyeStabilityResult(
        scored_seconds=scored_seconds,
        deviation_seconds=deviation_seconds,
        deviation_ratio=deviation_ratio,
        transition_count=transition_count,
        transitions_per_min=transitions_per_min,
        score=score,
    )
