"""Injectable settings for the meditation monitor.

Kept as one place for every tunable so the future parent `mindlab` app can
override anything (data location, thresholds, weights) without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TargetEyeState = Literal["closed", "open", "none"]


def _default_data_dir() -> Path:
    return Path.home() / ".mindlab_monitor"


@dataclass
class CalibrationSettings:
    min_confidence: float = 0.5
    center_tolerance: float = 0.18  # fraction of frame width/height
    # Fraction of frame width the face mesh's bounding box (cheek-to-cheek,
    # tighter than the full head) should span. The lower bound especially
    # was found too strict in manual testing — it forced sitting
    # uncomfortably close just to stop being flagged "too far".
    target_face_width_range: tuple[float, float] = (0.10, 0.40)
    hold_seconds: float = 1.5
    override_after_seconds: float = 12.0


@dataclass
class EyeSettings:
    target_state: TargetEyeState = "closed"
    ear_closed_threshold: float = 0.21
    settle_seconds: float = 15.0
    min_dwell_seconds: float = 1.0
    # Penalty caps: 100% sustained-deviation time maxes out `deviation_penalty_cap`;
    # `reference_transitions_per_min` sustained transitions/min maxes out `transition_penalty_cap`.
    deviation_penalty_cap: float = 60.0
    transition_penalty_cap: float = 40.0
    reference_transitions_per_min: float = 5.0


@dataclass
class MotionSettings:
    analysis_hz: float = 12.0
    downscale_width: int = 320
    # x face height, below face bbox. Original 1.4/1.2 defaults landed on the
    # stomach in real testing — a typical webcam bust-shot has the chest
    # starting just below the chin/neck, not over a full face-height down.
    chest_roi_y_offset_factor: float = 0.3
    chest_roi_height_factor: float = 0.9  # x face height
    # EMA weight (0-1) applied to the face bbox before deriving the chest ROI
    # from it — lower means smoother/more lag, higher means more responsive
    # but jitterier. Reduces both visible ROI-box shake and noise it would
    # otherwise inject into the breath signal.
    bbox_smoothing_alpha: float = 0.25
    breath_window_seconds: float = 50.0
    breath_min_seconds_before_estimate: float = 35.0
    breath_update_seconds: float = 5.0
    breath_min_bpm: float = 3.0
    breath_max_bpm: float = 30.0
    # Minimum short-term (stillness_smoothing_seconds) smoothed stillness
    # score required to accept a sample into the breath-signal buffer at
    # all. Brief movement (shifting posture, reaching to end the session)
    # just gets skipped rather than sitting inside the FFT window and
    # getting misread as a fast "breath" — see MotionTracker.update.
    breath_quality_min_stillness: float = 75.0
    # Empirical scale for mean-abs-diff -> 0-100 stillness score; tune against
    # real footage using scripts/debug_movement_breath.py.
    movement_sensitivity: float = 8.0
    stillness_smoothing_seconds: float = 2.0


@dataclass
class StabilityWeights:
    eyes: float = 0.4
    movement: float = 0.4
    breath: float = 0.2


@dataclass
class Settings:
    data_dir: Path = field(default_factory=_default_data_dir)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    eyes: EyeSettings = field(default_factory=EyeSettings)
    motion: MotionSettings = field(default_factory=MotionSettings)
    stability_weights: StabilityWeights = field(default_factory=StabilityWeights)
    default_duration_minutes: int = 10
    # Digital zoom applied uniformly to capture, calibration, and preview
    # (see CameraPipeline._apply_zoom) so a comfortable sitting distance can
    # be made to visually fill the silhouette without physically moving.
    # User-adjustable live via a slider on the setup screen.
    default_camera_zoom: float = 2.0
    bell_start_asset: str = "audio/bell_start.wav"
    bell_end_asset: str = "audio/bell_end.wav"
    silhouette_asset: str = "silhouette/seated_silhouette.png"
