import math

import numpy as np

from mindlab_monitor.config import MotionSettings
from mindlab_monitor.vision.movement_breath import MotionTracker, compute_chest_roi
from mindlab_monitor.vision.types import BBox

_FRAME_W, _FRAME_H = 320, 240
_FACE_BBOX = BBox(x=0.4, y=0.05, width=0.2, height=0.15)


def test_compute_chest_roi_sits_below_face_and_within_frame():
    settings = MotionSettings()
    roi = compute_chest_roi(_FACE_BBOX, _FRAME_W, _FRAME_H, settings)
    face_bottom_px = (_FACE_BBOX.y + _FACE_BBOX.height) * _FRAME_H
    assert roi.y0 >= face_bottom_px
    assert 0 <= roi.x0 < roi.x1 <= _FRAME_W
    assert 0 <= roi.y0 < roi.y1 <= _FRAME_H


def _flat_frame(value: int = 128) -> np.ndarray:
    return np.full((_FRAME_H, _FRAME_W, 3), value, dtype=np.uint8)


def test_identical_frames_score_near_perfect_stillness():
    settings = MotionSettings(downscale_width=_FRAME_W)
    tracker = MotionTracker(settings)
    frame = _flat_frame()
    tracker.update(frame, _FACE_BBOX, timestamp=0.0)
    tick = tracker.update(frame, _FACE_BBOX, timestamp=0.1)
    assert tick.stillness_score is not None
    assert tick.stillness_score > 99.0


def test_large_diff_outside_roi_lowers_stillness_score():
    settings = MotionSettings(downscale_width=_FRAME_W)
    tracker = MotionTracker(settings)
    frame_a = _flat_frame(128)
    frame_b = _flat_frame(128)
    frame_b[0:20, 0:20] = 250  # corner, well outside the chest ROI
    tracker.update(frame_a, _FACE_BBOX, timestamp=0.0)
    tick = tracker.update(frame_b, _FACE_BBOX, timestamp=0.1)
    assert tick.stillness_score is not None
    assert tick.stillness_score < 99.0


def test_breath_bpm_recovers_known_synthetic_frequency():
    settings = MotionSettings(
        downscale_width=_FRAME_W,
        breath_window_seconds=10.0,
        breath_min_seconds_before_estimate=6.0,
        breath_update_seconds=1.0,
        breath_min_bpm=3.0,
        breath_max_bpm=30.0,
    )
    tracker = MotionTracker(settings)
    roi = compute_chest_roi(_FACE_BBOX, _FRAME_W, _FRAME_H, settings)

    target_hz = 0.2  # 12 breaths/min
    dt = 0.1
    n_ticks = 160  # 16 seconds
    last_tick = None
    for i in range(n_ticks):
        t = i * dt
        frame = _flat_frame(128)
        intensity = int(128 + 40 * math.sin(2 * math.pi * target_hz * t))
        frame[roi.y0 : roi.y1, roi.x0 : roi.x1] = intensity
        last_tick = tracker.update(frame, _FACE_BBOX, timestamp=t)

    assert last_tick is not None
    assert last_tick.breath_bpm is not None
    assert abs(last_tick.breath_bpm - 12.0) <= 3.0


def test_movement_burst_does_not_corrupt_breath_estimate():
    settings = MotionSettings(
        downscale_width=_FRAME_W,
        breath_window_seconds=10.0,
        breath_min_seconds_before_estimate=6.0,
        breath_update_seconds=1.0,
        breath_min_bpm=3.0,
        breath_max_bpm=30.0,
        breath_quality_min_stillness=75.0,
        movement_sensitivity=8.0,
    )
    tracker = MotionTracker(settings)
    roi = compute_chest_roi(_FACE_BBOX, _FRAME_W, _FRAME_H, settings)
    target_hz = 0.2  # 12 breaths/min
    dt = 0.1

    def make_frame(t: float, disturb: bool) -> np.ndarray:
        frame = _flat_frame(128)
        intensity = int(128 + 40 * math.sin(2 * math.pi * target_hz * t))
        frame[roi.y0 : roi.y1, roi.x0 : roi.x1] = intensity
        if disturb:
            # Large alternating disturbance well outside the ROI (e.g.
            # shifting posture) — should drop the short-term stillness score
            # below the quality gate, so these ticks get skipped from the
            # breath-signal buffer entirely rather than corrupting the FFT.
            frame[0:40, 0:40] = 250 if int(t / dt) % 2 == 0 else 5
        return frame

    # Phase 1: settle into a stable estimate with a clean signal.
    t, last_tick = 0.0, None
    for _ in range(160):  # 16s
        last_tick = tracker.update(make_frame(t, disturb=False), _FACE_BBOX, timestamp=t)
        t += dt
    assert last_tick.breath_bpm is not None
    stable_bpm = last_tick.breath_bpm

    # Phase 2: sustained large movement burst outside the ROI. The gate
    # should stop feeding the FFT within ~stillness_smoothing_seconds, so
    # the estimate should stay close to the pre-burst value, not jump.
    for _ in range(150):  # 15s
        last_tick = tracker.update(make_frame(t, disturb=True), _FACE_BBOX, timestamp=t)
        t += dt

    assert abs(last_tick.breath_bpm - stable_bpm) <= 1.0
