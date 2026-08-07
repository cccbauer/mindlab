"""Platform-agnostic OpenCV-only movement/stillness + breath-rate estimation.

Deliberately avoids MediaPipe Pose / BlazePose: there's no comparable mobile
(tflite-runtime) reference implementation for pose, so instead everything
here runs off a face bounding box (which every `FaceBackend`, desktop or
mobile, produces) plus plain frame-differencing. This keeps the algorithm
identical on desktop and mobile.

Movement and breathing are measured over disjoint regions on purpose: the
chest ROI (used for the breath signal) is excluded from the stillness
measurement, so normal breathing doesn't get scored as "movement".
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from mindlab_monitor.config import MotionSettings
from mindlab_monitor.vision.types import BBox


@dataclass(frozen=True)
class ChestRoiPixels:
    x0: int
    y0: int
    x1: int
    y1: int


def compute_chest_roi(face_bbox: BBox, frame_width: int, frame_height: int, settings: MotionSettings) -> ChestRoiPixels:
    face_h_px = face_bbox.height * frame_height
    x0 = int(face_bbox.x * frame_width)
    x1 = int((face_bbox.x + face_bbox.width) * frame_width)
    y0 = int((face_bbox.y + face_bbox.height) * frame_height + settings.chest_roi_y_offset_factor * face_h_px)
    y1 = int(y0 + settings.chest_roi_height_factor * face_h_px)
    x0 = max(0, min(x0, frame_width - 1))
    x1 = max(x0 + 1, min(x1, frame_width))
    y0 = max(0, min(y0, frame_height - 1))
    y1 = max(y0 + 1, min(y1, frame_height))
    return ChestRoiPixels(x0, y0, x1, y1)


@dataclass(frozen=True)
class MotionTick:
    timestamp: float
    stillness_score: float | None  # None until a previous frame exists to diff against
    breath_bpm: float | None  # None until enough signal has accumulated
    chest_roi: ChestRoiPixels
    raw_breath_signal: float | None  # this tick's raw signal sample, for diagnostics/plotting


class MotionTracker:
    """Stateful, call `update()` once per analysis tick with the latest frame
    (BGR) and the current face bbox (normalized 0-1, or None if no face)."""

    def __init__(self, settings: MotionSettings) -> None:
        self._settings = settings
        self._prev_gray: np.ndarray | None = None
        self._stillness_buffer: deque[tuple[float, float]] = deque()
        self._breath_buffer: deque[tuple[float, float]] = deque()
        self._last_breath_estimate_ts: float = 0.0
        self._last_breath_bpm: float | None = None
        self._smoothed_bbox: BBox | None = None

    def _smoothed_face_bbox(self, face_bbox: BBox | None) -> BBox | None:
        # Face-landmark detection jitters slightly frame to frame even when
        # you're still. Deriving the chest ROI straight from the raw bbox
        # makes the ROI itself jitter, which then samples slightly different
        # skin/clothing/background each tick — noise on top of the real
        # breath signal, not just a visual annoyance. An EMA stabilizes it.
        if face_bbox is None:
            self._smoothed_bbox = None
            return None
        if self._smoothed_bbox is None:
            self._smoothed_bbox = face_bbox
            return self._smoothed_bbox
        alpha = self._settings.bbox_smoothing_alpha
        prev = self._smoothed_bbox
        self._smoothed_bbox = BBox(
            x=alpha * face_bbox.x + (1 - alpha) * prev.x,
            y=alpha * face_bbox.y + (1 - alpha) * prev.y,
            width=alpha * face_bbox.width + (1 - alpha) * prev.width,
            height=alpha * face_bbox.height + (1 - alpha) * prev.height,
        )
        return self._smoothed_bbox

    def update(self, frame_bgr: np.ndarray, face_bbox: BBox | None, timestamp: float | None = None) -> MotionTick:
        ts = time.time() if timestamp is None else timestamp
        h, w = frame_bgr.shape[:2]
        scale = self._settings.downscale_width / w
        small = cv2.resize(frame_bgr, (self._settings.downscale_width, max(1, int(h * scale))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sh, sw = gray.shape[:2]

        smoothed_bbox = self._smoothed_face_bbox(face_bbox)
        roi = compute_chest_roi(smoothed_bbox, w, h, self._settings) if smoothed_bbox is not None else None
        roi_small = None
        if roi is not None:
            roi_small = ChestRoiPixels(
                x0=int(roi.x0 * scale), y0=int(roi.y0 * scale), x1=int(roi.x1 * scale), y1=int(roi.y1 * scale)
            )
            roi_small = ChestRoiPixels(
                x0=max(0, min(roi_small.x0, sw - 1)),
                y0=max(0, min(roi_small.y0, sh - 1)),
                x1=max(roi_small.x0 + 1, min(roi_small.x1, sw)),
                y1=max(roi_small.y0 + 1, min(roi_small.y1, sh)),
            )

        stillness_score = None
        raw_breath_signal = None
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            # Signed difference, not absolute: raw mean brightness in the ROI
            # is dominated by ambient lighting/auto-exposure and barely
            # budges for subtle chest motion (only harsh movement showed up
            # in testing). Frame-to-frame *change* is far more sensitive to
            # small motion, and a signed diff is proportional to the
            # derivative of position — a sinusoidal breath motion still
            # shows up at the same fundamental frequency in the derivative
            # (just phase-shifted), unlike an absolute-value diff, which
            # would fold negative half-cycles over and double the apparent
            # frequency.
            signed_diff = gray.astype(np.float32) - self._prev_gray.astype(np.float32)
            abs_diff = np.abs(signed_diff)

            if roi_small is not None:
                raw_breath_signal = float(
                    np.mean(signed_diff[roi_small.y0 : roi_small.y1, roi_small.x0 : roi_small.x1])
                )
                abs_diff[roi_small.y0 : roi_small.y1, roi_small.x0 : roi_small.x1] = 0.0

            mean_abs_diff = float(np.mean(abs_diff))
            raw_stillness = max(0.0, 100.0 - (mean_abs_diff / self._settings.movement_sensitivity) * 100.0)
            self._stillness_buffer.append((ts, raw_stillness))
            self._trim(self._stillness_buffer, self._settings.stillness_smoothing_seconds)
            stillness_score = sum(v for _, v in self._stillness_buffer) / len(self._stillness_buffer)

            # Only feed the breath signal while recently still (short-term
            # smoothed score, not the whole-window average): a brief
            # movement burst — shifting posture, reaching to end the
            # session — just gets skipped instead of sitting inside the FFT
            # window forever. np.interp bridges smoothly over the resulting
            # gap, since movements are typically short relative to the
            # window (~50s) and this only pauses accumulation, it doesn't
            # invalidate everything already collected.
            if roi_small is not None and stillness_score >= self._settings.breath_quality_min_stillness:
                self._breath_buffer.append((ts, raw_breath_signal))
                self._trim(self._breath_buffer, self._settings.breath_window_seconds)

        self._prev_gray = gray

        if roi is not None and ts - self._last_breath_estimate_ts >= self._settings.breath_update_seconds:
            self._last_breath_estimate_ts = ts
            new_estimate = self._estimate_breath_bpm()
            # `None` means "don't trust this window" (not enough data yet,
            # or too much movement — see the quality gate below), not "no
            # breathing" — hold the last good reading rather than blanking
            # it out or replacing it with movement-corrupted noise.
            if new_estimate is not None:
                self._last_breath_bpm = new_estimate

        return MotionTick(
            timestamp=ts,
            stillness_score=stillness_score,
            breath_bpm=self._last_breath_bpm,
            chest_roi=roi if roi is not None else ChestRoiPixels(0, 0, 0, 0),
            raw_breath_signal=raw_breath_signal,
        )

    @staticmethod
    def _trim(buf: deque[tuple[float, float]], window_seconds: float) -> None:
        if not buf:
            return
        cutoff = buf[-1][0] - window_seconds
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def _estimate_breath_bpm(self) -> float | None:
        s = self._settings
        if len(self._breath_buffer) < 2:
            return None
        span = self._breath_buffer[-1][0] - self._breath_buffer[0][0]
        if span < s.breath_min_seconds_before_estimate:
            return None

        timestamps = np.array([t for t, _ in self._breath_buffer])
        values = np.array([v for _, v in self._breath_buffer])

        sample_rate_hz = len(timestamps) / span
        n_uniform = max(16, int(span * sample_rate_hz))
        uniform_ts = np.linspace(timestamps[0], timestamps[-1], n_uniform)
        uniform_values = np.interp(uniform_ts, timestamps, values)

        # Linear detrend (not just mean subtraction): slow drift from
        # auto-exposure/lighting changes or gradual posture shift otherwise
        # dominates the low-frequency end of the spectrum and gets
        # mistaken for an implausibly slow "breath rate".
        relative_ts = uniform_ts - uniform_ts[0]
        trend_coeffs = np.polyfit(relative_ts, uniform_values, deg=1)
        detrended = uniform_values - np.polyval(trend_coeffs, relative_ts)
        windowed = detrended * np.hanning(len(detrended))

        freqs = np.fft.rfftfreq(len(windowed), d=span / n_uniform)
        magnitudes = np.abs(np.fft.rfft(windowed))

        min_hz = s.breath_min_bpm / 60.0
        max_hz = s.breath_max_bpm / 60.0
        band = (freqs >= min_hz) & (freqs <= max_hz)
        if not np.any(band):
            return None

        band_freqs = freqs[band]
        band_mags = magnitudes[band]
        peak_freq = band_freqs[int(np.argmax(band_mags))]
        return float(peak_freq * 60.0)
