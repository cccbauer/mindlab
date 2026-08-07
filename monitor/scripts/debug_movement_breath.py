"""Manual verification tool: live webcam window showing the chest ROI box,
a scrolling plot of the raw breath signal, and the current stillness score
+ FFT breath-rate estimate.

Starts with a CALIBRATION phase: hold still and take 3 slow, deep breaths.
This (a) gives the tracker a strong, unambiguous signal to bootstrap from
instead of starting from an empty buffer, and (b) records the full
amplitude range of a real deep breath, which sets the chart's fixed y-axis
scale so normal breathing's amplitude reads meaningfully relative to it
(instead of a rolling auto-scale that resets constantly and can't tell you
whether the current breathing is even producing much signal).

Once calibration ends, the elapsed-time counter resets to 0 and CSV
logging switches to "running" phase — that's the sync point to use when
reporting a paced-breathing schedule back for comparison (e.g. "0-60s at
10bpm, 60-120s at 15bpm").

Run: uv run python scripts/debug_movement_breath.py
Press 'q' to quit.

CSV columns: phase, elapsed_s, stillness_score, breath_bpm, raw_breath_signal
"""

from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from mindlab_monitor.config import MotionSettings
from mindlab_monitor.vision.mediapipe_backend import MediaPipeFaceBackend
from mindlab_monitor.vision.movement_breath import MotionTracker

_CHART_HEIGHT = 100
_CHART_HISTORY = 300
_LOG_DIR = Path(__file__).resolve().parent.parent / "debug_logs"
_CALIBRATION_SECONDS = 30.0


def main() -> None:
    backend = MediaPipeFaceBackend()
    backend.warmup()
    tracker = MotionTracker(MotionSettings())
    signal_history: deque[float] = deque(maxlen=_CHART_HISTORY)

    _LOG_DIR.mkdir(exist_ok=True)
    log_path = _LOG_DIR / f"breath_{time.strftime('%Y%m%dT%H%M%S')}.csv"
    log_file = log_path.open("w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["phase", "elapsed_s", "stillness_score", "breath_bpm", "raw_breath_signal"])
    print(f"Logging to {log_path}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    phase = "calibrating"
    phase_start_ts = time.time()
    calib_min, calib_max = None, None
    chart_lo, chart_hi = None, None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            ts = time.time()
            elapsed = ts - phase_start_ts
            face_result = backend.process(frame, ts)
            face_bbox = face_result.bbox if face_result is not None else None
            tick = tracker.update(frame, face_bbox, timestamp=ts)

            if phase == "calibrating" and tick.raw_breath_signal is not None:
                calib_min = tick.raw_breath_signal if calib_min is None else min(calib_min, tick.raw_breath_signal)
                calib_max = tick.raw_breath_signal if calib_max is None else max(calib_max, tick.raw_breath_signal)

            if phase == "calibrating" and elapsed >= _CALIBRATION_SECONDS:
                phase = "running"
                phase_start_ts = ts
                elapsed = 0.0
                if calib_min is not None and calib_max is not None and calib_max > calib_min:
                    pad = (calib_max - calib_min) * 0.15
                    chart_lo, chart_hi = calib_min - pad, calib_max + pad
                    print(f"Calibration done. Deep-breath signal range: [{calib_min:.2f}, {calib_max:.2f}]")
                else:
                    print("Calibration done, but no clear signal range captured (face not detected?).")
                print("Timer reset to 0 — breathe normally now.")

            log_writer.writerow(
                [phase, f"{elapsed:.2f}", tick.stillness_score, tick.breath_bpm, tick.raw_breath_signal]
            )

            h, w = frame.shape[:2]
            roi = tick.chest_roi
            if face_bbox is not None:
                cv2.rectangle(frame, (roi.x0, roi.y0), (roi.x1, roi.y1), (255, 200, 0), 2)
            if tick.raw_breath_signal is not None:
                # Plot the actual signal the tracker feeds into the FFT
                # (signed frame-to-frame diff in the ROI), not an
                # independently-recomputed raw-intensity approximation.
                signal_history.append(tick.raw_breath_signal)

            if phase == "calibrating":
                remaining = max(0.0, _CALIBRATION_SECONDS - elapsed)
                cv2.putText(frame, "CALIBRATING", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    "Hold still. Take 3 SLOW deep breaths in/out.",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    frame, f"Time left: {remaining:.0f}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                )
            else:
                stillness_text = f"{tick.stillness_score:.0f}/100" if tick.stillness_score is not None else "-"
                breath_text = f"{tick.breath_bpm:.1f} bpm" if tick.breath_bpm is not None else "estimating..."
                cv2.putText(
                    frame, f"Elapsed: {elapsed:.0f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f"Stillness: {stillness_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Breath: {breath_text}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2
                )

            chart = np.zeros((_CHART_HEIGHT, w, 3), dtype=np.uint8)
            if len(signal_history) >= 2:
                values = np.array(signal_history)
                if chart_lo is not None and chart_hi is not None:
                    # Fixed scale from the calibration deep breaths, so
                    # normal-breathing amplitude reads meaningfully relative
                    # to a known "full range" instead of an auto-scale that
                    # always fills the chart regardless of actual signal size.
                    lo, hi = chart_lo, chart_hi
                else:
                    lo, hi = values.min(), values.max()
                span = max(1e-6, hi - lo)
                normalized = np.clip((values - lo) / span, 0.0, 1.0)
                xs = np.linspace(0, w - 1, len(normalized)).astype(np.int32)
                ys = (_CHART_HEIGHT - 1 - normalized * (_CHART_HEIGHT - 1)).astype(np.int32)
                pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
                cv2.polylines(chart, [pts], isClosed=False, color=(0, 200, 255), thickness=2)

            combined = np.vstack([frame, chart])
            cv2.imshow("Movement/Breath Debug (q to quit)", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        backend.close()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
