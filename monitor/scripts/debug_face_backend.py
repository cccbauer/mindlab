"""Manual verification tool: live webcam window overlaying eye landmarks,
EAR value, and the eyes-open/closed classification from the desktop
MediaPipe backend. Blink on camera and confirm the label flips correctly.

Run: uv run python scripts/debug_face_backend.py
Press 'q' to quit.
"""

from __future__ import annotations

import time

import cv2

from mindlab_monitor.config import EyeSettings
from mindlab_monitor.vision.eyes import combined_ear, eyes_open_from_ear
from mindlab_monitor.vision.mediapipe_backend import MediaPipeFaceBackend


def main() -> None:
    eye_settings = EyeSettings()
    backend = MediaPipeFaceBackend()
    backend.warmup()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            result = backend.process(frame, time.time())

            if result is None:
                cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                x0, y0 = int(result.bbox.x * w), int(result.bbox.y * h)
                x1, y1 = int((result.bbox.x + result.bbox.width) * w), int((result.bbox.y + result.bbox.height) * h)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

                for pt in (*result.eye_points.left, *result.eye_points.right):
                    cv2.circle(frame, (int(pt[0] * w), int(pt[1] * h)), 2, (0, 255, 255), -1)

                ear = combined_ear(result.eye_points)
                eyes_open = eyes_open_from_ear(ear, eye_settings.ear_closed_threshold)
                label = "OPEN" if eyes_open else "CLOSED"
                color = (0, 255, 0) if eyes_open else (0, 165, 255)
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}  Eyes: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

            cv2.imshow("Face Backend Debug (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        backend.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
