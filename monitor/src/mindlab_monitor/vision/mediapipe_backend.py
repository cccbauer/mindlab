from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mindlab_monitor.vision.face_backend import FaceBackend
from mindlab_monitor.vision.types import BBox, EyePoints, FaceResult

# Classic 6-point EAR ordering per eye: [outer, top1, top2, inner, bottom1, bottom2].
# Indices into MediaPipe's 478-point face mesh landmark topology (unchanged
# between the legacy `solutions.face_mesh` API and the current Tasks API's
# FaceLandmarker — both use the same underlying mesh).
_RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)
_LEFT_EYE_IDX = (362, 385, 387, 263, 373, 380)

_MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "models" / "face_landmarker.task"


class MediaPipeFaceBackend(FaceBackend):
    """Desktop face/eye-landmark backend using MediaPipe's Tasks API
    (`mediapipe.tasks.vision.FaceLandmarker`).

    mediapipe 1.0.0 removed the older `mediapipe.solutions.face_mesh` API
    entirely — only the Tasks API remains, which requires a pre-built
    `.task` model bundle (vendored under `assets/models/`) rather than
    working out of the box like the old solutions API did.

    Only imports `mediapipe` inside methods that need it (not at module
    load) so this module can be imported harmlessly on platforms without
    mediapipe installed (e.g. during mobile-only test runs).
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._landmarker = None

    def warmup(self) -> None:
        import mediapipe as mp

        if not _MODEL_PATH.exists():
            raise RuntimeError(
                f"Missing MediaPipe face landmarker model at {_MODEL_PATH}. "
                "Download it from https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> FaceResult | None:
        import mediapipe as mp

        if self._landmarker is None:
            self.warmup()
        assert self._landmarker is not None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        def pts(indices: tuple[int, ...]) -> tuple[tuple[float, float], ...]:
            return tuple((landmarks[i].x, landmarks[i].y) for i in indices)

        return FaceResult(
            bbox=BBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y),
            eye_points=EyePoints(left=pts(_LEFT_EYE_IDX), right=pts(_RIGHT_EYE_IDX)),
            # FaceLandmarker's IMAGE-mode result doesn't expose a per-frame
            # detection score; presence of landmarks already implies it
            # cleared min_face_detection_confidence.
            confidence=1.0,
            timestamp=timestamp,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
