from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mindlab_monitor.vision.types import FaceResult


class FaceBackend(ABC):
    """Face detection + eye-landmark backend.

    One implementation per platform: `mediapipe_backend.py` on desktop,
    `tflite_backend.py` on mobile (Phase 2). Both must return the same
    `FaceResult` shape so everything downstream (EAR, chest-ROI motion,
    calibration) is platform-agnostic.
    """

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def process(self, frame_bgr: np.ndarray, timestamp: float) -> FaceResult | None: ...

    @abstractmethod
    def close(self) -> None: ...
