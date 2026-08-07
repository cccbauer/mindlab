from __future__ import annotations

import threading

import cv2
import numpy as np

from mindlab_monitor.capture.base import FrameSource


class Cv2FrameSource(FrameSource):
    """Desktop webcam capture via OpenCV, running in a background thread.

    Only the most recently grabbed frame is kept — if analysis is slower
    than the camera's frame rate, older frames are simply dropped instead
    of building up a backlog.
    """

    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise RuntimeError(f"Could not open camera index {self._camera_index}")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest_frame = None

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def _run(self) -> None:
        assert self._cap is not None
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                continue
            with self._lock:
                self._latest_frame = frame
