"""Owns the camera + face backend and fans out each analysis tick to
subscribers (the calibration screen, then the session engine once a session
starts) so the camera is only ever opened once and the face backend is only
warmed up once per app run."""

from __future__ import annotations

import threading
import time
from typing import Callable

import cv2
import numpy as np

from mindlab_monitor.capture.base import FrameSource
from mindlab_monitor.vision.face_backend import FaceBackend
from mindlab_monitor.vision.types import FaceResult

TickCallback = Callable[[np.ndarray, "FaceResult | None", float], None]


def _apply_zoom(frame: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    crop_w, crop_h = int(w / zoom), int(h / zoom)
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (w, h))


class CameraPipeline:
    def __init__(self, frame_source: FrameSource, face_backend: FaceBackend, hz: float, zoom: float = 1.0) -> None:
        self._frame_source = frame_source
        self._face_backend = face_backend
        self._period = 1.0 / hz
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._subscribers_lock = threading.Lock()
        self._subscribers: list[TickCallback] = []
        # Plain float, read fresh each tick — safe to adjust live from the UI
        # thread (e.g. a zoom slider) without any locking; worst case one
        # tick reads a slightly stale value mid-drag.
        self.zoom = zoom

    def start(self) -> None:
        if self._thread is not None:
            return
        self._frame_source.start()
        self._face_backend.warmup()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._face_backend.close()
        self._frame_source.stop()

    def subscribe(self, callback: TickCallback) -> Callable[[], None]:
        with self._subscribers_lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._subscribers_lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def _run(self) -> None:
        while not self._stop_event.is_set():
            tick_start = time.time()
            frame = self._frame_source.read()
            if frame is not None:
                frame = _apply_zoom(frame, self.zoom)
                ts = time.time()
                face_result = self._face_backend.process(frame, ts)
                with self._subscribers_lock:
                    subscribers = list(self._subscribers)
                for cb in subscribers:
                    cb(frame, face_result, ts)
            elapsed = time.time() - tick_start
            time.sleep(max(0.0, self._period - elapsed))
