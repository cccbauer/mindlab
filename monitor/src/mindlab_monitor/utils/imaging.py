from __future__ import annotations

import cv2
import numpy as np


def frame_to_jpeg_bytes(frame_bgr: np.ndarray, quality: int = 70) -> bytes:
    """Flet's `ft.Image.src` accepts raw bytes directly (no base64 encoding
    needed) as of Flet 0.86."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame")
    return buf.tobytes()
