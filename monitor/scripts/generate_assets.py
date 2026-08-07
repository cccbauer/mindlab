"""One-off generator for the placeholder bell sounds + silhouette guide PNG.

Not part of the runtime app — run manually (`uv run python scripts/generate_assets.py`)
whenever these placeholder assets need to be regenerated. Real bell recordings
or artwork can simply replace the files this writes under `assets/`.
"""

from __future__ import annotations

import wave
from pathlib import Path

import cv2
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent.parent / "src" / "mindlab_monitor" / "assets"
SAMPLE_RATE = 44100


def _bell_tone(duration_s: float, fundamental_hz: float) -> np.ndarray:
    """Classic additive bell synthesis: a handful of inharmonic partials,
    each an independently decaying sine, summed together."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    partials = [
        (1.0, 1.0, 2.0),
        (0.55, 2.0, 3.2),
        (0.35, 2.76, 4.0),
        (0.22, 4.07, 5.2),
        (0.12, 5.4, 6.5),
    ]
    signal = np.zeros_like(t)
    for amplitude, freq_ratio, decay_rate in partials:
        signal += amplitude * np.sin(2 * np.pi * fundamental_hz * freq_ratio * t) * np.exp(-decay_rate * t)
    signal /= np.max(np.abs(signal))
    fade_samples = int(0.01 * SAMPLE_RATE)
    signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
    return signal


def _write_wav(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    int_signal = (signal * 32767 * 0.9).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int_signal.tobytes())


def _draw_silhouette(path: Path) -> None:
    """Simple seated-meditator outline (head + shoulders + crossed-leg base)
    used as the on-screen alignment guide. Transparent background, light
    gray stroke, no fill, so it can overlay the live camera preview."""
    w, h = 480, 640
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    color = (210, 210, 210, 235)  # light gray, mostly opaque, BGRA
    thickness = 4

    head_center = (w // 2, int(h * 0.22))
    head_radius = int(h * 0.09)
    cv2.circle(canvas, head_center, head_radius, color, thickness)

    neck_bottom = (w // 2, head_center[1] + head_radius + int(h * 0.02))
    shoulder_y = neck_bottom[1] + int(h * 0.05)
    shoulder_half_width = int(w * 0.22)
    cv2.line(canvas, neck_bottom, (neck_bottom[0], shoulder_y), color, thickness)

    torso_bottom_y = int(h * 0.72)
    torso_half_width_top = shoulder_half_width
    torso_half_width_bottom = int(w * 0.3)
    pts_torso = np.array(
        [
            (w // 2 - torso_half_width_top, shoulder_y),
            (w // 2 + torso_half_width_top, shoulder_y),
            (w // 2 + torso_half_width_bottom, torso_bottom_y),
            (w // 2 - torso_half_width_bottom, torso_bottom_y),
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [pts_torso], isClosed=True, color=color, thickness=thickness)

    knee_y = int(h * 0.92)
    knee_half_width = int(w * 0.42)
    cv2.line(canvas, (w // 2 - torso_half_width_bottom, torso_bottom_y), (w // 2 - knee_half_width, knee_y), color, thickness)
    cv2.line(canvas, (w // 2 + torso_half_width_bottom, torso_bottom_y), (w // 2 + knee_half_width, knee_y), color, thickness)
    cv2.line(canvas, (w // 2 - knee_half_width, knee_y), (w // 2 + knee_half_width, knee_y), color, thickness)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    _write_wav(ASSETS_DIR / "audio" / "bell_start.wav", _bell_tone(2.5, fundamental_hz=440.0))
    _write_wav(ASSETS_DIR / "audio" / "bell_end.wav", _bell_tone(3.2, fundamental_hz=330.0))
    _draw_silhouette(ASSETS_DIR / "silhouette" / "seated_silhouette.png")
    print("Generated bell_start.wav, bell_end.wav, seated_silhouette.png")


if __name__ == "__main__":
    main()
