from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Normalized (0-1) bounding box, origin top-left."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class EyePoints:
    """Normalized (0-1) landmark points used for eye-aspect-ratio, per eye.

    Six points per eye in the classic EAR ordering:
    [outer_corner, top_1, top_2, inner_corner, bottom_1, bottom_2]
    """

    left: tuple[tuple[float, float], ...]
    right: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FaceResult:
    bbox: BBox
    eye_points: EyePoints
    confidence: float
    timestamp: float
