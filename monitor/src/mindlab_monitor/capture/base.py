from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FrameSource(ABC):
    """Abstract source of BGR frames (numpy arrays), latest-frame-only.

    Implementations run their own background capture thread/callback and
    drop older frames rather than queueing, so `read()` always reflects the
    most recent frame available and analysis never falls behind.
    """

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the latest available frame, or None if none yet."""
