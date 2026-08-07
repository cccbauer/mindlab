"""Public API for embedding this module inside a larger app.

`build_view(page)` constructs the monitor, mounts it on `page`, and starts
it (opens the camera, shows the setup screen) — it returns the same root
control it already added, purely for convenience/reference; callers should
not add it to the page again. Importing this package never opens a camera or
launches a Flet app by itself — only calling `build_view`/`main` does.
"""

from __future__ import annotations

import flet as ft

from mindlab_monitor.app import MonitorApp, main
from mindlab_monitor.config import Settings

__all__ = ["build_view", "main", "MonitorApp", "Settings"]


def build_view(page: ft.Page, settings: Settings | None = None) -> ft.Control:
    app = MonitorApp(page, settings=settings)
    page.add(app.root)
    return app.start()
