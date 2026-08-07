"""Marshal a UI mutation from a background thread onto Flet's event loop.

The camera pipeline runs its own thread; Flet's async runtime doesn't pick
up control mutations made directly from a foreign thread (the change is
applied to the control's state, but no repaint is scheduled — it only
becomes visible on the next repaint triggered some other way, e.g. dragging
the window). `page.run_task` is Flet's documented thread-safe way to
schedule work back onto the page's own loop.
"""

from __future__ import annotations

from typing import Callable

import flet as ft


def run_on_ui(page: ft.Page, fn: Callable[[], None]) -> None:
    async def _runner() -> None:
        fn()

    page.run_task(_runner)
