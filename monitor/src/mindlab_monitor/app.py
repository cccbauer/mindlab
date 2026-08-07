"""Flet entry point + app controller.

`main(page)` is what `flet run` calls for standalone use. `MonitorApp` owns
the camera pipeline, session engine, bell, and history store, and drives
navigation between the three screens (setup/calibration -> live session ->
summary) by swapping the content of a single root container — it never
touches `page.controls` directly, so it can also be mounted inside a larger
page (see `build_view` in `__init__.py`) instead of owning the whole window.

Importing this module must never have the side effect of starting a camera
or launching `ft.app()` — that only happens when `main()`/`start()` runs.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from typing import Callable

import flet as ft

from mindlab_monitor.audio.bell import BellPlayer
from mindlab_monitor.capture.desktop_cv2 import Cv2FrameSource
from mindlab_monitor.config import Settings
from mindlab_monitor.session.engine import SessionEngine
from mindlab_monitor.session.history_store import HistoryStore
from mindlab_monitor.session.models import SessionSummary
from mindlab_monitor.ui.views.session_view import build_session_view
from mindlab_monitor.ui.views.setup_view import build_setup_view
from mindlab_monitor.ui.views.summary_view import build_summary_view
from mindlab_monitor.vision.calibration import CalibrationTracker
from mindlab_monitor.vision.mediapipe_backend import MediaPipeFaceBackend
from mindlab_monitor.vision.pipeline import CameraPipeline

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


class MonitorApp:
    """Construct, then call `start()` once `root` has been mounted onto a
    page (added to `page.controls` or a parent container) — `start()` is
    what actually opens the camera and navigates to the first screen."""

    def __init__(self, page: ft.Page, settings: Settings | None = None) -> None:
        self.page = page
        self.settings = settings or Settings()

        self.frame_source = Cv2FrameSource()
        self.face_backend = MediaPipeFaceBackend()
        self.pipeline = CameraPipeline(
            self.frame_source,
            self.face_backend,
            hz=self.settings.motion.analysis_hz,
            zoom=self.settings.default_camera_zoom,
        )
        self.history_store = HistoryStore(self.settings.data_dir)
        self.bell = BellPlayer(page, self.settings.bell_start_asset, self.settings.bell_end_asset)
        self.calibration_tracker = CalibrationTracker(self.settings.calibration)
        self.session_engine: SessionEngine | None = None

        self._cleanup_fns: list[Callable[[], None]] = []
        self.root = ft.Container(padding=24, expand=True, alignment=ft.Alignment.CENTER)

    def start(self) -> ft.Control:
        """Opens the camera and shows the setup screen. Call only after
        `self.root` has been mounted on a page. Returns `self.root`."""
        try:
            self.pipeline.start()
        except RuntimeError as exc:
            self._show_camera_error(exc)
            return self.root
        self.goto_setup()
        return self.root

    def _show_camera_error(self, exc: Exception) -> None:
        self.root.content = ft.Column(
            [
                ft.Icon(ft.Icons.VIDEOCAM_OFF, size=48, color=ft.Colors.RED_400),
                ft.Text("Couldn't access the camera", size=20, weight=ft.FontWeight.BOLD),
                ft.Text(str(exc), size=13, color=ft.Colors.ON_SURFACE_VARIANT, text_align=ft.TextAlign.CENTER),
                ft.Text(
                    "On macOS: System Settings -> Privacy & Security -> Camera, "
                    "grant access to this app/terminal, then restart it.",
                    size=13,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Button("Retry", on_click=lambda _: self.start()),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        )
        if self.root.page is not None:
            self.root.update()

    def add_cleanup(self, fn: Callable[[], None]) -> None:
        self._cleanup_fns.append(fn)

    def _run_cleanup(self) -> None:
        for fn in self._cleanup_fns:
            fn()
        self._cleanup_fns = []

    def _set_view(self, builder: Callable[[], ft.Control]) -> None:
        # Run the OLD view's cleanup before building the NEW view, not after —
        # otherwise the new view's freshly-registered subscriptions (added by
        # `builder()`) would land in `_cleanup_fns` before `_run_cleanup` runs
        # and get torn down immediately.
        self._run_cleanup()
        self.root.content = builder()
        if self.root.page is not None:
            self.root.update()

    def goto_setup(self) -> None:
        self._set_view(lambda: build_setup_view(self))

    def start_session(self, duration_minutes: int) -> None:
        self.session_engine = SessionEngine(self.pipeline, self.settings, self.history_store)
        self._set_view(lambda: build_session_view(self, duration_minutes))

    def goto_summary(self, summary: SessionSummary) -> None:
        self._set_view(lambda: build_summary_view(self, summary))

    def shutdown(self) -> None:
        # Safe to call more than once (e.g. on_close AND on_disconnect AND
        # atexit all firing) — pipeline/frame-source/face-backend stop()s
        # are all idempotent.
        self._run_cleanup()
        self.pipeline.stop()


def main(page: ft.Page) -> None:
    page.title = "Meditation Monitor"
    page.window.width = 640
    page.window.height = 820
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    app = MonitorApp(page)
    # Belt-and-suspenders camera release: on_close fires on a normal window
    # close, on_disconnect if the client detaches without a clean close, and
    # atexit as a last resort if neither Flet callback fires before the
    # process exits (e.g. quitting via the app menu/Cmd+Q).
    page.on_close = lambda _: app.shutdown()
    page.on_disconnect = lambda _: app.shutdown()
    atexit.register(app.shutdown)
    page.add(app.root)
    app.start()


if __name__ == "__main__":
    ft.run(main, assets_dir=str(ASSETS_DIR))
