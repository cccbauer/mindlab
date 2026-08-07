from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import flet as ft

from mindlab_monitor.session.engine import EngineTick
from mindlab_monitor.session.models import SessionSummary
from mindlab_monitor.ui.components.metrics_panel import MetricsPanel
from mindlab_monitor.ui.components.silhouette_overlay import CameraPreview
from mindlab_monitor.utils.ui_thread import run_on_ui
from mindlab_monitor.vision.movement_breath import compute_chest_roi

if TYPE_CHECKING:
    from mindlab_monitor.app import MonitorApp


def build_session_view(app: "MonitorApp", duration_minutes: int) -> ft.Control:
    preview = CameraPreview(app.settings.silhouette_asset, show_status=False, show_silhouette=False)
    metrics = MetricsPanel()

    def handle_end_click(_: ft.ControlEvent) -> None:
        app.session_engine.stop_session()

    end_button = ft.Button("End Session", on_click=handle_end_click, width=180)

    def on_frame(frame, face_result, ts) -> None:
        # Runs on the camera pipeline's background thread. Draw the chest
        # ROI the breath signal is actually sampled from, so it's visible
        # instead of an invisible/implicit region of the frame.
        if face_result is not None:
            h, w = frame.shape[:2]
            roi = compute_chest_roi(face_result.bbox, w, h, app.settings.motion)
            frame = frame.copy()
            cv2.rectangle(frame, (roi.x0, roi.y0), (roi.x1, roi.y1), (255, 200, 0), 2)
            cv2.putText(
                frame, "breath ROI", (roi.x0, max(0, roi.y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1
            )
        run_on_ui(app.page, lambda: preview.update_frame(frame))

    def on_tick(tick: EngineTick) -> None:
        # Also fires on the pipeline thread (SessionEngine's tick handler
        # runs as a pipeline subscriber).
        run_on_ui(app.page, lambda: metrics.update_from_tick(tick))

    def on_complete(summary: SessionSummary) -> None:
        # May fire from the pipeline thread (duration elapsed) or the UI
        # thread (manual "End Session" click) — safe to marshal either way.
        run_on_ui(app.page, lambda: (app.bell.play_end(), app.goto_summary(summary)))

    unsubscribe_frame = app.pipeline.subscribe(on_frame)
    app.add_cleanup(unsubscribe_frame)

    app.bell.play_start()
    app.session_engine.start_session(
        planned_duration_seconds=duration_minutes * 60, on_tick=on_tick, on_complete=on_complete
    )

    return ft.Row(
        [
            preview.control,
            ft.Container(width=32),
            ft.Column(
                [metrics.control, ft.Container(height=24), end_button],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
