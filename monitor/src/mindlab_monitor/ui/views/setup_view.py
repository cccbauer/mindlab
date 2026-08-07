from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from mindlab_monitor.ui.components.silhouette_overlay import CameraPreview
from mindlab_monitor.ui.theme import CALIBRATION_COLORS, CALIBRATION_MESSAGES
from mindlab_monitor.utils.ui_thread import run_on_ui

if TYPE_CHECKING:
    from mindlab_monitor.app import MonitorApp

_DURATION_OPTIONS_MINUTES = (5, 10, 15, 20, 30, 45, 60)


def build_setup_view(app: "MonitorApp") -> ft.Control:
    app.calibration_tracker.reset()
    preview = CameraPreview(app.settings.silhouette_asset, show_status=True)

    duration_dropdown = ft.Dropdown(
        label="Session length",
        value=str(app.settings.default_duration_minutes),
        options=[ft.dropdown.Option(str(m), f"{m} min") for m in _DURATION_OPTIONS_MINUTES],
        width=160,
    )

    def handle_start(_: ft.ControlEvent) -> None:
        app.start_session(int(duration_dropdown.value))

    start_button = ft.Button("Start", disabled=True, on_click=handle_start, width=160)

    zoom_label = ft.Text(f"Zoom: {app.pipeline.zoom:.1f}x", size=13)

    def handle_zoom_change(e: ft.ControlEvent) -> None:
        app.pipeline.zoom = float(e.control.value)
        zoom_label.value = f"Zoom: {app.pipeline.zoom:.1f}x"
        zoom_label.update()

    zoom_slider = ft.Slider(
        min=1.0,
        max=6.0,
        # Wide-angle webcams need aggressive digital zoom-in to make a face
        # fill the frame — there's no optical zoom-out to compensate with,
        # so headroom goes up (crops smaller, upscales more), not down.
        value=app.pipeline.zoom,
        width=220,
        on_change=handle_zoom_change,
    )

    def apply_tick(frame, face_result) -> None:
        preview.update_frame(frame)
        state = app.calibration_tracker.update(face_result)
        message = CALIBRATION_MESSAGES[state.status]
        color = CALIBRATION_COLORS[state.status]
        if state.override_available and not state.can_start:
            message = message + " You can start anyway."
        if state.face_width_fraction is not None:
            message = message + f" (face width: {state.face_width_fraction:.0%})"
        preview.update_status(message, color)
        if start_button.disabled == state.can_start:
            start_button.disabled = not state.can_start
            start_button.update()

    def on_pipeline_tick(frame, face_result, ts) -> None:
        # Runs on the camera pipeline's background thread — marshal the
        # actual control mutations onto Flet's event loop (see run_on_ui).
        run_on_ui(app.page, lambda: apply_tick(frame, face_result))

    unsubscribe = app.pipeline.subscribe(on_pipeline_tick)
    app.add_cleanup(unsubscribe)

    return ft.Column(
        [
            ft.Text("Align yourself with the silhouette", size=22, weight=ft.FontWeight.BOLD),
            preview.control,
            ft.Column(
                [zoom_label, zoom_slider],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=0,
            ),
            ft.Row([duration_dropdown, start_button], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=20,
        scroll=ft.ScrollMode.AUTO,
    )
