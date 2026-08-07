from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from mindlab_monitor.session.models import SessionSummary

if TYPE_CHECKING:
    from mindlab_monitor.app import MonitorApp


def _fmt(value: float | None, suffix: str = "", digits: int = 0) -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def build_summary_view(app: "MonitorApp", summary: SessionSummary) -> ft.Control:
    minutes, seconds = divmod(int(summary.duration_seconds), 60)

    def handle_new_session(_: ft.ControlEvent) -> None:
        app.goto_setup()

    return ft.Column(
        [
            ft.Text("Session Summary", size=26, weight=ft.FontWeight.BOLD),
            ft.Text(f"Duration: {minutes}m {seconds}s", size=16),
            ft.Container(height=8),
            ft.Text(f"Stability score: {summary.stability_score:.0f}/100", size=28, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text(f"Eyes stability: {_fmt(summary.eye_score)}/100"),
            ft.Text(f"Movement stillness: {_fmt(summary.movement_score)}/100"),
            ft.Text(f"Breath regularity: {_fmt(summary.breath_score)}/100"),
            ft.Text(f"Average breath rate: {_fmt(summary.avg_breath_bpm, ' bpm', 1)}"),
            ft.Container(height=24),
            ft.Button("New Session", on_click=handle_new_session, width=180),
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=8,
    )
