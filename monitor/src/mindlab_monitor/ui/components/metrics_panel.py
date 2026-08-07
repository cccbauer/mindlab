from __future__ import annotations

import flet as ft

from mindlab_monitor.session.engine import EngineTick


class MetricsPanel:
    def __init__(self) -> None:
        self.timer_text = ft.Text("--:--", size=44, weight=ft.FontWeight.BOLD)
        self.eyes_text = ft.Text("Eyes: —", size=16)
        self.stillness_bar = ft.ProgressBar(value=0, width=260)
        self.stillness_text = ft.Text("Stillness: —", size=16)
        self.breath_text = ft.Text("Breath: estimating…", size=16)
        self.stability_text = ft.Text("Stability: —", size=26, weight=ft.FontWeight.BOLD)

        self.control = ft.Column(
            [
                self.timer_text,
                ft.Divider(),
                self.stability_text,
                ft.Container(height=8),
                self.eyes_text,
                self.stillness_text,
                self.stillness_bar,
                self.breath_text,
            ],
            spacing=6,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def update_from_tick(self, tick: EngineTick) -> None:
        minutes, seconds = divmod(int(max(0.0, tick.remaining_seconds)), 60)
        self.timer_text.value = f"{minutes:02d}:{seconds:02d}"

        if not tick.face_detected:
            self.eyes_text.value = "Eyes: no face detected"
        else:
            self.eyes_text.value = f"Eyes: {'open' if tick.eyes_open else 'closed'}"

        if tick.stillness_score is not None:
            self.stillness_bar.value = max(0.0, min(1.0, tick.stillness_score / 100))
            self.stillness_text.value = f"Stillness: {tick.stillness_score:.0f}/100"
        else:
            self.stillness_text.value = "Stillness: —"

        self.breath_text.value = (
            f"Breath: {tick.breath_bpm:.1f} breaths/min" if tick.breath_bpm is not None else "Breath: estimating…"
        )
        self.stability_text.value = f"Stability: {tick.live_stability_score:.0f}/100"
        self.control.update()
