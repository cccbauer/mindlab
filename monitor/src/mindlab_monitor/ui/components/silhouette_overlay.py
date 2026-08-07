from __future__ import annotations

import numpy as np
import flet as ft

from mindlab_monitor.ui.theme import PREVIEW_HEIGHT, PREVIEW_WIDTH
from mindlab_monitor.utils.imaging import frame_to_jpeg_bytes


class CameraPreview:
    """Live camera feed with a translucent silhouette guide + status banner
    layered on top, used by both the setup/calibration screen and (without
    the status banner) the live session screen."""

    def __init__(self, silhouette_asset_path: str, show_status: bool = True, show_silhouette: bool = True) -> None:
        self.image = ft.Image(
            # Placeholder until the first camera frame arrives.
            src=silhouette_asset_path,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
            fit=ft.BoxFit.COVER,
            border_radius=12,
            # Without this, Flutter blanks the widget while each new frame's
            # bytes decode, producing a visible flicker at 10-15fps.
            gapless_playback=True,
        )
        self.silhouette = ft.Image(
            src=silhouette_asset_path,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
            fit=ft.BoxFit.CONTAIN,
            opacity=0.55,
            visible=show_silhouette,
        )
        self.status_text = ft.Text(
            "",
            color=ft.Colors.WHITE,
            size=16,
            weight=ft.FontWeight.W_600,
            text_align=ft.TextAlign.CENTER,
        )
        self.status_banner = ft.Container(
            content=self.status_text,
            bgcolor=ft.Colors.with_opacity(0.55, ft.Colors.BLACK),
            padding=10,
            border_radius=8,
            bottom=16,
            left=16,
            right=16,
            visible=show_status,
        )
        self.control = ft.Container(
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
            bgcolor=ft.Colors.BLACK,
            border_radius=12,
            content=ft.Stack(
                [self.image, self.silhouette, self.status_banner],
                width=PREVIEW_WIDTH,
                height=PREVIEW_HEIGHT,
            ),
        )

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        self.image.src = frame_to_jpeg_bytes(frame_bgr)
        self.image.update()

    def update_status(self, text: str, color: str) -> None:
        self.status_text.value = text
        self.status_banner.bgcolor = ft.Colors.with_opacity(0.55, color)
        self.status_banner.update()
