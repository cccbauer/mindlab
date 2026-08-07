import flet as ft

CALIBRATION_COLORS: dict[str, str] = {
    "no_face": ft.Colors.RED_400,
    "off_center": ft.Colors.AMBER_400,
    "too_far": ft.Colors.AMBER_400,
    "too_close": ft.Colors.AMBER_400,
    "not_frontal": ft.Colors.AMBER_400,
    "ready": ft.Colors.GREEN_400,
}

CALIBRATION_MESSAGES: dict[str, str] = {
    "no_face": "We can't see your face — check lighting and camera position.",
    "off_center": "Center yourself in the frame.",
    "too_far": "Move closer to the camera.",
    "too_close": "Move back from the camera.",
    "not_frontal": "Face the camera directly.",
    "ready": "Looking good — hold still.",
}

PREVIEW_WIDTH = 480
PREVIEW_HEIGHT = 640
