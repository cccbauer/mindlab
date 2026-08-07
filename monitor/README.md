# mindlab-monitor

A camera-based meditation stability monitor: tracks eyes-open/closed, movement/stillness, and breath rate during a timed session, with a start/end bell and a per-session stability score. Built as an importable Python package (`mindlab_monitor`) so it can later be embedded inside the larger `mindlab` app, but also runs standalone.

See [`DEVLOG.md`](DEVLOG.md) for the running history of decisions and what's been tuned so far, and the plan under `.claude/plans/` for the phased architecture (desktop-first with MediaPipe; mobile phase 2 planned around a TFLite face backend since MediaPipe itself can't run on Android/iOS).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --extra desktop --group dev
```

## Run the app

```bash
uv run flet run src/mindlab_monitor/app.py
```

## Run tests

```bash
uv run pytest
```

## Debug/tuning tools

```bash
uv run python scripts/debug_face_backend.py      # live EAR/eyes-open-closed overlay
uv run python scripts/debug_movement_breath.py   # chest ROI + breath signal + calibration phase
```

`debug_movement_breath.py` starts with a 30s calibration phase (hold still, take 3 slow deep breaths) before resetting its on-screen timer and logging to `debug_logs/` — useful for validating breath-rate accuracy against a manually counted, paced breathing schedule.
