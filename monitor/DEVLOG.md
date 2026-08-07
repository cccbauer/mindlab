# Dev Log — Meditation Monitor

A running log of what was built, what broke, and why we made the calls we made. Newest entries at the bottom.

## 2026-08-06 — Project kickoff & architecture

**Goal**: a Flet app that watches a meditator via camera (+ mic later), tracks eyes-open/closed, movement/stillness, and breath rate, and reports a "stability" score for the session. Meant to eventually be embedded as a module inside a larger `mindlab` app, so it's built as a real importable package (`src/mindlab_monitor/`), not a script.

Before writing any code, researched two things that shaped the whole plan:

- **Flet has no desktop camera control at all**, on any version. `ft.Camera` (added in 0.81+) is iOS/Android/web only. Desktop capture is permanently `cv2.VideoCapture` + pushing frames into an `ft.Image`.
- **`mediapipe` (the Python package) cannot run on Android/iOS** — no wheels exist anywhere for it, and it's not on Flet's mobile wheel index. Mobile face detection would need a from-scratch port onto raw `.tflite` models via `tflite-runtime`. There's a decent open-source reference for this on the *face* side (`patlevin/face-detection-tflite`, MIT), but **no comparable reference for MediaPipe's Pose/BlazePose** — reimplementing that would be an ~800-1500 line undertaking with nothing to build from.

Decision: drop MediaPipe Pose entirely, on both platforms. Movement/stillness and breath rate are instead computed with a single **platform-agnostic OpenCV-only algorithm** (frame differencing / ROI intensity) anchored off the face bounding box, which every face backend (desktop or mobile) can produce. This means the only thing that needs a per-platform implementation at all is face/eye detection — everything else is shared code. Real feature-fidelity tradeoff (no per-limb movement diagnostics) accepted in exchange for something that can actually ship on mobile later.

Phase 1 (this log) = desktop only, real MediaPipe, prove the whole pipeline end-to-end. Phase 2 (later) = port the mobile TFLite face backend. Phase 3 = polish/history/mic breath backup.

Scaffolded the package: `capture/` (camera), `vision/` (face backend, EAR/eyes, movement+breath), `session/` (engine, stability scoring, history), `audio/` (bell), `ui/` (Flet views), `config.py` (all tunables, injectable). Built with `uv`, pinned to Python 3.12 (safer wheel availability for mediapipe/opencv than 3.13 at the time).

## 2026-08-06/07 — Getting Phase 1 to actually run

Building the code was the easy part — every one of these was found by actually launching the app, not by reading docs:

- **`mediapipe` 1.0.0 removed the legacy `solutions.face_mesh` API entirely.** Only the Tasks API (`mediapipe.tasks.vision.FaceLandmarker`) remains, which needs a pre-built `.task` model bundle. Downloaded Google's official `face_landmarker.task` and vendored it under `assets/models/`, rewrote the backend on the Tasks API. Same underlying 478-point mesh topology, so the EAR landmark indices didn't need to change.
- **`ft.Audio` isn't in Flet core** despite it looking like it should be — still a separate `flet-audio` package (`from flet_audio import Audio`).
- **Flet 0.86's `ft.Image` dropped `src_base64`** in favor of `src` accepting raw bytes directly. Actually simpler once found.
- **`ElevatedButton`/`OutlinedButton` are deprecated** in favor of a unified `Button`.
- **Real ordering bug in view navigation**: switching views called `build_new_view(app)` (which registers the new view's cleanup callbacks) *before* running the old view's cleanup — so a brand new screen's camera-frame subscription was torn down immediately by the previous screen's cleanup. Fixed by making `_set_view` take a builder function and run cleanup *before* calling it.
- **Camera permission crash**: a denied/pending macOS camera permission raised an unhandled `RuntimeError` and crashed the whole page. Added a friendly "Couldn't access the camera" screen with a Retry button instead.

## 2026-08-07 — Live testing, round 1: rendering

- **Preview froze after one frame.** Root cause: Flet's async runtime (0.86) doesn't repaint from a control mutation made on a foreign thread — the camera pipeline's background thread was calling `.update()` directly. The state changed but no repaint got scheduled; it only became visible if something else forced a repaint (confirmed by: dragging the window "unfroze" it). Fixed by marshaling every pipeline-thread UI mutation through `page.run_task` (`utils/ui_thread.py::run_on_ui`).
- **Flicker** once frames were updating: Flutter blanks the `Image` widget while decoding each new frame's bytes unless `gapless_playback=True` is set. One-line fix.

## 2026-08-07 — Live testing, round 2: calibration & framing

- **Calibration required standing uncomfortably close.** Default `target_face_width_range` (0.18-0.32 of frame width) was tuned blind, no real camera data. Loosened to (0.10, 0.40) and added a live "(face width: NN%)" readout to the status banner so both of us could see real numbers instead of guessing again.
- **User's webcam is a fixed wide-angle lens** — no optical zoom, so a normal sitting distance still leaves the face small in frame. Rather than trying to guess a universal "correct" distance, added a **live digital zoom slider** (`CameraPipeline.zoom`, applied as a center-crop+resize before face detection *and* display, so calibration/breath-ROI/preview all stay consistent with what's zoomed). Started at 1.4x/max 3x, then raised to 2.0x default / 6x max ceiling once it was clear wide-angle lenses need more headroom than expected.
- Removed the silhouette overlay from the live session screen — it was only ever meant for the setup/alignment step.

## 2026-08-07 — Live testing, round 3: breath signal was fundamentally not working

This was the big one, several compounding issues found via `scripts/debug_movement_breath.py` (built specifically for this kind of back-and-forth tuning):

1. **Chest ROI was on the stomach.** Original offset (1.4x / 1.2x face-heights below the chin) was tuned blind; real testing showed it landed well below the actual chest. Repositioned to 0.3x / 0.9x.
2. **Signal was raw average brightness in the ROI** — actively suppressed by the webcam's auto-exposure, which holds overall brightness steady. Only harsh movement showed up; subtle breathing didn't move the needle at all ("timeseries stops if I move subtly, only harsh movements move it"). Switched to **signed frame-to-frame difference** in the ROI instead — far more sensitive to small motion, and since it's proportional to the derivative of position, a sinusoidal breath motion still shows up at the *same* fundamental frequency (just phase-shifted) — unlike an absolute-value diff, which would fold negative half-cycles over and double the apparent frequency.
3. **Slow drift dominating the FFT**: only mean-subtraction was applied before the FFT, not detrending. Auto-exposure/lighting drift over a 35-50s window can dominate the low-frequency end of the spectrum and get mistaken for an implausibly slow "breath". Switched to a proper linear detrend (least-squares fit, subtract the trend line).
4. **ROI box visibly jittering.** Face-landmark detection wobbles slightly frame to frame even when still; deriving the ROI straight from the raw bbox propagated that jitter into which pixels get sampled — noise on top of the real signal, not just cosmetic. Added an EMA smoother (`bbox_smoothing_alpha`) on the face bbox before deriving the ROI from it.
5. **Real validation looked great, then broke at the end.** Manual test: counted 10 breaths/min, algorithm reported 9.6 — excellent. Then jumped to 22 right at the end of the session. Cause: a single burst of movement (posture shift, reaching to end the session) can dominate the FFT window and get misread as a fast "breath" — the FFT has no way to know a spike was movement, not breathing.

   First attempt: gate the *decision* to refresh the estimate on the whole window's *average* stillness. User correctly pushed back — a brief movement gets diluted by 45+ seconds of good data in an average, so it wouldn't even trigger the gate in realistic scenarios. Reworked to gate *admission* instead: only feed a sample into the breath-signal buffer at all while the *short-term* (2s) smoothed stillness score is above threshold. A movement burst just gets skipped rather than sitting in the FFT input; `np.interp`'s uniform resampling bridges smoothly over the resulting gap. Added `test_movement_burst_does_not_corrupt_breath_estimate` to lock this in.

## 2026-08-07 — Interactive calibration tooling

Added to `scripts/debug_movement_breath.py` for doing real paced-breathing validation together:

- **CSV logging** of every tick (phase, elapsed seconds, stillness, breath bpm, raw signal) to `debug_logs/breath_<timestamp>.csv`, so a real session can be checked against a self-reported breathing schedule after the fact instead of eyeballing a live window.
- **A calibration phase**: 30s of "hold still, take 3 slow deep breaths" before the real test starts. Two purposes: (a) gives the tracker a strong, unambiguous signal to bootstrap from instead of an empty buffer, and (b) the deep-breath signal's min/max range sets a **fixed chart y-axis scale**, so normal breathing's amplitude reads meaningfully relative to a known "full range" instead of a rolling auto-scale that always fills the chart regardless of actual signal size. The elapsed-time counter resets to 0 when calibration ends — that's the sync point for reporting a paced-breathing schedule back for comparison.

Also, separately: found and cleaned up **9 orphaned background processes** accumulated from repeated `kill $PID`-on-the-wrapper-process test restarts (each still holding the camera open, one with 12+ minutes of accumulated CPU time) — `uv run`/`flet run` spawn child processes that don't die with the parent. Added `page.on_close` + `page.on_disconnect` + `atexit` shutdown hooks so the app itself reliably releases the camera on quit, independent of how it's closed.

## 2026-08-07 — First real paced-breathing validation, and an adaptive window

Ran a real paced-breathing test via the calibration tooling above: 0-60s targeting 8bpm, 60-120s targeting 17bpm, 120-180s targeting 3bpm (self-counted). Read the logged CSV against the reported schedule:

- 8bpm target → settled at **8.4bpm**. 17bpm target → settled at **16.8bpm**. Both within ~5%, once each segment had ~35-50s to settle — the core algorithm is solid for normal/moderate breathing rates.
- 3bpm target → **never converged**, bounced between 7.2 and 21.6bpm the whole segment. Root cause: at 3 breaths/min, one cycle is 20s, and the (then-fixed) 50s window only covers 2.5 cycles — not enough for the FFT to resolve reliably, made worse by 3.0bpm sitting exactly on the search band's lower edge (`breath_min_bpm`), where FFT peak-picking is least reliable. Also visible: brief spurious jumps right after each target-rate change (expected — a fixed-window FFT can't distinguish "two rates mixed in one window" from a genuine new rate until the old one ages out; much less of a concern for real gradually-changing breathing than this artificial abrupt-switching test).

Fix needed a longer window for slow breathing, but a longer window always costs first-reading latency. Proposed a fixed longer window (90s); user pushed back with a better idea — **make the window adaptive**: start at a fast/responsive minimum, only grow it once a reading comes back both slow *and* stable (earned, not default), and shrink it quickly again if breathing speeds back up or stillness drops (movement, discomfort, coming out of the session). Implemented as `MotionTracker._adapt_window`, re-evaluated every `breath_update_seconds`:

- Grows toward `breath_window_max_seconds` (100s) only when the latest estimate is slow (`< breath_window_grow_bpm_threshold`, 8bpm) *and* short-term stillness is comfortably high (`>= breath_window_grow_min_stillness`, 85 — stricter than the sample-admission gate, since growing is a bigger commitment than admitting one sample).
- Shrinks back toward `breath_window_min_seconds` (45s) immediately if stillness drops below the admission gate, or the estimate comes back fast (`> breath_window_shrink_bpm_threshold`, 14bpm).
- The "enough data yet?" check (`breath_min_span_fraction`, 0.7) now scales with the *current* window instead of a fixed constant, so it tightens/loosens as the window moves.

Net effect: typical/faster breathers get the original fast ~30s-ish first reading; someone settling into slow, stable meditative breathing gradually earns a longer, more accurate window; any sign of movement or speeding back up snaps the window back down for responsiveness. Added `test_window_grows_for_slow_stable_breathing_then_shrinks_on_movement` and exposed `breath_window_seconds` on `MotionTick` (now shown on-screen and logged in the debug tool) so the adaptation is directly observable during the next real test.

## Status

Phase 1 core loop (camera → calibration → session → summary → history) works end-to-end. Breath-rate estimation validated against a real paced-breathing test for normal/moderate rates (8bpm→8.4, 17bpm→16.8); slow-breathing accuracy (~3bpm) fix (adaptive window) implemented but not yet re-validated against real data. Phase 2 (mobile TFLite backend) not started.
