# mindlab

Umbrella workspace for mindlab apps, each a self-contained Flet app sharing one Python/Flet
version so they can eventually be embedded together:

- **`monitor/`** — camera/mic-based meditation stability monitor.
- **`mindwear/`** — real-time EEG neurofeedback on the Emotiv EPOC X, running the frozen DMNELF
  EFP decoder (PDA = CEN − DMN). See `mindwear/README.md`.

Both target **Python 3.12** and **Flet 0.86.5**, managed as a single [uv](https://docs.astral.sh/uv/)
workspace (`pyproject.toml` at this root, `[tool.uv.workspace]`).

## Setup

```bash
uv sync --all-packages   # installs every member's dependencies into one shared .venv
```

`uv sync` alone (no `--all-packages`) only sets up the root's own (empty) dependencies — the root
project isn't a package itself, it just ties the members' lockfile resolution together, so members
need to be synced explicitly. Per-member setup (`cd monitor && uv sync`, `cd mindwear && uv sync`)
also works independently if you only need one app.

## Adding a new app

Create `<app>/pyproject.toml` (see `monitor/` for a packaged app via `uv_build`, or `mindwear/` for
a non-packaged one via `[tool.uv] package = false`), pin `requires-python = "==3.12.*"` and Flet to
the version above, then add `"<app>"` to this root's `[tool.uv.workspace] members`.
