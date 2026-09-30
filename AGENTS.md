# Notes for agents working in this repo

## Comment style

Write comments for a reader who knows yt_idv reasonably well and Blender/Cycles
a little — not an expert, not a beginner.

- Keep comments short, usually one line, occasionally up to three.
- Do comment non-obvious fixes and gotchas, stated as the fact or constraint:
  `# save() makes Cycles re-sync the whole scene on the next render; save_render() doesn't.`
- Don't write investigation narratives, measurements/timings, "confirmed
  directly", history of earlier implementations, or "see X's docstring"
  cross-references. Put those in the commit message.
- Don't comment what the code plainly does. Explain a gotcha once, at the
  most relevant spot, not in every place it matters.
- Docstrings: a one-line summary, plus only the API facts a caller needs
  (units, row order, alpha mode, return shape).
- Operator class docstrings are Blender UI tooltips; keep them to a sentence.

## Running things

- Tests: `PYTHONPATH= uv run --group ytidv pytest -q`
- Headless render (real `.blend`, real yt_idv, GPU):
  `PYTHONPATH= uv run --group ytidv python scripts/yt_idv_exploration/blender_scene/render_headless.py`
  — limit frames with `RENDER_START_FRAME`/`RENDER_END_FRAME`. Frame 1 pays a
  multi-minute dataset load. Scene/dataset/look changes go in `scene_setup.py`.
- For quick pipeline experiments, set `renderer_mode = "SAMPLE_CONSTANT"` to
  skip yt_idv entirely (seconds per run instead of minutes).

## Gotchas when debugging

- Don't run a test render while another render is running: they share
  `work_dir` (`/tmp/cycles_volume_override`) and `render_session` sweeps
  leftover `frame_*` directories. Set `prefs.work_dir` to a private directory.
- Checking output for the volume: compare RGB only. PNGs are RGBA, and the
  constant alpha channel hides large RGB differences in a plain mean.
- Per-frame slowdowns are almost always Cycles' persistent-data cache being
  invalidated. Timestamp `bpy.app.handlers.render_stats` and look for
  "Synchronizing object" / "Updating Lights | Computing tree" to see which
  render re-synced, then bisect the steps before it.
- `/` is usually close to full; clean up scratch renders and EXRs.
