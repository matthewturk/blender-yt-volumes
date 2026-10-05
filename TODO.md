# TODO

## GUI usability (Blender UI on OpenGL)
- yt_idv's EGL context fails (EGL_BAD_ACCESS) when Blender's GPU backend is
  OpenGL; Vulkan works. Add a clear error telling the user to switch.
- The user's chezmoi `config.json` sets `gpu_backend: OPENGL`, which may undo
  the switch on restart.

## Cycles settings for GUI F12
Example .blends don't save Cycles samples/device/denoiser; batch sets them from
the TOML. A fresh scene uses 4096 samples and OpenImageDenoise on the CPU
(render denoise is separate from viewport denoise), giving ~50 s per Cycles
pass (two per frame) vs <1 s at 16 samples with OptiX denoising.
- Have the example generators save samples=16, device=GPU, denoiser=OPTIX.
- Maybe warn when a yt_idv render uses high samples or CPU denoising.

## Demo script (~/misc-transfer/yt_idv_demo_script.md)
- Prep: set GPU backend to Vulkan, Render denoiser to OptiX, low max samples,
  Rescale Colormap on. Regenerate the Kindle HTML.

## Upstream
- yt_idv: PRs 284–288 + the two fix patches (~/misc-transfer/yt_idv_patches/)
  need to land; then `uv lock --upgrade-package yt_idv`.
- Local yt_idv branch `integration-284-288` is not pushed (on purpose).

## Switching render mode can leave yt_idv rendering blank
Reported: after changing render mode, yt_idv frames come back fully blank
(not a flat color) until restart. Fixed: the signature is recorded only after
a render succeeds, and the colormap range is per volume and cleared on a mode
change. Remaining suspects in `yt_idv_renderer.py`, if it recurs:
- A render that returns blank without raising is still cached.
- `_yt_tf_applied` skips reconfiguring a no-argument yt TF whose key matches,
  even if another mode has since changed `render_method`/`tf_min`/`tf_max`
  on the component. Edits to a TF .py file also don't show until something
  in the signature changes.
- The zero-TF path leaves `tf_min, tf_max = 2, 3` on the component.
