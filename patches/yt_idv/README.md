# yt_idv patches

Changes to yt_idv made while working on this add-on, kept here as diffs for
yt_idv PRs. Apply from a yt_idv checkout with `git apply <diff>`.

## Not yet in yt_idv

Both are against upstream `main` (74fac48) and also apply cleanly to the
`exact_tf` and `remove_padding` branches. yt_idv's tests pass with each
(217 passed, 1 skipped).

- **`early-ray-termination.diff`**: a `ray_is_done(curr_color)` hook in
  `ray_tracing.frag.glsl`, beside `sample_texture` and `cleanup_phase`. The
  transfer-function shader stops a ray at 0.999 accumulated opacity; max
  intensity and projection never stop early. On zoom.blend's frames with an
  opaque end, about 12% less GPU time (frame 225: 2.57 -> 2.28 s; frame
  240: 1.81 -> 1.59 s); images unchanged (max difference 4e-4).
- **`occlusion-culling.diff`** (prototype; apply after early termination):
  the transfer-function pass gets a `lequal` depth test, and a
  `fragment_depth()` hook writes a fragment's block entry depth where its ray
  went opaque and the far plane elsewhere, declared `depth_greater`
  (`GL_ARB_conservative_depth`, guarded, so macOS compiles it without the
  speedup). Blocks behind an opaque pixel are then rejected before their
  shader runs; blending is front to back, so they would have added nothing.
  About 5-7% more on the same frames. It changes what the transfer-function
  pass's depth holds (`USE_DB`): which matters only for several components
  in one yt_idv scene, so upstream it would be better as an opt-in trait on
  `BlockRendering`.

## Already in yt_idv (`applied/`)

- `dataset-units.diff`, `dataset-units-fix.diff`: merged upstream (scene
  files record the dataset's unit registry).
- `exact-tf-step.diff`: on the `exact_tf` branch (exact emission-absorption
  per step, needed for opaque transfer functions).
- `atlas-no-padding.diff`, `atlas-no-padding-test-fix.diff`: on the
  `remove_padding` branch (lookups clamped to each block's texels instead of
  padding the atlas; the test fix gives the test a defined transfer-function
  range, since the default log range starts at log(0) and draws nothing on
  macOS).

## Not prepared

- yt_idv's default `tf_log = True` with `tf_min = 0` takes log(0) in the
  transfer-function shader; whether that draws depends on the driver (it does
  on NVIDIA/Linux, not on macOS). A default of `tf_log = False`, or a small
  positive `tf_min`, would make it consistent.
