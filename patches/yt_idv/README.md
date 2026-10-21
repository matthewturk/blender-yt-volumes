# yt_idv patches

Changes to yt_idv made while working on this add-on, kept here as diffs for
yt_idv PRs. Apply from a yt_idv checkout with `git apply <diff>`.

## Worth upstreaming

Measured on zoom.blend on 2026-10-21 (EEVEE, 4 samples per cell, fade off);
only the saved-scene kd walk changed frame times noticeably.

- **A compiled walk for saved scenes**, from either diff below: the Python
  walk took ~86 ms per draw on dd0087 (40150 blocks), the compiled one ~1 ms,
  with identical order. `viewpoint-order-arrays.diff` needs yt's
  `viewpoint_order_arrays` (the `kdtree_walk_speedup` PR); `lod-cut.diff`
  carries its own `utilities/kd_walk.pyx` and needs no yt change, but also
  brings the LOD cut, which didn't pay (see below). The walk alone could be
  split out of it.

## Not yet in yt_idv

Both are against upstream `main` (74fac48) and also apply cleanly to the
`exact_tf` and `remove_padding` branches. yt_idv's tests pass with each
(217 passed, 1 skipped).

Measured again on 2026-10-21 at 4 samples per cell with fade off,
early termination plus occlusion culling saved only 3-5% of yt_idv's GPU
time (images bit-identical): few rays reach 0.999 opacity there. They pay
off only for heavily opaque transfer functions.

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
- **`viewpoint-order-arrays.diff`** (against `remove_padding`, 6f085c8):
  scene files' saved kd-tree is ordered by yt's compiled
  `amr_kdtools.viewpoint_order_arrays` when yt has it (the
  `kdtree_walk_speedup` branch), instead of the Python walk.
- **`lod-cut.diff`** (against `~/dxl/yt_idv`'s working tree of 2026-10-21:
  `interruptible` plus the uncommitted `--interior-nodes` export and
  `viewpoint-order-arrays.diff`, which this replaces): scene files' `lod/`
  coarse interior-node data. `BlockCollection` loads it after the leaves (in
  the atlases and vertex array, not `block_data`, not re-saved);
  `BlockRendering.lod_pixels` draws a kd node smaller than that on screen as
  its coarse block. The walk is a new Cython `utilities/kd_walk.pyx`, used for
  all saved scenes. 18 new tests; the suite otherwise passes except
  `test_geographic_matches_spherical`, which fails without it too.
  Benchmarked on `dd0087_scene_gz.zip` (2725 coarse bricks): zoomed out it
  draws 288 bricks instead of 40150 with no visible change but little time
  saved (frame 1 at 240×135: 13.1 -> 9.8 ms); deep in the zoom the coarse
  nodes are hundreds of pixels on screen, so it barely cuts, and where it
  does the dense core washes out (parent grids hold averages). Not worth
  upstreaming for zoom-style movies.
- **`preintegrated-tf.diff`** (prototype; against `remove_padding` with
  `viewpoint-order-arrays.diff` applied): a `tf_preintegrated` trait on
  `BlockRendering`. Each step integrates the transfer function between the
  field at its two ends (a block's first step starts on its boundary texels),
  from a running-integral texture at the table's resolution. On zoom.blend at
  4 samples per cell it matches the standard path's mean error with half its
  worst-pixel error, at ~20% more GPU time per sample; at 2 samples it is ~35%
  cheaper than standard 4 with better worst pixels. Only pays off where
  standard sampling visibly bands.

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

## Data

- `dd0087_block_sizes.npy`: the (40150, 3) int16 block shapes of
  `dd0087_scene4.zip`, for a texture-packer test on real AMR blocks (3 to
  161 texels per axis, 16646 distinct shapes, median anisotropy 3.4). The
  current packer fills 60% of its atlas on them, against 85%+ on the
  synthetic sizes its tests use; a layer-by-depth variant tried here did
  worse (20%), so a better packer needs real 3D packing.

## Not prepared

- yt_idv's default `tf_log = True` with `tf_min = 0` takes log(0) in the
  transfer-function shader; whether that draws depends on the driver (it does
  on NVIDIA/Linux, not on macOS). A default of `tf_log = False`, or a small
  positive `tf_min`, would make it consistent.
