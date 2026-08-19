"""First exploration: does yt_idv's offscreen Maximum Intensity Projection
even work at all, end to end -- real sample dataset, real offscreen OpenGL
context, real render, real saved image?

Run inside the throwaway venv (see setup_ytidv_venv.sh), NOT this repo's own
dev venv (no `yt`/`yt_idv` there -- see yt_idv_renderer.py's module
docstring):

    source /path/to/ytidv-venv/bin/activate
    python scripts/yt_idv_exploration/explore_basic_render.py

Confirmed findings (see images/01_*.png, images/02_*.png):
  - `rc.add_scene(dd, "density", no_ghost=True)`'s component already
    defaults to `render_method == "max_intensity"` -- confirmed by printing
    it below, matching yt_idv/yt_idv/shaders/shaderlist.yaml's
    `component_shaders.block_rendering.default_value`. This add-on's own
    `yt_idv_renderer.py` still sets it explicitly rather than relying on
    that default holding forever.
  - `scene.camera.update(position=..., focus=..., up=..., fov=...,
    aspect_ratio=...)` works and produces a genuinely different image --
    confirmed via `np.allclose`, not just "it didn't crash".
  - Steady-state render cost at 256x256 on this machine (software OSMesa,
    no GPU needed): ~0.11s/frame after the first (slower, ~0.44s) call.
  - The visible image is the domain box's silhouette (a hexagonal/pentagon
    outline from most viewing angles -- 2-3 visible faces of a cube) filled
    with a dim, mostly-uniform dark-red/near-black-red tone almost
    everywhere, plus a bright blue/green/white MIP-colormapped blob at the
    galaxy's actual density peak. The dim fill across the WHOLE box
    (not just near the blob) suggests the current colormap gives even very
    low densities a nonzero floor color/alpha rather than true transparency
    -- exactly the kind of thing a real transfer function (the documented
    "next step") would let you control; not investigated further here.
"""

import time

import numpy as np
import yt

import yt_idv

ds = yt.load_sample("IsolatedGalaxy")
dd = ds.all_data()

rc = yt_idv.render_context("osmesa", width=256, height=256)
scene = rc.add_scene(dd, "density", no_ghost=True)
component = scene.components[0]
print("default render_method:", component.render_method)

t0 = time.time()
image1 = rc.run()
t1 = time.time()
print(f"first render: shape={image1.shape} dtype={image1.dtype} took={t1 - t0:.3f}s")
yt.write_bitmap(image1[:, :, :3], "01_first_render_square_256_osmesa.png")

scene.camera.update(
    position=np.array([0.9, 0.9, 0.9], dtype="f4"),
    focus=np.array([0.5, 0.5, 0.5], dtype="f4"),
    up=np.array([0.0, 0.0, 1.0], dtype="f4"),
    fov=60.0,
    aspect_ratio=1.0,
)
t2 = time.time()
image2 = rc.run()
t3 = time.time()
print(f"after camera move: shape={image2.shape} took={t3 - t2:.3f}s")
print("images differ after camera move:", not np.allclose(image1, image2))
yt.write_bitmap(image2[:, :, :3], "02_camera_moved_square_256_osmesa.png")

t4 = time.time()
for _ in range(3):
    rc.run()
t5 = time.time()
print(f"steady-state render cost: {(t5 - t4) / 3:.3f}s/frame")
