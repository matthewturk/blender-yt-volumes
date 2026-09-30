"""Checks that yt_idv's offscreen max-intensity render works end to end, and
that `scene.camera.update(...)` changes the image.

Run inside the throwaway venv (see setup_ytidv_venv.sh):

    source /path/to/ytidv-venv/bin/activate
    python scripts/yt_idv_exploration/explore_basic_render.py
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
