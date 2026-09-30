"""Demonstrates two yt_idv bugs at non-square resolutions.

Run inside the throwaway venv (see setup_ytidv_venv.sh).

Bug 1 (EGL and OSMesa): `SceneGraph.image` calls `arr.swapaxes(0, 1)` without
reassigning, so non-square renders come back shaped (width, height, 4). Fixed
by swapping on the caller's side (`yt_idv_renderer._fix_yt_idv_axis_order`).

Bug 2 (OSMesa only): non-square renders are corrupted (tiled garbage), and
swapping OSMesaMakeCurrent's width/height does not fix it. Unresolved, so
`yt_idv_renderer.DEFAULT_BACKEND` is "egl".
"""

import numpy as np
import yt

import yt_idv

ds = yt.load_sample("IsolatedGalaxy")
dd = ds.all_data()

W, H = 400, 100  # deliberately extreme so any row/column mixup is obvious

# --- Bug 1: EGL, wrong axis order, clean data ------------------------------
rc = yt_idv.render_context("egl", width=W, height=H)
scene = rc.add_scene(dd, "density", no_ghost=True)
scene.camera.update(position=[1.6, 1.1, 2.0], focus=[0.5, 0.5, 0.5], up=[0, 1, 0])
raw = rc.run()
print(f"EGL raw shape (requested W={W} H={H}):", raw.shape)  # -> (400, 100, 4), WRONG
yt.write_bitmap(raw[:, :, :3], "08_extreme_aspect_raw_wrong_axis_order.png")
fixed = raw.swapaxes(0, 1)
print("EGL fixed shape:", fixed.shape)  # -> (100, 400, 4), matches (H, W)
yt.write_bitmap(fixed[:, :, :3], "extreme_aspect_egl_fixed_standalone.png")

# --- Bug 2: OSMesa, corrupted data at any non-square size ------------------
# EGL and OSMesa can't coexist in one process (PYOPENGL_PLATFORM is read once);
# run this in a separate process.
"""
rc2 = yt_idv.render_context("osmesa", width=W, height=H)
scene2 = rc2.add_scene(dd, "density", no_ghost=True)
img2 = rc2.run()
print("OSMesa shape:", img2.shape)
yt.write_bitmap(img2[:, :, :3], "06_bug2_osmesa_corrupted_nonsquare.png")
"""
