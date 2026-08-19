"""Fourth exploration: chasing the period-4 banding artifact found while
testing camera motion (scripts/yt_idv_exploration/blender_scene/) at
non-square resolutions, even through EGL (bug 1 already fixed).

Run inside the throwaway venv (see setup_ytidv_venv.sh).

Investigation, in order:

1. Dump a raw scanline (not just look at the picture) -- reveals an EXACT
   period-4 pattern: `[value, 0, 0, value]` repeating. Not proportional to
   scene content, not random noise.
2. Hypothesis: ray-marching under-sampling (`component.sample_factor`,
   default 1.0, documented up to 20.0 in the component's own GUI slider).
   Tested at 1.0, 4.0, 10.0 -- **pixel-identical output every time**
   (images/10_*.png, images/11_*.png). This DISPROVES the under-sampling
   explanation; don't reach for it again without new evidence.
3. Confirm it's specifically tied to non-square dimensions: same
   camera/dataset, SQUARE resolution -- scanline is smooth and continuous,
   no dropout pattern at all (images/12_*.png).

Root cause still open at the end of this script -- being investigated
independently (candidates: EGL surface/multisample config, a width/height
mixup in yt_idv's own viewport or grid-expansion geometry shader
specifically for non-square viewports). Worked around in the real add-on
module (`yt_idv_renderer.yt_idv_max_intensity_renderer`) by rendering
square and center-cropping -- see that function's docstring.
"""

import numpy as np
import yt

import yt_idv

ds = yt.load_sample("IsolatedGalaxy")
dd = ds.all_data()

W, H = 320, 240
CAMERA_KWARGS = dict(position=[1.6, 1.1, 2.0], focus=[0.5, 0.5, 0.5], up=[0, 1, 0])


def scanline(image, row):
    return list(np.round(image[row, 100:140, 0], 3))


# --- Step 1 + 2: sample_factor sweep, all pixel-identical -----------------
for sample_factor in (1.0, 4.0, 10.0):
    rc = yt_idv.render_context("egl", width=W, height=H)
    scene = rc.add_scene(dd, "density", no_ghost=True)
    scene.camera.update(**CAMERA_KWARGS)
    scene.components[0].sample_factor = sample_factor
    image = rc.run().swapaxes(0, 1)  # bug 1's fix
    print(f"sample_factor={sample_factor}: scanline={scanline(image, H // 2)}")
    yt.write_bitmap(image[:, :, :3], f"10_bug3_sample_factor_{sample_factor:g}.png")

# --- Step 3: same scene, square resolution -> clean ------------------------
rc_square = yt_idv.render_context("egl", width=256, height=256)
scene_square = rc_square.add_scene(dd, "density", no_ghost=True)
scene_square.camera.update(**CAMERA_KWARGS)
image_square = rc_square.run()  # square -- no axis-order ambiguity either
print(f"square 256x256: scanline={scanline(image_square, 128)}")
yt.write_bitmap(image_square[:, :, :3], "12_bug3_square_256_confirmed_clean_no_banding.png")
