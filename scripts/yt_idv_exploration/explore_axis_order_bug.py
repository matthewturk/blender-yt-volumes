"""Second exploration: found two real yt_idv bugs by testing a NON-square
resolution, which every one of yt_idv's own examples/tests avoids (they all
use e.g. width=height=1024) -- this add-on can't avoid it, Cycles renders
are essentially never square (1920x1080, etc).

Run inside the throwaway venv (see setup_ytidv_venv.sh).

Bug 1 (both backends, WRONG SHAPE, data itself is fine): `SceneGraph.image`
reads the framebuffer via `glReadPixels(0, 0, width, height, GL_RGBA,
GL_FLOAT)`, which comes back shaped `(width, height, 4)`, then calls
`arr.swapaxes(0, 1)` without reassigning it -- a complete no-op, since
`ndarray.swapaxes` returns a new view rather than mutating in place. Every
non-square render comes back axis-swapped relative to the conventional
`(height, width, 4)` shape. `images/05_bug1_egl_raw_wrong_axis_order.png`
is what naively saving that raw array looks like at an extreme 400x100
aspect ratio -- a WIDE request rendered as a TALL, squished image.
Confirmed the data itself is otherwise correct: `.swapaxes(0, 1)` on the
caller's side (exactly what `yt_idv_renderer._fix_yt_idv_axis_order` does)
fixes it completely for EGL.

Bug 2 (OSMesa ONLY, genuinely CORRUPTED data, not just mislabeled): the
OSMesa backend produces visibly tiled/repeated garbage at non-square sizes
(images/06_bug2_osmesa_corrupted_nonsquare.png) -- not a clean transpose of
a correct image, which rules out "it's just bug 1 again". Tried the obvious
suspect fix (`OSMesaMakeCurrent`'s `width`/`height` arguments look swapped
relative to the real OSMesa C API signature `(ctx, buffer, type, width,
height)` -- yt_idv passes `(self.height, self.width)`) by monkeypatching a
corrected subclass; it did NOT fix it
(images/07_bug2_osmesa_still_corrupted_after_argswap_attempt.png is
visually near-identical to the unfixed version) -- so the real cause is
still open, something deeper in how OSMesa's software rasterizer is fed
this buffer. NOT chased further. This is why `yt_idv_renderer.
DEFAULT_BACKEND` is `"egl"`, not `"osmesa"`, despite EGL needing a working
GPU + driver stack that OSMesa doesn't.

A third, smaller, ALSO-unresolved artifact: even through EGL (bug 1 fixed),
non-square renders show faint fine-grained vertical/horizontal banding
overlaid on an otherwise-correct image
(images/09_extreme_aspect_via_real_module_fixed.png, produced through the
real `yt_idv_renderer.yt_idv_max_intensity_renderer` -- see
explore_stub_end_to_end.py). Not investigated (candidates not yet tried:
GL_PACK_ALIGNMENT, MSAA interacting with max_intensity's GL_MAX blend
equation, yt_idv's own viewport handling).
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
# NOTE: EGL and OSMesa can't coexist in one process -- PYOPENGL_PLATFORM is
# read once at OpenGL's first import, so switching backends mid-process
# raises AttributeError on the second one. Run this section in a SEPARATE
# process/invocation from the EGL section above if you want both; shown
# together here only for documentation.
"""
rc2 = yt_idv.render_context("osmesa", width=W, height=H)
scene2 = rc2.add_scene(dd, "density", no_ghost=True)
img2 = rc2.run()
print("OSMesa shape:", img2.shape)
yt.write_bitmap(img2[:, :, :3], "06_bug2_osmesa_corrupted_nonsquare.png")
"""
