"""Fifth exploration: building a real transfer function for yt_idv's
`transfer_function` render_method -- isocontour layers at fixed densities
in log space, the standard yt volume-rendering technique
(`yt.ColorTransferFunction.add_layers`), not a placeholder colormap.

Run inside the throwaway venv (see setup_ytidv_venv.sh).

The hard part wasn't the transfer function API itself -- it's figuring out
WHAT NUMERIC RANGE `tf_min`/`tf_max` actually operate in. They are NOT the
field's raw physical units (density in g/cm**3, ~1e-31 to ~1e-23 for
IsolatedGalaxy). `yt_idv.scene_data.block_collection.BlockCollection.
_load_textures` LINEARLY normalizes field data to [0, 1] by the dataset's
own absolute min/max BEFORE it ever reaches the GPU texture -- confirmed by
reading that source directly, not guessed. `tf_min`/`tf_max` (and
`yt.ColorTransferFunction`'s own `x_bounds`) operate in THIS
normalized-fraction space.

Even knowing that, picking values still took two tries:

1. First guess: `tf_min=1e-4, tf_max=1.0` (natural log of both, `tf_log=True`)
   -- image is essentially all black except a single bright pixel
   (`images/14_tf_first_guess_too_narrow_near_black.png`). The dataset's
   REAL visible signal turned out to live far below 1e-4 of the normalized
   range -- most of [1e-4, 1.0] is empty for this dataset/field.
2. Cross-checked against `max_intensity` mode's own auto-printed
   `cmap_min`/`cmap_max` (the actual per-view max of this SAME normalized
   quantity, from an EARLIER exploration run) -- on the order of 1e-4 to
   1e-2, not close to 1.0. Retried with `tf_min=1e-6, tf_max=2e-3`
   (`images/15_tf_working_isocontours.png`) -- real nested isocontour shells
   appear, colored by the colormap gradient, exactly what "isocontours at
   fixed densities in log space" should look like.

These bounds (`yt_idv_renderer.TF_VALUE_MIN`/`TF_VALUE_MAX`) are tuned by
trial and error for THIS dataset/field, not derived from its statistics --
see that module's docstring "Known rough edges".
"""

import numpy as np
import yt

import yt_idv

ds = yt.load_sample("IsolatedGalaxy")
dd = ds.all_data()

CAMERA_KWARGS = dict(position=[1.6, 1.1, 2.0], focus=[0.5, 0.5, 0.5], up=[0, 1, 0])


def render_with_tf(value_min, value_max, n_layers=5, colormap="viridis", out_name="tf_test.png"):
    rc = yt_idv.render_context("egl", width=256, height=256)
    scene = rc.add_scene(dd, "density", no_ghost=True)
    component = scene.components[0]
    print("field's raw (unnormalized) min_val/max_val:", component.data.min_val, component.data.max_val)

    component.render_method = "transfer_function"
    component.tf_min = value_min
    component.tf_max = value_max
    component.tf_log = True

    tf = yt.ColorTransferFunction((np.log(value_min), np.log(value_max)))
    tf.add_layers(n_layers, colormap=colormap)
    rgba = np.stack([f.y for f in tf.funcs], axis=-1)
    component.transfer_function.data = np.clip(rgba * 255, 0, 255).astype("u1").reshape(256, 1, 4)

    scene.camera.update(**CAMERA_KWARGS)
    image = rc.run()
    print(f"tf_min={value_min}, tf_max={value_max}: alpha max={image[..., 3].max():.4f}, "
          f"nonzero-alpha px={int((image[..., 3] > 0.01).sum())}/{image.shape[0] * image.shape[1]}")
    yt.write_bitmap(image[:, :, :3], out_name)


# 1. First guess -- too narrow/too high, nearly all black.
render_with_tf(1e-4, 1.0, out_name="14_tf_first_guess_too_narrow_near_black.png")

# 2. Working range, found by bracketing max_intensity's own observed
#    per-view max (~1e-3 to 1e-2 of the normalized range).
render_with_tf(1e-6, 2e-3, out_name="15_tf_working_isocontours.png")
