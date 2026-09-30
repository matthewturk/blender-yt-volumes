"""Builds an isocontour transfer function (`yt.ColorTransferFunction.add_layers`)
for yt_idv's `transfer_function` render_method.

Run inside the throwaway venv (see setup_ytidv_venv.sh).

`tf_min`/`tf_max` are in normalized [0, 1] space, not physical units: yt_idv
linearly normalizes field data by the dataset's min/max before upload. For
IsolatedGalaxy, 1e-6 to 2e-3 works; 1e-4 to 1.0 renders nearly black.
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


# 1. Too high, nearly all black.
render_with_tf(1e-4, 1.0, out_name="14_tf_first_guess_too_narrow_near_black.png")

# 2. Working range.
render_with_tf(1e-6, 2e-3, out_name="15_tf_working_isocontours.png")
