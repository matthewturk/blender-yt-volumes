import bpy
import numpy as np
import pytest

from yt_volumes import properties, tf_plot, transfer_function_nodes

pytest.importorskip("yt")
pytest.importorskip("matplotlib")


@pytest.fixture
def volume():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    obj = bpy.data.objects.new("V", None)
    bpy.context.scene.collection.objects.link(obj)
    obj.volume_override.is_volume_override = True
    yield obj
    tf_plot.unregister()
    properties.unregister()


def test_yt_layers_plot_is_the_rendered_transfer_function(volume):
    props = volume.volume_override
    props.renderer_mode = "YT_IDV_TRANSFER_FUNCTION"
    props.tf_fit_range = False
    props.tf_log_min, props.tf_log_max = -26.0, -20.0
    props.tf_n_layers = 5

    tf = tf_plot.transfer_function(volume)

    assert tuple(tf.x_bounds) == (-26.0, -20.0)
    alpha = tf.funcs[3].y
    peaks = np.nonzero((alpha[1:-1] > alpha[:-2]) & (alpha[1:-1] >= alpha[2:]))[0]
    assert len(peaks) == 5


def test_plot_draws_into_a_preview_icon(volume):
    props = volume.volume_override
    props.renderer_mode = "YT_IDV_TRANSFER_FUNCTION"
    props.tf_fit_range = False

    assert tf_plot.plot(volume) is None

    preview = tf_plot._preview(volume)
    assert tuple(preview.image_size) == (tf_plot.WIDTH, tf_plot.HEIGHT)
    pixels = np.empty(tf_plot.WIDTH * tf_plot.HEIGHT * 4, dtype=np.float32)
    preview.image_pixels_float.foreach_get(pixels)
    assert pixels.reshape(-1, 4)[:, :3].std() > 0.05  # a picture, not a blank
    assert tf_plot.icon(volume) == preview.icon_id
    assert not bpy.data.images and not bpy.data.textures


def test_display_resampling_keeps_thin_layers(volume):
    import yt

    tf = yt.ColorTransferFunction((-30.0, -10.0), nbins=8192, grey_opacity=True)
    tf.add_layers(12, w=1e-4)  # far thinner than a display bin
    shown = tf_plot._for_display(tf)
    assert shown.funcs[3].y.size == tf_plot.PLOT_BINS
    assert (shown.funcs[3].y > 0.5).sum() >= 12


def test_geometry_nodes_plot_uses_the_sampled_table(volume):
    props = volume.volume_override
    props.renderer_mode = "YT_IDV_GN_TRANSFER_FUNCTION"
    props.tf_fit_range = False
    transfer_function_nodes.add_transfer_function_modifier(volume)
    bpy.context.view_layer.update()

    tf = tf_plot.transfer_function(volume)

    assert tuple(tf.x_bounds) == (props.tf_log_min, props.tf_log_max)
    assert tf.funcs[3].y[-1] == pytest.approx(0.8, abs=0.01)  # the starter curve's top
    assert tf.funcs[3].y[0] == pytest.approx(0.0, abs=1e-6)


def test_other_modes_have_nothing_to_plot(volume):
    volume.volume_override.renderer_mode = "YT_IDV_MAX_INTENSITY"
    assert tf_plot.transfer_function(volume) is None
