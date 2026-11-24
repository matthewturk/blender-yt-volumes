import json

import bpy
import numpy as np
import pytest

from yt_volumes import annotations, change_tracking, probe_analysis, probes, properties
from yt_volumes import yt_idv_renderer as yir
from yt_volumes.external_render import FrameParams

from test_yt_idv_renderer import _block_scene, _identity_volume, _subset_collection


def _registered_volume(custom_params):
    collection = _subset_collection()
    scene = _block_scene(collection)
    yir.scale_scene_to_subset(scene)
    matrix = [[2.0, 0, 0, 1.0], [0, 2.0, 0, -3.0], [0, 0, 2.0, 0.5], [0, 0, 0, 1]]
    volume = _identity_volume(matrix_world=matrix)
    volume.custom_params = custom_params
    yir._volume_scene_cache[volume.dataset.dataset_id] = (scene, None, "auto")
    return scene, collection, volume


@pytest.mark.parametrize("custom_params", [
    {},
    {"yt_idv_zoom_target": "0.45 0.47 0.46", "yt_idv_zoom_level": 2.0},
    {"yt_idv_zoom_target": "0.45 0.47 0.46", "yt_idv_zoom_level": 1.0, "yt_idv_shot_space": 3.0},
])
def test_world_to_unitary_agrees_with_the_render_registration(custom_params):
    scene, collection, volume = _registered_volume(custom_params)
    try:
        camera_world = np.array([0.3, -2.0, 0.9])
        render_volume = yir._shot_space_volume(scene, volume, custom_params)
        model_frame = yir.apply_zoom(scene, custom_params, render_volume, camera_world)
        A_model, b_model = yir._world_to_model(render_volume, model_frame)
        A, b = yir.world_to_unitary(volume)
        offset = np.asarray(collection.applied_scale_offset)
        for world in np.random.default_rng(0).uniform(-2.0, 2.0, (5, 3)):
            unitary = A @ world + b
            model = (unitary - offset) / collection.applied_scale_ratio
            np.testing.assert_allclose(model, A_model @ (world - b_model), rtol=1e-9, atol=1e-9)
    finally:
        yir._volume_scene_cache.clear()


def test_world_to_unitary_needs_a_render():
    with pytest.raises(LookupError):
        yir.world_to_unitary(_identity_volume())


def test_to_pixels_marks_nan_as_empty():
    pixels = probe_analysis.to_pixels(np.array([[1.0, np.nan]]))
    assert pixels.shape == (1, 2, 4)
    np.testing.assert_array_equal(pixels[0, 0], [1, 1, 1, 1])
    np.testing.assert_array_equal(pixels[0, 1], [0, 0, 0, 0])


_calls = []


def counting_analysis(ctx):
    _calls.append(ctx)
    return probe_analysis.ProbeResult(values=np.full((2, 3), float(len(_calls))), meta={"Value Max": 2.0})


def test_run_recomputes_only_for_a_new_key():
    change_tracking.reset()
    probe_analysis.clear_cache()
    _calls.clear()
    spec = f"{__name__}:counting_analysis"
    key = probe_analysis.probe_key("d", (0.5, 0.5, 0.5), 0.1, spec, {})

    result, fresh = probe_analysis.run("p", key, lambda: "ctx", spec)
    assert fresh and len(_calls) == 1 and result.values.shape == (2, 3, 4)
    assert probe_analysis.run("p", key, lambda: "ctx", spec)[1] is False
    assert len(_calls) == 1

    bigger = probe_analysis.probe_key("d", (0.5, 0.5, 0.5), 0.2, spec, {})
    assert probe_analysis.run("p", bigger, lambda: "ctx", spec)[1] is True
    assert len(_calls) == 2
    # back to the first sphere: cached, but the Image holds the other result
    result, fresh = probe_analysis.run("p", key, lambda: "ctx", spec)
    assert fresh and len(_calls) == 2 and result.values[0, 0, 0] == 1.0


def test_phase_plot_fits_ranges_once_and_marks_empty_bins():
    yt_testing = pytest.importorskip("yt.testing")
    ds = yt_testing.fake_random_ds(16, fields=("density", "temperature"), units=("g/cm**3", "K"))
    center, radius = ds.domain_center, ds.quan(0.3, "unitary")
    state = {}
    ctx = probe_analysis.ProbeContext(
        ds=ds, sphere=ds.sphere(center, radius), center=center, radius=radius,
        frame=FrameParams(1, 0.0), params={"x_bins": "16", "y_bins": "8", "z_field": "gas, cell_mass"}, state=state,
    )
    result = probe_analysis.phase_plot(ctx)
    assert result.values.shape == (8, 16)
    assert np.isnan(result.values).any() and np.isfinite(result.values).any()
    assert {"x_range", "y_range", "value_range"} <= set(state)
    assert result.meta["X Min"] < result.meta["X Max"]
    assert "density" in result.meta["X Title"]
    json.dumps(state)

    fitted = dict(state)
    ctx.sphere = ds.sphere(center, radius / 2)
    probe_analysis.phase_plot(ctx)
    assert state == fitted

    ctx.params["fit"] = "every"
    probe_analysis.phase_plot(ctx)
    assert state["x_range"] != fitted["x_range"]


@pytest.fixture
def blender_probe():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    annotations.register()
    probes.register()
    change_tracking.reset()
    probe_analysis.clear_cache()
    _calls.clear()
    volume = bpy.data.objects.new("V", None)
    volume.empty_display_type = "CUBE"
    bpy.context.scene.collection.objects.link(volume)
    volume.volume_override.is_volume_override = True
    probe = bpy.data.objects.new("P", None)
    probe.empty_display_type = "SPHERE"
    bpy.context.scene.collection.objects.link(probe)
    probe.yt_probe.is_probe = True
    probe.yt_probe.volume = volume
    probe.yt_probe.analysis = f"{__name__}:counting_analysis"
    yield volume, probe
    probes.unregister()
    annotations.unregister()
    properties.unregister()
    yir._volume_scene_cache.clear()


def _fake_load(volume):
    import yt.testing

    ds = yt.testing.fake_random_ds(8)
    collection = _subset_collection()
    yir._volume_scene_cache[properties.ensure_dataset_uuid(volume)] = (_block_scene(collection), ds, "auto")


def test_update_probe_writes_the_image_and_panel_inputs(blender_probe):
    pytest.importorskip("yt")
    volume, probe = blender_probe
    scene = bpy.context.scene
    with pytest.raises(LookupError):
        probes.update_probe(scene, probe)
    _fake_load(volume)

    assert probes.update_probe(scene, probe) is True
    image = probe.yt_probe.image
    assert tuple(image.size) == (3, 2) and image.is_float and image["Value Max"] == 2.0

    panel = annotations.add_from_library(scene, probes.PANEL_GROUP)
    annotations._set_input(panel.modifiers[0], "Image", image)
    assert probes.panels(image) == [(panel, panel.modifiers[0])]
    assert probes.update_probe(scene, probe) is False  # nothing moved
    probe.location.x += 0.1
    bpy.context.view_layer.update()
    assert probes.update_probe(scene, probe) is True
    item = next(i for i in panel.modifiers[0].node_group.interface.items_tree if getattr(i, "name", "") == "Value Max")
    assert getattr(panel.modifiers[0].properties.inputs, item.identifier).value == 2.0
    assert len(_calls) == 2


def test_add_probe_links_a_panel_to_its_image(blender_probe):
    volume, _ = blender_probe
    probe = probes.add_probe(bpy.context, volume)
    assert probe.yt_probe.volume == volume and probe.empty_display_type == "SPHERE"
    [(panel, _)] = probes.panels(probe.yt_probe.image)
    assert panel.users_scene[0] == annotations.annotation_scene(bpy.context.scene)
