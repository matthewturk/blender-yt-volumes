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

    panel = annotations.add_from_library(scene, probes.PANEL_GROUP, probe=probe)
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


def _input(obj, name):
    modifier = obj.modifiers[0]
    item = next(i for i in modifier.node_group.interface.items_tree if getattr(i, "name", "") == name)
    return getattr(modifier.properties.inputs, item.identifier).value


def _ds_with_time():
    import yt

    rng = np.random.default_rng(1)
    data = {"density": (rng.uniform(1.0, 2.0, (8, 8, 8)), "g/cm**3"), "temperature": (rng.uniform(1e3, 1e4, (8, 8, 8)), "K")}
    return yt.load_uniform_grid(data, (8, 8, 8), length_unit="kpc", time_unit="Myr", sim_time=3.0)


def _context(ds, params=None, state=None, frame=1):
    center, radius = ds.domain_center, ds.quan(0.3, "unitary")
    return probe_analysis.ProbeContext(
        ds=ds, sphere=ds.sphere(center, radius), center=center, radius=radius,
        frame=FrameParams(frame, 0.0), params=params or {}, state={} if state is None else state,
    )


def test_quantities_returns_named_scalars_with_units():
    pytest.importorskip("yt")
    ds = _ds_with_time()
    ctx = _context(ds, {"fields": "gas, density; gas, temperature"})
    result = probe_analysis.quantities(ctx)
    assert result.values is None
    expected = ctx.sphere.mean(("gas", "temperature"), weight=("gas", "mass"))
    assert result.scalars["temperature"] == pytest.approx(float(expected))
    assert result.meta["temperature Units"] == "K"
    assert set(result.scalars) == {"density", "temperature"}

    ctx.params.update(stat="max", fields="gas, temperature")
    assert probe_analysis.quantities(ctx).scalars["temperature"] == pytest.approx(float(ctx.sphere.max(("gas", "temperature"))))
    with pytest.raises(ValueError):
        probe_analysis.quantities(_context(ds, {"stat": "median"}))


def test_history_adds_a_point_per_run_and_passes_the_inner_values_through():
    pytest.importorskip("yt")
    ds = _ds_with_time()
    state = {}
    params = {"fields": "gas, temperature", "x": "frame"}
    first = probe_analysis.history(_context(ds, params, state, frame=1))
    assert first.values.shape == (1, 1, 4)
    second = probe_analysis.history(_context(ds, params, state, frame=5))
    pixels = second.values
    assert pixels.shape == (1, 2, 4)
    np.testing.assert_allclose(pixels[0, :, 0], [1.0, 5.0])
    assert pixels[0, 1, 1] == pytest.approx(np.log10(second.scalars["temperature"]), rel=1e-6)
    assert (pixels[0, :, 3] == 1.0).all()
    assert second.meta["Cursor X"] == 5.0 and second.meta["X Max"] == 5.0
    assert second.meta["Y Title"] == "log temperature (K)"
    json.dumps(state)

    # rerunning a frame replaces its point; time on x is the dataset's
    probe_analysis.history(_context(ds, params, state, frame=5))
    assert len(state["points"]) == 2
    fixed = probe_analysis.history(_context(ds, {**params, "x_range": "1, 100", "y_range": "3, 5"}, {}, frame=1))
    assert (fixed.meta["X Min"], fixed.meta["X Max"], fixed.meta["Y Min"], fixed.meta["Y Max"]) == (1, 100, 3, 5)
    timed = probe_analysis.history(_context(ds, {"x": "time", "time_units": "Myr"}, {}, frame=1))
    assert timed.values[0, 0, 0] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        probe_analysis.history(_context(ds, {"quantity": "pressure"}, {}))


def scalar_analysis(ctx):
    _calls.append(ctx)
    return probe_analysis.ProbeResult(
        scalars={"temperature": 100.0 * ctx.frame.frame_number}, meta={"temperature Units": "K"}
    )


def test_a_new_ds_for_the_same_dataset_reruns_the_probe(blender_probe):
    pytest.importorskip("yt")
    volume, probe = blender_probe
    scene = bpy.context.scene
    _fake_load(volume)
    assert probes.update_probe(scene, probe) is True
    assert probes.update_probe(scene, probe) is False
    _fake_load(volume)  # as a time-varying loader does each frame
    assert probes.update_probe(scene, probe) is True


def test_scalars_go_on_the_probe_and_into_labels(blender_probe):
    pytest.importorskip("yt")
    volume, probe = blender_probe
    scene = bpy.context.scene
    probe.yt_probe.analysis = f"{__name__}:scalar_analysis"
    probe.yt_probe.update = "EVERY_FRAME"
    _fake_load(volume)
    label = annotations.add_from_library(scene, probes.LABEL_GROUP, probe=probe)
    assert _input(label, "Probe") == probe

    assert probes.update_probe(scene, probe) is True
    assert probe.yt_probe.image is None  # values only
    assert probe["yt_values"]["temperature"] == pytest.approx(100.0)
    assert _input(label, "Value") == pytest.approx(100.0) and _input(label, "Units") == "K"
    assert probes.update_probe(scene, probe) is False  # same frame


def test_a_driver_in_the_annotation_scene_sees_the_new_value(blender_probe):
    pytest.importorskip("yt")
    volume, probe = blender_probe
    scene = bpy.context.scene
    probe.yt_probe.analysis = f"{__name__}:scalar_analysis"
    probe.yt_probe.update = "EVERY_FRAME"
    _fake_load(volume)
    probes.update_probe(scene, probe)
    overlay = annotations.add_annotation_scene(scene)
    follower = bpy.data.objects.new("Follower", None)
    overlay.collection.objects.link(follower)
    driver = follower.driver_add("location", 0).driver
    driver.type = "AVERAGE"
    variable = driver.variables.new()
    variable.targets[0].id = probe
    variable.targets[0].data_path = '["yt_values"]["temperature"]'

    scene.frame_set(3)
    probes.update_probe(scene, probe)
    depsgraph = overlay.view_layers[0].depsgraph
    depsgraph.update()
    assert follower.evaluated_get(depsgraph).location.x == pytest.approx(300.0)


def test_bake_keyframes_each_frames_values(blender_probe):
    pytest.importorskip("yt")
    volume, probe = blender_probe
    scene = bpy.context.scene
    probe.yt_probe.analysis = f"{__name__}:scalar_analysis"
    probe.yt_probe.update = "EVERY_FRAME"
    _fake_load(volume)
    scene.frame_set(2)

    assert probes.bake(scene, probe, range(1, 5)) == ["temperature"]
    assert scene.frame_current == 2
    for frame in (1, 4, 2):  # only keyframes can bring back frame 1's value
        scene.frame_set(frame)
        assert probe["yt_values"]["temperature"] == pytest.approx(100.0 * frame)


@pytest.mark.parametrize("kind, group, image", [
    ("PHASE", probes.PANEL_GROUP, True), ("HISTORY", probes.LINE_GROUP, True), ("VALUES", probes.LABEL_GROUP, False),
])
def test_each_kind_of_probe_gets_its_annotation(blender_probe, kind, group, image):
    volume, _ = blender_probe
    probe = probes.add_probe(bpy.context, volume, kind)
    [(panel, modifier)] = probes.panels(probe=probe)
    assert modifier.node_group.name == group and _input(panel, "Probe") == probe
    assert (probe.yt_probe.image is not None) == image
    assert probe.yt_probe.analysis == probes.KINDS[kind][1]


def test_probe_annotations_need_a_probe_but_not_a_units_object(blender_probe):
    volume, probe = blender_probe
    assert annotations.unavailable(probes.LINE_GROUP, volume) == "Select a probe"
    assert annotations.unavailable(probes.LINE_GROUP, volume, probe) is None
    assert annotations.unavailable("yt Time Label", volume) is not None  # needs the units object
