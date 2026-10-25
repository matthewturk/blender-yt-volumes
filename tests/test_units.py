import json
import pathlib
import sys
import zipfile

import bpy
import numpy as np
import pytest

from yt_volumes import properties, units
from yt_volumes import yt_idv_renderer as yir

yt = pytest.importorskip("yt")


@pytest.fixture
def scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    units.register()
    yield bpy.context.scene
    units.unregister()
    properties.unregister()


def _volume(scene, size=1.0):
    obj = bpy.data.objects.new("Volume", None)
    obj.empty_display_type = "CUBE"
    obj.empty_display_size = size
    obj.volume_override.is_volume_override = True
    scene.collection.objects.link(obj)
    return obj


def _kpc_dataset():
    # a 10 kpc box: 1 unitary = 10 kpc
    return yt.load_uniform_grid({"density": np.ones((4, 4, 4))}, (4, 4, 4), length_unit="kpc", bbox=np.array([[0.0, 10.0]] * 3))


def test_scene_per_unitary_follows_the_box_the_scale_and_the_zoom(scene):
    obj = _volume(scene)  # a 2-unit cube
    assert units.scene_per_unitary(obj, scene) == pytest.approx(2.0)
    obj.scale = (10.0, 10.0, 10.0)
    bpy.context.view_layer.update()
    assert units.scene_per_unitary(obj, scene) == pytest.approx(20.0)
    obj.volume_override.zoom_target = "max"
    obj.volume_override.zoom_level = 1.5
    assert units.scene_per_unitary(obj, scene) == pytest.approx(20.0 * 10**1.5)


def test_data_units_convert_through_the_registry(scene):
    obj = _volume(scene, size=5.0)  # a 10-unit cube
    obj.volume_override.data_units = "kpc, pc, cm, Msun"
    obj["yt_units"] = {"registry": _kpc_dataset().unit_registry.to_json()}

    units.update(obj, scene)

    group = obj["yt_units"]
    assert group["unitary_in_kpc"] == pytest.approx(10.0)
    assert group["scene_per_kpc"] == pytest.approx(1.0)
    assert group["scene_per_pc"] == pytest.approx(1e-3)
    assert group["scene_per_cm"] == pytest.approx(1.0 / 3.0856775814913673e21, rel=1e-6)
    assert "scene_per_Msun" not in group  # not a length


def test_scene_files_carry_their_units(tmp_path):
    path = tmp_path / "scene.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("scene.json", json.dumps({"format_version": 1}))
    assert units.scene_file_units(str(path)) is None

    units.add_units_to_scene_file(str(path), _kpc_dataset())

    registry = units.scene_file_units(str(path))
    assert units.unitary_in(registry, "kpc") == pytest.approx(10.0)
    with pytest.raises(ValueError):
        units.add_units_to_scene_file(str(path), _kpc_dataset())


def test_a_scene_json_dataset_entry_is_read_too(tmp_path):
    path = tmp_path / "scene.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("scene.json", json.dumps({"format_version": 1, "dataset": units.dataset_info(_kpc_dataset())}))
    assert units.unitary_in(units.scene_file_units(str(path)), "pc") == pytest.approx(1e4)


def test_recording_a_loaded_dataset_takes_its_registry_and_cube(scene, monkeypatch):
    obj = _volume(scene)
    ds = _kpc_dataset()
    collection = type("C", (), {"applied_scale_offset": np.zeros(3), "applied_scale_ratio": 0.25})()
    component = type("Comp", (), {"name": "block_rendering", "data": collection})()
    monkeypatch.setitem(yir._volume_scene_cache, "d", (type("S", (), {"components": [component]})(), ds, "egl"))

    units.record(obj, "d", scene)

    # the 2-unit cube spans a quarter of the domain: 8 scene units per unitary, 10 kpc
    assert obj["yt_units"]["scene_per_unitary"] == pytest.approx(8.0)
    assert obj["yt_units"]["scene_per_kpc"] == pytest.approx(0.8)


def test_a_driver_reads_the_conversion_after_a_frame_change(scene):
    obj = _volume(scene)
    obj["yt_units"] = {"registry": _kpc_dataset().unit_registry.to_json()}
    obj.volume_override.zoom_target = "0.5, 0.5, 0.5"
    obj.volume_override.zoom_level = 0.0
    obj.volume_override.keyframe_insert("zoom_level", frame=1)
    obj.volume_override.zoom_level = 2.0
    obj.volume_override.keyframe_insert("zoom_level", frame=3)
    units.update(obj, scene)  # the property exists before a driver uses it, as in the UI
    follower = bpy.data.objects.new("Follower", None)
    scene.collection.objects.link(follower)
    driver = follower.driver_add("scale", 0).driver
    driver.type = "AVERAGE"
    variable = driver.variables.new()
    variable.targets[0].id = obj
    variable.targets[0].data_path = '["yt_units"]["scene_per_kpc"]'

    scene.frame_set(3)
    bpy.context.view_layer.update()

    assert driver.is_valid
    assert follower.scale[0] == pytest.approx(2.0 / 10.0 * 100.0, rel=1e-5)  # 0.2 per kpc, zoomed 100x


def test_a_subset_scenes_cube_is_its_bounding_boxs_longest_side(tmp_path):
    path = tmp_path / "subset.zip"
    with zipfile.ZipFile(path, "w") as zf:
        state = {"format_version": 1, "data_objects": [{"attributes": {"_bbox": {"__array__": "arrays/000000.npy"}}}]}
        zf.writestr("scene.json", json.dumps(state))
        with zf.open("arrays/000000.npy", "w") as f:
            np.lib.format.write_array(f, np.array([[0.4, 0.45, 0.42], [0.48, 0.5, 0.47]]))
    assert units.scene_file_cube(str(path)) == pytest.approx(0.08)


def _reader(scene, units_object, attribute):
    """A mesh whose Geometry Nodes store `attribute`, sampled from `units_object`'s vertex 0, as "k"."""
    tree = bpy.data.node_groups.new("Read Units", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = tree.nodes, tree.links
    group_in, group_out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    info = nodes.new("GeometryNodeObjectInfo")
    info.inputs["Object"].default_value = units_object
    named = nodes.new("GeometryNodeInputNamedAttribute")
    named.data_type = "FLOAT"
    named.inputs["Name"].default_value = attribute
    sample = nodes.new("GeometryNodeSampleIndex")
    sample.data_type, sample.domain = "FLOAT", "POINT"
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type, store.domain = "FLOAT", "POINT"
    store.inputs["Name"].default_value = "k"
    links.new(info.outputs["Geometry"], sample.inputs["Geometry"])
    links.new(named.outputs["Attribute"], sample.inputs["Value"])
    links.new(group_in.outputs[0], store.inputs["Geometry"])
    links.new(sample.outputs["Value"], store.inputs["Value"])
    links.new(store.outputs["Geometry"], group_out.inputs[0])
    mesh = bpy.data.meshes.new("Reader")
    mesh.vertices.add(1)
    reader = bpy.data.objects.new("Reader", mesh)
    scene.collection.objects.link(reader)
    reader.modifiers.new("Read", "NODES").node_group = tree
    return reader


def _read_k(reader):
    evaluated = reader.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return evaluated.data.attributes["k"].data[0].value


def test_geometry_nodes_read_the_units_object(scene):
    obj = _volume(scene)
    obj.volume_override.data_units = "kpc, pc"
    obj["yt_units"] = {"registry": _kpc_dataset().unit_registry.to_json()}
    obj.volume_override.zoom_target = "0.5, 0.5, 0.5"
    obj.volume_override.zoom_level = 0.0
    obj.volume_override.keyframe_insert("zoom_level", frame=1)
    obj.volume_override.zoom_level = 2.0
    obj.volume_override.keyframe_insert("zoom_level", frame=3)
    scene.frame_set(1)

    units_object = units.add_units_object(obj, scene)
    reader = _reader(scene, units_object, "scene_per_kpc")
    bpy.context.view_layer.update()
    assert _read_k(reader) == pytest.approx(0.2, rel=1e-6)  # 2 scene units per 10 kpc
    assert units_object.hide_render

    scene.frame_set(3)
    bpy.context.view_layer.update()
    assert _read_k(reader) == pytest.approx(20.0, rel=1e-6)  # zoomed 100x

    obj.volume_override.data_units = "pc"
    units.update(obj, scene)
    names = {a.name for a in units_object.data.attributes}
    assert "scene_per_pc" in names and "scene_per_kpc" not in names


def _registered_cube(obj, unitary, offset, ratio):
    """Volume-cube coordinates of unitary points, via the renderer's own registration."""
    from yt_volumes import operators

    volume = type("V", (), {"matrix_world": [list(r) for r in obj.matrix_world], "bounding_box_local": operators._empty_cube_bounding_box_local(obj)})()
    transform = units.unitary_transform(obj, bpy.context.scene)
    origin, linear, world_origin = transform
    world = (np.asarray(unitary) - origin) @ linear.T + world_origin
    A, b = yir._world_to_model(volume)
    return (world - b) @ A.T, (np.asarray(unitary) - offset) / ratio


def test_unitary_transform_matches_the_renderers_registration(scene):
    obj = _volume(scene, size=1.5)
    obj.location = (3.0, -2.0, 1.0)
    obj.rotation_euler = (0.3, -0.7, 1.1)
    obj.scale = (2.0, 2.0, 2.0)
    obj["yt_units"] = {"cube_in_unitary": 0.25, "cube_origin_unitary": [0.4, 0.3, 0.5]}
    bpy.context.view_layer.update()
    points = np.array([[0.4, 0.3, 0.5], [0.65, 0.55, 0.75], [0.5, 0.41, 0.62]])

    registered, expected = _registered_cube(obj, points, np.array([0.4, 0.3, 0.5]), 0.25)

    assert registered == pytest.approx(expected, abs=1e-12)


def test_unitary_transform_follows_the_volume_zoom(scene):
    obj = _volume(scene)
    target = np.array([0.25, 0.5, 0.75])
    unzoomed = units.unitary_transform(obj, scene)
    obj.volume_override.zoom_target = "0.25, 0.5, 0.75"
    obj.volume_override.zoom_level = 3.0

    origin, linear, world_origin = units.unitary_transform(obj, scene)

    # The target stays put and everything around it grows 1000x.
    assert origin == pytest.approx(target)
    assert world_origin == pytest.approx(unzoomed[1] @ (target - unzoomed[0]) + unzoomed[2])
    assert linear == pytest.approx(unzoomed[1] * 1000.0)


def test_unitary_transform_waits_for_a_max_target_and_skips_spheres(scene):
    obj = _volume(scene)
    obj.volume_override.zoom_target = "max"
    assert units.unitary_transform(obj, scene) is None
    obj["yt_units"] = {"zoom_max_unitary": [0.1, 0.2, 0.3]}
    assert units.unitary_transform(obj, scene)[0] == pytest.approx([0.1, 0.2, 0.3])
    obj.volume_override.zoom_target = ""
    obj.empty_display_type = "SPHERE"
    assert units.unitary_transform(obj, scene) is None


def test_unitary_transform_uses_deep_zoom_shot_space(scene, monkeypatch):
    from yt_volumes import operators

    obj = _volume(scene)
    monkeypatch.setattr(operators, "_deep_zoom_params", lambda s: {
        "yt_idv_shot_space": 2.0, "yt_idv_zoom_target": "0.1, 0.2, 0.3", "yt_idv_zoom_level": 6.0})

    origin, linear, world_origin = units.unitary_transform(obj, scene)

    assert origin == pytest.approx([0.1, 0.2, 0.3]) and world_origin == pytest.approx(0.0)
    assert linear == pytest.approx(np.eye(3) * 2e6)


def _placed(scene, units_object, unitary, location=(0.0, 0.0, 0.0)):
    """Evaluated positions of a mesh, at `location`, whose vertices "yt Unitary to Scene" moves to `unitary`."""
    mesh = bpy.data.meshes.new("Placed")
    mesh.vertices.add(len(unitary))
    units.store_unitary_positions(mesh, unitary)
    obj = bpy.data.objects.new("Placed", mesh)
    obj.location = location
    scene.collection.objects.link(obj)
    tree = bpy.data.node_groups.new("Place", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, link = tree.nodes, tree.links.new
    group_in, group_out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    convert = nodes.new("GeometryNodeGroup")
    convert.node_tree = units.unitary_node_group()
    convert.inputs["Units Object"].default_value = units_object
    for socket, name in (("Unitary", "unitary"), ("Unitary Low", "unitary_low")):
        named = nodes.new("GeometryNodeInputNamedAttribute")
        named.data_type = "FLOAT_VECTOR"
        named.inputs["Name"].default_value = name
        link(named.outputs["Attribute"], convert.inputs[socket])
    set_position = nodes.new("GeometryNodeSetPosition")
    link(group_in.outputs[0], set_position.inputs["Geometry"])
    link(convert.outputs["Position"], set_position.inputs["Position"])
    link(set_position.outputs["Geometry"], group_out.inputs[0])
    obj.modifiers.new("Place", "NODES").node_group = tree
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    local = np.array([v.co for v in evaluated.data.vertices], dtype=np.float64)
    return local + np.asarray(location)


def test_geometry_nodes_place_unitary_positions_exactly_at_any_zoom(scene):
    obj = _volume(scene)
    obj.location = (5.0, 1.0, -2.0)
    target = np.array([0.123456789012345, 0.678901234567890, 0.314159265358979])
    obj.volume_override.zoom_target = ", ".join(repr(float(v)) for v in target)
    obj.volume_override.zoom_level = 9.0
    bpy.context.view_layer.update()
    units_object = units.add_units_object(obj, scene)
    # Points 1e-10 apart: 0.1 scene units apart at this zoom, far below float32 in unitary.
    unitary = target + np.array([[0.0, 0.0, 0.0], [1e-10, 0.0, 0.0], [0.0, -3e-10, 2e-10]])

    placed = _placed(scene, units_object, unitary, location=(-1.0, 2.0, 0.5))

    origin, linear, world_origin = units.unitary_transform(obj, scene)
    expected = (unitary - origin) @ linear.T + world_origin
    assert placed == pytest.approx(expected, abs=1e-5)
    assert placed[1] - placed[0] == pytest.approx([0.2, 0.0, 0.0], abs=1e-5)  # a 2-unit cube, 1e9x


def test_split_keeps_float64_digits():
    values = np.array([0.123456789012345, 1e-12, 0.999999999999])
    high, low = units.split(values)
    assert high.dtype == low.dtype == np.float32
    assert high.astype(np.float64) + low.astype(np.float64) == pytest.approx(values, rel=1e-14, abs=1e-20)


def test_a_typed_position_fills_both_inputs(scene):
    tree = bpy.data.node_groups.new("Uses", "GeometryNodeTree")
    node = tree.nodes.new("GeometryNodeGroup")
    node.node_tree = units.unitary_node_group()
    units.set_unitary_inputs(node, [0.123456789012345, 0.5, 1e-11])
    total = np.array(node.inputs["Unitary"].default_value, dtype=np.float64) + np.array(node.inputs["Unitary Low"].default_value)
    assert total == pytest.approx([0.123456789012345, 0.5, 1e-11], rel=1e-14, abs=1e-24)


def test_arranged_groups_flow_left_to_right(scene):
    tree = units.unitary_node_group()
    for link in tree.links:
        assert link.from_node.location.x < link.to_node.location.x
    group_out = next(n for n in tree.nodes if n.bl_idname == "NodeGroupOutput")
    assert group_out.location.x == max(n.location.x for n in tree.nodes)
