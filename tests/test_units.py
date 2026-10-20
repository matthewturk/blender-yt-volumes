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
