import bpy
import pytest

from cycles_volume_override import properties


@pytest.fixture
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    yield bpy.context.scene
    properties.unregister()


def test_to_custom_params_includes_fixed_and_extra_fields(clean_scene):
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    obj.volume_override.density_scale = 2.5
    obj.volume_override.vdb_filepath = "/tmp/smoke.vdb"
    extra = obj.volume_override.custom_params_extra.add()
    extra.key = "grid_name"
    extra.value = "density"

    params = obj.volume_override.to_custom_params()
    assert params["density_scale"] == 2.5
    assert params["vdb_filepath"] == "/tmp/smoke.vdb"
    assert params["grid_name"] == "density"
    assert len(params["absorption"]) == 3


def test_flagged_objects_filters_by_flag(clean_scene):
    scene = clean_scene
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    flagged = bpy.context.active_object
    flagged.name = "Flagged"
    flagged.volume_override.is_volume_override = True

    bpy.ops.mesh.primitive_cube_add(location=(2, 0, 0))
    plain = bpy.context.active_object
    plain.name = "Plain"

    names = {obj.name for obj in properties.flagged_objects(scene)}
    assert names == {"Flagged"}
