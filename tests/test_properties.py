import bpy
import pytest

from yt_volumes import properties


@pytest.fixture
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    yield bpy.context.scene
    properties.unregister()


def test_to_custom_params_includes_fixed_and_extra_fields(clean_scene):
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    obj.volume_override.is_volume_override = True
    obj.volume_override.tf_n_layers = 7
    obj.volume_override.dataset_path = "/tmp/dataset"
    obj.volume_override.dataset_kind = "yt"
    extra = obj.volume_override.custom_params_extra.add()
    extra.key = "field_name"
    extra.value = "density"

    params = obj.volume_override.to_custom_params()
    assert params["yt_idv_tf_n_layers"] == 7
    assert params["renderer_mode"] == "YT_IDV_MAX_INTENSITY"
    assert params["field_name"] == "density"
    assert params["yt_idv_samples_per_cell"] == 4.0
    # dataset identity is tracked separately by build_dataset_params().
    assert "dataset_path" not in params
    assert "dataset_kind" not in params


def test_ensure_dataset_uuid_generates_once_and_persists(clean_scene):
    bpy.ops.object.empty_add(type='CUBE')
    obj = bpy.context.active_object
    assert obj.volume_override.dataset_uuid == ""

    first = properties.ensure_dataset_uuid(obj)
    assert first
    second = properties.ensure_dataset_uuid(obj)
    assert second == first


def test_flagged_objects_filters_by_flag(clean_scene):
    scene = clean_scene
    bpy.ops.object.empty_add(type='CUBE', location=(0, 0, 0))
    flagged = bpy.context.active_object
    flagged.name = "Flagged"
    flagged.volume_override.is_volume_override = True

    bpy.ops.object.empty_add(type='CUBE', location=(2, 0, 0))
    plain = bpy.context.active_object
    plain.name = "Plain"

    names = {obj.name for obj in properties.flagged_objects(scene)}
    assert names == {"Flagged"}


def _flag(name, scene):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    obj.volume_override.is_volume_override = True
    return obj


def test_flagged_objects_follows_renames_deletes_and_unflags(clean_scene):
    scene = clean_scene
    a, b = _flag("A", scene), _flag("B", scene)
    assert properties.flagged_objects(scene) == [a, b]

    a.name = "C"
    assert [o.name for o in properties.flagged_objects(scene)] == ["B", "C"]

    # Swapping names keeps the object count, and "A" now names an unflagged object.
    plain = bpy.data.objects.new("plain", None)
    scene.collection.objects.link(plain)
    properties.flagged_objects(scene)
    a.name, plain.name = "A", "C"
    assert [o.name for o in properties.flagged_objects(scene)] == ["A", "B"]

    b.volume_override.is_volume_override = False
    assert [o.name for o in properties.flagged_objects(scene)] == ["A"]

    bpy.data.objects.remove(a)
    assert properties.flagged_objects(scene) == []


def test_flagged_objects_survives_a_delete_and_add_between_calls(clean_scene):
    scene = clean_scene
    a = _flag("A", scene)
    assert properties.flagged_objects(scene) == [a]
    bpy.data.objects.remove(a)
    b = _flag("B", scene)
    assert properties.flagged_objects(scene) == [b]


def test_flagged_objects_skips_objects_outside_the_scene(clean_scene):
    scene = clean_scene
    v = _flag("V", scene)
    scene.collection.objects.unlink(v)
    assert properties.flagged_objects(scene) == []
    child = bpy.data.collections.new("child")
    scene.collection.children.link(child)
    child.objects.link(v)
    assert properties.flagged_objects(scene) == [v]


def test_flagged_objects_is_per_scene(clean_scene):
    other = bpy.data.scenes.new("Other")
    v = _flag("V", other)
    assert properties.flagged_objects(clean_scene) == []
    assert properties.flagged_objects(other) == [v]


def test_flagged_objects_rescans_after_loading_a_file(clean_scene, tmp_path):
    _flag("V", clean_scene)
    path = str(tmp_path / "flagged.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.data.objects.new("plain", None)  # same object count as the saved file
    assert properties.flagged_objects(bpy.context.scene) == []
    bpy.ops.wm.open_mainfile(filepath=path)
    assert [o.name for o in properties.flagged_objects(bpy.context.scene)] == ["V"]


def test_renderer_modes_keep_the_numbers_saved_files_use(clean_scene):
    obj = bpy.data.objects.new("V", None)
    numbers = {}
    for mode in ("YT_IDV_MAX_INTENSITY", "SAMPLE_CONSTANT", "YT_IDV_GN_TRANSFER_FUNCTION"):
        obj.volume_override.renderer_mode = mode
        numbers[mode] = obj.volume_override["renderer_mode"]
    assert numbers == {"YT_IDV_MAX_INTENSITY": 0, "SAMPLE_CONSTANT": 6, "YT_IDV_GN_TRANSFER_FUNCTION": 7}



def test_add_volume_makes_a_flagged_cube_with_a_transfer_function(clean_scene):
    from yt_volumes import transfer_function_nodes

    assert bpy.ops.volume_override.add_volume() == {"FINISHED"}

    obj = bpy.context.active_object
    assert (obj.type, obj.empty_display_type) == ("EMPTY", "CUBE")
    assert obj.volume_override.is_volume_override
    assert obj.volume_override.renderer_mode == "YT_IDV_GN_TRANSFER_FUNCTION"
    assert all(node is not None for node in transfer_function_nodes.find_nodes(obj))


def test_a_saved_transfer_function_evaluates_without_the_add_on(clean_scene, tmp_path):
    from yt_volumes import transfer_function_nodes

    bpy.ops.volume_override.add_volume()
    path = str(tmp_path / "volume.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    properties.unregister()
    try:
        bpy.ops.wm.open_mainfile(filepath=path)
        obj = bpy.data.objects["yt Volume"]
        assert not hasattr(obj, "volume_override")
        rgba = transfer_function_nodes.sample(obj, bpy.context.evaluated_depsgraph_get())
        assert rgba[-1] == pytest.approx([0.988, 0.998, 0.645, 0.8], abs=2e-3)
    finally:
        properties.register()  # the fixture unregisters


def test_choosing_a_yt_mode_sets_finer_sampling_unless_set_by_hand(clean_scene):
    props = bpy.data.objects.new("V", None).volume_override
    assert props.samples_per_cell == 4.0
    props.renderer_mode = "YT_IDV_TRANSFER_FUNCTION"
    assert props.samples_per_cell == 16.0
    props.renderer_mode = "YT_IDV_PROJECTION"
    assert props.samples_per_cell == 4.0
    props.samples_per_cell = 8.0
    props.renderer_mode = "YT_IDV_YT_TRANSFER_FUNCTION"
    assert props.samples_per_cell == 8.0
