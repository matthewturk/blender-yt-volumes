import os

import bpy
import numpy as np
import pytest

from yt_volumes import compositor_build as cb


@pytest.fixture
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    yield bpy.context.scene


def test_ensure_compositor_tree_creates_group_and_image_output_on_blender_5_api(clean_scene):
    # Blender 5.0+: nothing auto-creates compositing_node_group, and its "Image"
    # output socket replaces the classic Composite node.
    assert hasattr(clean_scene, "compositing_node_group"), (
        "this test's premise (Blender >= 5.0 API) no longer holds -- "
        "update it for the Blender version actually running it"
    )
    assert clean_scene.compositing_node_group is None

    tree = cb.ensure_compositor_tree(clean_scene)

    assert clean_scene.compositing_node_group is tree
    image_sockets = [
        item for item in tree.interface.items_tree
        if item.item_type == "SOCKET" and item.in_out == "OUTPUT" and item.name == "Image"
    ]
    assert len(image_sockets) == 1

    # Idempotent.
    tree_again = cb.ensure_compositor_tree(clean_scene)
    assert tree_again is tree
    image_sockets_again = [
        item for item in tree.interface.items_tree
        if item.item_type == "SOCKET" and item.in_out == "OUTPUT" and item.name == "Image"
    ]
    assert len(image_sockets_again) == 1


def test_clear_managed_nodes_only_removes_prefixed_nodes():
    tree = bpy.data.node_groups.new("T", "CompositorNodeTree")
    ours = tree.nodes.new("CompositorNodeAlphaOver")
    ours.name = cb._node_name("Over", "0", "Foo")
    other = tree.nodes.new("CompositorNodeAlphaOver")
    other.name = "MyOwnNode"

    cb._clear_managed_nodes(tree)

    remaining = {n.name for n in tree.nodes}
    assert remaining == {"MyOwnNode"}


def test_link_final_output_wires_group_output_on_blender_5_api(clean_scene):
    tree = cb.ensure_compositor_tree(clean_scene)
    rl = tree.nodes.new("CompositorNodeRLayers")

    sink = cb._link_final_output(clean_scene, tree, rl.outputs["Image"])

    assert sink.bl_idname == "NodeGroupOutput"
    assert sink.name.startswith(cb.NODE_PREFIX)
    image_input = sink.inputs["Image"]
    assert image_input.is_linked
    assert image_input.links[0].from_socket == rl.outputs["Image"]


def test_configure_single_file_output_on_blender_5_api(clean_scene, tmp_path):
    tree = cb.ensure_compositor_tree(clean_scene)
    node = tree.nodes.new("CompositorNodeOutputFile")

    socket = cb._configure_single_file_output(node, str(tmp_path), "vo_beauty_")

    assert node.directory == str(tmp_path)
    assert node.file_name == "vo_beauty_####"
    assert len(node.file_output_items) == 1
    item = node.file_output_items[0]
    # Regression: a named item writes a named EXR part, which
    # bpy.data.images.load() fails to decode.
    assert item.name == ""
    assert item.override_node_format is True
    assert item.format.file_format == "OPEN_EXR"
    assert item.format.color_depth == "32"
    assert socket == node.inputs[0]


def test_pass1_depth_exr_round_trips_through_bpy_data_images_load(clean_scene, tmp_path):
    # Regression: a named file_output_items entry made bpy.data.images.load()
    # silently decode the depth EXR as 0x0 / 0 channels.
    scene = clean_scene
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(1.5708, 0, 0))
    scene.camera = bpy.context.object
    bpy.ops.object.light_add(type="SUN")
    scene.render.resolution_x = 8
    scene.render.resolution_y = 8
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    bpy.context.view_layer.use_pass_z = True

    scene.frame_set(1)
    beauty_path, depth_path = cb.build_pass1_capture_graph(scene, str(tmp_path), 1)
    bpy.ops.render.render(write_still=False)

    for path in (beauty_path, depth_path):
        img = bpy.data.images.load(path, check_existing=False)
        try:
            assert img.size[0] == 8 and img.size[1] == 8, (
                f"{path}: decoded as {img.size[:]} -- the named-part-EXR bug reproduces as (0, 0)"
            )
            assert img.channels > 0
        finally:
            bpy.data.images.remove(img)


def test_set_linear_colorspace_forces_a_genuine_passthrough_tag(clean_scene):
    # The .new() default ("Linear Rec.709") bakes an sRGB encode into the file
    # at .save() time; "Non-Color" is a true passthrough.
    img = bpy.data.images.new("t", width=2, height=2, alpha=True, float_buffer=True)
    assert img.colorspace_settings.name == "Linear Rec.709"  # the buggy default, confirmed unchanged
    cb.set_linear_colorspace(img)
    assert img.colorspace_settings.name == "Non-Color"


def test_set_linear_colorspace_prevents_the_save_time_encode_bake(clean_scene, tmp_path):
    # Regression: the default colorspace bakes an sRGB encode into the EXR.
    # Read back with OpenEXR directly, bypassing Blender's loader.
    import OpenEXR
    import Imath

    def save_and_read_back(colorspace_fn):
        img = bpy.data.images.new("t", width=2, height=2, alpha=True, float_buffer=True)
        if colorspace_fn is not None:
            colorspace_fn(img)
        px = np.full((2, 2, 4), 0.283, dtype=np.float32)
        px[..., 3] = 1.0
        img.pixels.foreach_set(px.ravel())
        path = str(tmp_path / f"probe_{colorspace_fn is not None}.exr")
        img.filepath_raw = path
        img.file_format = "OPEN_EXR"
        img.save()
        bpy.data.images.remove(img)
        exr = OpenEXR.InputFile(path)
        r = np.frombuffer(exr.channel("R", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
        return float(r[0])

    buggy = save_and_read_back(None)
    fixed = save_and_read_back(cb.set_linear_colorspace)
    assert buggy == pytest.approx(0.568, abs=0.01), (
        f"expected the OLD default to bake sRGB-encode(0.283)=0.568 on disk, got {buggy}"
    )
    assert fixed == pytest.approx(0.283, abs=1e-4), (
        f"expected set_linear_colorspace to give a true passthrough (0.283 in, 0.283 on disk), got {fixed}"
    )


def test_compositor_image_node_round_trips_a_known_linear_value_correctly(clean_scene, tmp_path):
    # Beauty-layer semantics: linear data gets exactly one gamma correction
    # (linear 0.5 -> sRGB 188/255).
    scene = clean_scene
    scene.render.resolution_x = 4
    scene.render.resolution_y = 4
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    out_path = str(tmp_path / "known_value_render")
    scene.render.filepath = out_path

    src = bpy.data.images.new("src", width=4, height=4, alpha=True, float_buffer=True)
    cb.set_linear_colorspace(src)
    px = np.full((4, 4, 4), 0.5, dtype=np.float32)
    px[..., 3] = 1.0
    src.pixels.foreach_set(px.ravel())
    exr_path = str(tmp_path / "known_value.exr")
    src.filepath_raw = exr_path
    src.file_format = "OPEN_EXR"
    src.save()
    bpy.data.images.remove(src)

    tree = cb.ensure_compositor_tree(scene)
    img_node = cb._make_image_node(tree, "Test_KnownValue", exr_path)
    socket = cb._color_output_socket(img_node)
    cb._link_final_output(scene, tree, socket)

    bpy.ops.render.render(write_still=True)

    from PIL import Image as PILImage
    out = np.array(PILImage.open(out_path + ".png"))
    assert tuple(out[0, 0, :3]) == (188, 188, 188), (
        f"expected linear 0.5 -> sRGB 188 (textbook value), got {tuple(out[0, 0, :3])}"
    )


def test_compositor_volume_layer_image_node_passes_through_display_ready_value_unchanged(clean_scene, tmp_path):
    # Volume-layer semantics: display-ready colormap data must survive
    # unchanged -- the read-side decode cancels the view transform's encode.
    scene = clean_scene
    scene.render.resolution_x = 4
    scene.render.resolution_y = 4
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    out_path = str(tmp_path / "volume_value_render")
    scene.render.filepath = out_path

    src = bpy.data.images.new("src", width=4, height=4, alpha=True, float_buffer=True)
    cb.set_linear_colorspace(src)  # what write_render_result does
    value = 0.283
    px = np.full((4, 4, 4), value, dtype=np.float32)
    px[..., 3] = 1.0
    src.pixels.foreach_set(px.ravel())
    exr_path = str(tmp_path / "volume_value.exr")
    src.filepath_raw = exr_path
    src.file_format = "OPEN_EXR"
    src.save()
    bpy.data.images.remove(src)

    tree = cb.ensure_compositor_tree(scene)
    img_node = cb._make_image_node(
        tree, "Test_VolumeValue", exr_path, cb.set_display_referred_colorspace,
    )
    socket = cb._color_output_socket(img_node)
    cb._link_final_output(scene, tree, socket)

    bpy.ops.render.render(write_still=True)

    from PIL import Image as PILImage
    out = np.array(PILImage.open(out_path + ".png"))
    expected = round(value * 255)
    assert tuple(int(c) for c in out[0, 0, :3]) == (expected, expected, expected), (
        f"expected display-ready {value} to survive as {expected} unchanged, got {tuple(out[0, 0, :3])}"
    )


def _write_exr(path, value, size=2):
    img = bpy.data.images.new("tmp_src", width=size, height=size, alpha=True, float_buffer=True)
    cb.set_linear_colorspace(img)
    px = np.full((size, size, 4), value, dtype=np.float32)
    px[..., 3] = 1.0
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = "OPEN_EXR"
    img.save()
    bpy.data.images.remove(img)


def test_load_fresh_image_reuses_the_same_datablock_across_calls(clean_scene, tmp_path):
    # Regression (perf): remove() + load() per call invalidated Cycles'
    # persistent-data cache.
    path_a = str(tmp_path / "a.exr")
    path_b = str(tmp_path / "b.exr")
    _write_exr(path_a, 0.2)
    _write_exr(path_b, 0.8)

    img_a = cb._load_fresh_image(path_a, "Test_Reused")
    pointer_after_first = img_a.as_pointer()
    buf_a = np.empty(2 * 2 * img_a.channels, dtype=np.float32)
    img_a.pixels.foreach_get(buf_a)

    img_b = cb._load_fresh_image(path_b, "Test_Reused")
    buf_b = np.empty(2 * 2 * img_b.channels, dtype=np.float32)
    img_b.pixels.foreach_get(buf_b)

    assert img_b.as_pointer() == pointer_after_first, "must reuse the same datablock, not recreate it"
    assert np.allclose(buf_a.reshape(2, 2, -1)[..., 0], 0.2, atol=1e-5)
    assert np.allclose(buf_b.reshape(2, 2, -1)[..., 0], 0.8, atol=1e-5)
    assert len([i for i in bpy.data.images if i.name.startswith("Test_Reused")]) == 1


def test_build_pass1_capture_graph_returns_paths_that_actually_get_written(clean_scene, tmp_path):
    scene = clean_scene
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(1.5708, 0, 0))
    scene.camera = bpy.context.object
    bpy.ops.object.light_add(type="SUN")
    scene.render.resolution_x = 8
    scene.render.resolution_y = 8
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    bpy.context.view_layer.use_pass_z = True

    scene.frame_set(3)
    beauty_path, depth_path = cb.build_pass1_capture_graph(scene, str(tmp_path), scene.frame_current)
    bpy.ops.render.render(write_still=False)

    assert os.path.exists(beauty_path), f"expected {beauty_path}, dir has {os.listdir(tmp_path)}"
    assert os.path.exists(depth_path), f"expected {depth_path}, dir has {os.listdir(tmp_path)}"
    assert beauty_path.endswith("vo_beauty_0003.exr")
    assert depth_path.endswith("vo_depth_0003.exr")


def test_build_or_update_graph_composites_named_part_beauty_and_result_exrs(clean_scene, tmp_path):
    # Regression: an Image node reading a named-part EXR has no "Image"
    # output, so hardcoding outputs.get("Image") crashed the link.
    scene = clean_scene
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(1.5708, 0, 0))
    scene.camera = bpy.context.object
    bpy.ops.object.light_add(type="SUN")
    scene.render.resolution_x = 8
    scene.render.resolution_y = 8
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    bpy.context.view_layer.use_pass_z = True

    scene.frame_set(1)
    beauty_path, depth_path = cb.build_pass1_capture_graph(scene, str(tmp_path), 1)
    bpy.ops.render.render(write_still=False)
    assert os.path.exists(beauty_path)

    h, w = 8, 8
    rgba = np.zeros((h, w, 4), dtype="float32")
    rgba[..., 1] = 0.5  # solid green, premultiplied alpha 0.5
    rgba[..., 3] = 0.5
    result_path = str(tmp_path / "vo_result_Cube_0001.exr")
    img = bpy.data.images.new("vo_result_tmp", width=w, height=h, alpha=True, float_buffer=True)
    cb.set_linear_colorspace(img)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = result_path
    img.file_format = "OPEN_EXR"
    img.save()
    bpy.data.images.remove(img)

    cb.build_or_update_graph(scene, beauty_path, [("Cube", result_path)])
    scene.render.engine = "BLENDER_WORKBENCH"
    final_path = str(tmp_path / "final.exr")
    scene.render.filepath = final_path
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    bpy.ops.render.render(write_still=True)  # would raise TypeError before the fix

    assert os.path.exists(final_path)


# --- reuse-in-place instead of rebuild-every-frame (perf) ---------------------
# Rebuilding the graphs every frame invalidated Cycles' persistent-data cache.

def test_build_pass1_capture_graph_reuses_nodes_across_calls(clean_scene, tmp_path):
    scene = clean_scene
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(1.5708, 0, 0))
    scene.camera = bpy.context.object
    bpy.ops.object.light_add(type="SUN")
    scene.render.resolution_x = 8
    scene.render.resolution_y = 8
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    bpy.context.view_layer.use_pass_z = True

    tree = cb.ensure_compositor_tree(scene)

    dir1 = tmp_path / "frame1"
    dir1.mkdir()
    beauty_path1, depth_path1 = cb.build_pass1_capture_graph(scene, str(dir1), 1)
    # Compare .as_pointer(): bpy wrappers aren't cached, so `is` is unreliable.
    rlayers_ptr = tree.nodes[cb._node_name("Pass1RLayers")].as_pointer()
    beauty_ptr = tree.nodes[cb._node_name("Pass1BeautyOut")].as_pointer()

    dir2 = tmp_path / "frame2"
    dir2.mkdir()
    scene.frame_set(2)  # #### substitution uses scene.frame_current, not the frame_number arg below
    beauty_path2, depth_path2 = cb.build_pass1_capture_graph(scene, str(dir2), 2)

    assert tree.nodes[cb._node_name("Pass1RLayers")].as_pointer() == rlayers_ptr, "node identity must be preserved, not recreated"
    assert tree.nodes[cb._node_name("Pass1BeautyOut")].as_pointer() == beauty_ptr
    assert beauty_path1 != beauty_path2  # different frame_dir each time -- the one thing that DOES need to change
    assert beauty_path1.startswith(str(dir1)) and beauty_path2.startswith(str(dir2))

    bpy.ops.render.render(write_still=False)
    assert os.path.exists(beauty_path2), f"expected {beauty_path2!r} after the reused-node render, dir has {os.listdir(dir2)}"


def test_pass1_and_pass2_graphs_coexist_across_interleaved_frames(clean_scene, tmp_path):
    # Regression (perf): each pass's rebuild swept the other pass's nodes,
    # forcing a full rebuild of both graphs every frame.
    scene = clean_scene
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(1.5708, 0, 0))
    scene.camera = bpy.context.object
    bpy.ops.object.light_add(type="SUN")
    scene.render.resolution_x = 8
    scene.render.resolution_y = 8
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    bpy.context.view_layer.use_pass_z = True

    tree = cb.ensure_compositor_tree(scene)
    beauty_path = _build_solid_volume_image(tmp_path, "beauty0", green=0.1)
    vol_path = _build_solid_volume_image(tmp_path, "vol0", green=0.2)

    frame1_dir = tmp_path / "frame1"
    frame1_dir.mkdir()
    real_beauty_path, _ = cb.build_pass1_capture_graph(scene, str(frame1_dir), 1)
    bpy.ops.render.render(write_still=False)
    cb.build_or_update_graph(scene, beauty_path, [("A", vol_path)])
    bpy.ops.render.render(write_still=False)

    rlayers_ptr = tree.nodes[cb._node_name("Pass1RLayers")].as_pointer()
    beauty_out_ptr = tree.nodes[cb._node_name("Pass1BeautyOut")].as_pointer()
    pass2_beauty_ptr = tree.nodes[cb._node_name("Beauty")].as_pointer()
    vol_ptr = tree.nodes[cb._node_name("Vol", "A")].as_pointer()

    frame2_dir = tmp_path / "frame2"
    frame2_dir.mkdir()
    scene.frame_set(2)
    cb.build_pass1_capture_graph(scene, str(frame2_dir), 2)
    bpy.ops.render.render(write_still=False)
    cb.build_or_update_graph(scene, beauty_path, [("A", vol_path)])
    bpy.ops.render.render(write_still=False)

    assert tree.nodes[cb._node_name("Pass1RLayers")].as_pointer() == rlayers_ptr, (
        "pass 1's own nodes must survive pass 2's build/render, not get swept away by it"
    )
    assert tree.nodes[cb._node_name("Pass1BeautyOut")].as_pointer() == beauty_out_ptr
    assert tree.nodes[cb._node_name("Beauty")].as_pointer() == pass2_beauty_ptr, (
        "pass 2's own nodes must survive pass 1's build/render, not get swept away by it"
    )
    assert tree.nodes[cb._node_name("Vol", "A")].as_pointer() == vol_ptr


def test_build_or_update_graph_mutes_pass1_file_outputs_and_pass1_unmutes_them(clean_scene, tmp_path):
    # Otherwise pass 1's persisting File Output nodes re-write their EXRs
    # during pass 2's render.
    scene = clean_scene
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(1.5708, 0, 0))
    scene.camera = bpy.context.object
    bpy.context.view_layer.use_pass_z = True

    tree = cb.ensure_compositor_tree(scene)
    frame_dir = tmp_path / "frame1"
    frame_dir.mkdir()
    cb.build_pass1_capture_graph(scene, str(frame_dir), 1)
    assert tree.nodes[cb._node_name("Pass1BeautyOut")].mute is False
    assert tree.nodes[cb._node_name("Pass1DepthOut")].mute is False

    beauty_path = _build_solid_volume_image(tmp_path, "beauty1", green=0.1)
    vol_path = _build_solid_volume_image(tmp_path, "vol1", green=0.2)
    cb.build_or_update_graph(scene, beauty_path, [("A", vol_path)])
    assert tree.nodes[cb._node_name("Pass1BeautyOut")].mute is True
    assert tree.nodes[cb._node_name("Pass1DepthOut")].mute is True

    scene.frame_set(2)
    cb.build_pass1_capture_graph(scene, str(frame_dir), 2)
    assert tree.nodes[cb._node_name("Pass1BeautyOut")].mute is False
    assert tree.nodes[cb._node_name("Pass1DepthOut")].mute is False


def test_pass1_and_pass2_graphs_share_exactly_one_final_output(clean_scene, tmp_path):
    # Regression: pass 1's own output node shadowed pass 2's, so frames
    # saved without the volume.
    scene = clean_scene
    bpy.ops.mesh.primitive_cube_add()
    bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(1.5708, 0, 0))
    scene.camera = bpy.context.object
    bpy.context.view_layer.use_pass_z = True
    output_type = "NodeGroupOutput" if cb._is_group_api(scene) else "CompositorNodeComposite"

    tree = cb.ensure_compositor_tree(scene)
    beauty_path = _build_solid_volume_image(tmp_path, "beauty1", green=0.1)
    vol_path = _build_solid_volume_image(tmp_path, "vol1", green=0.2)
    for frame in (1, 2):
        frame_dir = tmp_path / f"frame{frame}"
        frame_dir.mkdir()
        scene.frame_set(frame)
        cb.build_pass1_capture_graph(scene, str(frame_dir), frame)
        cb.build_or_update_graph(scene, beauty_path, [("A", vol_path)])

        outputs = [n for n in tree.nodes if n.bl_idname == output_type]
        assert [n.name for n in outputs] == [cb._node_name("Composite")]
        (link,) = [l for l in tree.links if l.to_node == outputs[0]]
        assert link.from_node.name == cb._node_name("Over", "0", "A")


def test_build_pass1_capture_graph_removes_obsolete_pass1_output_node(clean_scene, tmp_path):
    # Trees from older versions (e.g. saved .blends) still have this node.
    scene = clean_scene
    bpy.context.view_layer.use_pass_z = True
    tree = cb.ensure_compositor_tree(scene)
    cb.build_pass1_capture_graph(scene, str(tmp_path), 1)
    stale = cb._link_final_output(scene, tree, tree.nodes[cb._node_name("Pass1RLayers")].outputs["Image"])
    stale.name = cb._node_name("Pass1Composite")

    cb.build_pass1_capture_graph(scene, str(tmp_path), 2)

    assert tree.nodes.get(cb._node_name("Pass1Composite")) is None


def _build_solid_volume_image(tmp_path, name, green, alpha=0.5):
    h, w = 8, 8
    rgba = np.zeros((h, w, 4), dtype="float32")
    rgba[..., 1] = green
    rgba[..., 3] = alpha
    path = str(tmp_path / f"{name}.exr")
    img = bpy.data.images.new(f"{name}_tmp", width=w, height=h, alpha=True, float_buffer=True)
    cb.set_linear_colorspace(img)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = path
    img.file_format = "OPEN_EXR"
    img.save()
    bpy.data.images.remove(img)
    return path


def test_build_or_update_graph_reuses_nodes_when_same_objects_same_order(clean_scene, tmp_path):
    scene = clean_scene
    tree = cb.ensure_compositor_tree(scene)
    beauty_path = _build_solid_volume_image(tmp_path, "beauty", green=0.1)
    vol_a_path1 = _build_solid_volume_image(tmp_path, "a1", green=0.3)
    vol_b_path1 = _build_solid_volume_image(tmp_path, "b1", green=0.4)

    cb.build_or_update_graph(scene, beauty_path, [("A", vol_a_path1), ("B", vol_b_path1)])
    beauty_ptr = tree.nodes[cb._node_name("Beauty")].as_pointer()
    vol_a_ptr = tree.nodes[cb._node_name("Vol", "A")].as_pointer()
    vol_b_ptr = tree.nodes[cb._node_name("Vol", "B")].as_pointer()
    over_node = tree.nodes.get(cb._node_name("Over", "0", "A"))
    assert over_node is not None
    over_ptr = over_node.as_pointer()

    vol_a_path2 = _build_solid_volume_image(tmp_path, "a2", green=0.35)
    vol_b_path2 = _build_solid_volume_image(tmp_path, "b2", green=0.45)
    cb.build_or_update_graph(scene, beauty_path, [("A", vol_a_path2), ("B", vol_b_path2)])

    assert tree.nodes[cb._node_name("Beauty")].as_pointer() == beauty_ptr
    assert tree.nodes[cb._node_name("Vol", "A")].as_pointer() == vol_a_ptr
    assert tree.nodes[cb._node_name("Vol", "B")].as_pointer() == vol_b_ptr
    assert tree.nodes[cb._node_name("Over", "0", "A")].as_pointer() == over_ptr
    assert tree.nodes[cb._node_name("Vol", "A")].image.filepath_raw.endswith("a2.exr")
    assert tree.nodes[cb._node_name("Vol", "B")].image.filepath_raw.endswith("b2.exr")


def test_build_or_update_graph_rebuilds_when_object_set_changes(clean_scene, tmp_path):
    scene = clean_scene
    tree = cb.ensure_compositor_tree(scene)
    beauty_path = _build_solid_volume_image(tmp_path, "beauty", green=0.1)
    vol_a_path = _build_solid_volume_image(tmp_path, "a", green=0.3)

    cb.build_or_update_graph(scene, beauty_path, [("A", vol_a_path)])
    assert tree.nodes.get(cb._node_name("Vol", "B")) is None

    vol_b_path = _build_solid_volume_image(tmp_path, "b", green=0.4)
    cb.build_or_update_graph(scene, beauty_path, [("A", vol_a_path), ("B", vol_b_path)])

    assert tree.nodes.get(cb._node_name("Vol", "A")) is not None
    assert tree.nodes.get(cb._node_name("Vol", "B")) is not None


def test_build_or_update_graph_rebuilds_when_order_changes(clean_scene, tmp_path):
    # Depth order changed (e.g. camera moved): Alpha Over chain must follow.
    scene = clean_scene
    tree = cb.ensure_compositor_tree(scene)
    beauty_path = _build_solid_volume_image(tmp_path, "beauty", green=0.1)
    vol_a_path = _build_solid_volume_image(tmp_path, "a", green=0.3)
    vol_b_path = _build_solid_volume_image(tmp_path, "b", green=0.4)

    cb.build_or_update_graph(scene, beauty_path, [("A", vol_a_path), ("B", vol_b_path)])
    assert tree.nodes.get(cb._node_name("Over", "0", "A")) is not None
    assert tree.nodes.get(cb._node_name("Over", "1", "B")) is not None

    cb.build_or_update_graph(scene, beauty_path, [("B", vol_b_path), ("A", vol_a_path)])

    assert tree.nodes.get(cb._node_name("Over", "0", "A")) is None
    assert tree.nodes.get(cb._node_name("Over", "0", "B")) is not None
    assert tree.nodes.get(cb._node_name("Over", "1", "A")) is not None
