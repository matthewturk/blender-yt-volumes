import os

import bpy
import numpy as np
import pytest

from cycles_volume_override import compositor_build as cb


@pytest.fixture
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    yield bpy.context.scene


def test_ensure_compositor_tree_creates_group_and_image_output_on_blender_5_api(clean_scene):
    # Pins the CURRENT behavior on Blender >= 5.0 (which this test suite runs
    # under, via pip bpy==5.2.0): ensure_compositor_tree() must create the
    # scene's compositing_node_group datablock (nothing auto-creates it, unlike
    # classic scene.use_nodes) and make sure it has an "Image" OUTPUT socket
    # on its interface, since that -- fed by a NodeGroupOutput node -- is what
    # 5.0+ uses in place of the classic CompositorNodeComposite node. See
    # module docstring and README "Blender 5.0+ compositor". If you run this
    # same test suite against a real Blender < 5.0, this test should be
    # skipped/updated since the premise (>= 5.0 API) no longer holds there.
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

    # Idempotent: calling again on an already-set-up scene doesn't create a
    # second tree or a duplicate "Image" socket.
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
    # Pins a real bug found on a real Blender 5.x install: a non-empty item
    # name makes Blender write this file as a NAMED EXR part, which
    # bpy.data.images.load() (operators.read_depth_buffer's read path)
    # reproducibly fails to decode -- see this function's docstring and
    # test_pass1_depth_exr_round_trips_through_bpy_data_images_load below
    # for the full end-to-end reproduction.
    assert item.name == ""
    assert item.override_node_format is True
    assert item.format.file_format == "OPEN_EXR"
    assert item.format.color_depth == "32"
    assert socket == node.inputs[0]


def test_pass1_depth_exr_round_trips_through_bpy_data_images_load(clean_scene, tmp_path):
    # End-to-end reproduction of a real bug found on a real Blender 5.x
    # install: operators.read_depth_buffer() crashed with "index 0 is out of
    # bounds for axis 2 with size 0" because bpy.data.images.load() silently
    # decoded the depth EXR as 0x0/0-channel. Traced to
    # _configure_single_file_output writing a NAMED file_output_items entry;
    # this test exercises the exact same bpy.data.images.load() call
    # operators.read_depth_buffer() makes, against a real render, and would
    # fail the same way if that regressed.
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


def test_set_linear_colorspace_accepts_this_blenders_ocio_config(clean_scene):
    # Pins the actual bug found while porting to Blender 5.0+: the classic
    # hardcoded name "Linear" no longer exists in the OCIO config pip
    # bpy==5.2.0 ships (it's called "Linear Rec.709" there) -- assigning it
    # directly raises TypeError. set_linear_colorspace() must resolve to
    # whatever name this Blender's config actually accepts instead of
    # blowing up on a fresh image datablock.
    img = bpy.data.images.new("t", width=2, height=2, alpha=True, float_buffer=True)
    cb.set_linear_colorspace(img)
    assert img.colorspace_settings.name in cb.LINEAR_COLORSPACE_CANDIDATES
    assert img.colorspace_settings.is_data is False


def test_build_pass1_capture_graph_returns_paths_that_actually_get_written(clean_scene, tmp_path):
    # End-to-end on the real (5.0+) API: render a trivial scene through the
    # pass-1 graph and confirm the two paths build_pass1_capture_graph()
    # predicts are exactly the files Blender wrote -- no globbing needed.
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
    # End-to-end regression for a real bug found while porting: pass 1's
    # beauty EXR is written via a named CompositorNodeOutputFile item, which
    # on Blender 5.0+ makes the Image node that reads it back expose its
    # output under THAT name instead of the generic "Image" -- while a
    # per-object result (written the way operators.write_render_result()
    # actually does, plain Image.save(), no part name) keeps the generic
    # name. build_or_update_graph must composite both correctly; hardcoding
    # `.outputs.get("Image")` for the beauty side returns None and crashes
    # the very next link (this is what _color_output_socket exists to fix).
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
