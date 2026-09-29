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


def test_set_linear_colorspace_forces_a_genuine_passthrough_tag(clean_scene):
    # See the comment block above LINEAR_COLORSPACE_CANDIDATES in
    # compositor_build.py for the full story: `bpy.data.images.new()`'s own
    # default ("Linear Rec.709") looks like a passthrough tag but actually
    # bakes an sRGB-shaped encode into the file at `.save()` time -- fixed by
    # actively forcing "Non-Color" instead, which set_linear_colorspace does
    # unconditionally now (no longer a no-op).
    img = bpy.data.images.new("t", width=2, height=2, alpha=True, float_buffer=True)
    assert img.colorspace_settings.name == "Linear Rec.709"  # the buggy default, confirmed unchanged
    cb.set_linear_colorspace(img)
    assert img.colorspace_settings.name == "Non-Color"


def test_set_linear_colorspace_prevents_the_save_time_encode_bake(clean_scene, tmp_path):
    # THE direct regression test for bug 1 (see the comment block above
    # LINEAR_COLORSPACE_CANDIDATES): saving a `.new()` float image at its own
    # default colorspace bakes an sRGB-shaped encode into the EXR on disk.
    # Verified via direct OpenEXR inspection, bypassing Blender's own image
    # loader entirely, so this doesn't depend on trusting bpy's colorspace
    # auto-detection on read.
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
    # Beauty-layer semantics: genuinely linear data, written via
    # set_linear_colorspace (matching write_render_result -- though beauty
    # itself is actually written natively by Blender's own File Output node,
    # not this function, this test stands in for "some genuinely-linear EXR"
    # generically) and read back the same way (the default `colorspace_fn`
    # on `_make_image_node`, used for the beauty node in build_or_update_graph).
    # Textbook reference value: linear 0.5 sRGB-encodes to 188/255 -- exactly
    # ONE real gamma correction (Standard view transform's own, at final
    # output), which is what genuinely-linear data actually needs to display
    # correctly. See the comment block above LINEAR_COLORSPACE_CANDIDATES in
    # compositor_build.py for the two-bugs-stacked story this test used to be
    # entangled with.
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
    # Volume-layer semantics: already-display-ready colormap data (NOT
    # scene-linear), written via set_linear_colorspace (matching
    # write_render_result -- a true passthrough on disk) and read back via
    # set_display_referred_colorspace (matching build_or_update_graph's
    # per-object Image node). Standard view transform's one mandatory encode
    # at final output must come out canceled by the compensating decode this
    # read-side tag applies, netting to IDENTITY: a genuinely-linear-style
    # single-gamma-correction test (the one above) would be WRONG here --
    # colormap bytes must survive unchanged, not get gamma-corrected as if
    # they were raw light data. See the comment block above
    # LINEAR_COLORSPACE_CANDIDATES in compositor_build.py.
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


# --- reuse-in-place instead of rebuild-every-frame (perf) ---------------------
# Real perf bug found doing an optimization pass on real per-frame timings:
# unconditionally tearing down and rebuilding these graphs every frame, even
# though their SHAPE usually never changes, was found to invalidate Cycles'
# own use_persistent_data session/BVH cache just as badly as switching render
# engines did. Both graph builders now reuse existing nodes (just updating
# properties) when nothing structural actually changed, and only fall back to
# a real rebuild when it did.

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
    # .as_pointer() (the underlying C pointer, stable regardless of Python
    # wrapper churn) -- bpy's RNA wrapper objects aren't singleton-cached,
    # so plain `is` on two separate `tree.nodes[name]` lookups is NOT a
    # reliable identity check even for the exact same underlying node.
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
    # .as_pointer() -- see test_build_pass1_capture_graph_reuses_nodes_across_calls
    # for why plain `is` doesn't reliably check bpy node identity.
    beauty_ptr = tree.nodes[cb._node_name("Beauty")].as_pointer()
    vol_a_ptr = tree.nodes[cb._node_name("Vol", "A")].as_pointer()
    vol_b_ptr = tree.nodes[cb._node_name("Vol", "B")].as_pointer()
    over_node = tree.nodes.get(cb._node_name("Over", "0", "A"))
    assert over_node is not None
    over_ptr = over_node.as_pointer()

    vol_a_path2 = _build_solid_volume_image(tmp_path, "a2", green=0.35)
    vol_b_path2 = _build_solid_volume_image(tmp_path, "b2", green=0.45)
    cb.build_or_update_graph(scene, beauty_path, [("A", vol_a_path2), ("B", vol_b_path2)])

    # Same object names, same order -- nodes reused in place, not recreated.
    assert tree.nodes[cb._node_name("Beauty")].as_pointer() == beauty_ptr
    assert tree.nodes[cb._node_name("Vol", "A")].as_pointer() == vol_a_ptr
    assert tree.nodes[cb._node_name("Vol", "B")].as_pointer() == vol_b_ptr
    assert tree.nodes[cb._node_name("Over", "0", "A")].as_pointer() == over_ptr
    # But the image CONTENT was refreshed to the new call's paths.
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
    # Same SET of objects, different order -- e.g. the camera moved such
    # that two volumes swapped which one is closer. Must NOT be silently
    # treated as "topology unchanged": the Alpha Over chain's compositing
    # order has to match, or occlusion comes out wrong.
    scene = clean_scene
    tree = cb.ensure_compositor_tree(scene)
    beauty_path = _build_solid_volume_image(tmp_path, "beauty", green=0.1)
    vol_a_path = _build_solid_volume_image(tmp_path, "a", green=0.3)
    vol_b_path = _build_solid_volume_image(tmp_path, "b", green=0.4)

    cb.build_or_update_graph(scene, beauty_path, [("A", vol_a_path), ("B", vol_b_path)])
    assert tree.nodes.get(cb._node_name("Over", "0", "A")) is not None
    assert tree.nodes.get(cb._node_name("Over", "1", "B")) is not None

    cb.build_or_update_graph(scene, beauty_path, [("B", vol_b_path), ("A", vol_a_path)])

    # The old A-first chain nodes must be gone, replaced by a B-first chain.
    assert tree.nodes.get(cb._node_name("Over", "0", "A")) is None
    assert tree.nodes.get(cb._node_name("Over", "0", "B")) is not None
    assert tree.nodes.get(cb._node_name("Over", "1", "A")) is not None
