"""Compositor node graph: the pass-1 beauty (read back from EXR) composited
with N external volume layers via a chain of premultiplied Alpha Over nodes.

Supports both compositor scripting APIs, detected at runtime:

  Classic (Blender 3.x/4.x): `scene.node_tree`, `CompositorNodeComposite` as
  the sink, `CompositorNodeOutputFile.base_path` / `.file_slots[i].path`.

  Blender 5.0+: `scene.compositing_node_group` (a standalone
  `CompositorNodeTree`, not auto-created); the sink is a `NodeGroupOutput`
  feeding an interface OUTPUT socket (no Composite node exists); File Output
  uses `.directory` + `.file_name` (a `####` run is replaced by the frame
  number, not auto-appended) and `file_output_items`, with per-item format
  overrides since the node's own format is locked to OPEN_EXR_MULTILAYER.

Graph shape (all nodes prefixed `VO_`):

    Image(VO_Beauty) --\\
                         AlphaOver(farthest volume) --\\
    Image(VO_Vol_<far>)-/                              AlphaOver(next) -- ... -- [sink]
                                          Image(VO_Vol_<near>)-/

Volumes are layered back-to-front by camera distance; volumes interleaved in
depth along the same ray are not blended physically.
"""

import os

import bpy

NODE_PREFIX = "VO_"
FINAL_OUTPUT_SOCKET_NAME = "Image"

# "Linear" doesn't exist in Blender 5.0's default OCIO config. "Non-Color" is
# the only tag that is a true passthrough on both save and load; the others
# are fallbacks for configs without it.
LINEAR_COLORSPACE_CANDIDATES = ("Non-Color", "Linear Rec.709", "scene_linear")

# Colorspace handling:
# - Saving a float image tagged "Linear Rec.709" (the `images.new()` default)
#   bakes an sRGB encode into the EXR, so result EXRs are written "Non-Color".
# - The beauty pass is scene-linear and is read as "Non-Color"; the Standard
#   view transform applies the one sRGB encode it needs.
# - Volume layers (yt_idv colormap output) are already display-referred, so
#   they are read as "sRGB" to cancel the Standard view transform's encode.


def _is_group_api(scene):
    return hasattr(scene, "compositing_node_group")


def set_linear_colorspace(image):
    """Tag `image` with a passthrough colorspace (no transform on save or load)."""
    for name in LINEAR_COLORSPACE_CANDIDATES:
        try:
            image.colorspace_settings.name = name
            return
        except TypeError:
            continue
    raise RuntimeError(
        f"none of {LINEAR_COLORSPACE_CANDIDATES} is a valid colorspace name "
        "on this Blender's OCIO config"
    )


def set_display_referred_colorspace(image):
    """Tag a display-referred volume layer as "sRGB" (see colorspace notes above)."""
    image.colorspace_settings.name = "sRGB"


def ensure_compositor_tree(scene):
    """Return the scene's compositor node tree, creating it if needed."""
    if _is_group_api(scene):
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new(f"{scene.name} Compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
        if not any(
            item.item_type == "SOCKET" and item.in_out == "OUTPUT" and item.name == FINAL_OUTPUT_SOCKET_NAME
            for item in tree.interface.items_tree
        ):
            tree.interface.new_socket(
                name=FINAL_OUTPUT_SOCKET_NAME, in_out="OUTPUT", socket_type="NodeSocketColor",
            )
        return tree
    scene.use_nodes = True
    return scene.node_tree


def _node_name(*parts):
    return NODE_PREFIX + "_".join(parts)


_PASS1_NODE_PREFIX = _node_name("Pass1")


def _clear_managed_nodes(tree, exclude_prefix=None, only_prefix=None):
    """Remove `VO_`-prefixed nodes, optionally limited to `only_prefix` or
    skipping `exclude_prefix`.

    Both passes share one tree, so each pass's sweep must exclude the other
    pass's nodes; otherwise they delete each other's graphs every frame and
    force full rebuilds.
    """
    for node in list(tree.nodes):
        if not node.name.startswith(NODE_PREFIX):
            continue
        if only_prefix is not None and not node.name.startswith(only_prefix):
            continue
        if exclude_prefix is not None and node.name.startswith(exclude_prefix):
            continue
        tree.nodes.remove(node)


def _load_fresh_image(path, node_label, colorspace_fn=set_linear_colorspace):
    """Point the Image datablock named `node_label` at `path`, reusing it
    across calls.

    Removing and re-loading an Image datablock every frame invalidates
    Cycles' persistent-data cache, so the datablock is reloaded in place.
    """
    img = bpy.data.images.get(node_label)
    if img is None:
        img = bpy.data.images.load(path, check_existing=False)
        img.name = node_label
    else:
        img.filepath = path
        img.reload()
    colorspace_fn(img)
    img.source = "FILE"
    return img


def _make_image_node(tree, name, path, colorspace_fn=set_linear_colorspace):
    node = tree.nodes.new("CompositorNodeImage")
    node.name = name
    node.label = name
    node.image = _load_fresh_image(path, name, colorspace_fn)
    return node


def _color_output_socket(image_node):
    """The Image node's color output socket.

    Blender 5.0+ names it after the EXR part name when the part is named, so
    fall back to `outputs[0]` when there is no "Image" socket.
    """
    return image_node.outputs.get("Image") or image_node.outputs[0]


def _make_alpha_over(tree, name, background_socket, foreground_socket):
    node = tree.nodes.new("CompositorNodeAlphaOver")
    node.name = name
    node.label = name
    # Inputs are already premultiplied. Classic API uses `use_premultiply`;
    # Blender 5.x uses a "Straight Alpha" input socket.
    if hasattr(node, "use_premultiply"):
        node.use_premultiply = False
    straight_alpha_input = node.inputs.get("Straight Alpha")
    if straight_alpha_input is not None:
        straight_alpha_input.default_value = False
    tree.links.new(background_socket, node.inputs[0])
    tree.links.new(foreground_socket, node.inputs[1])
    return node


def _link_final_output(scene, tree, source_socket):
    """Wire `source_socket` to the final output: a Composite node (classic)
    or a `NodeGroupOutput` on the "Image" interface socket (5.0+).
    """
    if _is_group_api(scene):
        group_out = tree.nodes.new("NodeGroupOutput")
        group_out.name = _node_name("Composite")
        group_out.label = _node_name("Composite")
        tree.links.new(source_socket, group_out.inputs[FINAL_OUTPUT_SOCKET_NAME])
        return group_out
    composite = tree.nodes.new("CompositorNodeComposite")
    composite.name = _node_name("Composite")
    composite.label = _node_name("Composite")
    tree.links.new(source_socket, composite.inputs[0])
    return composite


def _configure_single_file_output(node, directory, name_prefix):
    """Configure a fresh File Output node to write one full-float OPEN_EXR
    per frame, named `<name_prefix><frame>.exr`. Returns its input socket.

    The item name is left empty: a named item makes Blender write a named
    EXR part, which fails to load through `bpy.data.images.load()` +
    `.pixels` (size (0, 0)).
    """
    if hasattr(node, "file_output_items"):
        node.directory = directory
        node.file_name = f"{name_prefix}####"
        item = node.file_output_items.new("RGBA", "")
        item.override_node_format = True
        item.format.file_format = "OPEN_EXR"
        item.format.color_depth = "32"
        return node.inputs[0]
    node.base_path = directory
    node.format.file_format = "OPEN_EXR"
    node.format.color_depth = "32"
    node.file_slots[0].path = name_prefix
    return node.inputs[0]


_VOLUME_ORDER_KEY = "vo_volume_order"


def build_or_update_graph(scene, beauty_exr_path, volume_layers):
    """Build the pass-2 graph, or update the existing Image nodes in place if
    the same ordered volume layers are already built.

    `volume_layers`: list of (object_name, exr_path), ordered farthest from
    the camera first. Pass `[]` to wire beauty straight to the output.

    Creating/deleting nodes invalidates Cycles' persistent-data cache, so a
    rebuild only happens when the layer names or their order change. The
    built order is stored on the tree under `_VOLUME_ORDER_KEY`.
    """
    tree = ensure_compositor_tree(scene)

    expected_order = [obj_name for obj_name, _ in volume_layers]
    beauty_node = tree.nodes.get(_node_name("Beauty"))
    topology_matches = (
        beauty_node is not None
        and list(tree.get(_VOLUME_ORDER_KEY, [])) == expected_order
    )

    if topology_matches:
        # Reassigning `.image`, even to the same datablock, invalidates
        # Cycles' persistent-data cache, so only assign when it differs.
        # bpy returns a fresh wrapper per access, so compare `.as_pointer()`.
        beauty_img = _load_fresh_image(beauty_exr_path, _node_name("Beauty"), set_linear_colorspace)
        if beauty_node.image is None or beauty_node.image.as_pointer() != beauty_img.as_pointer():
            beauty_node.image = beauty_img
        for obj_name, exr_path in volume_layers:
            vol_node = tree.nodes[_node_name("Vol", obj_name)]
            vol_img = _load_fresh_image(
                exr_path, _node_name("Vol", obj_name), set_display_referred_colorspace,
            )
            if vol_node.image is None or vol_node.image.as_pointer() != vol_img.as_pointer():
                vol_node.image = vol_img
        _mute_pass1_file_outputs(tree)
        return tree

    _clear_managed_nodes(tree, exclude_prefix=_PASS1_NODE_PREFIX)

    beauty_node = _make_image_node(tree, _node_name("Beauty"), beauty_exr_path)
    current_output = _color_output_socket(beauty_node)

    for i, (obj_name, exr_path) in enumerate(volume_layers):
        vol_node = _make_image_node(
            tree, _node_name("Vol", obj_name), exr_path, set_display_referred_colorspace,
        )
        alpha_over = _make_alpha_over(
            tree, _node_name("Over", str(i), obj_name),
            current_output, _color_output_socket(vol_node),
        )
        current_output = alpha_over.outputs.get("Image")

    _link_final_output(scene, tree, current_output)
    tree[_VOLUME_ORDER_KEY] = expected_order

    _layout_nodes(tree)
    _mute_pass1_file_outputs(tree)
    return tree


def _mute_pass1_file_outputs(tree):
    """Mute pass 1's File Output nodes so pass 2's render doesn't rewrite
    their EXRs. Muting doesn't invalidate the persistent-data cache.
    """
    beauty_out = tree.nodes.get(_node_name("Pass1BeautyOut"))
    if beauty_out is not None:
        beauty_out.mute = True
    depth_out = tree.nodes.get(_node_name("Pass1DepthOut"))
    if depth_out is not None:
        depth_out.mute = True


def _redirect_single_file_output(node, directory):
    """Point an already-configured File Output node at a new directory.

    Calling `_configure_single_file_output` again would add a duplicate item
    on the 5.0+ API.
    """
    if hasattr(node, "file_output_items"):
        node.directory = directory
    else:
        node.base_path = directory


def build_pass1_capture_graph(scene, output_dir, frame_number):
    """Pass-1 graph: Render Layers -> File Output for Combined and Depth
    (full-float EXRs). Requires `view_layer.use_pass_z = True`.

    Returns (beauty_path, depth_path) for this frame. Built once; later calls
    only repoint the output directory, to avoid invalidating Cycles'
    persistent-data cache.
    """
    tree = ensure_compositor_tree(scene)

    # Remove the obsolete pass-1 output node from trees built by older versions.
    stale_sink = tree.nodes.get(_node_name("Pass1Composite"))
    if stale_sink is not None:
        tree.nodes.remove(stale_sink)

    render_layers = tree.nodes.get(_node_name("Pass1RLayers"))
    if render_layers is None:
        _clear_managed_nodes(tree, only_prefix=_PASS1_NODE_PREFIX)

        render_layers = tree.nodes.new("CompositorNodeRLayers")
        render_layers.name = _node_name("Pass1RLayers")

        beauty_out = tree.nodes.new("CompositorNodeOutputFile")
        beauty_out.name = _node_name("Pass1BeautyOut")
        beauty_socket = _configure_single_file_output(beauty_out, output_dir, "vo_beauty_")
        tree.links.new(render_layers.outputs["Image"], beauty_socket)

        depth_out = tree.nodes.new("CompositorNodeOutputFile")
        depth_out.name = _node_name("Pass1DepthOut")
        depth_socket = _configure_single_file_output(depth_out, output_dir, "vo_depth_")
        tree.links.new(render_layers.outputs["Depth"], depth_socket)

        # No final-output node for pass 1: Blender would use it as the active
        # output during pass 2 too, dropping the volume layers.

        _layout_nodes(tree)
    else:
        _redirect_single_file_output(tree.nodes[_node_name("Pass1BeautyOut")], output_dir)
        _redirect_single_file_output(tree.nodes[_node_name("Pass1DepthOut")], output_dir)

    # Pass 2 mutes these; unmute so pass 1 writes its EXRs.
    tree.nodes[_node_name("Pass1BeautyOut")].mute = False
    tree.nodes[_node_name("Pass1DepthOut")].mute = False

    # Both APIs write a 4-digit zero-padded frame number.
    frame_suffix = f"{frame_number:04d}"
    beauty_path = os.path.join(output_dir, f"vo_beauty_{frame_suffix}.exr")
    depth_path = os.path.join(output_dir, f"vo_depth_{frame_suffix}.exr")
    return beauty_path, depth_path


def _layout_nodes(tree):
    """Cosmetic left-to-right layout of the managed nodes."""
    managed = [n for n in tree.nodes if n.name.startswith(NODE_PREFIX)]
    x = 0
    for node in sorted(managed, key=lambda n: n.name):
        node.location = (x, 0)
        x += 220
