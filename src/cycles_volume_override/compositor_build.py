"""Programmatic compositor node graph: Cycles beauty (read back from EXR,
NOT a live Render Layers node -- see operators.py's two-pass rationale)
composited with N external-volume-renderer layers via chained,
premultiplied Alpha Over, no Z-Combine anywhere.

Supports BOTH compositor scripting APIs, detected at runtime (no Blender
version string ever parsed -- see `_is_group_api` and README "Blender 5.0+
compositor"):

  Classic (Blender 3.x/4.x): `scene.node_tree` (auto-created by
  `scene.use_nodes = True`), `CompositorNodeComposite` as the graph's sink,
  `CompositorNodeOutputFile.base_path` / `.file_slots[i].path`.

  Blender 5.0+: the compositor tree became an independent `CompositorNodeTree`
  datablock (`scene.compositing_node_group`, not auto-created), the graph's
  sink is a `NodeTreeInterface`-defined OUTPUT socket fed by a
  `NodeGroupOutput` node (there is no `CompositorNodeComposite` node type at
  all anymore), and `CompositorNodeOutputFile` gained a `file_output_items`
  collection: `.directory` + `.file_name` (a node-level filename template --
  literal `#` runs in it get zero-padded-frame-number substituted, same
  convention as `output_properties.filepath`; NOT auto-appended for you the
  way classic `file_slots[i].path` is) replace `.base_path`; `item.name` is
  now purely an in-container part/layer label, NOT the on-disk filename.
  `node.format.file_format` is locked to `'OPEN_EXR_MULTILAYER'` once you try
  to set it directly (the node's own format is the multi-part-EXR container
  format); getting a plain single-layer OPEN_EXR out of one item requires
  `item.override_node_format = True` + configuring `item.format` instead.
  All empirically confirmed against pip `bpy==5.2.0` -- see
  `scripts/verify_zpass.py` (`get_or_create_compositor_tree`/
  `_make_file_output`), which independently reached the same shape.

Graph shape per frame (all nodes prefixed `VO_` so this coexists with any
of your own unrelated compositor nodes in the same tree):

    Image(VO_Beauty) --\\
                         AlphaOver(farthest volume) --\\
    Image(VO_Vol_<far>)-/                              AlphaOver(next) -- ... -- [sink]
                                          Image(VO_Vol_<near>)-/

`[sink]` is `CompositorNodeComposite` (classic) or `NodeGroupOutput` wired to
the tree's "Image" interface output socket (5.0+) -- see `_link_final_output`.

Ordering resolves "Open question 1" (multiple flagged volumes per frame):
each volume is composited independently against the same Cycles beauty/Z,
then chained back-to-front (farthest camera-distance first) via Alpha Over,
so a nearer volume correctly draws over a farther one. This does NOT handle
volume-through-volume self-occlusion within a single ray interval -- two
volumes that are actually interleaved in depth along the same ray will still
just layer in distance-sorted order, not blend physically correctly. Noted
as a v1 limitation in the README alongside the solid-fully-enclosed-by-
volume case.
"""

import os

import bpy

NODE_PREFIX = "VO_"
FINAL_OUTPUT_SOCKET_NAME = "Image"

# Blender < 5.0's default (Filmic-based) OCIO config names its scene-linear
# working space "Linear". Blender 5.0 ships a different default OCIO config
# where that exact name no longer exists at all (confirmed empirically
# against pip bpy==5.2.0 -- assigning "Linear" raises `TypeError`; its
# equivalent is named "Linear Rec.709" there instead). Every OCIO config
# Blender ships also defines a portable `scene_linear` ROLE alias that
# resolves to whichever space that config actually calls it, so it's tried
# last and should keep working even against some future config neither of
# these two literal names matches.
LINEAR_COLORSPACE_CANDIDATES = ("Linear", "Linear Rec.709", "scene_linear")


def _is_group_api(scene):
    return hasattr(scene, "compositing_node_group")


def set_linear_colorspace(image):
    """Assign `image`'s input color space to whichever name this Blender's
    loaded OCIO config actually uses for "scene-linear, no view transform"
    -- see `LINEAR_COLORSPACE_CANDIDATES`. Used for every EXR this add-on
    reads or writes as raw linear data (beauty/volume-layer reads here,
    per-object render results in operators.py) so none of it gets an
    unwanted display/view color transform applied.
    """
    for name in LINEAR_COLORSPACE_CANDIDATES:
        try:
            image.colorspace_settings.name = name
            return
        except TypeError:
            continue
    raise RuntimeError(
        f"none of {LINEAR_COLORSPACE_CANDIDATES!r} is a valid color space/role "
        "name on this Blender -- its OCIO config uses some other name; add it "
        "to LINEAR_COLORSPACE_CANDIDATES"
    )


def ensure_compositor_tree(scene):
    """Version-agnostic accessor for the scene's compositor node tree --
    creates it if it doesn't exist yet, exactly like classic
    `scene.use_nodes = True` used to do implicitly.
    """
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


def _clear_managed_nodes(tree):
    for node in list(tree.nodes):
        if node.name.startswith(NODE_PREFIX):
            tree.nodes.remove(node)


def _load_fresh_image(path, node_label):
    """Load `path` as a brand-new Image datablock, replacing any datablock
    left over from a previous frame under the same name -- this is the
    "reloaded fresh each frame, not cached" requirement. We give the image
    datablock a per-node stable name (not the frame-numbered filename) so
    each call cleanly replaces the previous frame's data instead of
    accumulating one orphaned Image datablock per frame over an animation.
    """
    existing = bpy.data.images.get(node_label)
    if existing is not None:
        bpy.data.images.remove(existing)
    img = bpy.data.images.load(path, check_existing=False)
    img.name = node_label
    set_linear_colorspace(img)
    img.source = "FILE"
    return img


def _make_image_node(tree, name, path):
    node = tree.nodes.new("CompositorNodeImage")
    node.name = name
    node.label = name
    node.image = _load_fresh_image(path, name)
    return node


def _color_output_socket(image_node):
    """The Image node's primary RGBA output socket.

    Normally named "Image" -- true for any plain, non-multi-part EXR, which
    covers every per-object result `operators.write_render_result()` writes
    (plain `bpy.data.images.new()` + `.save()`, no part name involved). But
    Blender 5.0+ names this node's output after the EXR's internal
    part/channel-group name instead when the part itself was named --
    confirmed empirically true of this module's OWN beauty/depth EXRs, which
    get their part name from `file_output_items`'s item name (see
    `_configure_single_file_output`) via a real `CompositorNodeOutputFile`
    node rather than a plain `Image.save()`. There is exactly one color
    output socket either way (no separate "Alpha" output appears in the
    named-part case), so falling back to `outputs[0]` when "Image" isn't
    present covers both without this module needing to know or replicate
    whatever name it gave that file_output_items entry.
    """
    return image_node.outputs.get("Image") or image_node.outputs[0]


def _make_alpha_over(tree, name, background_socket, foreground_socket):
    node = tree.nodes.new("CompositorNodeAlphaOver")
    node.name = name
    node.label = name
    # Our inputs are already premultiplied per the external_render.py
    # contract (RenderResult.alpha_mode == "PREMULTIPLIED") and Cycles'
    # Combined pass is premultiplied by convention -- so we must NOT let
    # Alpha Over re-convert straight-alpha-to-premultiplied. Historically a
    # node-level `use_premultiply` checkbox (default off = "treat as already
    # premultiplied", which is what we want); Blender 5.x moved this to a
    # "Straight Alpha" input socket instead (confirmed default value is
    # already `False`, i.e. already-premultiplied -- this explicit set is
    # belt-and-suspenders, not correctness-critical). Handle both, default-
    # safe either way since both APIs default to the behavior we want.
    if hasattr(node, "use_premultiply"):
        node.use_premultiply = False
    straight_alpha_input = node.inputs.get("Straight Alpha")
    if straight_alpha_input is not None:
        straight_alpha_input.default_value = False
    tree.links.new(background_socket, node.inputs[0])
    tree.links.new(foreground_socket, node.inputs[1])
    return node


def _link_final_output(scene, tree, source_socket):
    """Wire `source_socket` to whatever this Blender's compositor treats as
    "the final composited result" -- a `CompositorNodeComposite` node
    (classic) or the tree's "Image" interface output socket via a
    `NodeGroupOutput` node (5.0+, where `CompositorNodeComposite` doesn't
    exist at all). Named/prefixed so `_clear_managed_nodes` tears it down
    and it gets rebuilt fresh next call, same as the classic Composite node
    always was.
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
    """Point `node` (a fresh `CompositorNodeOutputFile`) at a single
    full-float OPEN_EXR file per frame, named `<name_prefix><frame>.exr`.
    Returns the node's single data input socket to link a source into.

    Blender 5.0+: `directory` + `file_name` (a node-level filename template;
    a literal `####` run gets substituted with the zero-padded frame number
    -- NOT automatic, unlike classic `file_slots[i].path`, confirmed
    empirically) + one `file_output_items` entry, format overridden at the
    item level since the node's own `.format.file_format` is locked to
    'OPEN_EXR_MULTILAYER' once items exist.

    `item.name` is left **empty**, deliberately -- this is a real bug fix,
    not cosmetic. A non-empty item name makes Blender write this file as a
    named EXR "part" (confirmed via direct OpenEXR inspection: the channels
    end up grouped as e.g. `vo_depth_.R/.G/.B/.A` under that name instead of
    plain `R/G/B/A`). A NAMED part loads fine through the compositor's own
    Image node (used for the beauty EXR in pass 2 -- `_color_output_socket()`
    exists to shrug off its differently-named output socket) but empirically
    fails through the plain Python `bpy.data.images.load()` + `.pixels` path
    `operators.read_depth_buffer()` uses for the depth EXR: on a real
    Blender 5.x install this reproduced as `img.size == (0, 0)` /
    `img.channels == 0`, then a downstream `IndexError` -- i.e. Blender's
    Python-facing pixel-buffer API doesn't reliably support multi-part EXRs
    the way its internal compositor file reader does. An empty item name
    keeps the part unnamed with plain `RGBA` channels -- structurally
    identical to what `operators.write_render_result()`'s plain
    `Image.save()` already produces and which loads back fine -- so both
    read paths this add-on actually uses stay on the well-supported shape.
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


def build_or_update_graph(scene, beauty_exr_path, volume_layers):
    """(Re)builds the `VO_`-prefixed subgraph and links it to the final
    composited output.

    `volume_layers`: list of (object_name, exr_path), already ordered
    farthest-from-camera first, nearest last (see module docstring on
    Alpha Over chain ordering). Pass `[]` for a frame with no flagged
    volumes -- beauty is wired straight to the final output.
    """
    tree = ensure_compositor_tree(scene)
    _clear_managed_nodes(tree)

    beauty_node = _make_image_node(tree, _node_name("Beauty"), beauty_exr_path)
    current_output = _color_output_socket(beauty_node)

    for i, (obj_name, exr_path) in enumerate(volume_layers):
        vol_node = _make_image_node(tree, _node_name("Vol", obj_name), exr_path)
        alpha_over = _make_alpha_over(
            tree, _node_name("Over", str(i), obj_name),
            current_output, _color_output_socket(vol_node),
        )
        # AlphaOver is a regular processing node (not file-backed), so its
        # own output is always reliably named "Image" regardless of Blender
        # version -- unlike the Image nodes above, which read Blender-5.0-
        # renamed part names off disk (see _color_output_socket).
        current_output = alpha_over.outputs.get("Image")

    _link_final_output(scene, tree, current_output)

    _layout_nodes(tree)
    return tree


def build_pass1_capture_graph(scene, output_dir, frame_number):
    """Pass-1 graph: Render Layers -> File Output (Combined, full-float EXR)
    and Render Layers -> File Output (Depth/Z, full-float EXR).

    This is deliberately the ONLY thing pass 1's compositor does -- no
    Alpha Over, no volume layers (they don't exist yet; this pass is what
    produces the Z buffer they need). Requires `view_layer.use_pass_z = True`
    to already be set by the caller (operators.py), or the "Depth" output
    socket won't exist on the Render Layers node.

    Returns (beauty_path, depth_path) -- the exact files this frame's Pass 1
    wrote, for the caller to read back into a DepthBuffer / feed as
    `build_or_update_graph`'s `beauty_exr_path`.

    Implements spec item 3's requirement to read back passes via a File
    Output node rather than assuming in-memory access to render results is
    reliably available.
    """
    tree = ensure_compositor_tree(scene)
    _clear_managed_nodes(tree)

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

    # Also route Combined straight to the final output so pass 1's own render
    # result / viewport preview isn't blank -- irrelevant to what we
    # actually save (pass 2 does that), just avoids a jarring black frame
    # if you're watching the Image Editor while it renders.
    pass1_sink = _link_final_output(scene, tree, render_layers.outputs["Image"])
    pass1_sink.name = _node_name("Pass1Composite")

    _layout_nodes(tree)

    # Compute the exact path each File Output wrote so the caller doesn't
    # have to glob the output directory. Both APIs zero-pad to the same
    # width this project has always used (4 digits) -- classic via its own
    # automatic `####`-equivalent frame-number append, 5.0+ via the explicit
    # `####` this function put in `file_name` (see `_configure_single_file_output`).
    frame_suffix = f"{frame_number:04d}"
    beauty_path = os.path.join(output_dir, f"vo_beauty_{frame_suffix}.exr")
    depth_path = os.path.join(output_dir, f"vo_depth_{frame_suffix}.exr")
    return beauty_path, depth_path


def _layout_nodes(tree):
    """Cosmetic left-to-right layout for the nodes we manage, purely so the
    graph is readable if you open the Compositor workspace to inspect it.
    """
    managed = [n for n in tree.nodes if n.name.startswith(NODE_PREFIX)]
    x = 0
    for node in sorted(managed, key=lambda n: n.name):
        node.location = (x, 0)
        x += 220
