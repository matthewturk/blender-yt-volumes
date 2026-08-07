"""Programmatic compositor node graph: Cycles beauty (read back from EXR,
NOT a live Render Layers node -- see operators.py's two-pass rationale)
composited with N external-volume-renderer layers via chained,
premultiplied Alpha Over, no Z-Combine anywhere.

Targets Blender's CLASSIC compositor node API (`scene.node_tree`,
`CompositorNodeComposite`, `file_slots`/`base_path`) -- Blender 3.x/4.x.
This matches the project's own existing node-scripting idiom
(`node_tree.nodes.new(...)`, `.links.new(...)`).

KNOWN GAP, discovered empirically while building this (see README
"Blender 5.0+ compositor"): Blender 5.0 replaced this entire API --
`scene.node_tree` -> `scene.compositing_node_group`, `file_slots`/`base_path`
-> `file_output_items`/`directory`/`file_name`, and there is no
`CompositorNodeComposite` node type at all anymore (compositor trees became
generic node groups with a `NodeTreeInterface`-defined output, closer to
Geometry Nodes). `ensure_compositor_tree()` below raises a clear error if it
detects that shape rather than silently building a broken graph -- if your
Blender turns out to be 5.0+, this module needs a follow-up patch, tracked
as a README limitation.

Graph shape per frame (all nodes prefixed `VO_` so this coexists with any
of your own unrelated compositor nodes in the same tree):

    Image(VO_Beauty) --\\
                         AlphaOver(farthest volume) --\\
    Image(VO_Vol_<far>)-/                              AlphaOver(next) -- ... -- Composite
                                          Image(VO_Vol_<near>)-/

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


def ensure_compositor_tree(scene):
    if hasattr(scene, "compositing_node_group"):
        raise RuntimeError(
            "This Blender's compositor uses the >=5.0 node-group API "
            "(scene.compositing_node_group), not the classic scene.node_tree "
            "this module targets. See compositor_build.py's module docstring "
            "and README 'Blender 5.0+ compositor' -- this module needs a "
            "version-specific patch before it will work here."
        )
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
    img.colorspace_settings.name = "Linear"
    img.source = "FILE"
    return img


def _make_image_node(tree, name, path):
    node = tree.nodes.new("CompositorNodeImage")
    node.name = name
    node.label = name
    node.image = _load_fresh_image(path, name)
    return node


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
    # "Straight Alpha" input socket instead. Handle both, default-safe
    # either way since both APIs default to the behavior we want -- this is
    # a belt-and-suspenders explicit set, not a correctness-critical one.
    if hasattr(node, "use_premultiply"):
        node.use_premultiply = False
    straight_alpha_input = node.inputs.get("Straight Alpha")
    if straight_alpha_input is not None:
        straight_alpha_input.default_value = False
    tree.links.new(background_socket, node.inputs[0])
    tree.links.new(foreground_socket, node.inputs[1])
    return node


def build_or_update_graph(scene, beauty_exr_path, volume_layers):
    """(Re)builds the `VO_`-prefixed subgraph and links it to Composite.

    `volume_layers`: list of (object_name, exr_path), already ordered
    farthest-from-camera first, nearest last (see module docstring on
    Alpha Over chain ordering). Pass `[]` for a frame with no flagged
    volumes -- beauty is wired straight to Composite.
    """
    tree = ensure_compositor_tree(scene)
    _clear_managed_nodes(tree)

    beauty_node = _make_image_node(tree, _node_name("Beauty"), beauty_exr_path)
    current_output = beauty_node.outputs.get("Image")

    for i, (obj_name, exr_path) in enumerate(volume_layers):
        vol_node = _make_image_node(tree, _node_name("Vol", obj_name), exr_path)
        alpha_over = _make_alpha_over(
            tree, _node_name("Over", str(i), obj_name),
            current_output, vol_node.outputs.get("Image"),
        )
        current_output = alpha_over.outputs.get("Image")

    composite = tree.nodes.new("CompositorNodeComposite")
    composite.name = _node_name("Composite")
    composite.label = _node_name("Composite")
    tree.links.new(current_output, composite.inputs[0])

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
    beauty_out.base_path = output_dir
    beauty_out.format.file_format = "OPEN_EXR"
    beauty_out.format.color_depth = "32"
    beauty_out.file_slots[0].path = "vo_beauty_"
    tree.links.new(render_layers.outputs["Image"], beauty_out.inputs[0])

    depth_out = tree.nodes.new("CompositorNodeOutputFile")
    depth_out.name = _node_name("Pass1DepthOut")
    depth_out.base_path = output_dir
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_depth = "32"
    depth_out.file_slots[0].path = "vo_depth_"
    tree.links.new(render_layers.outputs["Depth"], depth_out.inputs[0])

    # Also route Combined straight to Composite so pass 1's own render
    # result / viewport preview isn't blank -- irrelevant to what we
    # actually save (pass 2 does that), just avoids a jarring black frame
    # if you're watching the Image Editor while it renders.
    composite = tree.nodes.new("CompositorNodeComposite")
    composite.name = _node_name("Pass1Composite")
    tree.links.new(render_layers.outputs["Image"], composite.inputs[0])

    _layout_nodes(tree)

    # Blender's File Output node appends "####" (the frame number, formatted
    # with the node's padding) to file_slots[i].path when rendering an
    # actual frame. Compute the exact path so the caller doesn't have to
    # glob the output directory.
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
