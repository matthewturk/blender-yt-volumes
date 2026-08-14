"""Empirical verification of Blender's Z (Depth) pass convention and the
Image pixel-buffer row order, run under the pip-installable `bpy` module.

This is the "verify before trusting" step required by the add-on spec
(properties.py/scene_sync.py consume its *findings*, not this script).
Run with:

    uv run python scripts/verify_zpass.py

Findings are printed AND written to scripts/verify_zpass_results.json so the
README can quote them and the add-on's DepthBuffer construction can cite
this file as the source of truth.

Design of the test scene (kept deliberately trivial, single mesh at a known
distance, per the spec):

  - Camera at world (0, 0, 10), identity rotation -> looks straight down -Z,
    reaches the world origin at ray-distance exactly 10.0. Local +Y (camera
    "up", since rotation is identity) == world +Y.
  - Background plane at z=0, perpendicular to the view axis, large enough to
    fill the frame -> every pixel's PLANAR (view-space Z) depth would read
    exactly 10.0 across the whole image, but a EUCLIDEAN (ray-distance) depth
    reads 10.0 only at the exact center pixel and increasingly more at the
    corners (longer hypotenuse). Comparing center vs. corner tells us which
    convention Blender uses -- empirically, not from memory.
  - Two small unlit-emission cards close to the camera, one at world +Y
    (should land in the "up" half of frame given the camera orientation
    above) colored pure red, one at world -Y colored pure blue. Comparing
    which half of the raw pixel-buffer *rows* is red vs. blue tells us the
    row order (TOP_LEFT_ORIGIN vs BOTTOM_LEFT_ORIGIN) without assuming it.
  - A third small emission card entirely outside the camera frustum (behind
    the camera) is not rendered -- instead we just sample a background
    pixel (no geometry in that ray at all) to read Blender's "no hit" Z
    sentinel value directly.
"""

import json
import math
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_EXR_BEAUTY = os.path.join(HERE, "beauty")
OUT_EXR_DEPTH = os.path.join(HERE, "depth")
RESULTS_PATH = os.path.join(HERE, "verify_zpass_results.json")

RES_X, RES_Y = 64, 64
CAMERA_DISTANCE = 10.0
DEFAULT_LENS_MM = 50.0
DEFAULT_SENSOR_MM = 36.0
# Background plane half-width chosen to comfortably cover the center of the
# frame (including the EDGE_FRACTION sample point below) while leaving the
# extreme corners of the frustum genuinely unhit -- so we get a real no-hit
# sentinel pixel, unlike the first version of this script (bg plane fully
# covered the frustum -> every pixel hit geometry, no sentinel to sample).
BG_HALF_WIDTH = 2.0
EDGE_FRACTION = 0.4  # fraction of the half-frame used for the "off-axis but still on background" sample


def half_frame_world_extent(lens_mm, sensor_mm, distance):
    """World-space half-width of the frame at `distance`, for a square sensor/resolution."""
    return (sensor_mm / 2.0 / lens_mm) * distance


def get_or_create_compositor_tree(scene):
    """Version-agnostic accessor for the scene's compositor node tree.

    Blender < 5.0: `scene.use_nodes = True` auto-creates `scene.node_tree`.
    Blender >= 5.0: the compositor tree became an independent datablock;
    `scene.use_nodes` is a deprecated no-op-ish bool and the real handle is
    `scene.compositing_node_group`, which must be created explicitly via
    `bpy.data.node_groups.new(name, 'CompositorNodeTree')` and assigned.
    This mirrors the same branch used in compositor_build.py so the add-on
    works across both APIs without knowing the target Blender version ahead
    of time.
    """
    if hasattr(scene, "compositing_node_group"):
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new(f"{scene.name}_Compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
        return tree
    scene.use_nodes = True
    return scene.node_tree


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 4
    scene.cycles.use_denoising = False
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    cam_data = bpy.data.cameras.new("VerifyCam")
    cam_data.lens = DEFAULT_LENS_MM
    cam_data.sensor_width = DEFAULT_SENSOR_MM
    cam_obj = bpy.data.objects.new("VerifyCam", cam_data)
    cam_obj.location = (0.0, 0.0, CAMERA_DISTANCE)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)  # identity -> looks down -Z, +Y is "up"
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Background plane perpendicular to view axis at z=0 -> distance 10 from
    # camera along the principal ray. Deliberately NOT large enough to fill
    # the whole frame (half-width BG_HALF_WIDTH < the frustum's half-extent
    # at this distance) so the extreme frame corners are genuine no-hit
    # pixels we can sample the Z sentinel from, while the EDGE_FRACTION
    # sample point (used for the Euclidean-vs-planar test) still lands on it.
    bpy.ops.mesh.primitive_plane_add(size=2.0 * BG_HALF_WIDTH, location=(0.0, 0.0, 0.0))
    bg = bpy.context.active_object
    bg.name = "BackgroundPlane"
    _emissive_material(bg, (0.6, 0.6, 0.6, 1.0), strength=1.0)

    # Small "up" card at world +Y, closer to camera than the background so it
    # occludes it -- pure red, unlit emission.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 2.0, 5.0))
    up_card = bpy.context.active_object
    up_card.name = "UpCard"
    up_card.rotation_euler = (math.radians(90.0), 0.0, 0.0)  # face the camera
    up_card.scale = (3.0, 3.0, 1.0)
    _emissive_material(up_card, (1.0, 0.0, 0.0, 1.0), strength=1.0)

    # Small "down" card at world -Y -- pure blue.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, -2.0, 5.0))
    down_card = bpy.context.active_object
    down_card.name = "DownCard"
    down_card.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    down_card.scale = (3.0, 3.0, 1.0)
    _emissive_material(down_card, (0.0, 0.0, 1.0, 1.0), strength=1.0)

    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True

    tree = get_or_create_compositor_tree(scene)
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")

    out_beauty = _make_file_output(tree, HERE, "beauty", "RGBA", rl.outputs["Image"])
    out_depth = _make_file_output(tree, HERE, "depth", "FLOAT", rl.outputs["Depth"])

    return scene


def _make_file_output(tree, directory, name, socket_type, source_output):
    """Version-agnostic single-slot, single-layer OPEN_EXR File Output node.

    Blender < 5.0: `node.base_path` + `node.file_slots[0].path`, and
    `node.format.file_format` directly settable to 'OPEN_EXR'.
    Blender >= 5.0: the node became multilayer-first -- `node.directory` +
    `node.file_output_items.new(type, name)` (which appends a matching named
    input socket; the node keeps one extra blank trailing "add new" socket at
    all times). `node.format.file_format` is now locked to
    'OPEN_EXR_MULTILAYER' once items exist; to get a plain single-layer EXR
    per item you must set `item.override_node_format = True` and configure
    `item.format` instead of the node-level format.

    `item`'s own name is left EMPTY (not `name`) -- a REAL bug found while
    porting the actual add-on (see compositor_build.py's
    `_configure_single_file_output` docstring), not just a cosmetic choice:
    a non-empty item name makes Blender write this file as a NAMED EXR
    "part", which `bpy.data.images.load()` + `.pixels` (what this script's
    primary read path, and the add-on's `read_depth_buffer()`, both use)
    fails to decode -- reproducibly returns a 0x0/0-channel image even
    though the file is valid and non-empty on disk (confirmed by directly
    comparing named vs. empty item names on this exact pip wheel). This is
    NOT the wheel-specific "broken EXR decoder" this function's caller used
    to blame it on -- it reproduces identically and was independently
    reported against a real Blender 5.x install. An empty item name keeps
    the part unnamed with plain, universally-loadable `RGBA` channels.
    """
    node = tree.nodes.new("CompositorNodeOutputFile")
    if hasattr(node, "file_output_items"):
        node.directory = directory
        node.file_name = name  # node-level filename base; item.name is only
                                # the in-container layer name, NOT the file
                                # name (defaults to "{blend_name}" otherwise,
                                # which collides across nodes when unsaved)
        item = node.file_output_items.new(socket_type, "")
        item.override_node_format = True
        item.format.file_format = "OPEN_EXR"
        item.format.color_depth = "32"
        socket = node.inputs[0]
    else:
        node.base_path = directory
        node.file_slots[0].path = name
        node.format.file_format = "OPEN_EXR"
        node.format.color_depth = "32"
        socket = node.inputs[0]
    tree.links.new(source_output, socket)
    return node


def _emissive_material(obj, rgba, strength):
    mat = bpy.data.materials.new(f"{obj.name}_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = rgba
    emit.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.append(mat)


def render(scene):
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=False)


def load_exr(path_prefix):
    """Read back a File-Output-written EXR as a (H, W, C) float32 array.

    Tries the SAME pathway the shipped add-on uses in real Blender first --
    `bpy.data.images.load(path).pixels` -- since that is both the documented
    approach (spec item 3) and what determines the real `row_order` the
    add-on will actually produce. Only falls back to the `OpenEXR` PyPI
    package if that pathway is broken.

    CORRECTION to an earlier version of this docstring: the (0, 0)/0-channel
    failure previously seen here was NOT a packaging defect specific to this
    pip wheel -- it was `_make_file_output` writing a NAMED EXR part (see its
    docstring), which `bpy.data.images.load()` doesn't reliably decode
    regardless of Blender build; independently reproduced against a real
    Blender 5.x install via the shipped add-on's own equivalent code path.
    Fixed there by using an empty item name, so the primary path here is now
    expected to succeed rather than fall back. This function's fallback
    branch is kept as a safety net, not because the primary path is expected
    to fail anymore.

    Returns (array, method_used).
    """
    candidates = [
        f for f in os.listdir(HERE)
        if f.startswith(os.path.basename(path_prefix)) and f.endswith(".exr")
    ]
    if not candidates:
        raise FileNotFoundError(f"no EXR written for prefix {path_prefix}")
    full = os.path.join(HERE, sorted(candidates)[-1])

    img = bpy.data.images.load(full, check_existing=False)
    w, h = img.size[0], img.size[1]
    if w > 0 and h > 0 and len(img.pixels) > 0:
        channels = img.channels or (len(img.pixels) // (w * h))
        arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, channels)
        bpy.data.images.remove(img)
        method = "bpy.data.images.load"
    else:
        if img.name in bpy.data.images:
            bpy.data.images.remove(img)
        import OpenEXR

        exr_file = OpenEXR.File(full)
        part = exr_file.parts[0]
        # Our File Output items are named 'beauty' (RGBA) or 'depth' (single
        # float); OpenEXR.File groups same-named channels into one array.
        arr = next(iter(part.channels.values())).pixels.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        method = "OpenEXR-fallback (bpy.data.images.load produced a 0x0 image on this wheel)"

    for c in candidates:
        os.remove(os.path.join(HERE, c))
    return arr, method


def main():
    scene = build_scene()
    render(scene)

    beauty, beauty_method = load_exr(OUT_EXR_BEAUTY)  # (H, W, 4) RGBA
    depth, depth_method = load_exr(OUT_EXR_DEPTH)  # (H, W, 1) or (H, W, 4)
    if depth.shape[-1] > 1:
        depth = depth[..., :1]
    depth = depth[..., 0]
    h, w = depth.shape

    results = {
        "readback_method_beauty": beauty_method,
        "readback_method_depth": depth_method,
    }
    if "fallback" in beauty_method:
        results["row_order_caveat"] = (
            "row_order below reflects the OpenEXR-fallback read, i.e. the "
            "raw on-disk EXR scanline order -- NOT bpy.data.images.load().pixels "
            "(which real Blender is documented to return bottom-left-origin, "
            "applying its own flip on load regardless of on-disk order). "
            "Re-run this script against a real Blender to close this loop; "
            "see README 'Verification caveats'."
        )

    # --- 1. Row order: is buffer row 0 the top or bottom of the image? ---
    red_channel = beauty[..., 0]
    blue_channel = beauty[..., 2]
    top_half_red = float(red_channel[: h // 2, :].mean())
    bottom_half_red = float(red_channel[h // 2 :, :].mean())
    top_half_blue = float(blue_channel[: h // 2, :].mean())
    bottom_half_blue = float(blue_channel[h // 2 :, :].mean())
    # World +Y ("up", given identity camera rotation) is RED. If red dominates
    # buffer-row-index 0..h/2 (numpy row 0 first), then buffer row 0 == image
    # top == world "up" == TOP_LEFT_ORIGIN. Otherwise BOTTOM_LEFT_ORIGIN.
    row0_is_top = top_half_red > bottom_half_red
    results["row_order"] = "TOP_LEFT_ORIGIN" if row0_is_top else "BOTTOM_LEFT_ORIGIN"
    results["row_order_evidence"] = {
        "top_half_red_mean": top_half_red,
        "bottom_half_red_mean": bottom_half_red,
        "top_half_blue_mean": top_half_blue,
        "bottom_half_blue_mean": bottom_half_blue,
    }

    # --- 2. Depth convention: Euclidean ray distance vs. planar view-Z? ---
    # EDGE_FRACTION of the half-frame maps to a known world-space off-axis
    # distance at CAMERA_DISTANCE (via the camera's actual lens/sensor, not
    # an eyeballed pixel index) -- chosen to land ON the background plane
    # (BG_HALF_WIDTH) while still being far enough off-axis to distinguish
    # planar from Euclidean.
    half_extent = half_frame_world_extent(DEFAULT_LENS_MM, DEFAULT_SENSOR_MM, CAMERA_DISTANCE)
    assert EDGE_FRACTION * half_extent < BG_HALF_WIDTH, "edge sample would miss the background plane"
    cy, cx = h // 2, w // 2
    center_z = float(depth[cy, cx])
    edge_offset_px = int(EDGE_FRACTION * (h / 2))
    edge_y, edge_x = cy - edge_offset_px, cx - edge_offset_px
    edge_z = float(depth[edge_y, edge_x])
    results["depth_center_value"] = center_z
    results["depth_edge_value"] = edge_z
    results["depth_expected_planar"] = CAMERA_DISTANCE
    off_axis_world = EDGE_FRACTION * half_extent
    results["depth_expected_euclidean"] = math.sqrt(CAMERA_DISTANCE ** 2 + 2 * off_axis_world ** 2)
    # Planar (view-space Z) reads ~10.0 at both center and edge. Euclidean
    # (ray distance) reads ~10.0 at dead center only, and strictly larger
    # off-axis (longer hypotenuse to the same distance-10 plane).
    is_euclidean = edge_z > center_z + 0.01
    results["depth_convention"] = "EUCLIDEAN" if is_euclidean else "PLANAR"

    # --- 3. "No hit" sentinel value for background/sky pixels. ---
    # The extreme frame corner: off-axis world extent there is
    # `half_extent` (>> BG_HALF_WIDTH by construction), guaranteed to miss
    # all geometry -> genuine no-hit / sky pixel.
    corner_z = float(depth[0, 0])
    max_z = float(depth.max())
    results["no_hit_value_corner_pixel"] = corner_z
    results["no_hit_value_observed_max"] = max_z

    # --- 4. Units: raw Z pass values are in Blender scene units; report the
    # scale_length used to interpret them as meters. ---
    results["scene_unit_system"] = scene.unit_settings.system
    results["scene_scale_length"] = float(scene.unit_settings.scale_length)

    print(json.dumps(results, indent=2))
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    for f in (OUT_EXR_BEAUTY, OUT_EXR_DEPTH):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
