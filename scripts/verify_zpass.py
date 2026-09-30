"""Verify Blender's Z (Depth) pass convention and Image pixel row order.

Renders a trivial scene (camera at z=10 looking down -Z, background plane at
z=0, red card at +Y, blue card at -Y) and reports row order, planar vs.
Euclidean depth, the no-hit Z sentinel, and scene units. Results are printed
and written to scripts/verify_zpass_results.json.

Run with:  uv run python scripts/verify_zpass.py
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
# Covers the EDGE_FRACTION sample but leaves frame corners unhit for the sentinel.
BG_HALF_WIDTH = 2.0
EDGE_FRACTION = 0.4  # fraction of the half-frame used for the "off-axis but still on background" sample


def half_frame_world_extent(lens_mm, sensor_mm, distance):
    """World-space half-width of the frame at `distance`, for a square sensor/resolution."""
    return (sensor_mm / 2.0 / lens_mm) * distance


def get_or_create_compositor_tree(scene):
    """Return the scene's compositor tree (Blender 5.0 `compositing_node_group` or older `node_tree`)."""
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

    # Background plane at z=0; deliberately smaller than the frame so corners are no-hit.
    bpy.ops.mesh.primitive_plane_add(size=2.0 * BG_HALF_WIDTH, location=(0.0, 0.0, 0.0))
    bg = bpy.context.active_object
    bg.name = "BackgroundPlane"
    _emissive_material(bg, (0.6, 0.6, 0.6, 1.0), strength=1.0)

    # Red card at world +Y, blue card at -Y, in front of the background.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 2.0, 5.0))
    up_card = bpy.context.active_object
    up_card.name = "UpCard"
    up_card.rotation_euler = (math.radians(90.0), 0.0, 0.0)  # face the camera
    up_card.scale = (3.0, 3.0, 1.0)
    _emissive_material(up_card, (1.0, 0.0, 0.0, 1.0), strength=1.0)

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
    """Single-layer OPEN_EXR File Output node, for Blender 5.0 and older APIs.

    On 5.0, a plain EXR per item needs `item.override_node_format = True`.
    The item name is left empty: a named EXR part loads as a 0x0 image via
    `bpy.data.images.load()`.
    """
    node = tree.nodes.new("CompositorNodeOutputFile")
    if hasattr(node, "file_output_items"):
        node.directory = directory
        node.file_name = name  # otherwise defaults to "{blend_name}", colliding across nodes
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
    """Read back a File-Output EXR as a (H, W, C) float32 array.

    Uses `bpy.data.images.load().pixels` (as the add-on does), falling back to
    the `OpenEXR` package. Returns (array, method_used).
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
    # Red is world +Y (image top), so red in rows 0..h/2 means TOP_LEFT_ORIGIN.
    row0_is_top = top_half_red > bottom_half_red
    results["row_order"] = "TOP_LEFT_ORIGIN" if row0_is_top else "BOTTOM_LEFT_ORIGIN"
    results["row_order_evidence"] = {
        "top_half_red_mean": top_half_red,
        "bottom_half_red_mean": bottom_half_red,
        "top_half_blue_mean": top_half_blue,
        "bottom_half_blue_mean": bottom_half_blue,
    }

    # --- 2. Depth convention: Euclidean ray distance vs. planar view-Z? ---
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
    # Planar depth is constant across the plane; Euclidean grows off-axis.
    is_euclidean = edge_z > center_z + 0.01
    results["depth_convention"] = "EUCLIDEAN" if is_euclidean else "PLANAR"

    # --- 3. "No hit" sentinel value for background/sky pixels. ---
    # Frame corner misses all geometry by construction.
    corner_z = float(depth[0, 0])
    max_z = float(depth.max())
    results["no_hit_value_corner_pixel"] = corner_z
    results["no_hit_value_observed_max"] = max_z

    # --- 4. Units: Z pass values are in scene units. ---
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
