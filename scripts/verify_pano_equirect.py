"""Determine Cycles' equirectangular panorama mapping: which camera-local
direction lands at which (u, v) image fraction.

Renders one marker per axis direction and reports its centroid (u, v).

Run with: uv run python scripts/verify_pano_equirect.py
"""

import json
import os

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "verify_pano_equirect_results.json")
RES_X, RES_Y = 128, 64  # standard 2:1 equirectangular aspect
CAM_LOC = (0.0, 0.0, 10.0)
DIST = 5.0

DIRECTIONS = {
    "forward(-Z)": (0.0, 0.0, -DIST),
    "back(+Z)": (0.0, 0.0, DIST),
    "up(+Y)": (0.0, DIST, 0.0),
    "down(-Y)": (0.0, -DIST, 0.0),
    "right(+X)": (DIST, 0.0, 0.0),
    "left(-X)": (-DIST, 0.0, 0.0),
}


def render_marker_direction(offset):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.film_transparent = True
    scene.world = bpy.data.worlds.new("W")
    scene.world.color = (0.0, 0.0, 0.0)

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.type = "PANO"
    cam_data.panorama_type = "EQUIRECTANGULAR"
    cam_obj = bpy.data.objects.new("Cam", cam_data)
    cam_obj.location = CAM_LOC
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    marker_loc = tuple(CAM_LOC[i] + offset[i] for i in range(3))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=marker_loc)
    marker = bpy.context.active_object
    mat = bpy.data.materials.new("m")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 50.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    marker.data.materials.append(mat)

    tree = _get_or_create_compositor_tree(scene)
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    _make_file_output(tree, HERE, "pano", "RGBA", rl.outputs["Image"])

    bpy.ops.render.render(write_still=False)

    candidates = [f for f in os.listdir(HERE) if f.startswith("pano") and f.endswith(".exr")]
    full = os.path.join(HERE, sorted(candidates)[-1])
    img = bpy.data.images.load(full, check_existing=False)
    w, h = img.size[0], img.size[1]
    if w > 0 and h > 0 and len(img.pixels) > 0:
        arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)
        bpy.data.images.remove(img)
        method = "bpy.data.images.load"
    else:
        if img.name in bpy.data.images:
            bpy.data.images.remove(img)
        import OpenEXR

        exr_file = OpenEXR.File(full)
        arr = next(iter(exr_file.parts[0].channels.values())).pixels.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        method = "OpenEXR-fallback"
    for c in candidates:
        os.remove(os.path.join(HERE, c))

    h, w = arr.shape[:2]  # authoritative regardless of which branch ran
    luminance = arr[..., :3].sum(axis=-1) if arr.shape[-1] >= 3 else arr[..., 0]
    if luminance.sum() <= 0:
        return None, method
    rows, cols = np.indices(luminance.shape)
    total = luminance.sum()
    row = float((rows * luminance).sum() / total)
    # Circular mean for column: a marker behind the camera straddles the u=0/1 seam.
    theta = 2 * np.pi * cols / w
    sin_mean = (np.sin(theta) * luminance).sum() / total
    cos_mean = (np.cos(theta) * luminance).sum() / total
    col = (np.arctan2(sin_mean, cos_mean) / (2 * np.pi)) % 1.0 * w

    u = col / w
    v_from_top = row / h
    return {"row": row, "col": col, "u": u, "v_from_top": v_from_top}, method


def _get_or_create_compositor_tree(scene):
    if hasattr(scene, "compositing_node_group"):
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new(f"{scene.name}_Compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
        return tree
    scene.use_nodes = True
    return scene.node_tree


def _make_file_output(tree, directory, name, socket_type, source_output):
    # Empty item name: a named EXR part loads as a 0x0 image via bpy.data.images.load().
    node = tree.nodes.new("CompositorNodeOutputFile")
    if hasattr(node, "file_output_items"):
        node.directory = directory
        node.file_name = name
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


def main():
    results = {}
    for name, offset in DIRECTIONS.items():
        info, method = render_marker_direction(offset)
        results[name] = info
        print(name, "->", info, f"[{method}]")
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
