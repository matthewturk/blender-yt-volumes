"""Verify that calc_matrix_camera() + matrix_world.inverted() projects world
points to the correct Cycles pixel, for PERSP and ORTHO cameras.

Renders an emissive marker at a known point and compares its centroid pixel
against the predicted projection.

Run with:  uv run python scripts/verify_camera_projection.py
"""

import json
import os

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "verify_camera_projection_results.json")
RES_X, RES_Y = 64, 48


def build_scene(cam_type, marker_world, **cam_kwargs):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = False
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.world = bpy.data.worlds.new("W")
    scene.world.color = (0.0, 0.0, 0.0)

    cam_data = bpy.data.cameras.new("VerifyCam")
    cam_data.type = cam_type
    cam_data.sensor_width = 36.0
    cam_data.sensor_height = 24.0
    cam_data.sensor_fit = "AUTO"
    for k, v in cam_kwargs.items():
        setattr(cam_data, k, v)
    cam_obj = bpy.data.objects.new("VerifyCam", cam_data)
    cam_obj.location = (0.0, 0.0, 10.0)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=marker_world)
    marker = bpy.context.active_object
    mat = bpy.data.materials.new("marker_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emit.inputs["Strength"].default_value = 50.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    marker.data.materials.append(mat)

    view_layer = bpy.context.view_layer
    tree = _get_or_create_compositor_tree(scene)
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    out_node = _make_file_output(tree, HERE, "marker", "RGBA", rl.outputs["Image"])

    return scene, cam_obj


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


def render_and_load(scene):
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=False)

    candidates = [f for f in os.listdir(HERE) if f.startswith("marker") and f.endswith(".exr")]
    full = os.path.join(HERE, sorted(candidates)[-1])

    img = bpy.data.images.load(full, check_existing=False)
    w, h = img.size[0], img.size[1]
    if w > 0 and h > 0 and len(img.pixels) > 0:
        arr = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)
        method = "bpy.data.images.load"
        bpy.data.images.remove(img)
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
    return arr, method


def predicted_pixel(scene, cam_obj, marker_world):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    proj = cam_obj.calc_matrix_camera(
        depsgraph,
        x=scene.render.resolution_x,
        y=scene.render.resolution_y,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    view = cam_obj.matrix_world.inverted()
    full = proj @ view
    p_world = Vector((*marker_world, 1.0))
    clip = full @ p_world
    ndc_x, ndc_y = clip.x / clip.w, clip.y / clip.w
    col = (ndc_x * 0.5 + 0.5) * scene.render.resolution_x
    row_from_bottom = (ndc_y * 0.5 + 0.5) * scene.render.resolution_y
    row_from_top = scene.render.resolution_y - row_from_bottom
    return col, row_from_bottom, row_from_top


def check_camera(cam_type, marker_world, **cam_kwargs):
    scene, cam_obj = build_scene(cam_type, marker_world, **cam_kwargs)
    arr, method = render_and_load(scene)
    luminance = arr[..., :3].sum(axis=-1) if arr.shape[-1] >= 3 else arr[..., 0]
    # Weighted centroid tracks the projected center better than argmax.
    rows, cols = np.indices(luminance.shape)
    total = luminance.sum()
    actual_row = float((rows * luminance).sum() / total)
    actual_col = float((cols * luminance).sum() / total)

    pred_col, pred_row_from_bottom, pred_row_from_top = predicted_pixel(scene, cam_obj, marker_world)

    err_top = abs(actual_row - pred_row_from_top)
    err_bottom = abs(actual_row - pred_row_from_bottom)
    matched_convention = "TOP_LEFT_ORIGIN" if err_top < err_bottom else "BOTTOM_LEFT_ORIGIN"

    col_error = float(abs(actual_col - pred_col))
    row_error = float(min(err_top, err_bottom))
    return {
        "camera_type": cam_type,
        "readback_method": method,
        "marker_world": list(marker_world),
        "actual_pixel_row_col": [actual_row, actual_col],
        "predicted_col": float(pred_col),
        "predicted_row_from_top": float(pred_row_from_top),
        "predicted_row_from_bottom": float(pred_row_from_bottom),
        "col_error_px": col_error,
        "row_error_px_matching_convention": row_error,
        "matched_row_convention": matched_convention,
        "PASS": bool(col_error < 1.5 and row_error < 1.5),
    }


def main():
    results = {
        "PERSP": check_camera("PERSP", (1.0, -0.5, 4.0), lens=50.0),
        "ORTHO": check_camera("ORTHO", (1.0, -0.5, 4.0), ortho_scale=8.0),
    }
    print(json.dumps(results, indent=2))
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    assert all(r["PASS"] for r in results.values()), "projection_matrix round-trip FAILED -- see printed results"


if __name__ == "__main__":
    main()
