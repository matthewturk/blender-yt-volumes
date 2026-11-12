"""pano_rays against the rays Cycles itself traces.

Cycles renders the camera at the origin inside a sphere whose material writes
each ray's direction to an AOV; the Position pass gives where each ray hits
the sphere, which checks the ray origins too.
"""

import os
import tempfile

import bpy  # noqa: I001 (bmesh needs bpy imported first)
import bmesh
import numpy as np
import pytest

from yt_volumes import compositor_build, operators, pano_rays

RADIUS = 2.0
AOV = "ray"


def _scene(panorama_type, size=(48, 24), **settings):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    cam_data = bpy.data.cameras.new("C")
    cam_data.type = "PANO"
    src = operators._panorama_settings_source(cam_data)
    src.panorama_type = panorama_type
    for name, value in settings.items():
        owner = cam_data.stereo if name.startswith("stereo_") else (cam_data if hasattr(cam_data, name) and not hasattr(src, name) else src)
        setattr(owner, name.removeprefix("stereo_"), value)
    camera = bpy.data.objects.new("C", cam_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return scene, camera


def _measure(scene, views):
    """{view: (directions, hit positions)} rendered by Cycles, rows bottom first."""
    mesh = bpy.data.meshes.new("sphere")
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=RADIUS)
    bm.to_mesh(mesh)
    bm.free()
    material = bpy.data.materials.new("m")
    if hasattr(material, "use_nodes") and not material.use_nodes:
        material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    geometry = nodes.new("ShaderNodeNewGeometry")
    flip = nodes.new("ShaderNodeVectorMath")
    flip.operation = "SCALE"
    flip.inputs["Scale"].default_value = -1.0
    aov = nodes.new("ShaderNodeOutputAOV")
    aov.aov_name = AOV
    emission = nodes.new("ShaderNodeEmission")  # nonzero, or rays pass through
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(geometry.outputs["Incoming"], flip.inputs[0])
    links.new(flip.outputs["Vector"], aov.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    mesh.materials.append(material)
    scene.collection.objects.link(bpy.data.objects.new("sphere", mesh))

    scene.render.engine = "CYCLES"
    scene.render.compositor_device = "CPU"
    cycles = scene.cycles
    cycles.device = "CPU"
    cycles.samples = 1
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.pixel_filter_type = "BOX"
    cycles.filter_width = 0.01
    layer = scene.view_layers[0]
    layer.use_pass_position = True
    a = layer.aovs.add()
    a.name, a.type = AOV, "COLOR"

    out_dir = tempfile.mkdtemp(prefix="pano_rays_test_")
    tree = compositor_build.ensure_compositor_tree(scene)
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    for prefix, socket_name in (("dir", AOV), ("pos", "Position")):
        node = tree.nodes.new("CompositorNodeOutputFile")
        socket = compositor_build._configure_single_file_output(node, out_dir, f"{prefix}_")
        tree.links.new(render_layers.outputs[socket_name], socket)
    bpy.ops.render.render(write_still=False)

    result = {}
    for view, suffix in views:
        data = []
        for prefix in ("dir", "pos"):
            image = bpy.data.images.load(os.path.join(out_dir, f"{prefix}_0001{suffix}.exr"))
            compositor_build.set_linear_colorspace(image)
            w, h, c = image.size[0], image.size[1], image.channels
            buf = np.empty(w * h * c, dtype=np.float32)
            image.pixels.foreach_get(buf)
            data.append(buf.reshape(h, w, c)[..., :3].astype(np.float64))
            bpy.data.images.remove(image)
        result[view] = tuple(data)
    return result


def _pixel_angles(directions):
    """Each pixel's angular size: the larger angle to its left and right
    neighbors that have rays.
    """
    has_ray = np.linalg.norm(directions, axis=-1) > 0.5
    step = np.where(has_ray[:, :-1] & has_ray[:, 1:], _angles(directions[:, :-1], directions[:, 1:]), 0.0)
    sizes = np.zeros(directions.shape[:2])
    sizes[:, :-1] = step
    sizes[:, 1:] = np.maximum(sizes[:, 1:], step)
    return sizes


def _angles(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(axis=-1)))


@pytest.mark.parametrize("panorama_type, settings", [
    ("EQUIRECTANGULAR", {}),
    ("EQUIRECTANGULAR", {"latitude_min": -0.5, "latitude_max": 1.2, "longitude_min": -2.0, "longitude_max": 1.0, "shift_x": 0.1}),
    ("FISHEYE_EQUIDISTANT", {"fisheye_fov": 3.6}),
    ("FISHEYE_EQUISOLID", {"fisheye_lens": 8.0, "fisheye_fov": 3.4}),
    ("FISHEYE_EQUISOLID", {"sensor_fit": "VERTICAL", "shift_y": -0.05}),
    ("FISHEYE_LENS_POLYNOMIAL", {"fisheye_polynomial_k1": -0.08, "fisheye_polynomial_k2": 0.0005, "fisheye_fov": 3.0}),
    ("MIRRORBALL", {}),
    ("CENTRAL_CYLINDRICAL", {"central_cylindrical_range_u_min": -2.0, "central_cylindrical_range_v_max": 0.6,
                             "central_cylindrical_radius": 1.5}),
    ("EQUIANGULAR_CUBEMAP_FACE", {}),
])
def test_rays_match_cycles(panorama_type, settings):
    scene, camera = _scene(panorama_type, **settings)
    directions, origins = pano_rays.ray_tables(scene, camera)[None]
    measured, _ = _measure(scene, [(None, "")])[None]

    has_ray = np.linalg.norm(measured, axis=-1) > 0.5
    computed_ray = np.linalg.norm(directions, axis=-1) > 0.5
    assert origins is None
    # pixels at the edge of a fisheye's image circle may land either way
    assert (has_ray != computed_ray).sum() <= 0.005 * has_ray.size + 2
    both = has_ray & computed_ray
    assert both.sum() > 0.3 * has_ray.size
    measured = measured / np.linalg.norm(measured, axis=-1, keepdims=True)
    # Cycles samples within about a hundredth of a pixel of its center
    error = _angles(directions[both], measured[both]) / _pixel_angles(directions)[both]
    assert error.max() < 0.02


@pytest.mark.parametrize("settings", [
    {},
    {"stereo_use_pole_merge": True},
    {"stereo_convergence_mode": "PARALLEL", "stereo_interocular_distance": 0.2},
    {"stereo_convergence_mode": "TOE", "stereo_pivot": "LEFT", "stereo_convergence_distance": 0.8},
])
def test_spherical_stereo_matches_cycles(settings):
    scene, camera = _scene("EQUIRECTANGULAR", stereo_use_spherical_stereo=True, **settings)
    scene.render.engine = "CYCLES"
    scene.render.use_multiview = True
    scene.render.views_format = "STEREO_3D"
    tables = pano_rays.ray_tables(scene, camera, views=("left", "right"))
    measured = _measure(scene, [("left", "_L"), ("right", "_R")])

    for view in ("left", "right"):
        directions, origins = tables[view]
        measured_dirs, measured_hits = measured[view]
        assert origins is not None
        measured_dirs = measured_dirs / np.linalg.norm(measured_dirs, axis=-1, keepdims=True)
        assert (_angles(directions, measured_dirs) / _pixel_angles(directions)).max() < 0.02
        # each computed origin lies on the line of the ray Cycles traced
        distance = np.linalg.norm(np.cross(origins - measured_hits, measured_dirs), axis=-1)
        assert distance.max() < 1e-4
