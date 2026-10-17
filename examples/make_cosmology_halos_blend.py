"""The scene for cosmology_halos.toml: an Enzo cosmology box whose gas shows
its filaments, with each output's yt_astro_analysis halos as spheres the size
of their virial radius.

    python examples/make_cosmology_halos_blend.py examples/cosmology_halos.toml [--finder hop|fof]
    python -m yt_volumes.render_series examples/cosmology_halos.toml

Needs yt_astro_analysis. Halo catalogs are cached next to the frames, in
halo_catalogs/. The halos of every output live in one mesh, with the frames
each output is shown on (render_series's own plan) as point attributes; a
Geometry Nodes group keeps the current frame's and puts spheres on them.
"""

import argparse
import copy
import math
import os
import tomllib

import bpy
import mathutils
import numpy as np

from yt_volumes import properties, render_series, render_volume, transfer_function_nodes

# Volume-weighted percentiles of the density spanned by the transfer function:
# deep in the voids, so the gas fades out well above the cut, to the densest
# cell, since yt_idv drops samples above the range (holes in the cores).
PERCENTILES = (30.0, 100.0)
OPACITY_SCALE = 6.0  # how see-through the gas is; Auto Exposure sets the brightness
# Color ramp stops (position, sRGB); color is weighted by the opacity curve,
# so the voids, transparent, stay dark.
COLOR_STOPS = ((0.0, (0.1, 0.35, 0.85)), (0.4, (0.3, 0.75, 1.0)), (0.8, (1.0, 1.0, 1.0)))
OPACITY_POINTS = ((0.0, 0.0), (0.15, 0.0), (0.4, 0.15), (0.7, 0.4), (1.0, 0.4))


def load_config(path):
    """render_series's config, merged and with paths resolved; not validated,
    since the .blend it names is what this script writes.
    """
    with open(path, "rb") as f:
        config = render_volume._merge(copy.deepcopy(render_volume.DEFAULTS), tomllib.load(f))
    render_volume._resolve_paths(config, os.path.dirname(os.path.abspath(path)))
    return config


def find_halos(ds, catalog_dir, finder):
    """`(positions, radii)` of `ds`'s halos in unitary units, running the halo
    finder only if its catalog isn't cached.
    """
    import yt
    from yt_astro_analysis.halo_analysis import HaloCatalog

    output_dir = os.path.join(catalog_dir, finder)
    path = os.path.join(output_dir, ds.basename, f"{ds.basename}.0.h5")
    if not os.path.exists(path):
        HaloCatalog(data_ds=ds, finder_method=finder, output_dir=output_dir).create()
    import h5py

    with h5py.File(path, "r") as f:
        # yt can't index an empty catalog, as at high redshift before anything collapsed
        if not f.attrs["num_halos"]:
            return np.zeros((0, 3)), np.zeros(0)
    ad = yt.load(path).all_data()
    positions = np.stack([ad["halos", f"particle_position_{axis}"].to("unitary").d for axis in "xyz"], axis=1)
    return positions, ad["halos", "virial_radius"].to("unitary").d


def halo_mesh(segments, halos_by_output):
    """A mesh of every shown output's halos: vertices in the volume cube's local
    space ([0, 1] unitary -> [-1, 1]) with `radius`, `frame_start` and `frame_end`.
    """
    points, radii, starts, ends = [], [], [], []
    for segment in segments:
        positions, radius = halos_by_output[segment.output_index]
        points.append(2.0 * positions - 1.0)
        radii.append(2.0 * radius)
        starts.append(np.full(len(radius), segment.start))
        ends.append(np.full(len(radius), segment.end))
    points, radii = np.concatenate(points), np.concatenate(radii)
    mesh = bpy.data.meshes.new("Halos")
    mesh.from_pydata(points.tolist(), [], [])
    for name, kind, values in (("radius", "FLOAT", radii), ("frame_start", "INT", np.concatenate(starts)),
                               ("frame_end", "INT", np.concatenate(ends))):
        attribute = mesh.attributes.new(name, kind, "POINT")
        attribute.data.foreach_set("value", values)
    return mesh


def halo_material():
    # Opaque: the volume is clipped at one depth per pixel, so it can't be layered
    # with a see-through surface inside the gas.
    material = bpy.data.materials.new("Halo")
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    color = (*transfer_function_nodes._srgb_to_linear((0.95, 0.6, 0.3)), 1.0)
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Emission Color"].default_value = color
    shader.inputs["Emission Strength"].default_value = 0.15
    shader.inputs["Roughness"].default_value = 0.6
    return material


def halo_nodes(material):
    """Spheres on the points whose frame range holds the current frame."""
    tree = bpy.data.node_groups.new("Halo Spheres", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, link = tree.nodes, tree.links.new
    group_input, output = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    time = nodes.new("GeometryNodeInputSceneTime")

    def attribute(name, kind):
        node = nodes.new("GeometryNodeInputNamedAttribute")
        node.data_type = kind
        node.inputs["Name"].default_value = name
        return node.outputs["Attribute"]

    started = nodes.new("FunctionNodeCompare")
    started.data_type, started.operation = "FLOAT", "GREATER_EQUAL"
    link(time.outputs["Frame"], started.inputs["A"])
    link(attribute("frame_start", "INT"), started.inputs["B"])
    not_ended = nodes.new("FunctionNodeCompare")
    not_ended.data_type, not_ended.operation = "FLOAT", "LESS_EQUAL"
    link(time.outputs["Frame"], not_ended.inputs["A"])
    link(attribute("frame_end", "INT"), not_ended.inputs["B"])
    shown = nodes.new("FunctionNodeBooleanMath")
    shown.operation = "NAND"  # selects the points to delete
    link(started.outputs["Result"], shown.inputs[0])
    link(not_ended.outputs["Result"], shown.inputs[1])
    delete = nodes.new("GeometryNodeDeleteGeometry")
    delete.domain = "POINT"
    link(group_input.outputs["Geometry"], delete.inputs["Geometry"])
    link(shown.outputs["Boolean"], delete.inputs["Selection"])

    sphere = nodes.new("GeometryNodeMeshIcoSphere")
    sphere.inputs["Subdivisions"].default_value = 3
    smooth = nodes.new("GeometryNodeSetShadeSmooth")
    painted = nodes.new("GeometryNodeSetMaterial")
    painted.inputs["Material"].default_value = material
    link(sphere.outputs["Mesh"], smooth.inputs["Geometry"])
    link(smooth.outputs["Geometry"], painted.inputs["Geometry"])
    instances = nodes.new("GeometryNodeInstanceOnPoints")
    link(delete.outputs["Geometry"], instances.inputs["Points"])
    link(painted.outputs["Geometry"], instances.inputs["Instance"])
    link(attribute("radius", "FLOAT"), instances.inputs["Scale"])
    link(instances.outputs["Instances"], output.inputs["Geometry"])
    return tree


def style_transfer_function(volume):
    """Filaments only, soft-edged: clear voids, a cool ramp rising through the
    filament densities, flat over the halo cores.
    """
    props = volume.volume_override
    props.renderer_mode = "YT_IDV_GN_TRANSFER_FUNCTION"
    props.tf_fit_range = True
    props.tf_percentile_low, props.tf_percentile_high = PERCENTILES
    props.gn_tf_opacity_scale = OPACITY_SCALE
    props.auto_exposure = True
    props.samples_per_cell = 8.0
    transfer_function_nodes.add_transfer_function_modifier(volume)
    ramp, curve = transfer_function_nodes.find_nodes(volume)
    elements = ramp.color_ramp.elements
    while len(elements) > 2:
        elements.remove(elements[-1])
    while len(elements) < len(COLOR_STOPS):
        elements.new(0.5)
    for element, (position, srgb) in zip(elements, COLOR_STOPS):
        element.position = position
        element.color = (*transfer_function_nodes._srgb_to_linear(srgb), 1.0)
    points = curve.mapping.curves[0].points
    while len(points) > 2:
        points.remove(points[1])
    points[0].location, points[1].location = OPACITY_POINTS[0], OPACITY_POINTS[-1]
    for location in OPACITY_POINTS[1:-1]:
        points.new(*location)
    curve.mapping.update()


def build(config_path, finder="hop"):
    config = load_config(config_path)
    outputs = render_series.discover_outputs(config["data"], config["time"])
    frames = render_volume.parse_frames(config["blender"]["frames"])
    segments = render_series.plan_segments(outputs, frames, config["time"])
    catalog_dir = os.path.join(os.path.dirname(config["blender"]["output"]), "halo_catalogs")

    import yt

    halos_by_output = {}
    for index in sorted({s.output_index for s in segments}):
        ds = yt.load(outputs[index].path)
        halos_by_output[index] = find_halos(ds, catalog_dir, finder)
        print(f"{ds.basename}: z = {float(ds.current_redshift):.2f}, {len(halos_by_output[index][1])} halos", flush=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    scene = bpy.context.scene
    first, last = frames
    scene.frame_start, scene.frame_end = frames
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = config["blender"]["resolution"]
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.world = world

    volume = bpy.data.objects.new(config["blender"]["volume"], None)
    volume.empty_display_type = "CUBE"
    volume.volume_override.is_volume_override = True
    # For editing in Blender (F12, Show in Viewport): the last output, on the frame
    # it's shown. render_series swaps in each frame's output itself.
    volume.volume_override.dataset_path = outputs[segments[-1].output_index].path
    scene.frame_current = last
    scene.collection.objects.link(volume)
    style_transfer_function(volume)

    halos = bpy.data.objects.new("Halos", halo_mesh(segments, halos_by_output))
    halos.parent = volume  # its vertices are in the volume cube's local space
    scene.collection.objects.link(halos)
    halos.modifiers.new("Halo Spheres", "NODES").node_group = halo_nodes(halo_material())

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(30))
    scene.collection.objects.link(sun)

    # A slow half orbit over the whole run, so the box turns as it evolves.
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.lens = 35.0
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new("TRACK_TO")
    track.target = volume
    track.track_axis, track.up_axis = "TRACK_NEGATIVE_Z", "UP_Y"
    for frame in range(first, last + 1, 6):
        angle = math.radians(-60 + 180 * (frame - first) / max(last - first, 1))
        camera.location = mathutils.Vector((3.8 * math.cos(angle), 3.8 * math.sin(angle), 1.8))
        camera.keyframe_insert("location", frame=frame)
    bpy.ops.wm.save_as_mainfile(filepath=config["blender"]["file"])
    print(f"wrote {config['blender']['file']}: {len(segments)} outputs over frames {first}-{last}")


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--finder", default="hop", choices=("hop", "fof"), help="yt_astro_analysis halo finder")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    build(args.config, args.finder)
