"""Build the .blend for examples/spherical_earth.toml.

    python examples/make_earth_blend.py OUT.blend [--frames N]

An Empty-Sphere of radius 1 stands in for the spherical dataset (its radius is
the data's outer radius, +z north, +x longitude 0). Around it: an opaque core
at the core-mantle boundary, a graticule just above the surface, rings at the
hotspots the synthetic plumes rise under, and a sun. By default the camera
orbits the globe; with --spin the globe turns once about its tilted axis in
front of a fixed camera, ending where it started so the movie loops.
"""

import argparse
import math
import os
import sys

import bpy
import mathutils
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spherical_earth as earth  # noqa: E402


def _emission(name, color, strength):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = strength
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs[0], output.inputs[0])
    return material


def _polyline(name, points, radius, material, closed=True):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 2
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, xyz in zip(spline.points, points):
        p.co = (*xyz, 1.0)
    spline.use_cyclic_u = closed
    curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _circle_about(axis, angular_radius_deg, radius, n=128):
    """Points of the small circle at `angular_radius_deg` around unit `axis`, on a sphere of `radius`."""
    axis = np.asarray(axis, dtype=float)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    a = math.radians(angular_radius_deg)
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)[:, None]
    return radius * (math.cos(a) * axis + math.sin(a) * (np.cos(t) * u + np.sin(t) * v))


def build(path, frames, spin=False, graticule_deg=30, tilt_deg=23.44, fps=24):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.fps = fps
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.world = world

    earth_empty = bpy.data.objects.new("Earth", None)
    earth_empty.empty_display_type = "SPHERE"
    earth_empty.empty_display_size = 1.0
    scene.collection.objects.link(earth_empty)
    if spin:
        # the axis leans sideways from the camera, like a desk globe
        tilt = bpy.data.objects.new("Tilt", None)
        tilt.rotation_euler = (0.0, math.radians(tilt_deg), 0.0)
        scene.collection.objects.link(tilt)
        earth_empty.parent = tilt

    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=earth.CMB_RADIUS)
    core = bpy.context.active_object
    core.name = "Core"
    bpy.ops.object.shade_smooth()
    core_material = bpy.data.materials.new("Core")
    core_material.diffuse_color = (0.35, 0.12, 0.05, 1.0)
    core_material.use_nodes = True
    core_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.12, 0.05, 1.0)
    core.data.materials.append(core_material)

    grid = _emission("Graticule", (0.6, 0.7, 0.8), 0.6)
    main_lines = _emission("Equator", (0.75, 0.85, 1.0), 1.5)
    shell = 1.004
    attached = [core]
    for lat in range(-90 + graticule_deg, 90, graticule_deg):
        material, radius = (main_lines, 0.0025) if lat == 0 else (grid, 0.0015)
        attached.append(_polyline(f"lat{lat}", _circle_about((0, 0, 1), 90 - lat, shell), radius, material))
    for lon in range(0, 180, graticule_deg):
        # each great circle through the poles covers longitudes lon and lon + 180
        material, radius = (main_lines, 0.0025) if lon == 0 else (grid, 0.0015)
        attached.append(_polyline(
            f"lon{lon}", _circle_about(earth.unit_vector(0.0, lon + 90.0), 90.0, shell), radius, material,
        ))

    ring = _emission("Hotspot", (1.0, 0.45, 0.1), 6.0)
    for name, (lat, lon) in earth.HOTSPOTS.items():
        attached.append(_polyline(name, _circle_about(earth.unit_vector(lat, lon), 5.0, 1.0), 0.004, ring))
    for obj in attached:
        obj.parent = earth_empty

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(-30))
    scene.collection.objects.link(sun)

    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.lens = 35.0
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new("TRACK_TO")
    track.target = earth_empty
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    hawaii_lon = earth.HOTSPOTS["Hawaii"][1]
    if spin:
        # the camera looks along +y, at longitude -90 before the globe turns
        camera.location = mathutils.Vector(4.2 * earth.unit_vector(15.0, -90.0))
        for frame in range(1, frames + 1):
            # eastward, as the Earth turns; starts with Hawaii facing the camera
            turn = -90.0 - hawaii_lon + 360.0 * (frame - 1) / frames
            earth_empty.rotation_euler = (0.0, 0.0, math.radians(turn))
            earth_empty.keyframe_insert("rotation_euler", frame=frame)
    else:
        for frame in range(1, frames + 1):
            lon = hawaii_lon + 360.0 * (frame - 1) / frames  # start over Hawaii, head east
            camera.location = mathutils.Vector(4.2 * earth.unit_vector(20.0, lon))
            camera.keyframe_insert("location", frame=frame)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--spin", action="store_true", help="turn the globe instead of orbiting the camera")
    parser.add_argument("--graticule", type=int, default=30, help="degrees between grid lines")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    build(args.out, args.frames, spin=args.spin, graticule_deg=args.graticule, fps=args.fps)


if __name__ == "__main__":
    main()
