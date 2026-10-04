"""Build the .blend for examples/deep_zoom.toml.

    python examples/make_deep_zoom_blend.py OUT.blend [--target X Y Z] [--frames N]

A 2x2x2 volume Empty at the origin, a camera that orbits the zoom target while
easing in, and an emissive ring around the target so the volume has Cycles
geometry to composite against. The zoom itself comes from the config's [zoom]
table; nothing here is scaled.
"""

import argparse
import math

import bpy
import mathutils

# DeeplyNestedZoom's density maximum, in unitary coordinates.
DEFAULT_TARGET = (0.49150733, 0.50526053, 0.49058056)


def build(path, target, frames):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, frames
    scene.render.engine = "CYCLES"  # yt_idv modes need it
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    scene.view_settings.view_transform = "Standard"

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    scene.world = world

    volume = bpy.data.objects.new("Volume", None)
    volume.empty_display_type = "CUBE"
    volume.empty_display_size = 1.0
    scene.collection.objects.link(volume)
    # unitary u -> world 2u - 1 for this cube
    focus_point = mathutils.Vector([2.0 * t - 1.0 for t in target])

    focus = bpy.data.objects.new("Focus", None)
    focus.location = focus_point
    scene.collection.objects.link(focus)

    ring_mesh = bpy.data.meshes.new("Ring")
    ring = bpy.data.objects.new("Ring", ring_mesh)
    scene.collection.objects.link(ring)
    bpy.context.view_layer.objects.active = ring
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=False, radius=0.35, segments=128)
    bm.to_mesh(ring_mesh)
    bm.free()
    ring.location = focus_point
    ring.rotation_euler = (math.radians(70), 0.0, math.radians(20))
    skin = ring.modifiers.new("Thickness", "SKIN")
    for v in ring_mesh.skin_vertices[0].data:
        v.radius = (0.006, 0.006)
    material = bpy.data.materials.new("RingGlow")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.4, 0.8, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 4.0
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs[0], output.inputs[0])
    ring_mesh.materials.append(material)

    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = 30.0
    camera_data.clip_start = 0.001
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    track = camera.constraints.new("TRACK_TO")
    track.target = focus
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # One and a half turns, easing from 3.5 to 1.2 box half-widths away.
    for frame in range(1, frames + 1, max(1, frames // 48)):
        s = (frame - 1) / max(1, frames - 1)
        angle = math.radians(30 + 540 * s)
        radius = 3.5 - 2.3 * (0.5 - 0.5 * math.cos(math.pi * s))
        height = 0.9 - 0.6 * s
        camera.location = focus_point + mathutils.Vector(
            (radius * math.cos(angle), radius * math.sin(angle), radius * height)
        )
        camera.keyframe_insert("location", frame=frame)

    bpy.ops.wm.save_as_mainfile(filepath=path)
    print(f"wrote {path}: target {target}, focus at world {tuple(focus_point)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--target", nargs=3, type=float, default=DEFAULT_TARGET)
    parser.add_argument("--frames", type=int, default=240)
    args = parser.parse_args()
    build(args.out, args.target, args.frames)


if __name__ == "__main__":
    main()
