"""Open a yt_idv window (`viewer.py`) on a flagged volume, its camera linked
both ways with the scene camera.
"""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Matrix

from . import operators, viewer

FOLLOW_INTERVAL = 0.1  # seconds between pose-file checks
VIEWER_HEIGHT = 720

# object name -> {"process", "work_dir", "pose_path", "camera_path", "mtime", "exchanged"};
# `exchanged` is the camera matrix last sent to or applied from the viewer.
_viewers = {}


def _follow():
    """Timer: apply each viewer's new pose to the scene camera, or send the
    scene camera to the viewer when Blender moved it.
    """
    camera = bpy.context.scene.camera
    for name, link in list(_viewers.items()):
        if link["process"].poll() is not None:
            _close(name)
            continue
        if camera is None:
            continue
        matrix, link["mtime"] = viewer.read_pose(link["pose_path"], link["mtime"])
        if matrix is not None:
            camera.matrix_world = Matrix(matrix.tolist())
            link["exchanged"] = np.array(camera.matrix_world)
            continue
        current = np.array(camera.matrix_world)
        # A camera that differs from the last pose exchanged moved in Blender (animation, scrubbing, a drag).
        if not np.allclose(current, link["exchanged"], rtol=0, atol=1e-6):
            viewer.write_pose(link["camera_path"], current)
            link["exchanged"] = current
    return FOLLOW_INTERVAL if _viewers else None


def _close(name):
    """Forget a viewer and remove its files; stops it if still running."""
    link = _viewers.pop(name)
    if link["process"].poll() is None:
        link["process"].terminate()
    shutil.rmtree(link["work_dir"], ignore_errors=True)


def viewer_config(scene, obj, width, height, pose_path, camera_path):
    """The `viewer.py` config for `obj`, starting from the scene camera."""
    if obj.volume_override.renderer_mode not in viewer.CONFIGURE:
        raise ValueError(f"the yt_idv viewer can't show {obj.volume_override.renderer_mode!r}")
    volume = operators.build_volume_params(obj)
    camera = dataclasses.asdict(operators.build_camera_params(scene, scene.camera))
    del camera["panorama"], camera["dof"]
    return {
        "dataset_path": volume.dataset.dataset_path,
        "dataset_kind": volume.dataset.dataset_kind,
        "renderer_mode": obj.volume_override.renderer_mode,
        # The viewer has no bpy to ask.
        "custom_params": dict(volume.custom_params, yt_idv_online_access=bpy.app.online_access),
        "volume": {"matrix_world": volume.matrix_world, "bounding_box_local": volume.bounding_box_local},
        "camera": camera,
        "width": width,
        "height": height,
        "pose_path": pose_path,
        "camera_path": camera_path,
        "title": f"yt_idv: {obj.name}",
    }


class VOLUMEOVERRIDE_OT_open_viewer(bpy.types.Operator):
    """Open a yt_idv window on this volume's data; the scene camera follows its view"""

    bl_idname = "volume_override.open_viewer"
    bl_label = "Open yt_idv Viewer"

    def execute(self, context):
        scene, obj = context.scene, context.object
        if scene.camera is None:
            self.report({"ERROR"}, "the scene needs a camera to follow the viewer")
            return {"CANCELLED"}
        if obj.name in _viewers:
            _close(obj.name)
        work = tempfile.mkdtemp(prefix="yt_volumes_viewer_")
        render = scene.render
        height = VIEWER_HEIGHT
        width = round(height * render.resolution_x * render.pixel_aspect_x / (render.resolution_y * render.pixel_aspect_y))
        try:
            config = viewer_config(
                scene, obj, width, height, os.path.join(work, "pose.json"), os.path.join(work, "blender_camera.json"),
            )
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        config_path = os.path.join(work, "viewer.json")
        with open(config_path, "w") as f:
            json.dump(config, f, default=float)
        # The viewer imports this add-on and yt_idv from wherever Blender found them.
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        env.pop("PYOPENGL_PLATFORM", None)  # pyglet opens a GLX window, not EGL
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer_main.py")
        process = subprocess.Popen([sys.executable, script, config_path], env=env)
        _viewers[obj.name] = {
            "process": process, "work_dir": work,
            "pose_path": config["pose_path"], "camera_path": config["camera_path"],
            "mtime": None, "exchanged": np.array(scene.camera.matrix_world),  # the viewer starts here
        }
        if not bpy.app.timers.is_registered(_follow):
            bpy.app.timers.register(_follow, first_interval=FOLLOW_INTERVAL)
        self.report({"INFO"}, "Loading the dataset in the yt_idv window; the scene camera will follow it")
        return {"FINISHED"}


CLASSES = (VOLUMEOVERRIDE_OT_open_viewer,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    if bpy.app.timers.is_registered(_follow):
        bpy.app.timers.unregister(_follow)
    for name in list(_viewers):
        _close(name)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
