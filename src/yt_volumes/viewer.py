"""A yt_idv window on a flagged volume's data, its camera linked both ways with Blender's.

    python -m yt_volumes.viewer config.json

`VOLUMEOVERRIDE_OT_open_viewer` writes the config and runs this with Blender's
Python: the dataset, the volume's look (`renderer_mode`, `custom_params`), its
registration and the scene camera, both in world space, and the window size.
As the camera moves (drag to orbit, scroll to move in), it writes the camera's
camera-to-world matrix to `pose_path`; when Blender moves the scene camera, it
writes the new matrix to `camera_path`, which the viewer follows. Zooms and
Deep Zoom shot space are registered as an F12 render registers them.
"""

import dataclasses
import json
import os
import sys
import types

import numpy as np

from . import yt_idv_renderer as yir
from .external_render import CameraParams, VolumeObjectParams

CONFIGURE = {
    "YT_IDV_MAX_INTENSITY": yir._configure_max_intensity,
    "YT_IDV_MAX_INTENSITY_ALPHA": yir._configure_max_intensity_alpha,
    "YT_IDV_TRANSFER_FUNCTION": yir._configure_transfer_function,
    "YT_IDV_PROJECTION": yir._configure_projection,
    "YT_IDV_THRESHOLD_FOG": yir._configure_threshold_fog,
    "YT_IDV_YT_TRANSFER_FUNCTION": yir._configure_yt_transfer_function,
    "YT_IDV_GN_TRANSFER_FUNCTION": yir._configure_gn_transfer_function,
}
# Refit when a zoomed camera gets this close to the target in model units;
# a refit puts it ~1 unit away (see yt_idv_renderer.apply_zoom).
REFIT_DISTANCE = 0.5


class Session:
    """A loaded scene and where its model space sits in the world."""

    def __init__(self, scene, ds, volume, camera, custom_params, width, height):
        self.scene, self.ds, self.volume, self.camera = scene, ds, volume, camera
        self.custom_params, self.width, self.height = custom_params, width, height
        self.zooming = bool(custom_params.get("yt_idv_zoom_target"))
        self.registered, self.frame = volume, None

    def refit(self, camera_world):
        """Register the model around `camera_world` (a camera-to-world 4x4) as
        an F12 render of that camera would, and put the yt_idv camera there.
        """
        self.registered = yir._shot_space_volume(self.scene, self.volume, self.custom_params)
        self.frame = yir.apply_zoom(self.scene, self.custom_params, self.registered, camera_world[:3, 3])
        camera = dataclasses.replace(self.camera, matrix_world=camera_world.tolist())
        self.scene.camera.update(**yir._registered_camera_kwargs(
            camera, self.registered, self.ds, self.width, self.height,
            model_bounds=yir._scene_model_bounds(self.scene), model_frame=self.frame,
        ))

    def camera_world(self):
        """The yt_idv camera as a camera-to-world 4x4, in float64."""
        A, b = yir._world_to_model(self.registered, self.frame)
        to_world = np.linalg.inv(A)
        camera = self.scene.camera
        position = np.asarray(camera.position, dtype=np.float64)
        forward = to_world @ (np.asarray(camera.focus, dtype=np.float64) - position)
        forward /= np.linalg.norm(forward)
        up = to_world @ np.asarray(camera.up, dtype=np.float64)
        up -= np.dot(up, forward) * forward
        up /= np.linalg.norm(up)
        # Blender cameras look down -Z with +Y up.
        matrix = np.eye(4)
        matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = np.cross(up, -forward), up, -forward
        matrix[:3, 3] = to_world @ position + b
        return matrix

    def update(self):
        """Refit a zoomed camera that has moved in, then fit the planes."""
        # Scrolling only moves the camera; the zoom stays Deep Zoom's (or the
        # volume's) alone. To have scroll drive the zoom instead, change it here.
        position = np.asarray(self.scene.camera.position, dtype=np.float64)
        if self.zooming and np.linalg.norm(position) < REFIT_DISTANCE:
            self.refit(self.camera_world())
            position = np.asarray(self.scene.camera.position, dtype=np.float64)
        # Fixed planes would clip everything once the camera moved in past the start's near plane.
        planes = yir.camera_planes(position, yir._scene_model_bounds(self.scene), self.frame)
        camera = self.scene.camera
        if (camera.near_plane, camera.far_plane) != (planes["near_plane"], planes["far_plane"]):
            camera.update(**planes)


def load_session(rc, config):
    """Load the dataset into `rc` as F12 would, with the volume's look, and
    put the camera where the scene camera is.
    """
    custom_params = config["custom_params"]
    dataset = types.SimpleNamespace(dataset_path=config["dataset_path"], dataset_kind=config["dataset_kind"])
    result = yir._default_volume_provider(yir.VolumeSetupContext(
        render_context=rc, dataset=dataset, custom_params=custom_params, frame=None,
        previous_scene=None, previous_ds=None,
    ))
    rc.scene = scene = result.scene
    configure = CONFIGURE[config["renderer_mode"]]
    context = {"frame": 0, "camera": None, "value": None, "ds": result.ds}
    for component in scene.components:
        yir._apply_samples_per_cell(component, custom_params)
        configure(component, dict(custom_params, _context=context))
        yir._apply_colormap_range(component, custom_params)
    volume = VolumeObjectParams(custom_params=custom_params, params_version=None, dataset=None, **config["volume"])
    camera = CameraParams(panorama=None, dof=None, **config["camera"])
    session = Session(scene, result.ds, volume, camera, custom_params, config["width"], config["height"])
    session.refit(np.array(camera.matrix_world, dtype=np.float64))
    session.update()
    return session


def write_pose(path, matrix):
    # Replaced whole, so the other side never reads a half-written file.
    temporary = f"{path}.tmp"
    with open(temporary, "w") as f:
        json.dump({"matrix_world": np.asarray(matrix).tolist()}, f)
    os.replace(temporary, path)


def read_pose(path, mtime):
    """`(matrix, mtime)` if `path` changed since `mtime`, else `(None, mtime)`."""
    try:
        new_mtime = os.path.getmtime(path)
    except OSError:
        return None, mtime
    if new_mtime == mtime:
        return None, mtime
    with open(path) as f:
        return np.array(json.load(f)["matrix_world"], dtype=np.float64), new_mtime


class Link:
    """The camera's two-way link with Blender, through `pose_path` (ours) and
    `camera_path` (Blender's).
    """

    def __init__(self, session, pose_path, camera_path):
        self.session, self.pose_path, self.camera_path = session, pose_path, camera_path
        self.sent, self.camera_mtime = None, None

    def receive(self):
        """Follow a camera Blender moved."""
        matrix, self.camera_mtime = read_pose(self.camera_path, self.camera_mtime)
        if matrix is not None:
            self.session.refit(matrix)
            # Blender already has this pose; don't send it back.
            self.sent = self.session.camera_world()

    def send(self):
        matrix = self.session.camera_world()
        if self.sent is None or not np.array_equal(matrix, self.sent):
            write_pose(self.pose_path, matrix)
            self.sent = matrix


def main(config_path):
    import pyglet
    import yt_idv

    with open(config_path) as f:
        config = json.load(f)
    rc = yt_idv.render_context(
        "pyglet", width=config["width"], height=config["height"], gui=False, title=config.get("title", "yt_idv"),
    )
    session = load_session(rc, config)
    link = Link(session, config["pose_path"], config["camera_path"])

    def tick(dt):
        link.receive()
        session.update()

    pyglet.clock.schedule_interval(tick, 1 / 60)
    pyglet.clock.schedule_interval(lambda dt: link.send(), 0.1)
    rc.run()


if __name__ == "__main__":
    main(sys.argv[1])
