import json
import math
import types

import bpy
import numpy as np
import pytest

from yt_volumes import change_tracking, properties, viewer, viewer_link
from yt_volumes.external_render import CameraParams, VolumeObjectParams

from test_yt_idv_renderer import _block_scene, _cube_bbox, _deep_zoom_shot_space, _full_domain_collection


@pytest.fixture
def scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    change_tracking.reset()
    yield bpy.context.scene
    properties.unregister()


def _volume_and_camera(scene):
    volume = bpy.data.objects.new("Volume", None)
    volume.empty_display_type = "CUBE"
    volume.location = (1.5, -2.0, 0.25)
    volume.rotation_euler = (0.3, -0.7, 1.1)
    volume.scale = (2.0, 2.0, 2.0)
    volume.volume_override.is_volume_override = True
    volume.volume_override.renderer_mode = "YT_IDV_MAX_INTENSITY"
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.location = (6.0, -5.0, 3.0)
    camera.rotation_euler = (1.1, 0.2, 0.8)
    for obj in (volume, camera):
        scene.collection.objects.link(obj)
    scene.camera = camera
    bpy.context.view_layer.update()
    return volume, camera


class _FakeCamera:
    """A yt_idv camera's update() and the attributes it sets."""

    near_plane, far_plane = 0.001, 20.0

    def update(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, np.asarray(value, dtype=np.float64) if name in ("position", "focus", "up") else value)


def _session(config, scene_):
    volume = VolumeObjectParams(custom_params=config["custom_params"], params_version=None, dataset=None, **config["volume"])
    camera = CameraParams(panorama=None, dof=None, **config["camera"])
    ds = types.SimpleNamespace(domain_center=np.full(3, 0.5))
    return viewer.Session(scene_, ds, volume, camera, config["custom_params"], config["width"], config["height"])


def _look_at(position, target, up=(0.0, 0.0, 1.0)):
    forward = np.asarray(target, dtype=np.float64) - position
    forward /= np.linalg.norm(forward)
    x = np.cross(forward, up)
    x /= np.linalg.norm(x)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2], matrix[:3, 3] = x, np.cross(x, forward), -forward, position
    return matrix


def test_viewer_config_carries_the_dataset_look_and_world_registration(scene, tmp_path):
    volume, camera = _volume_and_camera(scene)
    volume.volume_override.dataset_path = "/data/some_scene.zip"
    volume.volume_override.colormap = "arbre"
    config = viewer_link.viewer_config(scene, volume, 640, 360, str(tmp_path / "pose.json"), str(tmp_path / "camera.json"))

    assert config["dataset_path"] == "/data/some_scene.zip"
    assert config["renderer_mode"] in viewer.CONFIGURE
    assert config["custom_params"]["yt_idv_colormap"] == "arbre"
    np.testing.assert_allclose(config["camera"]["matrix_world"], np.array(camera.matrix_world), atol=1e-6)
    np.testing.assert_allclose(config["volume"]["matrix_world"], np.array(volume.matrix_world), atol=1e-6)


def test_the_viewer_hands_back_the_camera_it_was_given(scene, tmp_path):
    volume, camera = _volume_and_camera(scene)
    config = viewer_link.viewer_config(scene, volume, 1280, 720, str(tmp_path / "pose.json"), str(tmp_path / "camera.json"))
    session = _session(config, types.SimpleNamespace(camera=_FakeCamera(), components=[]))

    session.refit(np.array(config["camera"]["matrix_world"]))

    np.testing.assert_allclose(session.camera_world(), np.array(camera.matrix_world), atol=1e-5)


def test_planes_follow_the_camera_in(scene, tmp_path):
    volume, _ = _volume_and_camera(scene)
    config = viewer_link.viewer_config(scene, volume, 640, 360, str(tmp_path / "pose.json"), str(tmp_path / "camera.json"))
    session = _session(config, types.SimpleNamespace(camera=_FakeCamera(), components=[]))
    session.scene.camera.update(position=[0.5, -2.5, 0.5])

    session.update()
    assert session.scene.camera.near_plane == pytest.approx(2.4)  # 2.5 to the cube face, less the margin

    session.scene.camera.update(position=[0.5, 0.5, 0.5])  # scrolled into the middle of the data
    session.update()
    assert session.scene.camera.near_plane == pytest.approx(1e-5)
    assert session.scene.camera.far_plane == pytest.approx(math.sqrt(3) / 2 + 0.1)


def _deep_zoom_session(zoom, camera_world):
    shot_space = _deep_zoom_shot_space()
    anchor = np.array([0.491507330198, 0.505260534, 0.490580561])
    custom_params = {"yt_idv_shot_space": 2.0, "yt_idv_zoom_target": shot_space.format_anchor(anchor), "yt_idv_zoom_level": zoom}
    camera = {
        "matrix_world": camera_world.tolist(), "projection": "PERSP", "lens_mm": 50.0, "sensor_width_mm": 36.0,
        "sensor_height_mm": 24.0, "sensor_fit": "AUTO", "shift_x": 0.0, "shift_y": 0.0, "ortho_scale": None,
        "clip_start": 0.1, "clip_end": 100.0, "resolution_x": 640, "resolution_y": 360, "pixel_aspect_x": 1.0,
        "pixel_aspect_y": 1.0, "projection_matrix": None,
    }
    config = {
        "custom_params": custom_params, "camera": camera, "width": 640, "height": 360,
        "volume": {"matrix_world": np.eye(4).tolist(), "bounding_box_local": _cube_bbox()},
    }
    return _session(config, _block_scene(_full_domain_collection()))


def test_a_deep_zoom_camera_is_refit_as_it_scrolls_in():
    # Shot space at zoom 9.5, the camera 2.5 world units from the anchor (the world origin).
    start = _look_at(np.array([0.0, -2.5, 0.0]), np.zeros(3))
    session = _deep_zoom_session(9.5, start)
    session.scene.camera = _FakeCamera()
    session.refit(start)
    np.testing.assert_allclose(session.camera_world(), start, atol=1e-9)

    # Scroll in until the camera is 0.4 of the way to the target in model units.
    for _ in range(20):
        camera = session.scene.camera
        before = session.camera_world()
        step = 0.1 * (camera.focus - camera.position) / np.linalg.norm(camera.focus - camera.position)
        camera.update(position=camera.position + step)
        moved = session.camera_world()
        session.update()
        # a refit changes the numbers, never where the camera is
        np.testing.assert_allclose(session.camera_world(), moved, rtol=1e-9, atol=1e-12 * np.linalg.norm(before[:3, 3]))
        assert np.linalg.norm(session.scene.camera.position) >= viewer.REFIT_DISTANCE

    distance = np.linalg.norm(session.camera_world()[:3, 3])
    assert distance < 0.2  # each refit shrinks the step: 2.5 -> ~0.14 in 20 scrolls, still looking at the anchor
    np.testing.assert_allclose(-session.camera_world()[:3, 2], [0.0, 1.0, 0.0], atol=1e-6)


def test_the_viewer_follows_blenders_camera_without_echoing_it(scene, tmp_path):
    volume, camera = _volume_and_camera(scene)
    config = viewer_link.viewer_config(scene, volume, 640, 360, str(tmp_path / "pose.json"), str(tmp_path / "camera.json"))
    session = _session(config, types.SimpleNamespace(camera=_FakeCamera(), components=[]))
    session.refit(np.array(config["camera"]["matrix_world"]))
    link = viewer.Link(session, config["pose_path"], config["camera_path"])

    moved = _look_at(np.array([4.0, 6.0, -1.0]), np.array(volume.matrix_world)[:3, 3])
    viewer.write_pose(config["camera_path"], moved)  # Blender moved its camera
    link.receive()
    np.testing.assert_allclose(session.camera_world(), moved, atol=1e-5)
    link.send()
    assert not (tmp_path / "pose.json").exists()  # nothing sent back

    session.scene.camera.update(position=session.scene.camera.position * 1.1)  # a move in the window
    link.send()
    assert (tmp_path / "pose.json").exists()


class _RunningProcess:
    def poll(self):
        return None


def test_blender_applies_viewer_poses_and_sends_its_own_moves(scene, tmp_path, monkeypatch):
    _, camera = _volume_and_camera(scene)
    pose_path, camera_path = tmp_path / "pose.json", tmp_path / "camera.json"
    monkeypatch.setattr(viewer_link, "_viewers", {"Volume": {
        "process": _RunningProcess(), "work_dir": str(tmp_path),
        "pose_path": str(pose_path), "camera_path": str(camera_path),
        "mtime": None, "exchanged": np.array(camera.matrix_world),
    }})

    from_viewer = _look_at(np.array([3.0, -3.0, 2.0]), np.zeros(3))
    viewer.write_pose(str(pose_path), from_viewer)
    viewer_link._follow()
    np.testing.assert_allclose(np.array(camera.matrix_world), from_viewer, atol=1e-6)
    viewer_link._follow()
    assert not camera_path.exists()  # the viewer's own pose isn't sent back

    camera.location = (0.0, -9.0, 0.0)  # moved in Blender
    bpy.context.view_layer.update()  # as the UI does before timers run
    viewer_link._follow()
    sent = json.loads(camera_path.read_text())["matrix_world"]
    np.testing.assert_allclose(sent, np.array(camera.matrix_world), atol=1e-6)
    mtime = camera_path.stat().st_mtime_ns
    viewer_link._follow()
    assert camera_path.stat().st_mtime_ns == mtime  # unchanged, so not sent again


class _ExitedProcess:
    def poll(self):
        return 0


def test_a_closed_viewers_files_are_removed(scene, tmp_path, monkeypatch):
    _volume_and_camera(scene)
    work = tmp_path / "viewer"
    work.mkdir()
    (work / "pose.json").write_text("{}")
    monkeypatch.setattr(viewer_link, "_viewers", {"Volume": {
        "process": _ExitedProcess(), "work_dir": str(work), "pose_path": str(work / "pose.json"),
        "camera_path": str(work / "camera.json"), "mtime": None, "exchanged": np.eye(4),
    }})

    assert viewer_link._follow() is None  # nothing left to follow: the timer stops
    assert not work.exists() and not viewer_link._viewers


def test_viewer_main_loads_the_viewer_without_bpy():
    import os
    import pathlib
    import subprocess
    import sys

    script = pathlib.Path(viewer.__file__).with_name("viewer_main.py")
    code = (
        "import runpy, sys; viewer = runpy.run_path(sys.argv[1])['load_viewer'](); "
        "print('bpy' in sys.modules, callable(viewer.main))"
    )
    # sys.path as this test sees it, which can import a (pip) bpy
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    result = subprocess.run([sys.executable, "-c", code, str(script)], capture_output=True, text=True, env=env, check=True)
    assert result.stdout.split() == ["False", "True"]
