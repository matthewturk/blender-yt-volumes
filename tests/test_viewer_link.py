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


def test_the_viewer_takes_blenders_camera_and_sends_its_own_on_request(scene, tmp_path):
    volume, camera = _volume_and_camera(scene)
    config = viewer_link.viewer_config(scene, volume, 640, 360, str(tmp_path / "pose.json"), str(tmp_path / "camera.json"))
    session = _session(config, types.SimpleNamespace(camera=_FakeCamera(), components=[]))
    session.refit(np.array(config["camera"]["matrix_world"]))
    link = viewer.Link(session, config["pose_path"], config["camera_path"])

    moved = _look_at(np.array([4.0, 6.0, -1.0]), np.array(volume.matrix_world)[:3, 3])
    viewer.write_pose(config["camera_path"], moved)  # Send Camera to yt_idv
    link.receive()
    np.testing.assert_allclose(session.camera_world(), moved, atol=1e-5)
    assert not (tmp_path / "pose.json").exists()  # nothing goes back by itself

    link.send()  # Enter
    np.testing.assert_allclose(json.loads((tmp_path / "pose.json").read_text())["matrix_world"], moved, atol=1e-5)


class _RunningProcess:
    def poll(self):
        return None


def _link(tmp_path):
    return {
        "process": _RunningProcess(), "work_dir": str(tmp_path),
        "pose_path": str(tmp_path / "pose.json"), "camera_path": str(tmp_path / "camera.json"), "mtime": None,
    }


def test_blender_applies_a_sent_pose_and_sends_its_camera_only_on_request(scene, tmp_path, monkeypatch):
    volume, camera = _volume_and_camera(scene)
    monkeypatch.setattr(viewer_link, "_viewers", {"Volume": _link(tmp_path)})

    from_viewer = _look_at(np.array([3.0, -3.0, 2.0]), np.zeros(3))
    viewer.write_pose(str(tmp_path / "pose.json"), from_viewer)  # Enter in the window
    viewer_link._follow()
    np.testing.assert_allclose(np.array(camera.matrix_world), from_viewer, atol=1e-6)

    camera.location = (0.0, -9.0, 0.0)  # moved (or animated) in Blender
    bpy.context.view_layer.update()
    viewer_link._follow()
    assert not (tmp_path / "camera.json").exists()  # an animated camera can't pull the window back

    volume.volume_override.zoom_target = "0.5, 0.5, 0.5"
    volume.volume_override.zoom_level = 3.0
    viewer_link.send_camera(scene, volume)  # Send Camera to yt_idv
    sent = json.loads((tmp_path / "camera.json").read_text())
    np.testing.assert_allclose(sent["matrix_world"], np.array(camera.matrix_world), atol=1e-6)
    assert sent["zoom"] == {"yt_idv_shot_space": None, "yt_idv_zoom_target": "0.5, 0.5, 0.5", "yt_idv_zoom_level": 3.0}


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
        "camera_path": str(work / "camera.json"), "mtime": None,
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


def test_a_geometry_nodes_transfer_function_goes_to_the_viewer(scene, tmp_path):
    from yt_volumes import transfer_function_nodes

    volume, _ = _volume_and_camera(scene)
    volume.volume_override.renderer_mode = "YT_IDV_GN_TRANSFER_FUNCTION"
    transfer_function_nodes.add_transfer_function_modifier(volume)
    config = viewer_link.viewer_config(scene, volume, 640, 360, str(tmp_path / "pose.json"), str(tmp_path / "camera.json"))

    assert config["renderer_mode"] in viewer.CONFIGURE
    assert np.asarray(json.loads(json.dumps(config))["custom_params"]["yt_idv_gn_tf_rgba"]).shape == (256, 4)


def test_the_viewer_takes_blenders_zoom_and_keeps_its_camera(tmp_path):
    start = _look_at(np.array([0.0, -2.5, 0.0]), np.zeros(3))
    session = _deep_zoom_session(9.5, start)
    session.scene.camera = _FakeCamera()
    session.refit(start)
    link = viewer.Link(session, str(tmp_path / "pose.json"), str(tmp_path / "camera.json"))
    collection = session.scene.components[0].data
    ratio_before, position_before = collection.applied_scale_ratio, session.scene.camera.position.copy()

    zoom = {key: session.custom_params.get(key) for key in viewer.ZOOM_KEYS}
    zoom["yt_idv_zoom_level"] = 10.5  # a decade deeper, same camera
    viewer.write_pose(link.camera_path, start, zoom)
    link.receive()

    assert session.custom_params["yt_idv_zoom_level"] == 10.5
    np.testing.assert_allclose(session.camera_world(), start, atol=1e-9)
    # the zoom magnifies the data, not the camera's model position
    assert collection.applied_scale_ratio == pytest.approx(ratio_before / 10, rel=1e-9)
    np.testing.assert_allclose(session.scene.camera.position, position_before, rtol=1e-9)


def test_enter_in_the_window_sends_the_camera():
    import pyglet

    sent, captions = [], []
    link = types.SimpleNamespace(send=lambda: sent.append(True))
    rc = types.SimpleNamespace(set_caption=captions.append)
    on_key_press = viewer.send_on_key(link, rc, "yt_idv: Volume")

    assert on_key_press(pyglet.window.key.A, 0) is False and not sent  # other keys pass through to yt_idv
    assert on_key_press(pyglet.window.key.ENTER, 0) is True
    assert sent == [True] and captions == ["yt_idv: Volume: camera sent to Blender"]
