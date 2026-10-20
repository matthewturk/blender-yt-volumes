"""The Deep Zoom extension (extensions/deep_zoom), loaded as its own package."""

import json
import pathlib
import sys

import bpy
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "extensions"))
import deep_zoom  # noqa: E402
from deep_zoom import authoring, shot_space  # noqa: E402

from yt_volumes import operators  # noqa: E402

ANCHOR = np.array([0.491507330198, 0.505260534, 0.490580561])


@pytest.fixture
def scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    deep_zoom.register()
    scene = bpy.context.scene
    scene.deep_zoom.enabled = True
    yield scene
    deep_zoom.unregister()


# --- shot space math --------------------------------------------------------------


def test_world_round_trip_keeps_float64_precision_near_the_anchor():
    point = ANCHOR + np.array([3e-12, -1e-12, 2e-12])
    world = shot_space.to_world(point, ANCHOR, 11.0)
    np.testing.assert_allclose(world, [0.6, -0.2, 0.4], rtol=1e-4)  # 1e11 * 2 world units per unitary
    np.testing.assert_allclose(shot_space.to_unitary(world, ANCHOR, 11.0) - ANCHOR, point - ANCHOR, rtol=1e-9)


def test_anchor_text_round_trips_every_digit():
    assert np.array_equal(shot_space.parse_anchor(shot_space.format_anchor(ANCHOR)), ANCHOR)
    assert shot_space.parse_anchor(" max ") == "max"
    with pytest.raises(ValueError):
        shot_space.parse_anchor("0.1, 0.2")


@pytest.mark.parametrize("frame, expected", [(1, 0), (49, 0), (50, 1), (500, 2), (-5, 0)])
def test_active_shot_is_the_last_one_started(frame, expected):
    assert shot_space.active_shot([1, 50, 120], frame) == expected


def test_visible_boxes_keeps_boxes_near_the_current_scale():
    # nested boxes around the anchor, each a decade smaller than the last
    sizes = 10.0 ** -np.arange(15)
    left, right = ANCHOR - sizes[:, None] / 2, ANCHOR + sizes[:, None] / 2
    _, _, index = shot_space.visible_boxes(left, right, ANCHOR, zoom=8.0, min_size=0.02, max_size=50.0)
    # at zoom 8 a box of size s spans 2e8 * s world units: 20, 2, 0.2 and 0.02
    assert sorted(index.tolist()) == [7, 8, 9, 10]


def test_visible_boxes_prefers_the_nearest_when_capped():
    left = np.array([[5.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]]) / 2e8 + ANCHOR
    right = left + 1.0 / 2e8
    _, _, index = shot_space.visible_boxes(left, right, ANCHOR, zoom=8.0, max_count=2)
    assert index.tolist() == [1, 2]


def test_box_edges_and_faces_have_cube_topology():
    left, right = np.zeros((2, 3)), np.ones((2, 3))
    verts, edges = shot_space.box_edges(left, right)
    assert verts.shape == (16, 3) and edges.shape == (24, 2)
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    np.testing.assert_allclose(lengths, 1.0)  # every edge joins neighbouring corners
    _, faces = shot_space.box_faces(left, right)
    assert faces.shape == (12, 4)


# --- authoring in Blender -----------------------------------------------------------


def _camera(scene, keys):
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    for frame, location in keys:
        camera.location = location
        camera.keyframe_insert("location", frame=frame)
    return camera


def _camera_unitary(scene, camera, frame):
    scene.frame_set(frame)
    shot = authoring.active_shot(scene)
    return shot_space.to_unitary(
        np.array(camera.matrix_world.translation), shot_space.parse_anchor(shot.anchor),
        scene.deep_zoom.zoom, scene.deep_zoom.world_scale,
    )


def test_dive_keyframes_a_linear_zoom(scene):
    authoring.add_dive(scene, 1, 60, 6.0)
    assert authoring.zoom_at(scene, 1) == pytest.approx(0.0)
    assert authoring.zoom_at(scene, 31) == pytest.approx(3.0)
    assert authoring.zoom_at(scene, 61) == pytest.approx(6.0)


def test_new_shot_at_a_point_is_a_cut_that_keeps_the_view(scene):
    authoring.add_shot(scene, ANCHOR, 1, "Wide")
    authoring.add_dive(scene, 1, 100, 10.0)
    camera = _camera(scene, [(1, (0.0, -3.0, 1.0)), (40, (0.5, -2.0, 0.5)), (80, (0.2, -1.5, 0.3))])
    frames = (30, 45, 49, 50, 65, 80)
    world = {}
    for f in frames:
        scene.frame_set(f)
        world[f] = np.array(camera.matrix_world.translation)
    at_cut = _camera_unitary(scene, camera, 50)
    before = {f: _camera_unitary(scene, camera, f) for f in (30, 45, 49)}
    point = np.array([0.7, -0.2, 0.1])
    shot = authoring.add_shot_at(scene, point, 50, camera, "Deep")
    # the new anchor is the point, in float64
    expected_anchor = shot_space.to_unitary(point, ANCHOR, authoring.zoom_at(scene, 50))
    np.testing.assert_allclose(shot_space.parse_anchor(shot.anchor), expected_anchor, rtol=0, atol=1e-17)
    # frames before the cut are untouched, including the split segment
    for f in (30, 45, 49):
        scale = shot_space.world_per_unitary(authoring.zoom_at(scene, f))
        assert np.abs(_camera_unitary(scene, camera, f) - before[f]).max() * scale < 1e-6
    # at the cut the camera is where it was in the data
    scale = shot_space.world_per_unitary(authoring.zoom_at(scene, 50))
    assert np.abs(_camera_unitary(scene, camera, 50) - at_cut).max() * scale < 1e-6
    # after it, the same shot-space motion, relative to the new anchor (between keys too)
    for f in (50, 65, 80):
        scene.frame_set(f)
        np.testing.assert_allclose(np.array(camera.matrix_world.translation), world[f] - point, atol=1e-5)


def test_grid_map_draws_the_boxes_near_the_current_scale(scene, tmp_path):
    sizes = 10.0 ** -np.arange(15)
    path = tmp_path / "grids.npz"
    np.savez(path, left=ANCHOR - sizes[:, None] / 2, right=ANCHOR + sizes[:, None] / 2, level=np.arange(15))
    authoring.add_shot(scene, ANCHOR, 1)
    scene.deep_zoom.grid_file = str(path)
    scene.deep_zoom.zoom = 8.0
    obj = bpy.data.objects[authoring.GRID_MAP_NAME]
    assert not obj.hide_render
    assert len(obj.data.vertices) == 2 * 8  # levels 8-9: 2 and 0.2 units, within the 0.1-6 default
    assert len(obj.data.polygons) == 2 * 6
    fractions = np.empty(len(obj.data.vertices), dtype=np.float32)
    obj.data.attributes["level_fraction"].data.foreach_get("value", fractions)
    assert sorted(set(np.round(fractions * 14).astype(int))) == [8, 9]
    scene.deep_zoom.show_grid_map = False
    assert obj.hide_render


def test_exported_path_is_the_camera_in_unitary_float64(scene, tmp_path):
    authoring.add_shot(scene, ANCHOR, 1, "Wide")
    authoring.add_dive(scene, 1, 19, 9.0)  # zoom 9 at frame 20
    _camera(scene, [(1, (0.0, -3.0, 0.0)), (20, (0.0, -1.0, 0.0))])
    authoring.add_shot(scene, ANCHOR + 1e-9, 15, "Deep")
    scene.frame_start, scene.frame_end = 1, 20
    path = tmp_path / "path.json"
    authoring.export_camera_path(scene, path)
    data = json.loads(path.read_text())
    assert [f["shot"] for f in data["frames"]][13:16] == ["Wide", "Deep", "Deep"]
    last = data["frames"][-1]
    # frame 20: 1 world unit along -y at zoom 9, from the "Deep" anchor
    expected = ANCHOR + 1e-9 + np.array([0.0, -1.0, 0.0]) / (2.0 * 1e9)
    np.testing.assert_allclose(last["position"], expected, rtol=0, atol=1e-16)
    assert last["distance"] == pytest.approx(5e-10)
    np.testing.assert_allclose(last["forward"], [0.0, 0.0, -1.0], atol=1e-6)  # an unrotated camera


def test_yt_volumes_renders_the_playing_shot(scene):
    assert operators._deep_zoom_params(scene) == {}
    authoring.add_shot(scene, ANCHOR, 1, "Wide")
    authoring.add_shot(scene, "max", 30, "Peak")
    scene.deep_zoom.zoom = 4.0
    scene.frame_set(40)
    params = operators._deep_zoom_params(scene)
    assert params == {"yt_idv_shot_space": 2.0, "yt_idv_zoom_target": "max", "yt_idv_zoom_level": 4.0}
    assert params == authoring.volume_custom_params(scene)
    scene.deep_zoom.enabled = False
    assert operators._deep_zoom_params(scene) == {}


# --- when to re-anchor or rescale -----------------------------------------------------


def _tracking_camera(scene, camera_keys, focus_keys):
    camera = _camera(scene, camera_keys)
    focus = bpy.data.objects.new("Focus", None)
    scene.collection.objects.link(focus)
    for frame, location in focus_keys:
        focus.location = location
        focus.keyframe_insert("location", frame=frame)
    track = camera.constraints.new("TRACK_TO")
    track.target = focus
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    return camera, focus


def test_health_flags_float32_jitter_and_bad_scale(scene):
    authoring.add_shot(scene, ANCHOR, 1)
    camera, focus = _tracking_camera(scene, [(1, (0.0, -3.0, 0.0))], [(1, (0.0, 0.0, 0.0))])
    scene.frame_set(1)
    health = authoring.frame_health(scene, camera)
    assert health["distance"] == pytest.approx(3.0) and health["advice"] == []
    # the same view 1e5 units from the anchor: float32 can't hold it
    for obj, location in ((camera, (1e5, -3.0, 0.0)), (focus, (1e5, 0.0, 0.0))):
        obj.animation_data_clear()
        obj.location = location
    bpy.context.view_layer.update()
    assert [kind for kind, _ in authoring.frame_health(scene, camera)["advice"]] == ["reanchor"]
    # the focus 1e-4 units away: rescale by +4
    camera.location, focus.location = (0.0, -1e-4, 0.0), (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    advice = authoring.frame_health(scene, camera)["advice"]
    assert [kind for kind, _ in advice] == ["rescale"] and "+4.00" in advice[0][1]


def test_check_path_marks_where_the_problems_start(scene):
    authoring.add_shot(scene, ANCHOR, 1)
    # the camera closes on the anchor without the zoom following
    camera, _ = _tracking_camera(scene, [(1, (0.0, -3.0, 0.0)), (30, (0.0, -1e-3, 0.0))], [(1, (0.0, 0.0, 0.0))])
    scene.frame_start, scene.frame_end = 1, 30
    report = authoring.check_path(scene, camera)
    flagged = [f for f, kinds in report.items() if "rescale" in kinds]
    assert flagged and flagged == list(range(flagged[0], 31))
    markers = [(m.name, m.frame) for m in scene.timeline_markers]
    assert markers == [("DZ rescale", flagged[0])]
    authoring.check_path(scene, camera)  # re-checking replaces, not adds
    assert len(scene.timeline_markers) == 1


def test_rescale_changes_units_but_not_the_view(scene):
    authoring.add_shot(scene, ANCHOR, 1)
    authoring.add_dive(scene, 1, 60, 3.0)
    camera, focus = _tracking_camera(
        scene, [(1, (0.0, -3.0, 0.5)), (60, (0.2, -0.01, 0.05))], [(1, (0.0, 0.0, 0.0)), (60, (0.001, 0.0, 0.0))],
    )
    frames = (20, 39, 40, 50, 60)
    before = {f: (_camera_unitary(scene, camera, f), authoring.frame_health(scene, camera)["distance"]) for f in frames}
    authoring.rescale_at(scene, 40, 2.0, camera)
    for f in frames:
        position, distance = before[f]
        scale = shot_space.world_per_unitary(authoring.zoom_at(scene, f))
        # the same place in the data, to float32 at the new scale
        assert np.abs(_camera_unitary(scene, camera, f) - position).max() * scale < 1e-4
        scene.frame_set(f)
        expected = distance * (100.0 if f >= 40 else 1.0)
        assert authoring.frame_health(scene, camera)["distance"] == pytest.approx(expected, rel=1e-5)


def test_reanchor_at_the_focus_moves_what_the_camera_tracks_too(scene):
    authoring.add_shot(scene, ANCHOR, 1)
    camera, focus = _tracking_camera(
        scene, [(1, (5.0, -3.0, 0.0)), (40, (5.0, -2.0, 0.0))], [(1, (5.0, 0.0, 0.0)), (40, (5.0, 0.1, 0.0))],
    )
    scene.frame_set(20)
    target = authoring.focus_point(camera)
    offset = np.array(camera.matrix_world.translation) - target
    authoring.add_shot_at(scene, target, 20, camera)
    scene.frame_set(20)
    np.testing.assert_allclose(authoring.focus_point(camera), 0.0, atol=1e-5)  # the focus is the new anchor
    np.testing.assert_allclose(np.array(camera.matrix_world.translation), offset, atol=1e-5)


def test_new_shot_at_finest_grid_cuts_to_the_finest_box(scene, tmp_path):
    sizes = 10.0 ** -np.arange(12)
    finest = ANCHOR + 3e-10
    path = tmp_path / "grids.npz"
    np.savez(path, left=finest - sizes[:, None] / 2, right=finest + sizes[:, None] / 2, level=np.arange(12))
    scene.deep_zoom.grid_file = str(path)
    authoring.add_shot(scene, ANCHOR, 1, "Approach")
    authoring.add_dive(scene, 1, 99, 8.0)
    _camera(scene, [(1, (0.0, -3.0, 0.5)), (100, (0.0, -3.0, 0.5))])
    scene.frame_set(60)
    assert bpy.ops.deep_zoom.shot_at_finest() == {"FINISHED"}
    shots = scene.deep_zoom.shots
    assert [s.frame_start for s in shots] == [1, 60]
    np.testing.assert_allclose(shot_space.parse_anchor(shots[1].anchor), finest, rtol=0, atol=1e-17)


# --- where the grid structure comes from ------------------------------------------


def _save_scene_zip(path, left, right, dx):
    """A minimal SceneGraph.save zip: one block collection with float64 edges."""
    import io
    import zipfile

    buffer = io.BytesIO()
    np.save(buffer, np.stack([left, right, dx]))
    doc = {"data_objects": [{"attributes": {"_unscaled_edges": {"__array__": "edges.npy"}}}]}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("scene.json", json.dumps(doc))
        archive.writestr("edges.npy", buffer.getvalue())


@pytest.fixture
def volume(scene):
    from yt_volumes import properties

    properties.register()
    obj = bpy.data.objects.new("Volume", None)
    obj.empty_display_type = "CUBE"
    obj.volume_override.is_volume_override = True
    scene.collection.objects.link(obj)
    yield obj
    properties.unregister()


def _nested_boxes(n):
    sizes = 2.0 ** -np.arange(n)
    return ANCHOR - sizes[:, None] / 2, ANCHOR + sizes[:, None] / 2, sizes[:, None] / 8 * np.ones(3)


def test_grid_structure_reads_a_saved_scene_zip(scene, tmp_path):
    left, right, dx = _nested_boxes(5)
    path = tmp_path / "scene.zip"
    _save_scene_zip(path, left, right, dx)
    scene.deep_zoom.grid_file = str(path)
    grids = authoring._grids(scene)
    np.testing.assert_array_equal(grids["left"], left)
    assert list(grids["level"]) == [0, 1, 2, 3, 4]


def test_empty_grid_structure_uses_the_volumes_saved_scene(scene, volume, tmp_path):
    left, right, dx = _nested_boxes(3)
    path = tmp_path / "scene.zip"
    _save_scene_zip(path, left, right, dx)
    volume.volume_override.dataset_path = str(path)
    np.testing.assert_array_equal(authoring._grids(scene)["right"], right)


def test_empty_grid_structure_uses_the_boxes_yt_volumes_loaded(scene, volume, monkeypatch):
    from yt_volumes import grid_overlay

    assert authoring._grids(scene) is None
    boxes = {"left": np.zeros((1, 3)), "right": np.ones((1, 3)), "level": np.zeros(1, dtype=np.int32)}
    monkeypatch.setattr(grid_overlay, "loaded_boxes", lambda obj: boxes if obj == volume else None)
    assert authoring._grids(scene) is boxes


def test_anchor_at_densest_point_sets_the_selected_shot_from_the_volumes_data(scene, volume, monkeypatch):
    from yt_volumes import grid_overlay

    peak = ANCHOR + 1e-13
    monkeypatch.setattr(grid_overlay, "densest_point", lambda obj: peak if obj == volume else None)
    authoring.add_shot(scene, "0.5, 0.5, 0.5", 1, "Approach")
    assert bpy.ops.deep_zoom.anchor_densest() == {"FINISHED"}
    np.testing.assert_array_equal(shot_space.parse_anchor(scene.deep_zoom.shots[0].anchor), peak)


def test_anchor_at_densest_point_makes_the_first_shot(scene, volume, monkeypatch):
    from yt_volumes import grid_overlay

    monkeypatch.setattr(grid_overlay, "densest_point", lambda obj: ANCHOR)
    assert bpy.ops.deep_zoom.anchor_densest() == {"FINISHED"}
    assert [s.name for s in scene.deep_zoom.shots] == ["Shot 1"]
    np.testing.assert_array_equal(shot_space.parse_anchor(scene.deep_zoom.shots[0].anchor), ANCHOR)


def test_anchor_at_densest_point_needs_loaded_data(scene, volume):
    with pytest.raises(RuntimeError, match="render the flagged volume once"):
        bpy.ops.deep_zoom.anchor_densest()
    assert len(scene.deep_zoom.shots) == 0


# --- points ---------------------------------------------------------------------

from deep_zoom import points  # noqa: E402


def _anchored(scene, anchor=ANCHOR):
    authoring.add_shot(scene, shot_space.format_anchor(anchor), 1)


def test_points_follow_the_zoom_in_float64(scene):
    _anchored(scene)
    unitary = ANCHOR + np.array([[3e-12, -1e-12, 2e-12], [0.0, 0.0, 0.0], [1e-3, 0.0, 0.0]])
    scene.deep_zoom.zoom = 0.0
    scene.deep_zoom.keyframe_insert("zoom", frame=1)
    scene.deep_zoom.zoom = 11.0
    scene.deep_zoom.keyframe_insert("zoom", frame=11)
    obj = points.create_points(scene, unitary, point_names=["a", "b", "c"])

    scene.frame_set(11)

    coords = np.empty(9, dtype=np.float32)
    obj.data.vertices.foreach_get("co", coords)
    # float64 holds ANCHOR + 1e-12 only to ~2e-5 of the offset
    np.testing.assert_allclose(coords.reshape(3, 3)[:2], [[0.6, -0.2, 0.4], [0.0, 0.0, 0.0]], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(points.world_positions(obj)[0], [0.6, -0.2, 0.4], rtol=1e-4)
    scale = np.empty(3, dtype=np.float32)
    obj.data.attributes[points.SCALE_ATTRIBUTE].data.foreach_get("value", scale)
    assert scale[0] == pytest.approx(2e11, rel=1e-6)
    assert [g.name for g in obj.vertex_groups] == ["a", "b", "c"]


def test_a_constraint_follows_a_named_point(scene):
    _anchored(scene)
    obj = points.create_points(scene, ANCHOR + np.array([[0.0, 0.0, 0.0], [2e-9, 0.0, 0.0]]), point_names=["core", "clump"])
    follower = bpy.data.objects.new("Follower", None)
    scene.collection.objects.link(follower)
    constraint = follower.constraints.new("COPY_LOCATION")
    constraint.target, constraint.subtarget = obj, "clump"
    scene.deep_zoom.zoom = 9.0
    scene.deep_zoom.keyframe_insert("zoom", frame=5)

    scene.frame_set(5)
    bpy.context.view_layer.update()

    np.testing.assert_allclose(follower.matrix_world.translation, [4.0, 0.0, 0.0], rtol=1e-5)  # 2e-9 * 1e9 * 2


def test_points_hide_without_an_anchored_shot(scene):
    obj = points.create_points(scene, [[0.5, 0.5, 0.5]])
    assert obj.hide_viewport and obj.hide_render
    authoring.add_shot(scene, "max", 1)
    points.update_points(scene)
    assert obj.hide_viewport
    scene.deep_zoom.shots[0].anchor = "0.5, 0.5, 0.5"
    assert not obj.hide_viewport


@pytest.mark.parametrize("text", ["x,y,z,mass\n0.1,0.2,0.3,5\n0.4,0.5,0.6,7\n", "0.1 0.2 0.3 5\n0.4 0.5 0.6 7\n"])
def test_points_files_give_positions_and_extra_columns(tmp_path, text):
    path = tmp_path / "halos.csv"
    path.write_text(text)
    unitary, attributes = points.read_points_file(str(path))
    np.testing.assert_array_equal(unitary, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    assert list(attributes) == ["mass" if text.startswith("x") else "column_3"]
    np.testing.assert_array_equal(next(iter(attributes.values())), [5.0, 7.0])
