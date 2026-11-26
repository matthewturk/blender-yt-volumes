"""Provenance manifests: the PNG chunk, the Note summary, sidecars, and the pipeline."""

import json
import os
import struct
import types
import zlib

import bpy
import pytest

from yt_volumes import change_tracking, operators, provenance
from yt_volumes import yt_idv_renderer as yir


def _tiny_png(path):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\0\0\0\0"))
                + chunk(b"IEND", b""))


def test_png_text_round_trips_and_replaces_its_own_chunk(tmp_path):
    path = str(tmp_path / "a.png")
    _tiny_png(path)
    provenance.add_png_text(path, provenance.PNG_KEY, "first")
    provenance.add_png_text(path, provenance.PNG_KEY, json.dumps({"ünïcode": [1, 2]}))
    assert json.loads(provenance.read_png_text(path)) == {"ünïcode": [1, 2]}
    with open(path, "rb") as f:
        data = f.read()
    assert data.count(b"iTXt") == 1
    assert data.endswith(b"IEND\xaeB`\x82")


def _manifest(n_volumes):
    volumes = [{
        "object": f"Volume {i}", "renderer_mode": "YT_IDV_TRANSFER_FUNCTION", "field": "gas, density",
        "dataset": {"path": f"/data/run/DD{i:04d}/DD{i:04d}"}, "look_hash": "0123456789abcdef",
        "cmap_range": [1.234567e-30, 9.87654e-20], "zoom": {"zoom_target": "0.5, 0.5, 0.5", "zoom_level": 3.0},
    } for i in range(n_volumes)]
    return {"software": {"yt_volumes": "0.5.0"}, "frame": 12, "manifest_hash": "fedcba9876543210", "volumes": volumes}


@pytest.mark.parametrize("n_volumes", [1, 3, 8, 40])
def test_summary_fits_the_note(n_volumes):
    text = provenance.summary(_manifest(n_volumes))
    assert len(text.encode()) <= provenance.NOTE_MAX
    summary = json.loads(text)
    assert summary["manifest"] == "fedcba9876543210"
    assert summary["frame"] == 12


def test_summary_keeps_full_detail_for_one_volume():
    (volume,) = json.loads(provenance.summary(_manifest(1)))["volumes"]
    assert volume["ds"] == "DD0000"
    assert volume["cmap"] == [1.235e-30, 9.877e-20]
    assert volume["zoom"]["zoom_level"] == 3.0


def test_dataset_provenance_reads_the_rendered_field_and_dataset(monkeypatch):
    ds = types.SimpleNamespace(
        unit_registry=object(), unique_identifier=42, current_time="1.5 Myr",
        cosmological_simulation=1, current_redshift=7.25,
    )
    collection = types.SimpleNamespace(field=("gas", "density"), data_source="YTRegion (DD0046)")
    scene = types.SimpleNamespace(components=[types.SimpleNamespace(name="block_rendering", data=collection)])
    monkeypatch.setitem(yir._volume_scene_cache, "ds-id", (scene, ds, yir.DEFAULT_BACKEND))
    info = yir.dataset_provenance("ds-id")
    assert info["field"] == "gas, density"
    assert info["data_source"] == "YTRegion (DD0046)"
    assert info["current_redshift"] == 7.25
    assert info["unique_identifier"] == "42"
    assert yir.dataset_provenance("not loaded") is None


def test_dataset_provenance_skips_a_scene_files_stand_in(monkeypatch):
    stand_in = types.SimpleNamespace(domain_center=[0.5] * 3, unit_registry_json="{}")
    scene = types.SimpleNamespace(components=[types.SimpleNamespace(name="block_rendering", data=types.SimpleNamespace())])
    monkeypatch.setitem(yir._volume_scene_cache, "ds-id", (scene, stand_in, yir.DEFAULT_BACKEND))
    assert yir.dataset_provenance("ds-id") == {}


# --- Through the pipeline (SAMPLE_CONSTANT: no yt_idv) -----------------------


@pytest.fixture
def scene(tmp_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module="yt_volumes")
    change_tracking.reset()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.resolution_x = scene.render.resolution_y = 16
    bpy.ops.object.camera_add(location=(0, 0, 8))
    scene.camera = bpy.context.active_object
    bpy.ops.object.empty_add(type="CUBE")
    volume = bpy.context.active_object
    volume.volume_override.is_volume_override = True
    volume.volume_override.renderer_mode = "SAMPLE_CONSTANT"
    volume.volume_override.dataset_path = "/data/run/DD0046/DD0046"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    bpy.context.preferences.addons["yt_volumes"].preferences.work_dir = str(work_dir)
    scene.render.filepath = str(tmp_path / "frame_")
    yield scene
    bpy.ops.preferences.addon_disable(module="yt_volumes")


def _png_text(path, key):
    with open(path, "rb") as f:
        data = f.read()
    pos = 8
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        name, _, text = data[pos + 8:pos + 8 + length].partition(b"\0")
        if kind == b"tEXt" and name == key.encode():
            return text.decode("latin-1")
        pos += 12 + length
    return None


def test_a_rendered_png_carries_the_manifest_and_its_summary(scene, tmp_path):
    scene.render.use_stamp_note = True
    scene.render.stamp_note_text = ""
    scene.frame_set(3)
    operators._render_numbered_frame(bpy.context, write_still=True)
    png = str(tmp_path / "frame_0003.png")

    manifest = provenance.read(png)
    assert manifest["schema"] == provenance.SCHEMA
    assert manifest["frame"] == 3
    assert manifest["outputs"] == ["frame_0003.png"]
    assert manifest["software"]["blender"] == bpy.app.version_string
    (volume,) = manifest["volumes"]
    assert volume["renderer_mode"] == "SAMPLE_CONSTANT"
    assert volume["dataset"]["path"] == "/data/run/DD0046/DD0046"
    assert volume["look_hash"] == provenance.digest(volume["look"])
    assert manifest["cameras"][0]["lens_mm"] == pytest.approx(50.0)

    summary = json.loads(_png_text(png, "Note"))
    assert summary["manifest"] == manifest["manifest_hash"]
    assert summary["volumes"][0]["ds"] == "DD0046"
    assert not os.path.exists(str(tmp_path / "frame_0003.provenance.json"))
    # the scene's own Note settings are restored
    assert scene.render.use_stamp_note and scene.render.stamp_note_text == ""


def test_sidecar_json_when_asked(scene, tmp_path):
    scene.yt_volumes_provenance_sidecar = True
    operators._render_numbered_frame(bpy.context, write_still=True)
    with open(tmp_path / "frame_0001.provenance.json") as f:
        sidecar = json.load(f)
    assert sidecar == provenance.read(str(tmp_path / "frame_0001.png"))


def test_provenance_off_records_nothing(scene, tmp_path):
    scene.yt_volumes_provenance = False
    scene.yt_volumes_provenance_sidecar = True
    operators._render_numbered_frame(bpy.context, write_still=True)
    png = str(tmp_path / "frame_0001.png")
    assert provenance.read_png_text(png) is None
    assert _png_text(png, "Note") is None
    assert not os.path.exists(str(tmp_path / "frame_0001.provenance.json"))


def test_a_users_own_note_is_left_alone(scene, tmp_path):
    scene.render.use_stamp_note = True
    scene.render.stamp_note_text = "shot 12, take 3"
    operators._render_numbered_frame(bpy.context, write_still=True)
    png = str(tmp_path / "frame_0001.png")
    assert _png_text(png, "Note") == "shot 12, take 3"
    assert provenance.read(png)["frame"] == 1
    assert scene.render.stamp_note_text == "shot 12, take 3"


def test_burn_in_keeps_the_summary_off_the_frame(scene, tmp_path):
    scene.render.use_stamp = True
    operators._render_numbered_frame(bpy.context, write_still=True)
    png = str(tmp_path / "frame_0001.png")
    assert _png_text(png, "Note") is None
    assert provenance.read(png)["frame"] == 1


def test_batch_config_is_recorded(scene, tmp_path):
    provenance.set_batch("/configs/run.toml", {"data": {"path": "/data/x"}})
    try:
        operators._render_numbered_frame(bpy.context, write_still=True)
    finally:
        provenance.set_batch(None, None)
    batch = provenance.read(str(tmp_path / "frame_0001.png"))["batch"]
    assert batch == {"config_path": "/configs/run.toml", "config": {"data": {"path": "/data/x"}}}


def test_changing_the_note_each_frame_doesnt_make_cycles_resync(scene):
    for i in range(4):
        bpy.ops.mesh.primitive_uv_sphere_add(location=(i - 1.5, 0, -3), radius=0.4)
    stats = []

    def on_stats(text, *args):
        stats.append(str(text))

    bpy.app.handlers.render_stats.append(on_stats)
    resynced = []
    try:
        with operators.render_session(bpy.context):
            for frame in (1, 2, 3):
                scene.frame_set(frame)
                stats.clear()
                operators._render_numbered_frame(bpy.context, write_still=True)
                resynced.append(any("Building BVH" in s or "Synchronizing object" in s for s in stats))
    finally:
        bpy.app.handlers.render_stats.remove(on_stats)
    assert resynced == [True, False, False]
    assert not scene.render.use_stamp_note  # restored at session exit
