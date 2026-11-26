"""Provenance: a record of how each rendered frame was made.

The pipeline builds a manifest per frame (datasets, fields, looks, fitted
colormap ranges, zoom, camera, software versions, batch config) and stores it
three ways:

- a compact summary in the render's Note metadata, which Blender writes into
  every format's header (PNG text, EXR attributes) and into Image > Save As;
- the full manifest as a compressed `yt_volumes:provenance` iTXt chunk in PNG
  outputs;
- optionally, the full manifest as `<output>.provenance.json` next to each output.
"""

import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import os
import struct
import sys
import zlib

SCHEMA = "yt_volumes.provenance/1"
PNG_KEY = "yt_volumes:provenance"
SIDECAR_SUFFIX = ".provenance.json"
NOTE_MAX = 767  # RenderSettings.stamp_note_text holds 768 bytes, NUL included
METADATA_PROPERTY = "yt_volumes_provenance"
SIDECAR_PROPERTY = "yt_volumes_provenance_sidecar"

_batch = None  # set by render_volume for the length of a batch run


def set_batch(config_path, config):
    """Record the batch config in manifests until `set_batch(None, None)`."""
    global _batch
    _batch = None if config is None else {"config_path": config_path, "config": config}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value, length=16):
    """Short stable hash of a JSON-able value."""
    return hashlib.sha256(_canonical(value).encode()).hexdigest()[:length]


def _version(module_name, dist_name=None):
    module = sys.modules.get(module_name)
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(dist_name or module_name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None if version is None else str(version)


def software():
    import bpy

    addon = sys.modules[__package__].bl_info["version"]
    return {
        "blender": bpy.app.version_string,
        "yt_volumes": ".".join(map(str, addon)),
        "yt": _version("yt"),
        "yt_idv": _version("yt_idv"),
        "numpy": _version("numpy"),
    }


def camera_record(camera_obj, params, view=None):
    """The camera as rendered; `params` is its `CameraParams` (ray tables left out)."""
    record = {
        "object": camera_obj.name,
        "view": view,
        "matrix_world": params.matrix_world,
        "projection": params.projection,
        "lens_mm": params.lens_mm,
        "sensor_mm": [params.sensor_width_mm, params.sensor_height_mm],
        "sensor_fit": params.sensor_fit,
        "shift": [params.shift_x, params.shift_y],
        "ortho_scale": params.ortho_scale,
        "clip": [params.clip_start, params.clip_end],
        "resolution": [params.resolution_x, params.resolution_y],
    }
    if params.panorama is not None:
        record["panorama"] = {
            f.name: getattr(params.panorama, f.name) for f in dataclasses.fields(params.panorama)
            if f.name not in ("ray_directions", "ray_origins")
        }
    if params.dof is not None and params.dof.use_dof:
        record["dof"] = dataclasses.asdict(params.dof)
    return record


def volume_record(obj, request, cmap_range, dataset_info=None):
    """One volume's entry: what was rendered and with which look."""
    custom = request.volume.custom_params or {}
    dataset = request.volume.dataset
    record = {
        "object": obj.name,
        "renderer_mode": custom.get("renderer_mode"),
        "field": custom.get("yt_idv_field"),
        "dataset": {
            "path": dataset.dataset_path,
            "kind": dataset.dataset_kind or None,
            "id": dataset.dataset_id,
            "hash": dataset.version.hash if dataset.version else None,
        },
        "look_hash": digest(custom),
        "look": custom,
        "cmap_range": None if cmap_range is None else [float(v) for v in cmap_range],
        "matrix_world": request.volume.matrix_world,
    }
    zoom = {key[len("yt_idv_"):]: custom[key] for key in (
        "yt_idv_zoom_target", "yt_idv_zoom_level", "yt_idv_shot_space",
    ) if custom.get(key) not in (None, "")}
    if zoom:
        record["zoom"] = zoom
    if dataset_info:
        record["field"] = dataset_info.pop("field", None) or record["field"]
        record["dataset"].update(dataset_info)
    return record


def build_manifest(scene, frame, outputs, cameras, volumes):
    import bpy

    manifest = {
        "schema": SCHEMA,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "frame": frame,
        "scene": scene.name,
        "blend_file": bpy.data.filepath or None,
        "outputs": [os.path.basename(p) for p in outputs],
        "engine": scene.render.engine,
        "software": software(),
        "cameras": cameras,
        "volumes": volumes,
    }
    if scene.render.engine == "CYCLES":
        manifest["samples"] = scene.cycles.samples
    if _batch is not None:
        manifest["batch"] = _batch
    manifest["manifest_hash"] = digest({k: v for k, v in manifest.items() if k != "created"})
    return manifest


def summary(manifest, limit=NOTE_MAX):
    """Compact JSON for the Note metadata, at most `limit` bytes; drops detail
    (zoom, colormap ranges, dataset names, then per-volume entries) to fit.
    """
    def volume(v, level):
        entry = {"obj": v["object"], "mode": v["renderer_mode"], "look": v["look_hash"]}
        if level < 3:
            path = v["dataset"].get("path")
            entry["ds"] = os.path.basename(path) if path else v["dataset"].get("name")
            entry["field"] = v["field"]
        if level < 2 and v["cmap_range"] is not None:
            entry["cmap"] = [float(f"{x:.4g}") for x in v["cmap_range"]]
        if level < 1 and "zoom" in v:
            entry["zoom"] = v["zoom"]
        return entry

    base = {
        "yt_volumes": manifest["software"]["yt_volumes"],
        "frame": manifest["frame"],
        "manifest": manifest["manifest_hash"],
    }
    for level in range(4):
        text = _canonical({**base, "volumes": [volume(v, level) for v in manifest["volumes"]]})
        if len(text.encode()) <= limit:
            return text
    return _canonical({**base, "volumes": len(manifest["volumes"])})


# --- PNG text chunk ----------------------------------------------------------


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def add_png_text(path, key, text):
    """Insert (or replace) a compressed iTXt chunk `key` in the PNG at `path`."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    keyword = key.encode("latin-1")
    out, pos = [data[:8]], 8
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        end = pos + 12 + length
        body = data[pos + 8:pos + 8 + length]
        if kind == b"iTXt" and body.split(b"\0", 1)[0] == keyword:
            pos = end
            continue
        if kind == b"IEND":
            payload = keyword + b"\0\x01\x00\0\0" + zlib.compress(text.encode("utf-8"))
            out.append(_chunk(b"iTXt", payload))
        out.append(data[pos:end])
        pos = end
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"".join(out))
    os.replace(tmp, path)


def read_png_text(path, key=PNG_KEY):
    """The text of iTXt/tEXt chunk `key` in a PNG, or None."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 8
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        name, _, rest = body.partition(b"\0")
        if name.decode("latin-1") != key:
            continue
        if kind == b"tEXt":
            return rest.decode("latin-1")
        if kind == b"iTXt":
            compressed, rest = rest[0], rest[2:]
            text = rest.split(b"\0", 2)[2]
            return (zlib.decompress(text) if compressed else text).decode("utf-8")
    return None


def read(path):
    """The manifest of a rendered PNG, or of its sidecar JSON; None if neither."""
    sidecar = os.path.splitext(path)[0] + SIDECAR_SUFFIX
    if path.lower().endswith(".png"):
        text = read_png_text(path)
        if text is not None:
            return json.loads(text)
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            return json.load(f)
    return None


def write_outputs(manifest, outputs, sidecar):
    """Attach `manifest` to each written output: an iTXt chunk in PNGs, and
    a sidecar JSON if `sidecar`.
    """
    text = json.dumps(manifest, default=str)
    for path in outputs:
        if not os.path.exists(path):
            continue
        if path.lower().endswith(".png"):
            add_png_text(path, PNG_KEY, text)
        if sidecar:
            with open(os.path.splitext(path)[0] + SIDECAR_SUFFIX, "w") as f:
                json.dump(manifest, f, indent=2, default=str)


# --- Scene settings ----------------------------------------------------------


def enabled(scene):
    return bool(getattr(scene, METADATA_PROPERTY, True))


def sidecar_enabled(scene):
    return bool(getattr(scene, SIDECAR_PROPERTY, False))


def stamp_conflict(render):
    """Why the Note can't carry the summary, or None."""
    if render.use_stamp:
        return "Burn Into Image is on, so the note would be drawn on the frame"
    if render.use_stamp_note and render.stamp_note_text and not _is_ours(render.stamp_note_text):
        return "the scene has its own Note"
    return None


def _is_ours(text):
    return text.startswith('{"frame":')


def set_note(render, text):
    """Put `text` in the Note metadata; assigns only what differs."""
    if not render.use_stamp_note:
        render.use_stamp_note = True
    if render.stamp_note_text != text:
        render.stamp_note_text = text


def register():
    import bpy

    setattr(bpy.types.Scene, METADATA_PROPERTY, bpy.props.BoolProperty(
        name="Provenance Metadata",
        description=(
            "Record how each frame was made (datasets, looks, colormap ranges, camera, versions) "
            "in its metadata: a summary in the Note, and the full record in PNG outputs"
        ),
        default=True,
    ))
    setattr(bpy.types.Scene, SIDECAR_PROPERTY, bpy.props.BoolProperty(
        name="Provenance Sidecar JSON",
        description="Also write the full record next to each output as <output>.provenance.json",
        default=False,
    ))


def unregister():
    import bpy

    for name in (SIDECAR_PROPERTY, METADATA_PROPERTY):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)
