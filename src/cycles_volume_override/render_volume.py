"""Render a yt dataset into a Blender scene from one TOML config file.

    python -m cycles_volume_override.render_volume scene.toml [options]

The config names the dataset (and optional subset), the .blend file and the
Empty-Cube object to replace with the volume, and how it should look.
Command-line options override the file, and `--set section.key=value` reaches
any key. Relative paths resolve against the config file's directory. See
`examples/render_volume.toml` for every key.
"""

import argparse
import bisect
import copy
import json
import math
import os
import shutil
import sys
import tempfile
import tomllib
import types

DEFAULTS = {
    "data": {
        "sample": None,  # yt.load_sample name
        "path": None,  # yt.load path; exactly one of sample/path/scene
        "scene": None,  # a yt_idv scene saved with SceneGraph.save
        "series": None,  # glob or list of paths; render_series only
        "simulation": None,  # {parameter_file, code, ...}; render_series only
        "load_kwargs": {},
        "field": "density",
        "no_ghost": True,
        "subset": None,
    },
    "blender": {
        "file": None,
        "volume": None,
        "camera": None,
        "frames": None,  # default: the .blend's own frame range
        "output": "renders/frame_",
        "resolution": None,
        "samples": None,
        "device": "OPTIX",
        "work_dir": None,
    },
    "render": {
        "mode": "YT_IDV_MAX_INTENSITY",
        "colormap": None,
        "cmap_range": None,  # [min, max], or a curve [[frame, min, max], ...]
        "cmap_interpolation": "linear",  # between curve keys: "linear" or "log"
        "reset_cmap_each_frame": False,
        "yt_idv": {},
    },
    "zoom": {
        "target": None,  # "max", or [x, y, z] in unitary coordinates
        "level": 0.0,  # log10 magnification, or a [[frame, level], ...] curve
    },
    # Frame -> dataset mapping; render_series only.
    "time": {
        "by": "time",  # "time" or "index"
        "units": "code_time",  # units of `keys` values when by = "time"
        "keys": None,  # [[frame, value], ...]; default spans the first to last output
        "pick": "nearest",  # "nearest" or "previous"
    },
}

SUBSET_KINDS = ("region", "sphere", "box")
DEVICES = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI", "CPU")
# Modes whose colormap lives on the component rather than in a preference.
COMPONENT_COLORMAP_MODES = ("YT_IDV_MAX_INTENSITY", "YT_IDV_PROJECTION")
COLORMAP_PREFERENCE = {
    "YT_IDV_TRANSFER_FUNCTION": "tf_colormap",
    "YT_IDV_THRESHOLD_FOG": "fog_colormap",
    "YT_IDV_MAX_INTENSITY_ALPHA": "mip_alpha_colormap",
}
PATH_KEYS = (("data", "path"), ("data", "scene"), ("blender", "file"), ("blender", "output"), ("blender", "work_dir"))

_active_config = None  # read by volume_provider during a render
_active_scene = None  # the scene volume_provider built, for render()'s report
_applied_cmap_range = None  # last (min, max) volume_provider applied


class ConfigError(ValueError):
    pass


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_value(text):
    """TOML-parse a `--set` value, falling back to a bare string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_override(config, dotted_key, value):
    *parents, leaf = dotted_key.split(".")
    node = config
    for part in parents:
        if node.get(part) is None:
            node[part] = {}
        node = node[part]
        if not isinstance(node, dict):
            raise ConfigError(f"--set {dotted_key}: {part!r} is not a table")
    node[leaf] = value


def parse_frames(frames):
    """`7`, `"3-9"`, `[3, 9]` -> inclusive `(start, end)`; None stays None."""
    if frames is None:
        return None
    if isinstance(frames, int):
        return frames, frames
    if isinstance(frames, str):
        start, sep, end = frames.partition("-")
        try:
            return (int(start), int(end)) if sep else (int(start), int(start))
        except ValueError:
            raise ConfigError(f"blender.frames {frames!r} is not N or N-M") from None
    if isinstance(frames, list) and len(frames) == 2 and all(isinstance(f, int) for f in frames):
        return frames[0], frames[1]
    raise ConfigError(f"blender.frames {frames!r} is not N, 'N-M' or [N, M]")


def _validate_time(time):
    if time["by"] not in ("time", "index"):
        raise ConfigError(f"time.by must be 'time' or 'index', got {time['by']!r}")
    if time["pick"] not in ("nearest", "previous"):
        raise ConfigError(f"time.pick must be 'nearest' or 'previous', got {time['pick']!r}")
    keys = time["keys"]
    if keys is None:
        return
    if not (isinstance(keys, list) and len(keys) >= 2
            and all(isinstance(k, list) and len(k) == 2 for k in keys)):
        raise ConfigError("time.keys must be a list of at least two [frame, value] pairs")
    frames = [k[0] for k in keys]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ConfigError("time.keys frames must be strictly increasing")


def is_cmap_curve(value):
    return isinstance(value, list) and bool(value) and all(isinstance(k, list) and len(k) == 3 for k in value)


def interpolate_rows(keys, frame, log=False):
    """Values of `[[frame, v0, v1, ...], ...]` at `frame`: piecewise linear
    (or linear in log space), clamped at both ends. Returns a list.
    """
    frames = [k[0] for k in keys]
    if frame <= frames[0]:
        return [float(v) for v in keys[0][1:]]
    if frame >= frames[-1]:
        return [float(v) for v in keys[-1][1:]]
    i = bisect.bisect_right(frames, frame)
    (f0, *a), (f1, *b) = keys[i - 1], keys[i]
    w = (frame - f0) / (f1 - f0)
    if log:
        return [math.exp(math.log(x) + (math.log(y) - math.log(x)) * w) for x, y in zip(a, b)]
    return [x + (y - x) * w for x, y in zip(a, b)]


def cmap_range_at(render, frame):
    """`(min, max)` the config asks for at `frame`, or `(None, None)` for auto."""
    cmap_range = render["cmap_range"]
    if cmap_range is None:
        return None, None
    if is_cmap_curve(cmap_range):
        return tuple(interpolate_rows(cmap_range, frame, log=render["cmap_interpolation"] == "log"))
    return tuple(cmap_range)


def zoom_level_at(zoom, frame):
    level = zoom["level"]
    if isinstance(level, list):
        return interpolate_rows(level, frame)[0]
    return float(level)


def _zoom_target_spec(target):
    """The `zoom_target` property string; repr keeps float64 precision."""
    if target is None:
        return ""
    if target == "max":
        return "max"
    return ", ".join(repr(float(v)) for v in target)


def _validate_zoom(zoom):
    target, level = zoom["target"], zoom["level"]
    if target is not None and target != "max" and not (
        isinstance(target, list) and len(target) == 3 and all(isinstance(v, (int, float)) for v in target)
    ):
        raise ConfigError("zoom.target must be \"max\" or [x, y, z] in unitary coordinates")
    if isinstance(level, (int, float)):
        return
    if not (isinstance(level, list) and len(level) >= 2
            and all(isinstance(k, list) and len(k) == 2 for k in level)):
        raise ConfigError("zoom.level must be a number or a list of at least two [frame, level] pairs")
    frames = [k[0] for k in level]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ConfigError("zoom.level frames must be strictly increasing")


def _validate_cmap(render, series):
    cmap_range = render["cmap_range"]
    if render["cmap_interpolation"] not in ("linear", "log"):
        raise ConfigError("render.cmap_interpolation must be 'linear' or 'log'")
    if series and cmap_range in ("first", "first_last"):
        return  # render_series resolves these
    if cmap_range is None:
        return
    if is_cmap_curve(cmap_range):
        if len(cmap_range) < 2:
            raise ConfigError("a render.cmap_range curve needs at least two [frame, min, max] keys")
        frames = [k[0] for k in cmap_range]
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise ConfigError("render.cmap_range curve frames must be strictly increasing")
        if render["cmap_interpolation"] == "log" and any(v <= 0 for k in cmap_range for v in k[1:]):
            raise ConfigError("log render.cmap_interpolation needs positive min/max values")
        return
    if isinstance(cmap_range, list) and len(cmap_range) == 2 and all(isinstance(v, (int, float)) for v in cmap_range):
        return
    extra = ', "first" or "first_last"' if series else ""
    raise ConfigError(f"render.cmap_range must be [min, max], a [[frame, min, max], ...] curve{extra}")


def validate(config, series=False):
    """Raise ConfigError for anything that would only fail minutes into a render.

    `series=True` validates a render_series config instead of a single dataset.
    """
    data, blender, render = config["data"], config["blender"], config["render"]
    if series:
        if bool(data["series"]) == bool(data["simulation"]):
            raise ConfigError("set exactly one of data.series or data.simulation")
        if data["sample"] is not None or data["path"] is not None or data["scene"] is not None:
            raise ConfigError("data.sample/data.path/data.scene can't be combined with a series")
        if data["simulation"] and not {"parameter_file", "code"} <= set(data["simulation"]):
            raise ConfigError("data.simulation needs parameter_file and code")
        if blender["frames"] is None:
            raise ConfigError("blender.frames is required for a series")
        _validate_time(config["time"])
    else:
        if data["series"] or data["simulation"]:
            raise ConfigError("data.series/data.simulation are rendered with render_series")
        if sum(data[key] is not None for key in ("sample", "path", "scene")) != 1:
            raise ConfigError("set exactly one of data.sample, data.path or data.scene")
        if data["scene"] is not None and data["subset"] is not None:
            raise ConfigError("data.subset can't be applied to a saved data.scene")
    for key in ("file", "volume"):
        if not blender[key]:
            raise ConfigError(f"blender.{key} is required")
    if not os.path.isfile(blender["file"]):
        raise ConfigError(f"blender.file {blender['file']!r} does not exist")
    if blender["device"] not in DEVICES:
        raise ConfigError(f"blender.device must be one of {DEVICES}")
    start_end = parse_frames(blender["frames"])
    if start_end and start_end[0] > start_end[1]:
        raise ConfigError(f"blender.frames {blender['frames']!r} ends before it starts")
    if not render["mode"].startswith("YT_IDV_"):
        raise ConfigError(f"render.mode {render['mode']!r} is not a YT_IDV_* mode")
    _validate_cmap(render, series)
    _validate_zoom(config["zoom"])
    subset = data["subset"]
    if subset is not None:
        kind = subset.get("kind")
        if kind not in SUBSET_KINDS:
            raise ConfigError(f"data.subset.kind must be one of {SUBSET_KINDS}, got {kind!r}")
        required = {"region": ("half_width",), "sphere": ("radius",), "box": ("left_edge", "right_edge")}[kind]
        missing = [k for k in required if k not in subset]
        if missing:
            raise ConfigError(f"data.subset ({kind}) needs {', '.join(missing)}")


def _resolve(base_dir, value):
    return os.path.normpath(os.path.join(base_dir, os.path.expanduser(value)))


def _resolve_paths(config, base_dir, only=None):
    for section, key in PATH_KEYS:
        value = config.get(section, {}).get(key)
        if value and (only is None or f"{section}.{key}" in only):
            config[section][key] = _resolve(base_dir, value)
    data = config.get("data", {})
    if data.get("series") and (only is None or "data.series" in only):
        series = data["series"]
        data["series"] = [_resolve(base_dir, p) for p in series] if isinstance(series, list) else _resolve(base_dir, series)
    simulation = data.get("simulation")
    if isinstance(simulation, dict) and simulation.get("parameter_file") and (
        only is None or any(k.startswith("data.simulation") for k in only)
    ):
        simulation["parameter_file"] = _resolve(base_dir, simulation["parameter_file"])


def load_config(path=None, overrides=(), series=False):
    """Defaults <- TOML (or JSON) file <- `(dotted_key, value)` overrides, validated.

    Relative paths resolve against the config file's directory, or the current
    directory for paths given as overrides.
    """
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "rb") as f:
            loaded = json.load(f) if str(path).endswith(".json") else tomllib.load(f)
        config = _merge(config, loaded)
        _resolve_paths(config, os.path.dirname(os.path.abspath(path)))
    for dotted_key, value in overrides:
        apply_override(config, dotted_key, value)
    _resolve_paths(config, os.getcwd(), only={key for key, _ in overrides} if path is not None else None)
    unknown = {section: set(config[section]) - set(DEFAULTS[section]) for section in DEFAULTS}
    unknown = {s: keys for s, keys in unknown.items() if keys}
    unknown_sections = set(config) - set(DEFAULTS)
    if unknown or unknown_sections:
        names = sorted(f"{s}.{k}" for s, keys in unknown.items() for k in keys) + sorted(unknown_sections)
        raise ConfigError(f"unknown config keys: {', '.join(names)}")
    validate(config, series=series)
    return config


# --- Volume provider ----------------------------------------------------------


def _quantity(ds, value, units):
    """A config length (number, or [number, "unit"]) as a yt quantity."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], str):
        return ds.quan(value[0], value[1])
    return ds.quan(value, units)


def _center(ds, subset, field):
    center = subset.get("center", "max")
    if center == "max":
        return ds.find_max(field)[1]
    if center in ("c", "center"):
        return ds.domain_center
    return ds.arr(center, subset.get("units", "code_length"))


def build_data_source(ds, data_config):
    """The yt data object for `data_config` (the whole domain when there's no subset)."""
    subset = data_config["subset"]
    if subset is None:
        return ds.all_data()
    kind = subset["kind"]
    units = subset.get("units", "code_length")
    if kind == "box":
        return ds.box(ds.arr(subset["left_edge"], units), ds.arr(subset["right_edge"], units))
    center = _center(ds, subset, data_config["field"])
    if kind == "sphere":
        return ds.sphere(center, _quantity(ds, subset["radius"], units))
    half_width = _quantity(ds, subset["half_width"], units)
    return ds.region(center, center - half_width, center + half_width)


def _field(data_config):
    field = data_config["field"]
    return tuple(field) if isinstance(field, list) else field


def volume_provider(ctx):
    """`yt_idv_volume_provider` driven by the active config; builds the scene once."""
    from .yt_idv_renderer import VolumeSetupResult

    global _active_scene, _applied_cmap_range
    config = _active_config
    render = config["render"]
    if ctx.previous_scene is not None:
        changed = False
        if render["reset_cmap_each_frame"]:
            for component in ctx.previous_scene.components:
                component._cmap_bounds_invalid = True
        elif is_cmap_curve(render["cmap_range"]):
            changed = _apply_cmap_range(ctx.previous_scene, cmap_range_at(render, ctx.frame.frame_number))
        return VolumeSetupResult(scene=ctx.previous_scene, ds=ctx.previous_ds, changed=changed)

    data = config["data"]
    if data["scene"]:
        scene, ds = _load_saved_scene(data["scene"])
        ctx.render_context.scene = scene
    else:
        scene, ds = _build_scene(ctx, data)

    for component in scene.components:
        if render["colormap"] and render["mode"] in COMPONENT_COLORMAP_MODES:
            component.colormap.colormap_name = render["colormap"]
        component._cmap_bounds_invalid = True
    if not render["reset_cmap_each_frame"]:
        _apply_cmap_range(scene, cmap_range_at(render, ctx.frame.frame_number))
    _active_scene = scene
    return VolumeSetupResult(scene=scene, ds=ds, changed=True)


def _build_scene(ctx, data):
    import yt

    from .yt_idv_renderer import scale_scene_to_subset

    if data["path"]:
        ds = yt.load(data["path"], **data["load_kwargs"])
    else:
        ds = yt.load_sample(data["sample"], **data["load_kwargs"])
    data_source = build_data_source(ds, data)
    scene = ctx.render_context.add_scene(data_source, _field(data), no_ghost=data["no_ghost"])
    if data["subset"] is not None and data["subset"].get("scale", True):
        scale_scene_to_subset(scene)
    return scene, ds


def _load_saved_scene(path):
    """A scene from `SceneGraph.save`, scaled so its saved bbox fills the volume box."""
    import numpy as np
    from yt_idv.scene_graph import SceneGraph

    from .yt_idv_renderer import scale_scene_to_subset

    scene = SceneGraph.load(path)
    scale_scene_to_subset(scene)
    # the renderer only reads ds.domain_center, in unitary units
    return scene, types.SimpleNamespace(domain_center=np.full(3, 0.5))


def _apply_cmap_range(scene, cmap_range):
    """Pin every component to `(min, max)`; returns True if that changed it."""
    global _applied_cmap_range
    if cmap_range == _applied_cmap_range:
        return False
    for component in scene.components:
        component.fixed_cmap_min, component.fixed_cmap_max = cmap_range
        # Fixed bounds only take effect when yt_idv recomputes the range.
        component._cmap_bounds_invalid = True
    _applied_cmap_range = cmap_range
    return True


# --- Blender side -------------------------------------------------------------


def _configure_blender(config):
    import bpy

    blender, render = config["blender"], config["render"]
    bpy.ops.wm.open_mainfile(filepath=blender["file"])
    bpy.ops.preferences.addon_enable(module=__package__)
    scene = bpy.context.scene

    prefs = bpy.context.preferences.addons[__package__].preferences
    # Must be set before render_session opens: it picks compositor/view settings by mode.
    prefs.renderer_mode = render["mode"]
    temp_work_dir = None if blender["work_dir"] else tempfile.mkdtemp(prefix="render_volume_")
    prefs.work_dir = blender["work_dir"] or temp_work_dir
    colormap_pref = COLORMAP_PREFERENCE.get(render["mode"])
    if render["colormap"] and colormap_pref:
        setattr(prefs.yt_idv, colormap_pref, render["colormap"])
    for key, value in render["yt_idv"].items():
        if not hasattr(prefs.yt_idv, key):
            raise ConfigError(f"render.yt_idv.{key} is not a yt_idv preference")
        setattr(prefs.yt_idv, key, value)

    obj = bpy.data.objects.get(blender["volume"])
    if obj is None:
        raise ConfigError(f"no object named {blender['volume']!r} in {blender['file']}")
    obj.volume_override.is_volume_override = True
    obj.volume_override.yt_idv_volume_provider = f"{__name__}:volume_provider"
    obj.volume_override.zoom_target = _zoom_target_spec(config["zoom"]["target"])

    if blender["camera"]:
        camera = bpy.data.objects.get(blender["camera"])
        if camera is None or camera.type != "CAMERA":
            raise ConfigError(f"blender.camera {blender['camera']!r} is not a camera object")
        scene.camera = camera
    scene.render.engine = "CYCLES"
    if blender["resolution"]:
        scene.render.resolution_x, scene.render.resolution_y = blender["resolution"]
    if blender["samples"]:
        scene.cycles.samples = blender["samples"]
    scene.render.image_settings.file_format = "PNG"

    configure_cycles_device(scene, blender["device"])
    return scene, temp_work_dir


def configure_cycles_device(scene, device_type):
    """Render and denoise on `device_type`; prints what Cycles ended up using."""
    import bpy

    cycles = scene.cycles
    if device_type == "CPU":
        cycles.device = "CPU"
    else:
        # Headless sessions start with compute_device_type "NONE" whatever the .blend says.
        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        cycles_prefs.compute_device_type = device_type
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            device.use = device.type == device_type
        if not any(d.use for d in cycles_prefs.devices):
            raise ConfigError(f"blender.device {device_type!r}: Cycles found no such device")
        cycles.device = "GPU"
        # The .blend's denoiser is usually OpenImageDenoise on the CPU, which
        # dominates frame time at 1080p even when rendering on the GPU.
        if device_type == "OPTIX":
            cycles.denoiser = "OPTIX"
        if hasattr(cycles, "denoising_use_gpu"):
            cycles.denoising_use_gpu = True
    print(f"[render_volume] cycles device={cycles.device} ({device_type}), denoiser={cycles.denoiser}, "
          f"denoise_on_gpu={getattr(cycles, 'denoising_use_gpu', False)}", flush=True)


def _cmap_range(scene):
    """`[min, max]` the scene's first colormapped component used, or None."""
    for component in getattr(scene, "components", ()):
        low, high = getattr(component, "cmap_min", None), getattr(component, "cmap_max", None)
        if low is not None and high is not None:
            return [float(low), float(high)]
    return None


def write_report(path, config, written):
    """JSON summary of a run: dataset, frames written, and the colormap range used."""
    data = config["data"]
    report = {
        "dataset": data["path"] or data["sample"] or data["scene"],
        "frames": written,
        "cmap_range": _cmap_range(_active_scene),
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def render(config):
    """Render every configured frame; returns the written PNG paths."""
    import bpy

    from . import operators

    global _active_config
    _active_config = config
    scene, temp_work_dir = _configure_blender(config)
    volume = bpy.data.objects[config["blender"]["volume"]]
    start, end = parse_frames(config["blender"]["frames"]) or (scene.frame_start, scene.frame_end)
    prefix = config["blender"]["output"]
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    written = []
    try:
        with operators.render_session(bpy.context):
            for frame in range(start, end + 1):
                scene.frame_set(frame)
                if config["zoom"]["target"] is not None:
                    volume.volume_override.zoom_level = zoom_level_at(config["zoom"], frame)
                # The add-on's output isn't auto-numbered.
                scene.render.filepath = f"{prefix}{frame:04d}"
                bpy.ops.render.volume_override(animation=False)
                written.append(f"{scene.render.filepath}.png")
                print(f"[render_volume] frame {frame} -> {written[-1]}", flush=True)
    finally:
        if temp_work_dir:
            shutil.rmtree(temp_work_dir, ignore_errors=True)
    return written


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="python -m cycles_volume_override.render_volume",
        description="Render a yt dataset into a Blender scene's volume Empty.",
    )
    parser.add_argument("config", nargs="?", help="TOML config file (optional if options cover the required keys)")
    parser.add_argument("--sample", help="data.sample: yt.load_sample name")
    parser.add_argument("--path", help="data.path: dataset to yt.load")
    parser.add_argument("--field", help="data.field")
    parser.add_argument("--blend", help="blender.file")
    parser.add_argument("--volume", help="blender.volume: the Empty-Cube object to replace")
    parser.add_argument("--frames", help="blender.frames: N or N-M")
    parser.add_argument("--output", help="blender.output: path prefix for frame PNGs")
    parser.add_argument("--mode", help="render.mode, e.g. YT_IDV_MAX_INTENSITY_ALPHA")
    parser.add_argument("--colormap", help="render.colormap")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="any config key, e.g. data.subset.scale=false (repeatable)")
    parser.add_argument("--check", action="store_true", help="validate the config and print it; don't render")
    parser.add_argument("--report", metavar="PATH", help="write a JSON summary (frames, colormap range) here")
    return parser.parse_args(argv)


def _overrides(args):
    flags = {
        "sample": "data.sample", "path": "data.path", "field": "data.field",
        "blend": "blender.file", "volume": "blender.volume", "frames": "blender.frames",
        "output": "blender.output", "mode": "render.mode", "colormap": "render.colormap",
    }
    overrides = [(key, getattr(args, flag)) for flag, key in flags.items() if getattr(args, flag) is not None]
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(f"--set {item!r} is not KEY=VALUE")
        overrides.append((key.strip(), _parse_value(value.strip())))
    return overrides


def main(argv=None):
    if argv is None:
        # Inside Blender (`blender -b -P render_volume.py -- ...`) our args follow "--".
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = _parse_args(argv)
    try:
        config = load_config(args.config, _overrides(args))
    except ConfigError as exc:
        print(f"render_volume: {exc}", file=sys.stderr)
        return 2
    if args.check:
        import pprint

        pprint.pprint(config, sort_dicts=False)
        return 0
    written = render(config)
    if args.report:
        write_report(args.report, config, written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
