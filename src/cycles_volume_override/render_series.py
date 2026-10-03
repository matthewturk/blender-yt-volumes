"""Animate a yt dataset series through a Blender scene, one output at a time.

    python -m cycles_volume_override.render_series series.toml [options]

Takes a `render_volume` config whose `[data]` names a `series` (glob or list
of paths) or a `simulation` instead of one dataset, plus a `[time]` table
mapping frames to outputs. Consecutive frames that show the same output form
a segment, and each segment runs as its own `render_volume` process: yt_idv
never frees GPU textures, so a fresh process per dataset is what releases
them. See `examples/render_series.toml`.
"""

import argparse
import bisect
import dataclasses
import glob
import json
import os
import signal
import subprocess
import sys

from . import render_volume as rv


@dataclasses.dataclass
class Output:
    path: str
    time: float | None  # in time.units; None when mapping by index


@dataclasses.dataclass
class Segment:
    output_index: int
    path: str
    start: int
    end: int  # inclusive


# --- Frame -> output mapping (pure) ------------------------------------------


def interpolate_keys(keys, frame):
    """Piecewise-linear `[[frame, value], ...]` at `frame`, clamped at both ends."""
    return rv.interpolate_rows(keys, frame)[0]


def pick_output(values, target, pick):
    """Index into sorted `values` for `target`: the nearest, or the last one <= target."""
    i = bisect.bisect_right(values, target)
    if pick == "previous":
        return max(i - 1, 0)
    if i == 0:
        return 0
    if i == len(values):
        return len(values) - 1
    return i if values[i] - target < target - values[i - 1] else i - 1


def plan_segments(outputs, frames, time_config):
    """Group `frames` (inclusive `(start, end)`) into runs that show the same output."""
    start, end = frames
    if time_config["by"] == "index":
        values = list(range(len(outputs)))
    else:
        values = [o.time for o in outputs]
    keys = time_config["keys"] or [[start, values[0]], [end, values[-1]]]

    segments = []
    for frame in range(start, end + 1):
        index = pick_output(values, interpolate_keys(keys, frame), time_config["pick"])
        if segments and segments[-1].output_index == index and segments[-1].end == frame - 1:
            segments[-1].end = frame
        else:
            segments.append(Segment(index, outputs[index].path, frame, frame))
    return segments


def missing_runs(segment, prefix):
    """Inclusive `(start, end)` runs of the segment's frames with no PNG on disk yet."""
    runs = []
    for frame in range(segment.start, segment.end + 1):
        if os.path.exists(f"{prefix}{frame:04d}.png"):
            continue
        if runs and runs[-1][1] == frame - 1:
            runs[-1][1] = frame
        else:
            runs.append([frame, frame])
    return [tuple(r) for r in runs]


def segment_config(config, segment, frames, cmap_range):
    """The single-dataset `render_volume` config for one run of a segment."""
    out = json.loads(json.dumps(config))
    out["data"].update(path=segment.path, series=None, simulation=None)
    out["blender"]["frames"] = list(frames)
    out["render"]["cmap_range"] = cmap_range
    return out


# --- Output discovery (needs yt) ----------------------------------------------


def discover_outputs(data, time_config):
    """The series' outputs, sorted by time (or in yt's order when mapping by index)."""
    import yt

    if data["series"]:
        series = data["series"]
        paths = sorted(glob.glob(series)) if isinstance(series, str) else list(series)
    else:
        sim = data["simulation"]
        simulation = yt.load_simulation(sim["parameter_file"], sim["code"], find_outputs=sim.get("find_outputs", False))
        simulation.get_time_series(**sim.get("time_series", {}))
        paths = list(simulation.outputs)
    if not paths:
        raise rv.ConfigError("the series matched no outputs")
    if time_config["by"] == "index":
        return [Output(p, None) for p in paths]

    outputs = []
    for p in paths:
        ds = yt.load(p, **data["load_kwargs"])
        outputs.append(Output(p, float(ds.current_time.to(time_config["units"]))))
    return sorted(outputs, key=lambda o: o.time)


# --- Driver -------------------------------------------------------------------


def _run_segment(config_path, report_path):
    cmd = [sys.executable, "-m", "cycles_volume_override.render_volume", config_path, "--report", report_path]
    child = subprocess.Popen(cmd)
    try:
        return child.wait()
    except BaseException:
        # Ctrl-C or SIGTERM on the driver must not leave the segment rendering on its own.
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
        raise


def _render_run(state_dir, name, run_config):
    """Write `run_config`, render it in a child process, and return its report.

    Raises `SegmentFailed` if the child fails.
    """
    config_path = os.path.join(state_dir, f"{name}.json")
    report_path = os.path.join(state_dir, f"{name}.report.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    code = _run_segment(config_path, report_path)
    if code != 0:
        raise SegmentFailed(code, config_path)
    with open(report_path) as f:
        return json.load(f)


class SegmentFailed(Exception):
    def __init__(self, code, config_path):
        super().__init__(code, config_path)
        self.code, self.config_path = code, config_path


def _probe_cmap_curve(config, segments, state_dir):
    """Auto-scaled `[min, max]` at the first and last frame, as a two-key curve.

    The probe frames render into `state_dir`, not the output.
    """
    keys = []
    for label, seg, frame in (("first", segments[0], segments[0].start), ("last", segments[-1], segments[-1].end)):
        probe = segment_config(config, seg, (frame, frame), None)
        probe["blender"]["output"] = os.path.join(state_dir, f"probe_{label}_")
        print(f"[render_series] probing the {label} output's colormap range at frame {frame}", flush=True)
        low, high = _render_run(state_dir, f"probe_{label}", probe)["cmap_range"]
        keys.append([frame, low, high])
    return keys


def run(config, outputs, skip_existing=False, dry_run=False):
    """Render every segment; returns the process exit code."""
    frames = rv.parse_frames(config["blender"]["frames"])
    segments = plan_segments(outputs, frames, config["time"])
    prefix = config["blender"]["output"]
    for seg in segments:
        time = outputs[seg.output_index].time
        when = f" t={time:g} {config['time']['units']}" if time is not None else ""
        print(f"[render_series] frames {seg.start}-{seg.end}: {seg.path}{when}", flush=True)
    if dry_run:
        return 0

    state_dir = os.path.join(os.path.dirname(prefix) or ".", ".render_series")
    os.makedirs(state_dir, exist_ok=True)
    mode = config["render"]["cmap_range"]
    cmap_range = None if mode in ("first", "first_last") else mode
    saved_path = os.path.join(state_dir, "cmap_range.json")
    if mode in ("first", "first_last") and os.path.exists(saved_path):
        # A resumed run must reuse what the original run pinned.
        with open(saved_path) as f:
            cmap_range = json.load(f)
        print(f"[render_series] colormap range {cmap_range} (from an earlier run)", flush=True)

    try:
        if mode == "first_last" and cmap_range is None:
            cmap_range = _probe_cmap_curve(config, segments, state_dir)
            with open(saved_path, "w") as f:
                json.dump(cmap_range, f)
            print(f"[render_series] colormap curve {cmap_range}", flush=True)

        for n, seg in enumerate(segments, 1):
            runs = missing_runs(seg, prefix) if skip_existing else [(seg.start, seg.end)]
            for start, end in runs:
                print(f"[render_series] segment {n}/{len(segments)} frames {start}-{end}", flush=True)
                report = _render_run(state_dir, f"segment_{seg.start:04d}_{start:04d}-{end:04d}",
                                     segment_config(config, seg, (start, end), cmap_range))
                if mode == "first" and cmap_range is None:
                    cmap_range = report["cmap_range"]
                    with open(saved_path, "w") as f:
                        json.dump(cmap_range, f)
                    print(f"[render_series] colormap range pinned to {cmap_range}", flush=True)
    except SegmentFailed as exc:
        print(f"[render_series] segment failed (exit {exc.code}); rerun with: "
              f"python -m cycles_volume_override.render_volume {exc.config_path}", file=sys.stderr)
        return exc.code
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m cycles_volume_override.render_series",
        description="Animate a yt dataset series through a Blender scene.",
    )
    parser.add_argument("config", help="TOML config with data.series or data.simulation")
    parser.add_argument("--frames", help="blender.frames: N or N-M")
    parser.add_argument("--output", help="blender.output: path prefix for frame PNGs")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="any config key, e.g. time.pick=\"previous\" (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="print the frame -> output plan; don't render")
    parser.add_argument("--skip-existing", action="store_true", help="skip frames whose PNG already exists")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    # SIGTERM raises SystemExit, so _run_segment stops the running child too.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))

    overrides = [(k, v) for k, v in (("blender.frames", args.frames), ("blender.output", args.output)) if v is not None]
    try:
        for item in args.set:
            key, sep, value = item.partition("=")
            if not sep:
                raise rv.ConfigError(f"--set {item!r} is not KEY=VALUE")
            overrides.append((key.strip(), rv._parse_value(value.strip())))
        config = rv.load_config(args.config, overrides, series=True)
        outputs = discover_outputs(config["data"], config["time"])
    except rv.ConfigError as exc:
        print(f"render_series: {exc}", file=sys.stderr)
        return 2
    return run(config, outputs, skip_existing=args.skip_existing, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
