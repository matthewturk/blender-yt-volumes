"""Probe analyses: per-frame yt computations on a sphere whose result becomes
a float Image, e.g. a phase diagram shown by the "yt Image Panel" annotation.

A probe's analysis is a `"module.submodule:function"` string naming
`analysis(ctx: ProbeContext) -> ProbeResult`. It runs only when its key (the
dataset, sphere, analysis, params and, if asked, the frame) changes. Nothing
here imports bpy.
"""

import collections
import dataclasses
import importlib

import numpy as np

from . import change_tracking
from .external_render import FrameParams

RESULT_CACHE_SIZE = 32


@dataclasses.dataclass
class ProbeContext:
    """Passed to a probe's analysis."""

    ds: "object"  # the volume's yt dataset
    sphere: "object"  # ds.sphere(center, radius)
    center: "object"  # yt array, code_length
    radius: "object"  # yt quantity, code_length
    frame: FrameParams
    params: dict  # the probe's free-form key/value strings
    state: dict  # JSON-able, saved on the probe; e.g. ranges fitted on the first run


@dataclasses.dataclass
class ProbeResult:
    """Returned by a probe's analysis."""

    values: np.ndarray  # (h, w) or (h, w, 4) float; row 0 is the bottom. NaN marks empty pixels.
    # Copied to the Image's ID properties, and to inputs of the same name on panels showing it.
    meta: dict = dataclasses.field(default_factory=dict)


_results = collections.OrderedDict()  # key hash -> ProbeResult


def resolve(spec):
    """`"module.submodule:function"` as a callable."""
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"probe analysis must be a 'module.submodule:function' string, got {spec!r}")
    module_path, _, name = spec.rpartition(":")
    try:
        return getattr(importlib.import_module(module_path), name)
    except AttributeError as exc:
        raise AttributeError(f"module {module_path!r} has no attribute {name!r} (probe analysis {spec!r})") from exc


def to_pixels(values):
    """`(h, w, 4)` float32 RGBA: a 2D array goes in R, G and B, with alpha 0
    where it's NaN (and those pixels zeroed).
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        finite = np.isfinite(values)
        v = np.where(finite, values, 0.0)
        values = np.stack([v, v, v, finite.astype(np.float32)], axis=-1)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError(f"probe values must be (h, w) or (h, w, 4), got {values.shape}")
    return np.ascontiguousarray(np.nan_to_num(values, nan=0.0))


def probe_key(dataset_hash, center_unitary, radius_unitary, analysis, params, frame=None):
    """Everything a probe's result depends on, as a plain dict to hash."""
    return {
        "dataset": dataset_hash,
        "center": [float(c) for c in center_unitary],
        "radius": float(radius_unitary),
        "analysis": analysis,
        "params": dict(params),
        "frame": frame,
    }


def run(probe_id, key, make_context, analysis):
    """`(result, fresh)`: the cached result for `key`, or a new one from
    `analysis(make_context())`. `fresh` is False when the probe's Image
    already holds this result, so nothing needs writing.
    """
    version = change_tracking.version_for(probe_id, "probe", key, record=False)
    result = _results.get(version.hash)
    if result is None:
        result = resolve(analysis)(make_context())
        result = dataclasses.replace(result, values=to_pixels(result.values))
        _results[version.hash] = result
        while len(_results) > RESULT_CACHE_SIZE:
            _results.popitem(last=False)
    else:
        _results.move_to_end(version.hash)
    change_tracking.record_hash(probe_id, "probe", version.hash)
    return result, version.changed


def forget(probe_id):
    """Make `probe_id`'s next run write its Image again."""
    change_tracking.forget(probe_id, "probe")


def clear_cache():
    _results.clear()


def _field(spec):
    parts = [p.strip() for p in str(spec).split(",")]
    return tuple(parts) if len(parts) == 2 else parts[0]


def _label(ds, field, units):
    name = ds._get_field_info(field).name
    name = name[1] if isinstance(name, tuple) else name
    return f"log {name} ({units})" if units and units != "dimensionless" else f"log {name}"


def phase_plot(ctx: ProbeContext) -> ProbeResult:
    """2D profile of `z_field` (summed, or averaged by `weight_field`) over
    log `x_field` and log `y_field` in the sphere. Values are log10; axis and
    value ranges are fitted on the first run and kept in `state`, so data
    outside them is left out (Reset Fitted Ranges refits).

    Params: x_field (default "gas, density"), y_field ("gas, temperature"),
    z_field ("gas, mass"), weight_field (none), x_bins and y_bins (64),
    fit ("first", or "every" to refit each run, moving the axes).
    """
    import yt

    p = ctx.params
    x = _field(p.get("x_field", "gas, density"))
    y = _field(p.get("y_field", "gas, temperature"))
    z = _field(p.get("z_field", "gas, mass"))
    weight = _field(p["weight_field"]) if p.get("weight_field") else None
    bins = (int(p.get("x_bins", 64)), int(p.get("y_bins", 64)))

    state = ctx.state
    if p.get("fit", "first") == "every":
        state.clear()
    if "x_range" not in state or "y_range" not in state:
        for name, field in (("x", x), ("y", y)):
            data = ctx.sphere[field]
            positive = data[data > 0]
            if positive.size == 0:
                raise ValueError(f"phase_plot: {field} has no positive values in the sphere")
            state[f"{name}_units"] = str(data.units)
            state[f"{name}_range"] = [float(positive.min()), float(positive.max())]
    units = {x: state["x_units"], y: state["y_units"]}
    extrema = {x: tuple(state["x_range"]), y: tuple(state["y_range"])}
    profile = yt.create_profile(
        ctx.sphere, [x, y], [z], n_bins=bins, extrema=extrema, logs={x: True, y: True},
        units=units, weight_field=weight,
    )
    raw = np.asarray(profile[z].d, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(profile.used & (raw > 0), np.log10(raw), np.nan).T  # (y, x), row 0 at y min
    if "value_range" not in state:
        finite = values[np.isfinite(values)]
        state["value_range"] = [float(finite.min()), float(finite.max())] if finite.size else [0.0, 1.0]

    (x0, x1), (y0, y1) = (np.log10(state["x_range"]), np.log10(state["y_range"]))
    return ProbeResult(values=values, meta={
        "X Min": float(x0), "X Max": float(x1), "Y Min": float(y0), "Y Max": float(y1),
        "Value Min": state["value_range"][0], "Value Max": state["value_range"][1],
        "X Title": _label(ctx.ds, x, state["x_units"]), "Y Title": _label(ctx.ds, y, state["y_units"]),
    })
