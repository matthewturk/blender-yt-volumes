"""Probe analyses: per-frame yt computations on a sphere whose result becomes
a float Image (e.g. a phase diagram shown by the "yt Image Panel" annotation)
and named scalars (put on the probe for drivers and annotations).

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

    # (h, w) or (h, w, 4) float; row 0 is the bottom. NaN marks empty pixels. None: no Image.
    values: "np.ndarray | None" = None
    # Copied to the Image's ID properties, and to inputs of the same name on annotations showing the probe.
    meta: dict = dataclasses.field(default_factory=dict)
    # name -> float, put in the probe's `["yt_values"]` and synced like meta.
    scalars: dict = dataclasses.field(default_factory=dict)


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


def probe_key(dataset_hash, center_unitary, radius_unitary, analysis, params, frame=None, loaded=None):
    """Everything a probe's result depends on, as a plain dict to hash.
    `loaded` identifies the `ds` object, which a time-varying loader replaces
    without changing the dataset's version.
    """
    return {
        "dataset": dataset_hash,
        "loaded": loaded,
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
        if result.values is not None:
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
    fit ("first", or "every" to refit each run, moving the axes), and
    x_range, y_range (in the field's default units) and value_range (log10)
    as "low, high" to fix those ranges instead of fitting them, and z_units
    (e.g. "Msun").
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
    for name, field in (("x", x), ("y", y)):
        fixed = _range(p.get(f"{name}_range"))
        if fixed is not None:
            state[f"{name}_units"] = str(ctx.ds._get_field_info(field).units or "dimensionless")
            state[f"{name}_range"] = list(fixed)
    fixed_values = _range(p.get("value_range"))
    if fixed_values is not None:
        state["value_range"] = list(fixed_values)
    if "x_range" not in state or "y_range" not in state:
        for name, field in (("x", x), ("y", y)):
            data = ctx.sphere[field]
            positive = data[data > 0]
            if positive.size == 0:
                raise ValueError(f"phase_plot: {field} has no positive values in the sphere")
            state[f"{name}_units"] = str(data.units)
            low, high = float(positive.min()), float(positive.max())
            if high <= low * 1.0001:  # uniform data, e.g. a simulation's initial conditions
                low, high = low / 3.0, high * 3.0
            state[f"{name}_range"] = [low, high]
    units = {x: state["x_units"], y: state["y_units"]}
    if p.get("z_units"):
        units[z] = p["z_units"]
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
    state.setdefault("value_units", str(profile[z].units))

    (x0, x1), (y0, y1) = (np.log10(state["x_range"]), np.log10(state["y_range"]))
    return ProbeResult(values=values, meta={
        "X Min": float(x0), "X Max": float(x1), "Y Min": float(y0), "Y Max": float(y1),
        "Value Min": state["value_range"][0], "Value Max": state["value_range"][1],
        "X Title": _label(ctx.ds, x, state["x_units"]), "Y Title": _label(ctx.ds, y, state["y_units"]),
        "Value Title": _label(ctx.ds, z, state.get("value_units", "")),
    })


_STATS = ("mean", "sum", "min", "max")


def _short(field):
    return field[1] if isinstance(field, tuple) else field


def quantities(ctx: ProbeContext) -> ProbeResult:
    """Scalars of the data in the sphere, one per field, named by the field
    (e.g. `temperature`), in the field's default units (in meta as
    "<name> Units"). No Image.

    Params: fields ("gas, density; gas, temperature", ";"-separated),
    stat ("mean", "sum", "min" or "max"), weight_field ("gas, mass" for
    mean; "none" for unweighted), units (";"-separated, one per field, e.g.
    "Msun"; empty keeps the field's).
    """
    p = ctx.params
    stat = p.get("stat", "mean")
    if stat not in _STATS:
        raise ValueError(f"quantities: stat must be one of {_STATS}, got {stat!r}")
    weight = p.get("weight_field", "gas, mass")
    weight = None if weight.strip().lower() in ("", "none") else _field(weight)
    scalars, meta = {}, {}
    specs = [spec for spec in p.get("fields", "gas, density; gas, temperature").split(";") if spec.strip()]
    units = [u.strip() for u in p.get("units", "").split(";")]
    for i, spec in enumerate(specs):
        field = _field(spec)
        if stat == "mean":
            value = ctx.sphere.mean(field, weight=weight)
        else:
            value = getattr(ctx.sphere, stat)(field)
        if i < len(units) and units[i]:
            value = value.to(units[i])
        name = _short(field)
        scalars[name] = float(value.d)
        meta[f"{name} Units"] = str(value.units)
    return ProbeResult(scalars=scalars, meta=meta)


def _range(spec):
    """`"low, high"` as floats, or None if unset."""
    if not spec or not spec.strip():
        return None
    low, high = (float(v) for v in spec.split(","))
    return low, high


def history(ctx: ProbeContext) -> ProbeResult:
    """One scalar of another analysis over time, as an `(1, n, 4)` Image for
    "yt Line Panel": R is x (dataset time or frame), G the value, alpha 1.
    Each run adds a point to the history in `state`; the inner analysis's
    scalars and meta pass through.

    Params: of (the inner analysis, default quantities), quantity (which of
    its scalars; default its first), x ("time" or "frame"), time_units
    ("Myr"), log ("true" plots log10 of the value), x_range and y_range
    ("low, high" to fix an axis instead of fitting it to the points). Other
    params go to the inner analysis too.
    """
    p = ctx.params
    inner_spec = p.get("of", f"{__name__}:quantities")
    inner = resolve(inner_spec)
    inner_state = ctx.state.setdefault("inner", {})
    inner_result = inner(dataclasses.replace(ctx, state=inner_state))
    if not inner_result.scalars:
        raise ValueError(f"history: {inner_spec} returned no scalars")
    quantity = p.get("quantity") or next(iter(inner_result.scalars))
    if quantity not in inner_result.scalars:
        raise ValueError(f"history: {inner_spec} has no scalar {quantity!r}; it has {sorted(inner_result.scalars)}")
    value = inner_result.scalars[quantity]
    log = p.get("log", "true").strip().lower() in ("1", "true", "yes")
    if log:
        value = float(np.log10(value)) if value > 0 else float("nan")

    if p.get("x", "time") == "frame":
        x, x_title = float(ctx.frame.frame_number), "frame"
    else:
        units = p.get("time_units", "Myr")
        x, x_title = float(ctx.ds.current_time.to(units)), f"time ({units})"
    points = {float(k): v for k, v in ctx.state.get("points", [])}
    points[x] = value
    ordered = sorted((k, v) for k, v in points.items())
    ctx.state["points"] = [[k, v] for k, v in ordered]

    xs = np.array([k for k, _ in ordered])
    ys = np.array([v for _, v in ordered], dtype=np.float64)
    pixels = np.zeros((1, len(xs), 4), dtype=np.float32)
    pixels[0, :, 0] = xs
    pixels[0, :, 1] = np.nan_to_num(ys)
    pixels[0, :, 3] = np.isfinite(ys)
    finite = ys[np.isfinite(ys)]
    y_low, y_high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    pad = 0.05 * (y_high - y_low) or 0.5
    y_low, y_high = _range(p.get("y_range")) or (y_low - pad, y_high + pad)
    x_low, x_high = _range(p.get("x_range")) or (float(xs.min()), max(float(xs.max()), float(xs.min()) + 1.0))
    units = inner_result.meta.get(f"{quantity} Units", "")
    y_title = f"{quantity} ({units})" if units and units != "dimensionless" else quantity
    meta = dict(inner_result.meta)
    meta.update({
        "X Min": x_low, "X Max": x_high, "Y Min": y_low, "Y Max": y_high, "Cursor X": x,
        "X Title": x_title, "Y Title": f"log {y_title}" if log else y_title,
    })
    return ProbeResult(values=pixels, meta=meta, scalars=dict(inner_result.scalars))
