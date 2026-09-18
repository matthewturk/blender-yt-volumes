"""Verifies -- empirically, not just by describing it -- that the sphere
(real Cycles geometry) and the real yt_idv volume in render_real_demo.py's
scene genuinely co-exist and interact: renders the SAME real end-to-end
scene (`render_real_demo.build_scene()`) through the add-on's actual
two-pass pipeline TWICE per renderer mode -- once with the real external
depth clip wired up (the normal path) and once with it forced off -- and
diffs the two.

Why this is an actual test of "do they overlap", not just "does it look
plausible": the sphere's own Cycles beauty render (pass 1) never touches
the depth clip at all -- it's rendered once, before the volume renderer
even runs -- so it must come out identical between the two runs regardless
(sanity-checked below, not assumed). Only the VOLUME layer's own EXR
(`vo_result_FlaggedVolumeBox_*.exr`, read directly, before compositing) can
differ, and only the depth clip can make it differ. So: if disabling the
clip measurably raises the volume layer's own alpha specifically INSIDE
the sphere's on-screen silhouette (read straight from the beauty pass's own
alpha channel -- the sphere is the only opaque geometry in the scene), and
leaves it alone OUTSIDE that silhouette, that is direct evidence that (a)
the sphere and the volume genuinely occupy the same region of 3D space
along those rays -- there is something there to clip -- and (b) the clip is
doing real, spatially-localized work, not a global no-op or a uniform
effect that would show up everywhere.

Renderer mode matters here. `max_intensity` (MIP) keeps only the SINGLE
BRIGHTEST sample along each ray and discards where along the ray it came
from -- if that peak sample happens to already lie in front of the sphere,
clipping the ray behind it changes nothing about the pixel, so MIP can show
little or no measurable difference even where the geometry genuinely
overlaps. Not a bug -- an inherent property of what MIP even measures (see
yt_idv_renderer.py's own module docstring). `projection` (unweighted,
additive integration across the WHOLE ray) has no such blind spot: cutting
the ray off early necessarily removes some of the accumulated total, so a
measurable difference is expected wherever the geometry truly overlaps.
This script runs BOTH and reports the difference for each, so that
asymmetry is measured here, not just assumed.

Run with:

    env -u LD_PRELOAD uv run --group ytidv python \\
        scripts/yt_idv_exploration/blender_scene/verify_overlap.py

Output:
  - Printed per-mode numbers: mean volume-layer alpha inside vs. outside
    the sphere's silhouette, clipped vs. unclipped, and the resulting drop.
  - `images/29_verify_overlap_diff_<mode>.png` per mode -- red channel is
    how much alpha the depth clip removed at each pixel (should light up
    inside the sphere's silhouette, traced in faint green for registration,
    and stay dark everywhere else).
"""

import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
ADDON_SRC_PARENT = os.path.join(REPO_ROOT, "src")
IMAGES_DIR = os.path.normpath(os.path.join(HERE, "..", "images"))
WORK_DIR = os.path.join(HERE, "_verify_overlap_work")

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
sys.path.insert(0, ADDON_SRC_PARENT)
sys.path.insert(0, HERE)  # sibling module below

from cycles_volume_override import operators, properties, yt_idv_renderer  # noqa: E402

import render_real_demo  # noqa: E402 -- reuses build_scene()/_write_png()

RENDERER_MODES = ["projection", "max_intensity"]  # per the brief: MIP may show ~no difference
MODE_KEY_BY_NAME = render_real_demo.MODE_KEY_BY_NAME


class _StubPrefs:
    work_dir = WORK_DIR
    verbose_logging = False
    renderer_mode = None  # set per call, right before render_current_frame


def _load_exr_rgba(path):
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size[0], img.size[1]
        data = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)
    finally:
        bpy.data.images.remove(img)
    return data


def _render_once(renderer_mode, disable_depth_clip):
    """Runs the real two-pass pipeline once and returns (beauty_rgba,
    volume_rgba) as in-memory arrays -- read back immediately, before the
    NEXT call overwrites the same on-disk EXR paths (compositor_build.py
    writes fixed per-frame filenames, reused across calls in this loop).
    """
    real_set_external_depth_clip = yt_idv_renderer.set_external_depth_clip
    if disable_depth_clip:
        def _forced_off(scene, depth=None):
            # Real, documented contract (see set_external_depth_clip's own
            # docstring): depth=None disables clipping and restores the
            # unclipped baseline. Forcing it here, rather than skipping the
            # call outright, also correctly clears any clip state left over
            # from a PREVIOUS call against this same cached yt_idv context
            # (see yt_idv_renderer._get_context's own module-level caching).
            return real_set_external_depth_clip(scene, None)
        yt_idv_renderer.set_external_depth_clip = _forced_off
    try:
        _StubPrefs.renderer_mode = MODE_KEY_BY_NAME[renderer_mode]
        operators.render_current_frame(bpy.context)
    finally:
        yt_idv_renderer.set_external_depth_clip = real_set_external_depth_clip

    beauty = _load_exr_rgba(os.path.join(WORK_DIR, "vo_beauty_0001.exr"))
    volume = _load_exr_rgba(os.path.join(WORK_DIR, "vo_result_FlaggedVolumeBox_0001.exr"))
    return beauty, volume


def _safe_mean(arr):
    return float(arr.mean()) if arr.size else float("nan")


def _report_and_visualize(renderer_mode, mask, clipped_vol, unclipped_vol):
    clipped_alpha = clipped_vol[..., 3]
    unclipped_alpha = unclipped_vol[..., 3]

    in_clipped, in_unclipped = clipped_alpha[mask], unclipped_alpha[mask]
    out_clipped, out_unclipped = clipped_alpha[~mask], unclipped_alpha[~mask]

    in_drop = _safe_mean(in_unclipped) - _safe_mean(in_clipped)
    in_pct = 100.0 * in_drop / _safe_mean(in_unclipped) if _safe_mean(in_unclipped) > 1e-6 else float("nan")
    out_drop = _safe_mean(out_unclipped) - _safe_mean(out_clipped)

    print(f"[verify_overlap] mode={renderer_mode!r}")
    print(f"  sphere-silhouette pixels: {int(mask.sum())} of {mask.size}")
    print(f"  mean volume alpha INSIDE sphere silhouette:  unclipped={_safe_mean(in_unclipped):.4f} "
          f"clipped={_safe_mean(in_clipped):.4f}  drop={in_drop:.4f} ({in_pct:.1f}%)")
    print(f"  mean volume alpha OUTSIDE sphere silhouette: unclipped={_safe_mean(out_unclipped):.4f} "
          f"clipped={_safe_mean(out_clipped):.4f}  drop={out_drop:.4f}")

    # Visual diff: how much alpha the depth clip removed, per pixel -- for
    # a genuinely working, spatially-localized clip, this should light up
    # (mostly) inside the sphere's silhouette and stay dark elsewhere.
    diff = np.clip(unclipped_alpha - clipped_alpha, 0.0, 1.0)
    heat = np.zeros((*diff.shape, 4), dtype=np.float32)
    heat[..., 0] = diff  # red = alpha the clip removed
    heat[..., 1] = mask.astype(np.float32) * 0.15  # faint green outline = sphere silhouette, for registration
    heat[..., 3] = 1.0
    heat_path = os.path.join(IMAGES_DIR, f"29_verify_overlap_diff_{renderer_mode}.png")
    render_real_demo._write_png(heat_path, heat)
    print(f"  wrote {heat_path}")

    return {
        "mode": renderer_mode,
        "in_mask_unclipped_alpha": _safe_mean(in_unclipped),
        "in_mask_clipped_alpha": _safe_mean(in_clipped),
        "in_mask_drop_pct": in_pct,
        "out_mask_unclipped_alpha": _safe_mean(out_unclipped),
        "out_mask_clipped_alpha": _safe_mean(out_clipped),
    }


def main():
    properties.register()
    operators._get_prefs = lambda context: _StubPrefs()

    box, sphere, cam = render_real_demo.build_scene()
    print(f"[verify_overlap] scene built: box={box.name} sphere={sphere.name} cam={cam.name}")

    results = []
    for mode in RENDERER_MODES:
        print(f"[verify_overlap] === mode={mode} ===")
        beauty_unclipped, unclipped_vol = _render_once(mode, disable_depth_clip=True)
        beauty_clipped, clipped_vol = _render_once(mode, disable_depth_clip=False)

        # Sanity check this script's own premise: pass 1's Cycles beauty
        # render never touches the depth clip, so it MUST be the same
        # either way -- if it isn't, something is wrong with this script's
        # comparison, not with the add-on.
        max_beauty_diff = float(np.abs(beauty_unclipped - beauty_clipped).max())
        print(f"  max |beauty_unclipped - beauty_clipped| = {max_beauty_diff:.6f} (expect ~0 -- sanity check)")
        assert max_beauty_diff < 1e-4, (
            "Cycles beauty pass differed between the clipped/unclipped runs -- "
            "the comparison below is not isolating the depth clip's effect"
        )

        mask = beauty_clipped[..., 3] > 0.5  # sphere is the only opaque geometry
        results.append(_report_and_visualize(mode, mask, clipped_vol, unclipped_vol))

    print("[verify_overlap] summary (inside-sphere volume-alpha drop when the depth clip is enabled):")
    for r in results:
        print(f"  {r['mode']:16s} {r['in_mask_drop_pct']:6.1f}%  "
              f"(outside sphere: {r['out_mask_unclipped_alpha']:.4f} -> {r['out_mask_clipped_alpha']:.4f}, "
              "should stay ~unchanged)")


if __name__ == "__main__":
    main()
