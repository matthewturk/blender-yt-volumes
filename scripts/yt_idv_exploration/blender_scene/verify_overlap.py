"""Checks that the depth clip actually occludes the yt_idv volume behind
render_real_demo.py's sphere.

Renders `render_real_demo.build_scene()` through the two-pass pipeline twice
per renderer mode, with the depth clip on and forced off, and compares the
volume layer's alpha inside vs. outside the sphere's silhouette. The clip
should reduce alpha only inside the silhouette. MIP (`max_intensity`) may
show little difference, since the brightest sample can lie in front of the
sphere; `projection` should always show a drop.

Run with:

    env -u LD_PRELOAD uv run --group ytidv python \\
        scripts/yt_idv_exploration/blender_scene/verify_overlap.py

Output: per-mode alpha numbers, and `images/29_verify_overlap_diff_<mode>.png`
(red = alpha removed by the clip, faint green = sphere silhouette).
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

import render_real_demo  # noqa: E402

RENDERER_MODES = ["projection", "max_intensity"]
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
    """Run the two-pass pipeline once; return (beauty_rgba, volume_rgba).

    Read back immediately, since the next call overwrites the same EXR paths.
    """
    real_set_external_depth_clip = yt_idv_renderer.set_external_depth_clip
    if disable_depth_clip:
        def _forced_off(scene, depth=None):
            # depth=None disables clipping and clears clip state left on the
            # cached yt_idv context by a previous call.
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

    # Per-pixel alpha removed by the depth clip.
    diff = np.clip(unclipped_alpha - clipped_alpha, 0.0, 1.0)
    heat = np.zeros((*diff.shape, 4), dtype=np.float32)
    heat[..., 0] = diff  # red = alpha the clip removed
    heat[..., 1] = mask.astype(np.float32) * 0.15  # faint green = sphere silhouette
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

        # Pass 1 (Cycles) doesn't use the depth clip, so it must match.
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
