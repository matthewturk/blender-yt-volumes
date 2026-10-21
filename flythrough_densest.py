"""
Zoom from a saved scene's camera toward its densest cell, stop a fraction of the
box away from it, then orbit it slowly about the z axis with the horizon level.

    python flythrough_densest.py SCENE.zip [--out DIR] [--distance 0.2] ...

The scene is scaled first (BlockCollection.scale, PR 276), so the data source's
bounding box spans [0, 1] and distances are fractions of the box. Frames are
written as PNGs and, with --encode, as an MP4.

This needs yt_idv with the perspective-matrix fix (yt's
get_perspective_matrix squashes non-square frames vertically) and with scenes
that save their bounding box; for a scene saved without one, pass --bbox.
"""

import argparse
import contextlib
import io
import os
import subprocess
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import yt  # noqa: E402
from OpenGL import GL  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

import yt_idv  # noqa: E402
import yt_idv.scene_graph  # noqa: E402

Z_AXIS = np.array([0.0, 0.0, 1.0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("scene", help="scene file written by SceneGraph.save")
    p.add_argument("--out", default="flythrough_frames", help="frame directory")
    p.add_argument(
        "--distance",
        type=float,
        default=0.2,
        help="how far from the densest cell the zoom stops, in box widths",
    )
    p.add_argument(
        "--orbit-distance",
        type=float,
        help="keep zooming during the orbit, ending this far away (box widths); "
        "by default the orbit stays at --distance",
    )
    p.add_argument(
        "--deep-distance",
        type=float,
        help="after --distance, keep zooming at a constant logarithmic rate "
        "(--deep-frames-per-decade) to this distance, then orbit there; "
        "replaces --orbit-distance",
    )
    p.add_argument(
        "--deep-frames-per-decade",
        type=float,
        default=692.0,
        help="frames per factor of ten in distance during the deep zoom",
    )
    p.add_argument(
        "--deep-ease-frames",
        type=int,
        default=90,
        help="frames over which the deep zoom slows to a stop",
    )
    p.add_argument(
        "--near-fraction",
        type=float,
        help="set the near clipping plane to this fraction of the camera's "
        "distance from the target, every frame (needed for very close zooms)",
    )
    p.add_argument(
        "--near-min",
        type=float,
        default=1e-6,
        help="smallest near plane with --near-fraction; the camera's float32 "
        "matrices turn singular below about 1e-8",
    )
    p.add_argument(
        "--recenter-factor",
        type=float,
        help="each time the camera gets this many times closer (once it is "
        "within one box width), rescale the scene about the target with "
        "SceneGraph.rescale, keeping coordinates near the camera of order one "
        "for float32 precision; the near plane then follows --near-fraction "
        "in the rescaled coordinates",
    )
    p.add_argument("--zoom-frames", type=int, default=400)
    p.add_argument("--orbit-frames", type=int, default=900)
    p.add_argument(
        "--orbit-degrees", type=float, default=360.0, help="how far to orbit"
    )
    p.add_argument(
        "--orbit-ease-frames",
        type=int,
        default=90,
        help="frames over which the orbit's rotation speeds up from rest",
    )
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=60, help="frame rate of the MP4")
    p.add_argument(
        "--bbox",
        type=float,
        nargs=6,
        metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
        help="the data source's bounding box in unitary units, for a scene "
        "saved without one",
    )
    p.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="orbit this point (in the scaled box) instead of the densest cell",
    )
    p.add_argument(
        "--render-method",
        help="render with this method (e.g. projection) instead of the scene's",
    )
    p.add_argument(
        "--cmap-per-frame",
        action="store_true",
        help="recompute the colormap bounds from every frame (they flicker); "
        "otherwise the first frame's bounds are used throughout",
    )
    p.add_argument(
        "--cmap-plan",
        action="store_true",
        help="before rendering, sample the path to choose colormap bounds that "
        "vary smoothly over time (no flicker)",
    )
    p.add_argument(
        "--cmap-decades",
        type=float,
        default=6.0,
        help="dynamic range of the planned colormap, in decades",
    )
    p.add_argument(
        "--cmap-percentile",
        type=float,
        default=99.9,
        help="the planned upper bound follows this percentile of each sampled "
        "frame's values",
    )
    p.add_argument(
        "--cmap-sample-every", type=int, default=5, help="sample every Nth frame"
    )
    p.add_argument(
        "--cmap-smooth-frames",
        type=float,
        default=60.0,
        help="width (Gaussian sigma, in frames) of the smoothing over time",
    )
    p.add_argument("--encode", action="store_true", help="encode an MP4 with ffmpeg")
    p.add_argument(
        "--pipe",
        action="store_true",
        help="stream frames straight into ffmpeg (OUT.mp4) instead of writing PNGs",
    )
    p.add_argument(
        "--lossless",
        action="store_true",
        help="also encode a lossless (4:4:4, crf 0) MP4; many players can't open it",
    )
    p.add_argument(
        "--time-render-only",
        action="store_true",
        help="first time a pass over the path without reading back or saving",
    )
    return p.parse_args()


def scale_scene(data, bbox=None):
    """Scale the block collection so its data source's bounding box spans [0, 1]."""
    if bbox is not None:
        data._bbox = np.array(bbox, dtype="f8").reshape(2, 3)
    if data.scale:
        # a scene saved after scaling comes back scaled
        print("the scene is already scaled")
        return
    if data.bbox is None:
        raise SystemExit(
            "This scene was saved without its data source's bounding box; pass "
            "it with --bbox X0 Y0 Z0 X1 Y1 Z1 (unitary units)."
        )
    data.scale = True
    print(
        f"scaled by offset {np.round(data.applied_scale_offset, 5)}, "
        f"ratio {data.applied_scale_ratio:.5g}"
    )


def densest_cell(data):
    """
    The value and center of the densest cell the bitmaps keep (blocks also hold
    cells outside the data source, and cells covered by finer blocks).

    Textures hold vertex-centered values, so a cell's value is the mean of its
    8 corners, and cell k of a block is centered at left_edge + (k + 0.5) * dx.
    """
    left_edges = np.asarray(data.vertex_array["in_left_edge"].data, dtype="f8")
    widths = np.asarray(data.vertex_array["in_dx"].data, dtype="f8")
    best_value, best_center = -np.inf, None
    for vbo_i, tex in data.texture_objects.items():
        v = tex.data
        cells = (
            v[:-1, :-1, :-1]
            + v[1:, :-1, :-1]
            + v[:-1, 1:, :-1]
            + v[:-1, :-1, 1:]
            + v[1:, 1:, :-1]
            + v[1:, :-1, 1:]
            + v[:-1, 1:, 1:]
            + v[1:, 1:, 1:]
        ) / 8.0
        cells = np.where(data.bitmap_objects[vbo_i].data > 0, cells, -np.inf)
        k = np.unravel_index(np.argmax(cells), cells.shape)
        if cells[k] > best_value:
            best_value = cells[k]
            best_center = left_edges[vbo_i, :3] + (np.array(k) + 0.5) * widths[vbo_i]
    return best_value, best_center


def camera_path(
    start,
    target,
    distance,
    n_zoom,
    n_orbit,
    orbit_degrees,
    orbit_distance=None,
    ease_frames=0,
):
    """
    Camera positions: a zoom along the line from start toward target to
    distance away from it, then an orbit about the vertical (z) axis through
    target that ends orbit_distance away (by default, staying at distance).

    The distance follows one smooth, monotonic curve in log space through the
    start, distance at the end of the zoom and orbit_distance at the end, so
    there's no jolt where the zoom hands over to the orbit, and the rotation
    speeds up from rest over ease_frames.
    """
    if orbit_distance is None:
        orbit_distance = distance
    offset = start - target
    d0 = np.linalg.norm(offset)
    if not d0 > distance >= orbit_distance > 0:
        raise SystemExit(
            f"The distances must shrink: the camera starts {d0:.3g} from the "
            f"target, the zoom stops at {distance:.3g} and the orbit ends at "
            f"{orbit_distance:.3g}."
        )
    direction = offset / d0
    # the view direction can't be (nearly) vertical, since z is the up vector
    if abs(np.dot(direction, Z_AXIS)) > np.cos(np.radians(5.0)):
        raise SystemExit(
            "The camera is almost directly above or below the target, so the "
            "orbit about the z axis would be degenerate."
        )
    n = n_zoom + n_orbit
    log_d = PchipInterpolator(
        [0, n_zoom - 1, n - 1], np.log([d0, distance, orbit_distance])
    )(np.arange(n))
    distances = np.exp(log_d)
    # the rotation's speed ramps up from zero, then stays constant
    speed = np.minimum(1.0, (np.arange(n_orbit) + 1) / max(ease_frames, 1))
    angles = np.radians(orbit_degrees) * np.cumsum(speed) / speed.sum()
    positions = [target + direction * distances[i] for i in range(n_zoom)]
    for i, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        positions.append(target + rotation @ (direction * distances[n_zoom + i]))
    return positions


def deep_zoom_path(
    start,
    target,
    distance,
    n_zoom,
    deep_distance,
    frames_per_decade,
    deep_ease_frames,
    n_orbit,
    orbit_degrees,
    ease_frames=0,
):
    """
    Camera positions and distances: a zoom along the line from start toward
    target to distance away in n_zoom frames, then a zoom at a constant
    logarithmic rate (frames_per_decade) to deep_distance that slows to a stop
    over deep_ease_frames, then an orbit at deep_distance about the vertical (z)
    axis through target, speeding up from rest over ease_frames.

    The first zoom is a cubic in log distance that ends at the deep zoom's
    rate, so there's no jolt where one hands over to the other.
    """
    offset = start - target
    d0 = np.linalg.norm(offset)
    if not d0 > distance > deep_distance > 0:
        raise SystemExit(
            f"The distances must shrink: the camera starts {d0:.3g} from the "
            f"target, the zoom reaches {distance:.3g} and the deep zoom "
            f"{deep_distance:.3g}."
        )
    direction = offset / d0
    if abs(np.dot(direction, Z_AXIS)) > np.cos(np.radians(5.0)):
        raise SystemExit(
            "The camera is almost directly above or below the target, so the "
            "orbit about the z axis would be degenerate."
        )
    rate = -np.log(10.0) / frames_per_decade  # change in log distance per frame
    # the first zoom: cubic Hermite in log distance, in units of n_zoom - 1
    # frames, ending at the deep zoom's rate; the starting slope is chosen so
    # the two slopes average to the mean rate, clipped to stay monotonic
    y0, y1 = np.log(d0), np.log(distance)
    span = n_zoom - 1
    m1 = rate * span
    m0 = np.clip(2.0 * (y1 - y0) - m1, 3.0 * (y1 - y0), 0.0)
    u = np.arange(n_zoom) / span
    log_zoom = (
        (2 * u**3 - 3 * u**2 + 1) * y0
        + (u**3 - 2 * u**2 + u) * m0
        + (-2 * u**3 + 3 * u**2) * y1
        + (u**3 - u**2) * m1
    )
    # the deep zoom: constant rate, then slowing to rest
    total = np.log(distance / deep_distance)
    n_deep = int(round(total / -rate + deep_ease_frames / 2.0))
    speed = np.ones(n_deep)
    if deep_ease_frames > 0:
        speed[-deep_ease_frames:] = np.linspace(1.0, 0.0, deep_ease_frames + 2)[1:-1]
    log_deep = y1 - total * np.cumsum(speed) / speed.sum()
    distances = np.exp(np.concatenate([log_zoom, log_deep]))
    positions = [target + direction * d for d in distances]
    # the orbit, at deep_distance
    spin = np.minimum(1.0, (np.arange(n_orbit) + 1) / max(ease_frames, 1))
    angles = np.radians(orbit_degrees) * np.cumsum(spin) / spin.sum()
    end = direction * deep_distance
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        positions.append(target + rotation @ end)
    distances = np.concatenate([distances, np.full(n_orbit, deep_distance)])
    return positions, distances


class Recentering:
    """
    Keeps the scene's coordinates of order one around the target during a deep
    zoom. Once the camera is within one box width, the scene is rescaled about
    the target (SceneGraph.rescale) each time the camera gets factor times
    closer, so the model coordinates are (x - target) / ratio with the camera
    between 1 and factor away. The camera path stays in the original
    coordinates and is mapped into the current ones frame by frame.
    """

    def __init__(self, scene, distances, factor, near_fraction, near_min):
        self.scene = scene
        self.distances = distances
        self.factor = factor
        self.near_fraction = near_fraction
        self.near_min = near_min
        self.offset = np.zeros(3)
        self.ratio = 1.0

    def _wanted(self, k, target):
        d = self.distances[k]
        if d >= 1.0:
            return np.zeros(3), 1.0
        ratio = self.factor ** np.floor(np.log(d) / np.log(self.factor))
        return np.asarray(target, dtype="f8"), ratio

    def place(self, camera, positions, target, k):
        offset, ratio = self._wanted(k, target)
        if ratio != self.ratio or not np.array_equal(offset, self.offset):
            # from model = (x - self.offset) / self.ratio to (x - offset) / ratio
            self.scene.rescale((offset - self.offset) / self.ratio, ratio / self.ratio)
            self.offset, self.ratio = offset, ratio
        near = {}
        if self.near_fraction is not None:
            near_plane = self.near_fraction * self.distances[k] / ratio
            near["near_plane"] = float(max(near_plane, self.near_min))
        camera.update(
            position=(positions[k] - offset) / ratio,
            focus=(np.asarray(target) - offset) / ratio,
            up=Z_AXIS,
            **near,
        )


def place_camera(camera, positions, target, k, near_planes=None):
    if isinstance(near_planes, Recentering):
        near_planes.place(camera, positions, target, k)
        return
    near = {} if near_planes is None else {"near_plane": float(near_planes[k])}
    camera.update(position=positions[k], focus=target, up=Z_AXIS, **near)


def start_ffmpeg(movie, width, height, fps):
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo"]
        + ["-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps)]
        + ["-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "12"]
        + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", movie],
        stdin=subprocess.PIPE,
    )


def read_frame(scene, width, height):
    # SceneGraph.image reads the scene's framebuffer, or the window's when it
    # has none, top row first
    rgba = np.asarray(scene.image)
    if rgba.shape != (height, width, 4):
        raise RuntimeError(
            f"read a {rgba.shape} image for a {width}x{height} frame; this "
            "yt_idv doesn't have the non-square readback fix"
        )
    return (np.clip(rgba[:, :, :3], 0.0, 1.0) * 255).astype("u1")


def plan_cmap_bounds(
    rc, positions, target, every, percentile, decades, smooth, near_planes=None
):
    """
    Colormap bounds for every frame that vary smoothly over time: the upper
    bound follows a percentile of sampled frames' first-pass values, smoothed
    in log space, and the lower bound sits decades below it.
    """
    camera = rc.scene.camera
    component = rc.scene.components[0]
    stored = component.store_first_pass_fb
    component.store_first_pass_fb = True
    n = len(positions)
    samples = list(range(0, n, every))
    if samples[-1] != n - 1:
        samples.append(n - 1)
    sampled, highs = [], []
    for k in samples:
        place_camera(camera, positions, target, k, near_planes)
        rc.scene.render()
        fp = component.first_pass_fb_rgba
        values = fp[..., 0][fp[..., 3] > 0]
        values = values[values > 0]
        if values.size:
            sampled.append(k)
            highs.append(np.percentile(values, percentile))
    component.store_first_pass_fb = stored
    if not sampled:
        raise SystemExit("No sampled frame showed any data to set the colormap from.")
    log_high = np.interp(np.arange(n), sampled, np.log10(highs))
    log_high = gaussian_filter1d(log_high, smooth, mode="nearest")
    high = 10.0**log_high
    low = high / 10.0**decades
    print(
        f"planned colormap from {len(sampled)} sampled frames: upper bound "
        f"{high.min():.3g}-{high.max():.3g}, {decades:g} decades (normalized values)"
    )
    return low, high


def render_path(
    rc,
    positions,
    target,
    out=None,
    width=None,
    height=None,
    cmap_per_frame=False,
    cmap_bounds=None,
    near_planes=None,
    writer=None,
):
    camera = rc.scene.camera
    components = rc.scene.components
    bounds = []
    start = time.perf_counter()
    for k in range(len(positions)):
        place_camera(camera, positions, target, k, near_planes)
        if cmap_bounds is not None:
            low, high = cmap_bounds
            for component in components:
                component.cmap_min = low[k]
                component.cmap_max = high[k]
                component._cmap_bounds_invalid = False
            rc.scene.render()
        elif cmap_per_frame:
            for component in components:
                component._cmap_bounds_invalid = True
            # each recomputation prints the new bounds
            with contextlib.redirect_stdout(io.StringIO()):
                rc.scene.render()
            bounds.append((components[0].cmap_min, components[0].cmap_max))
        else:
            rc.scene.render()
        # Finish every frame. Queuing a long run of frames without waiting
        # (as the render-only pass did) left every later frame black, with no
        # GL error, on an NVIDIA RTX 2060 with big_scene (94,260 blocks).
        GL.glFinish()
        if writer is not None:
            writer.stdin.write(read_frame(rc.scene, width, height).tobytes())
        elif out is not None:
            frame = read_frame(rc.scene, width, height)
            Image.fromarray(frame).save(
                os.path.join(out, f"frame_{k:04d}.png"), compress_level=1
            )
    GL.glFinish()
    elapsed = time.perf_counter() - start
    if bounds:
        lows, highs = np.array(bounds, dtype="f8").T
        print(
            f"  per-frame colormap bounds: min {lows.min():.3g}-{lows.max():.3g}, "
            f"max {highs.min():.3g}-{highs.max():.3g} (normalized values)"
        )
    return elapsed


def encode(out, fps, lossless):
    frames = os.path.join(out, "frame_%04d.png")
    movies = [
        (
            f"{out}.mp4",
            ["-preset", "veryslow", "-crf", "12", "-pix_fmt", "yuv420p"],
        )
    ]
    if lossless:
        movies.append(
            (
                f"{out}_lossless.mp4",
                ["-preset", "veryslow", "-crf", "0", "-pix_fmt", "yuv444p"],
            )
        )
    for movie, options in movies:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps)]
            + ["-i", frames, "-c:v", "libx264"]
            + options
            + ["-movflags", "+faststart", movie],
            check=True,
        )
        print(f"wrote {movie}")


def main():
    args = parse_args()
    yt.set_log_level(40)
    args.out = os.path.normpath(args.out)
    if not args.pipe:
        os.makedirs(args.out, exist_ok=True)

    rc = yt_idv.render_context("egl", width=args.width, height=args.height)
    t = time.perf_counter()
    rc.scene = yt_idv.scene_graph.SceneGraph.load(args.scene)
    print(f"loaded {args.scene} in {time.perf_counter() - t:.1f} s")
    component = rc.scene.components[0]
    data = component.data
    if args.render_method is not None:
        component.render_method = args.render_method
    print(
        f"field range {data.min_val:.4g} to {data.max_val:.4g}; colormap values "
        "are normalized to it"
    )

    scale_scene(data, args.bbox)
    if args.target is not None:
        target = np.array(args.target)
        print(f"orbiting {target}")
    else:
        value, target = densest_cell(data)
        print(f"densest cell (normalized value {value:.4g}) at {np.round(target, 4)}")

    camera = rc.scene.camera
    camera.update(aspect_ratio=args.width / args.height)
    start = np.array(camera.position, dtype="f8")
    if args.deep_distance is not None:
        positions, distances = deep_zoom_path(
            start,
            target,
            args.distance,
            args.zoom_frames,
            args.deep_distance,
            args.deep_frames_per_decade,
            args.deep_ease_frames,
            args.orbit_frames,
            args.orbit_degrees,
            ease_frames=args.orbit_ease_frames,
        )
    else:
        positions = camera_path(
            start,
            target,
            args.distance,
            args.zoom_frames,
            args.orbit_frames,
            args.orbit_degrees,
            orbit_distance=args.orbit_distance,
            ease_frames=args.orbit_ease_frames,
        )
        distances = np.linalg.norm(np.array(positions) - target, axis=1)
    near_planes = None
    if args.recenter_factor is not None:
        near_planes = Recentering(
            rc.scene, distances, args.recenter_factor, args.near_fraction, args.near_min
        )
    elif args.near_fraction is not None:
        near_planes = np.maximum(args.near_fraction * distances, args.near_min)
    n = len(positions)
    print(
        f"{len(data.texture_objects)} blocks, {args.width}x{args.height}, "
        f"render_method={component.render_method}, "
        f"bindless={getattr(component, 'use_bindless_textures', False)}; "
        f"{n} frames ({n - args.orbit_frames} zooming, {args.orbit_frames} orbiting), "
        f"distance {distances[0]:.3g} to {distances[-1]:.3g}"
    )

    # compile the shaders before anything is timed
    rc.scene.render()
    GL.glFinish()
    cmap_bounds = None
    if args.cmap_plan:
        t = time.perf_counter()
        cmap_bounds = plan_cmap_bounds(
            rc,
            positions,
            target,
            args.cmap_sample_every,
            args.cmap_percentile,
            args.cmap_decades,
            args.cmap_smooth_frames,
            near_planes=near_planes,
        )
        print(f"  planned in {time.perf_counter() - t:.1f} s")
    if args.time_render_only:
        elapsed = render_path(
            rc,
            positions,
            target,
            cmap_per_frame=args.cmap_per_frame,
            cmap_bounds=cmap_bounds,
            near_planes=near_planes,
        )
        print(f"render only: {n} frames in {elapsed:.1f} s ({n / elapsed:.1f} fps)")
    writer = None
    if args.pipe:
        writer = start_ffmpeg(f"{args.out}.mp4", args.width, args.height, args.fps)
    elapsed = render_path(
        rc,
        positions,
        target,
        args.out,
        args.width,
        args.height,
        cmap_per_frame=args.cmap_per_frame,
        cmap_bounds=cmap_bounds,
        near_planes=near_planes,
        writer=writer,
    )
    label = "piped to ffmpeg" if args.pipe else "with saving"
    print(f"{label}: {n} frames in {elapsed:.1f} s ({n / elapsed:.1f} fps)")
    if writer is not None:
        writer.stdin.close()
        if writer.wait() != 0:
            raise SystemExit("ffmpeg failed")
        print(f"wrote {args.out}.mp4")
    elif args.encode:
        encode(args.out, args.fps, args.lossless)


if __name__ == "__main__":
    main()
