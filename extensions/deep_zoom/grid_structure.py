"""Export a dataset's grid structure for the Deep Zoom extension.

    python grid_structure.py DATASET OUT.npz
    python grid_structure.py --loader file.py:func OUT.npz
    python grid_structure.py --scene SCENE.zip OUT.npz

It writes float64 `left`/`right` edges in unitary coordinates and integer
`level`s, one row per box, so the authoring Blender can draw the hierarchy
without yt. A dataset needs yt; a yt_idv scene saved with SceneGraph.save
needs only numpy, and gives its blocks (kd-tree bricks) with levels from
their cell sizes. `load` needs only numpy.
"""

import argparse
import sys

import numpy as np


def grid_structure(ds):
    """`{"left", "right", "level"}` arrays for a grid-patch dataset."""
    index = ds.index
    return {
        "left": index.grid_left_edge.to("unitary").d.astype(np.float64),
        "right": index.grid_right_edge.to("unitary").d.astype(np.float64),
        "level": np.asarray(index.grid_levels, dtype=np.int32).ravel(),
    }


def scene_structure(path):
    """`{"left", "right", "level"}` for the blocks of a saved yt_idv scene;
    levels are log2 of the coarsest cell size over each block's.
    """
    import io
    import json
    import zipfile

    with zipfile.ZipFile(path) as archive:
        doc = json.loads(archive.read("scene.json"))
        collections = [d["attributes"] for d in doc["data_objects"] if "_unscaled_edges" in d.get("attributes", {})]
        if not collections:
            raise ValueError(f"{path} has no block collection saved with its float64 edges")
        edges = [np.load(io.BytesIO(archive.read(a["_unscaled_edges"]["__array__"]))) for a in collections]
    left, right, dx = (np.concatenate([e[i] for e in edges]) for i in range(3))
    cell = dx.max(axis=1)
    level = np.round(np.log2(cell.max() / cell)).astype(np.int32)
    return {"left": left, "right": right, "level": level}


def save(ds, path):
    np.savez(path, **grid_structure(ds))


def load(path):
    with np.load(path) as f:
        return {key: f[key] for key in ("left", "right", "level")}


def _resolve_loader(spec):
    """`"module:function"` or `"path/to/file.py:function"` as a callable."""
    import importlib
    import importlib.util
    import os

    target, _, name = spec.rpartition(":")
    if target.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(target))[0], target)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", nargs="?", help="anything yt.load accepts")
    parser.add_argument("out")
    parser.add_argument("--loader", help='"module:function" or "file.py:function" returning a dataset')
    parser.add_argument("--scene", help="a yt_idv scene saved with SceneGraph.save (needs only numpy)")
    args = parser.parse_args(argv)
    if sum(bool(v) for v in (args.dataset, args.loader, args.scene)) != 1:
        parser.error("give exactly one of DATASET, --loader or --scene")
    if args.scene:
        structure = scene_structure(args.scene)
        np.savez(args.out, **structure)
        print(f"wrote {args.out}: {len(structure['level'])} blocks, levels 0-{structure['level'].max()}")
        return 0
    if args.loader:
        ds = _resolve_loader(args.loader)()
    else:
        import yt

        ds = yt.load(args.dataset)
    save(ds, args.out)
    print(f"wrote {args.out}: {ds.index.num_grids} grids, levels 0-{ds.index.max_level}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
