"""Export a dataset's grid structure for the Deep Zoom extension.

    python grid_structure.py DATASET OUT.npz
    python grid_structure.py --loader file.py:func OUT.npz

Run it with any Python that has yt. It writes float64 `left`/`right` edges in
unitary coordinates and integer `level`s, one row per grid, so the authoring
Blender can draw the hierarchy without yt; `load` needs only numpy.
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
    args = parser.parse_args(argv)
    if bool(args.dataset) == bool(args.loader):
        parser.error("give exactly one of DATASET or --loader")
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
