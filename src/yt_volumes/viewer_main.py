"""Run the yt_idv viewer (`viewer.py`) in its own process:

    python viewer_main.py config.json

The package's __init__ imports bpy, which a plain Python either lacks or finds
as an unrelated pip bpy, so this loads `viewer` without it.
"""

import importlib
import os
import sys
import types


def load_viewer():
    """The `viewer` module, its package registered without running __init__."""
    package_dir = os.path.dirname(os.path.abspath(__file__))
    # Running a script puts its directory first on sys.path; the package's
    # modules would then also import as top-level names.
    if sys.path and os.path.abspath(sys.path[0] or os.curdir) == package_dir:
        sys.path.pop(0)
    name = os.path.basename(package_dir)
    package = types.ModuleType(name)
    package.__path__ = [package_dir]
    sys.modules[name] = package
    return importlib.import_module(f"{name}.viewer")


if __name__ == "__main__":
    load_viewer().main(sys.argv[1])
