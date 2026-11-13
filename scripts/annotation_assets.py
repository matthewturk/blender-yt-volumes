"""Helpers for scripts/export_annotations.sh, run under the "assets" uv group:

    python scripts/annotation_assets.py names <working.blend>
        print the file's node-group and material asset names, one per line;
        on stderr, which scene each node group is for, and warnings for
        groups nobody could add and groups not marked as assets
    python scripts/annotation_assets.py index
        write src/yt_volumes/assets/annotations.json from annotations.blend,
        warning of its groups nobody could add
"""

import importlib.util
import os
import sys

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "annotation_library", os.path.join(ROOT, "src", "yt_volumes", "annotation_library.py"))
library = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(library)


def _used(trees):
    """Names of the node groups any of `trees` uses as a group node."""
    return {node.node_tree.name for tree in trees for node in tree.nodes
            if node.bl_idname == "GeometryNodeGroup" and node.node_tree is not None}


def _report(trees, materials=(), source=None):
    """List what each group is for, then warn of groups nobody could add."""
    used = _used(trees)
    found, warnings = [], []
    for tree in trees:
        scene = library.scene_of(tree)  # raises for a group tagged for both scenes
        if scene is not None:
            where = "annotation scene" if scene == library.ANNOTATION_SCENE else "3D scene"
            found.append((tree.name, f"menu, {where}"))
        elif tree.name in used:
            found.append((tree.name, "helper, used by other groups"))
        else:
            found.append((tree.name, "no scene tag"))
            warnings.append(f'{tree.name!r} has no "Annotation Scene" or "3D Scene" tag and no '
                            "other group uses it, so nobody could add it")
    found += [(material.name, "material") for material in materials]
    print(f"Found {len(found)} asset{'s' * (len(found) != 1)}" + (f" in {source}:" if source else ":"), file=sys.stderr)
    width = max((len(name) for name, _ in found), default=0)
    for name, what in found:
        print(f"  {name:<{width}}  {what}", file=sys.stderr)
    return warnings


def names(blend):
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.node_groups, dst.materials = list(src.node_groups), list(src.materials)
    trees = [t for t in dst.node_groups if t.asset_data is not None]
    materials = [m for m in dst.materials if m.asset_data is not None]
    warnings = _report(trees, materials, os.path.basename(blend))
    used = _used(dst.node_groups)
    warnings += [f"{t.name!r} isn't marked as an asset, so it won't be exported" for t in dst.node_groups
                 if t.asset_data is None and t.name not in used and t.bl_idname == "GeometryNodeTree"]
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not trees and not materials:
        sys.exit(f"{blend} has no assets; mark the groups and materials to export as assets")
    for asset in trees + materials:
        print(asset.name)


def index():
    library.write_index()
    # write_index appended the library's groups
    for warning in _report(list(bpy.data.node_groups), source="the bundled library"):
        print(f"warning: {warning}", file=sys.stderr)


if __name__ == "__main__":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    command, *args = sys.argv[1:]
    {"names": names, "index": index}[command](*args)
