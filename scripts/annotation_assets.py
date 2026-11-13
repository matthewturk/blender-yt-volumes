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


def _report(trees):
    """Say what each group is for; warn of groups nobody could add."""
    used = _used(trees)
    for tree in trees:
        scene = library.scene_of(tree)  # raises for a group tagged for both scenes
        if scene is not None:
            where = "annotation scene" if scene == library.ANNOTATION_SCENE else "3D scene"
            print(f"  {tree.name}: menu, {where}", file=sys.stderr)
        elif tree.name in used:
            print(f"  {tree.name}: helper, used by other groups", file=sys.stderr)
        else:
            print(f'warning: {tree.name!r} has no "Annotation Scene" or "3D Scene" tag and no '
                  "other group uses it, so nobody could add it", file=sys.stderr)


def names(blend):
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.node_groups, dst.materials = list(src.node_groups), list(src.materials)
    trees = [t for t in dst.node_groups if t.asset_data is not None]
    materials = [m for m in dst.materials if m.asset_data is not None]
    _report(trees)
    used = _used(dst.node_groups)
    for tree in dst.node_groups:
        if tree.asset_data is None and tree.name not in used and tree.bl_idname == "GeometryNodeTree":
            print(f"warning: {tree.name!r} isn't marked as an asset, so it won't be exported", file=sys.stderr)
    for material in materials:
        print(f"  {material.name}: material", file=sys.stderr)
    if not trees and not materials:
        sys.exit(f"{blend} has no assets; mark the groups and materials to export as assets")
    for asset in trees + materials:
        print(asset.name)


def index():
    library.write_index()
    _report(list(bpy.data.node_groups))  # write_index appended the library's groups


if __name__ == "__main__":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    command, *args = sys.argv[1:]
    {"names": names, "index": index}[command](*args)
