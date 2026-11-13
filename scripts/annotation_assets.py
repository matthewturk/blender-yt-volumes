"""Helpers for scripts/export_annotations.sh, run under the "assets" uv group:

    python scripts/annotation_assets.py names <working.blend>
        print the file's node-group and material asset names, one per line,
        and on stderr which scene each node group is for
    python scripts/annotation_assets.py index
        write src/yt_volumes/assets/annotations.json from annotations.blend
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


def names(blend):
    with bpy.data.libraries.load(blend, link=False, assets_only=True) as (src, dst):
        dst.node_groups, dst.materials = list(src.node_groups), list(src.materials)
    for tree in dst.node_groups:
        scene = library.scene_of(tree)  # raises for a group tagged for both scenes
        where = {library.ANNOTATION_SCENE: "annotation scene", library.MAIN_SCENE: "3D scene"}.get(scene)
        note = f"menu, {where}" if where else 'helper, not in the menu (tag "Annotation Scene" or "3D Scene" to list it)'
        print(f"  {tree.name}: {note}", file=sys.stderr)
        print(tree.name)
    for material in dst.materials:
        print(f"  {material.name}: material", file=sys.stderr)
        print(material.name)
    if not dst.node_groups and not dst.materials:
        sys.exit(f"{blend} has no assets; mark the groups and materials to export as assets")


def index():
    for entry in library.write_index():
        print(f"  menu: {entry['name']} ({entry['scene'].lower()})", file=sys.stderr)


if __name__ == "__main__":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    command, *args = sys.argv[1:]
    {"names": names, "index": index}[command](*args)
