"""The bundled annotation library: the .blend the add-on appends annotation
node groups and materials from, and the index of the groups the Add yt
Annotation menu offers.

A node-group asset tagged "Annotation Scene" or "3D Scene" is a menu entry
for that scene; one with neither tag is a helper other groups use. This
module imports nothing from the add-on, so scripts/annotation_assets.py can
load it under nodebpy's Blender too.
"""

import functools
import json
import os

import bpy

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
BLEND = os.path.join(ASSETS_DIR, "annotations.blend")
INDEX = os.path.join(ASSETS_DIR, "annotations.json")
ANNOTATION_SCENE, MAIN_SCENE = "ANNOTATION", "MAIN"
SCENE_TAGS = {"Annotation Scene": ANNOTATION_SCENE, "3D Scene": MAIN_SCENE}


def scene_of(tree):
    """ANNOTATION_SCENE or MAIN_SCENE from `tree`'s asset tags; None for a helper."""
    tags = {tag.name for tag in tree.asset_data.tags} if tree.asset_data is not None else set()
    scenes = {SCENE_TAGS[tag] for tag in tags & SCENE_TAGS.keys()}
    if len(scenes) > 1:
        raise ValueError(f"{tree.name!r} is tagged for both scenes; give it one of {sorted(SCENE_TAGS)}")
    return scenes.pop() if scenes else None


def build_index(blend=BLEND):
    """Menu entries of `blend`'s node-group assets, sorted by name: dicts of
    name, scene, description and inputs ({socket name: socket type}). Appends
    the assets into the current session, so run it in a throwaway one.
    """
    with bpy.data.libraries.load(blend, link=False, assets_only=True) as (src, dst):
        dst.node_groups = list(src.node_groups)
    entries = []
    for tree in dst.node_groups:
        scene = scene_of(tree)
        if scene is not None:
            entries.append({
                "name": tree.name,
                "scene": scene,
                "description": tree.asset_data.description or tree.description,
                "inputs": {item.name: item.socket_type for item in tree.interface.items_tree
                           if item.item_type == "SOCKET" and item.in_out == "INPUT"},
            })
    return sorted(entries, key=lambda entry: entry["name"])


def write_index(blend=BLEND, path=INDEX):
    entries = build_index(blend)
    with open(path, "w") as f:
        json.dump(entries, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return entries


@functools.cache
def read_index(path=INDEX):
    """{name: entry} of the menu's annotations."""
    with open(path) as f:
        return {entry["name"]: entry for entry in json.load(f)}
