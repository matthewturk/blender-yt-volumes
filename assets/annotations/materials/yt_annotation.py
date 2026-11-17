# Material "yt Annotation", dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# The class is a recipe for the material's shader tree — build recreates the material and runs it into material.node_tree.
from bpy.types import ShaderNodeTree
from nodebpy import TreeBuilder
from nodebpy import shader as s
from nodebpy.builder import CustomShaderGroup


class YtAnnotation(CustomShaderGroup):
    _name = "Shader Nodetree"

    def _build_group(self, tree: TreeBuilder[ShaderNodeTree]) -> None:
        _material_output = s.MaterialOutput(
            s.Emission(s.Attribute(attribute_name="yt_annotation_color").o.color)
        )


MATERIAL = YtAnnotation
MATERIAL_NAME = "yt Annotation"

MATERIAL_ASSET_METADATA = {}
