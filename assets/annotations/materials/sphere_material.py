# Material "Sphere Material", dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# The class is a recipe for the material's shader tree — build recreates the material and runs it into material.node_tree.
from bpy.types import ShaderNodeTree
from nodebpy import TreeBuilder
from nodebpy import shader as s
from nodebpy.builder import CustomShaderGroup


class SphereMaterial(CustomShaderGroup):
    _name = "Shader Nodetree"

    def _build_group(self, tree: TreeBuilder[ShaderNodeTree]) -> None:
        principled_bsdf = s.PrincipledBSDF(
            emission_color=(0.07015842, 0.07015842, 0.07015842, 1.0),
            emission_strength=2.0,
        )
        _material_output = s.MaterialOutput(principled_bsdf)


MATERIAL = SphereMaterial
MATERIAL_NAME = "Sphere Material"

MATERIAL_PROPERTIES = {
    "roughness": 0.5,
}
