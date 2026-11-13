# Node-group asset "yt Annotation" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
import bpy
from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import CustomGeometryGroup

from .yt_annotation_view import YtAnnotationView


class YtAnnotation2(CustomGeometryGroup):
    _name = "yt Annotation"

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        _geometry = tree.inputs.geometry("Geometry")
        units_object = tree.inputs.object(
            "Units Object",
            description="The volume's units object (Units for Geometry Nodes)",
        )
        unit = tree.inputs.string(
            "Unit",
            "unitary",
            description="A unit listed in the volume's Data Units, or unitary",
        )
        size = tree.inputs.float(
            "Size", 1.0, description="Diameter of the dot, in Unit"
        )
        color = tree.inputs.color(
            "Color", (1.0, 1.0, 1.0, 1.0), description="Scene-linear color"
        )
        geometry_1 = tree.outputs.geometry("Geometry")

        yt_annotation_view = YtAnnotationView(
            **{"Units Object": units_object}, Unit=unit
        )
        transform_geometry = g.TransformGeometry(
            geometry=g.MeshCircle.n_gon(
                48, size * yt_annotation_view.o.frame_per_unit * 0.5
            ),
            translation=yt_annotation_view.o.screen_position,
        )
        (
            yt_annotation_view.o.in_front.switch.geometry(true=transform_geometry)
            >> g.StoreNamedAttribute.point.color(
                name="yt_annotation_color", value=color
            )
            >> g.SetMaterial(material=bpy.data.materials.get("yt Annotation"))
            >> geometry_1
        )


ASSET = YtAnnotation2

DATABLOCK_DEPENDENCIES = {
    "materials": ("yt Annotation",),
}
