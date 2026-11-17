# Node-group asset "yt Field of View" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
import bpy
from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import CustomGeometryGroup

from .yt_annotation_view import YtAnnotationView


class YtFieldOfView(CustomGeometryGroup):
    _name = "yt Field of View"

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
        length = tree.inputs.float(
            "Length",
            0.0,
            description="Length in Unit, which can be keyframed; 0 picks a round length up to Auto Max",
        )
        auto_max = tree.inputs.float(
            "Auto Max",
            0.2,
            description="With Length 0: the largest 1, 2 or 5 x 10^n at most this many frame widths",
        )
        line_width = tree.inputs.float(
            "Line Width", 0.004, description="Thickness of the bar, in frame widths"
        )
        label_size = tree.inputs.float(
            "Label Size", 0.02, description="Height of the label, in frame widths"
        )
        text_color = tree.inputs.color(
            "Text Color", (1.0, 1.0, 1.0, 1.0), description="Color of the bar and label"
        )
        geometry_1 = tree.outputs.geometry("Geometry")

        math_1 = g.Math.less_than(length, 0.000000000000000000000000000001)
        yt_annotation_view = YtAnnotationView(
            **{"Units Object": units_object}, Unit=unit
        )
        math_2 = yt_annotation_view.o.frame_width * auto_max
        math_3 = math_2.log(10.0).floor()
        math_4 = 10.0**math_3
        math_5 = math_2 / math_4
        math_6 = (
            1.0
            + g.Math.greater_than(math_5, 1.9999)
            + g.Math.greater_than(math_5, 4.9999).o.value * 3.0
        )
        switch = g.Switch.float(math_1, length, math_6 * math_4)
        math_7 = switch.o.output * yt_annotation_view.o.frame_per_unit
        switch_1 = g.Switch.float(
            math_1,
            (1.0 - switch.o.output.log(10.0).floor()).max(0.0),
            (math_3 * -1.0).max(0.0),
        )
        transform_geometry = g.TransformGeometry(
            geometry=g.Grid(
                size_x=math_7, size_y=line_width, vertices_x=2, vertices_y=2
            ),
            translation=g.CombineXYZ(x=math_7 * 0.5, y=line_width * 0.5),
        )
        transform_geometry_1 = (
            g.StringToCurves(
                string=g.JoinStrings(
                    (switch.o.output.to_string(switch_1), g.String(" "), unit)
                ),
                size=label_size,
                align_x="Center",
                align_y="Top",
            )
            >> g.RealizeInstances()
            >> g.FillCurve()
            >> g.TransformGeometry(
                translation=g.CombineXYZ(x=math_7 * 0.5, y=label_size * -0.3)
            )
        )
        (
            g.Switch.geometry(
                g.Math.greater_than(yt_annotation_view.o.frame_per_unit, 0.0),
                true=g.JoinGeometry((transform_geometry_1, transform_geometry)),
            )
            >> g.StoreNamedAttribute.point.color(
                name="yt_annotation_color", value=text_color
            )
            >> g.SetMaterial(material=bpy.data.materials.get("yt Annotation"))
            >> geometry_1
        )


ASSET = YtFieldOfView

ASSET_METADATA = {
    "description": "A labeled scale bar, sized at the volume's zoom target or middle",
    "tags": ("Annotation Scene",),
}

DATABLOCK_DEPENDENCIES = {
    "fonts": ("Bfont Regular",),
    "materials": ("yt Annotation",),
}
