# Node-group asset "yt Probe Label" (GeometryNodeTree).
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# Hand-written: re-exporting from a .blend inlines format_value from yt_image_panel.
import bpy
from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import CustomGeometryGroup

from .yt_image_panel import format_value


class YtProbeLabel(CustomGeometryGroup):
    _name = "yt Probe Label"

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        _geometry = tree.inputs.geometry("Geometry")
        _probe = tree.inputs.object("Probe", description="The probe whose value this shows")
        quantity = tree.inputs.string(
            "Quantity", "temperature", description="Which of the probe's values to show, e.g. temperature"
        )
        value = tree.inputs.float("Value", 0.0, description="The value; the probe sets it when it runs")
        units = tree.inputs.string("Units", "", description="The value's units; the probe sets them")
        label = tree.inputs.string("Label", "", description="Text before the value; empty for the Quantity")
        decimals = tree.inputs.integer("Decimals", 2, description="Decimal places")
        label_size = tree.inputs.float("Label Size", 0.02, description="Height of the label, in frame widths")
        text_color = tree.inputs.color("Text Color", (1.0, 1.0, 1.0, 1.0), description="Color of the label")
        geometry_1 = tree.outputs.geometry("Geometry")

        lead = g.Switch.string(g.Math.greater_than(label.length(), 0), quantity, label)
        suffix = g.Switch.string(
            g.Math.greater_than(units.length(), 0), g.String(""), g.JoinStrings((g.String(" "), units))
        )
        text = g.JoinStrings((lead, g.String(" = "), format_value(value, decimals), suffix))
        (
            g.StringToCurves(string=text, size=label_size, align_x="Left", align_y="Bottom")
            >> g.RealizeInstances()
            >> g.FillCurve()
            >> g.StoreNamedAttribute.point.color(name="yt_annotation_color", value=text_color)
            >> g.SetMaterial(material=bpy.data.materials.get("yt Annotation"))
            >> geometry_1
        )


ASSET = YtProbeLabel

ASSET_METADATA = {
    "description": "One of a probe's values as text, e.g. 'temperature = 1.2e4 K'",
    "tags": ("Annotation Scene",),
}

DATABLOCK_DEPENDENCIES = {
    "fonts": ("Bfont Regular",),
    "materials": ("yt Annotation",),
}
