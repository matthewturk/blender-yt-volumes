# Node-group asset "yt Time Label" (GeometryNodeTree).
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
import bpy
from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import CustomGeometryGroup


class YtTimeLabel(CustomGeometryGroup):
    _name = "yt Time Label"

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        _geometry = tree.inputs.geometry("Geometry")
        units_object = tree.inputs.object(
            "Units Object", description="The volume's units object (Units for Geometry Nodes)"
        )
        redshift = tree.inputs.boolean("Redshift", False, description="Show the redshift instead of the time")
        time_unit = tree.inputs.string("Time Unit", "Myr", description="s, yr, kyr, Myr or Gyr")
        decimals = tree.inputs.integer("Decimals", 1, description="Decimal places")
        prefix = tree.inputs.string("Prefix", "", description="Text before the value; empty for 't = ' or 'z = '")
        label_size = tree.inputs.float("Label Size", 0.025, description="Height of the label, in frame widths")
        text_color = tree.inputs.color("Text Color", (1.0, 1.0, 1.0, 1.0), description="Color of the label")
        geometry_1 = tree.outputs.geometry("Geometry")

        name = g.Switch.string(redshift, g.JoinStrings((g.String("time_in_"), time_unit)), g.String("redshift"))
        attribute = g.NamedAttribute.float(name)
        geometry = g.ObjectInfo(units_object).o.geometry
        value = g.SampleIndex.point.float(geometry, attribute.o.attribute)
        known = g.SampleIndex.point.boolean(geometry, attribute.o.exists)
        auto = g.Switch.string(redshift, g.String("t = "), g.String("z = "))
        lead = g.Switch.string(g.Math.greater_than(prefix.length(), 0), auto, prefix)
        unit = g.Switch.string(redshift, g.JoinStrings((g.String(" "), time_unit)), g.String(""))
        text = g.JoinStrings((lead, value.o.value.to_string(decimals), unit))
        (
            g.Switch.geometry(
                known,
                true=g.StringToCurves(string=text, size=label_size, align_x="Left", align_y="Bottom")
                >> g.RealizeInstances()
                >> g.FillCurve(),
            )
            >> g.StoreNamedAttribute.point.color(name="yt_annotation_color", value=text_color)
            >> g.SetMaterial(material=bpy.data.materials.get("yt Annotation"))
            >> geometry_1
        )


ASSET = YtTimeLabel

ASSET_METADATA = {
    "description": "The dataset's time or redshift, from the volume's units object",
    "tags": ("Annotation Scene",),
}

DATABLOCK_DEPENDENCIES = {
    "fonts": ("Bfont Regular",),
    "materials": ("yt Annotation",),
}
