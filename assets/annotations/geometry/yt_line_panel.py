# Node-group asset "yt Line Panel" (GeometryNodeTree).
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# Hand-written: re-exporting from a .blend inlines the helpers it imports from yt_image_panel.
import bpy
from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import CustomGeometryGroup

from .yt_image_panel import _axis, _callout, _frame_and_titles, callout_inputs


class YtLinePanel(CustomGeometryGroup):
    _name = "yt Line Panel"

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        _geometry = tree.inputs.geometry("Geometry")
        image = tree.inputs.image(
            "Image", description="A one-row float Image: x in R, y in G, alpha 0 for missing points"
        )
        width = tree.inputs.float("Width", 0.3, description="Width of the plot, in frame widths")
        aspect = tree.inputs.float("Aspect", 0.5, description="Height of the plot over its width")
        x_min = tree.inputs.float("X Min", 0.0, description="Axis value at the left edge")
        x_max = tree.inputs.float("X Max", 1.0, description="Axis value at the right edge")
        y_min = tree.inputs.float("Y Min", 0.0, description="Axis value at the bottom edge")
        y_max = tree.inputs.float("Y Max", 1.0, description="Axis value at the top edge")
        x_title = tree.inputs.string("X Title", "", description="Text under the x axis")
        y_title = tree.inputs.string("Y Title", "", description="Text left of the y axis")
        x_ticks = tree.inputs.float("X Ticks Every", 0.0, description="Spacing of the x ticks; 0 picks a round step")
        y_ticks = tree.inputs.float("Y Ticks Every", 0.0, description="Spacing of the y ticks; 0 picks a round step")
        x_decimals = tree.inputs.integer("X Decimals", 0, description="Decimal places of the x tick labels")
        y_decimals = tree.inputs.integer("Y Decimals", 0, description="Decimal places of the y tick labels")
        cursor_x = tree.inputs.float("Cursor X", 0.0, description="Where to draw the cursor, in x axis units")
        show_cursor = tree.inputs.boolean("Cursor", True, description="Draw a vertical line at Cursor X")
        label_size = tree.inputs.float("Label Size", 0.015, description="Height of the labels, in frame widths")
        text_color = tree.inputs.color(
            "Text Color", (1.0, 1.0, 1.0, 1.0), description="Color of the frame, ticks, labels and cursor"
        )
        line_color = tree.inputs.color("Line Color", (0.9847, 0.7997, 0.0182, 1.0), description="Color of the line")
        line_width = tree.inputs.float("Line Width", 0.0015, description="Width of the frame and ticks")
        probe, units_object, show_callout, radius = callout_inputs(tree)
        geometry_1 = tree.outputs.geometry("Geometry")

        height = width * aspect
        x_span = (x_max - x_min).max(1e-30)
        y_span = (y_max - y_min).max(1e-30)
        info = g.ImageInfo(image)
        count = info.o.width
        index = g.Position().o.position.x
        texture = g.ImageTexture(
            image, g.CombineXYZ((index + 0.5) / count.max(1.0), 0.5), interpolation="Closest", extension="EXTEND"
        )
        rgb = g.SeparateColor(texture.o.color)
        placed = g.CombineXYZ((rgb.o.red - x_min) / x_span * width, (rgb.o.green - y_min) / y_span * height)
        points = (
            g.MeshLine(count=count, offset=(1.0, 0.0, 0.0))
            >> g.SetPosition(position=placed)
            >> g.DeleteGeometry.point(selection=g.Math.less_than(texture.o.alpha, 0.5))
        )
        line = (
            points
            >> g.MeshToCurve()
            >> g.CurveToMesh(profile_curve=g.CurveCircle(radius=line_width * 0.75, resolution=4))
        )
        dots = g.RealizeInstances(g.InstanceOnPoints(
            points=points, instance=g.MeshCircle(vertices=12, radius=line_width * 1.8, fill_type="NGON")
        ))
        curve = g.StoreNamedAttribute.point.color(
            g.JoinGeometry((line, dots)), name="yt_annotation_color", value=line_color
        )

        tick = label_size * 0.5
        frame, x_label, y_label = _frame_and_titles(width, height, x_title, y_title, tick, label_size, line_width)
        x_axis = _axis(x_min, x_max, x_ticks, width, tick, label_size, x_decimals, vertical=False)
        y_axis = _axis(y_min, y_max, y_ticks, height, tick, label_size, y_decimals, vertical=True)
        cursor_at = (cursor_x - x_min) / x_span * width
        cursor = g.TransformGeometry(
            geometry=g.Grid(size_x=line_width, size_y=height, vertices_x=2, vertices_y=2),
            translation=g.CombineXYZ(x=cursor_at, y=height * 0.5),
        )
        cursor_shown = g.BooleanMath.l_and(
            show_cursor,
            g.BooleanMath.l_and(g.Math.greater_than(cursor_at, -1e-6), g.Math.less_than(cursor_at, width + 1e-6)),
        )
        text = g.StoreNamedAttribute.point.color(
            g.JoinGeometry((frame, x_axis, y_axis, x_label, y_label, g.Switch.geometry(cursor_shown, true=cursor))),
            name="yt_annotation_color", value=text_color,
        )
        callout = _callout(units_object, probe, radius, width, height, line_width, text_color)
        (
            g.JoinGeometry((text, curve, g.Switch.geometry(show_callout, true=callout)))
            >> g.SetMaterial(material=bpy.data.materials.get("yt Annotation"))
            >> geometry_1
        )


ASSET = YtLinePanel

ASSET_METADATA = {
    "description": "A one-row float Image as a line plot with axes and a cursor, such as a probe's time history",
    "tags": ("Annotation Scene",),
}

DATABLOCK_DEPENDENCIES = {
    "fonts": ("Bfont Regular",),
    "materials": ("yt Annotation",),
}
