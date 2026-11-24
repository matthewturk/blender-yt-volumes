# Node-group asset "yt Image Panel" (GeometryNodeTree).
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# Hand-written: re-exporting from a .blend replaces _axis with inline nodes.
import bpy
from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import CustomGeometryGroup

# matplotlib's viridis, scene-linear
VIRIDIS = (
    (0.000, (0.0580, 0.0004, 0.0887, 1.0)),
    (0.125, (0.0632, 0.0260, 0.1990, 1.0)),
    (0.250, (0.0431, 0.0848, 0.2588, 1.0)),
    (0.375, (0.0252, 0.1697, 0.2716, 1.0)),
    (0.500, (0.0148, 0.2813, 0.2639, 1.0)),
    (0.625, (0.0215, 0.4252, 0.2156, 1.0)),
    (0.750, (0.1123, 0.5852, 0.1212, 1.0)),
    (0.875, (0.4180, 0.7176, 0.0299, 1.0)),
    (1.000, (0.9847, 0.7997, 0.0182, 1.0)),
)


def _axis(low, high, every, length, tick, label_size, decimals, vertical):
    """Ticks and labels at multiples of `every` between `low` and `high`, along
    x from the origin (or up y, labels to the left, when `vertical`).
    """
    step = every.max(0.001)
    first = (low / step).ceil() * step
    span = high - low
    count = (((high - first) / step + 0.0001).floor() + 1.0).max(0.0) * g.Math.greater_than(span, 0.0)
    position = g.Position().o.position
    points = (
        g.MeshLine(count=count, start_location=g.CombineXYZ(first), offset=g.CombineXYZ(step))
        >> g.StoreNamedAttribute.point.float(name="yt_tick_value", value=position.x)
        >> g.SetPosition(position=g.CombineXYZ((position.x - low) / span * length))
    )
    for_each = g.ForEachGeometryElementZone(geometry=points)
    value = for_each.items.float(g.NamedAttribute.float("yt_tick_value").o.attribute, "Value")
    x = for_each.items.float(position.x, "X")
    generated = for_each.generated_items.geometry(name="Geometry")
    gap = label_size * 0.3
    if vertical:
        label = g.StringToCurves(string=value.output.to_string(decimals), size=label_size, align_x="Right", align_y="Middle")
        at = g.CombineXYZ(x=tick * -1.0 - gap, y=x.output)
        mark = g.TransformGeometry(
            geometry=g.Grid(size_x=tick, size_y=tick * 0.2, vertices_x=2, vertices_y=2),
            translation=g.CombineXYZ(x=tick * -0.5),
        )
    else:
        label = g.StringToCurves(string=value.output.to_string(decimals), size=label_size, align_x="Center", align_y="Top")
        at = g.CombineXYZ(x=x.output, y=tick * -1.0 - gap)
        mark = g.TransformGeometry(
            geometry=g.Grid(size_x=tick * 0.2, size_y=tick, vertices_x=2, vertices_y=2),
            translation=g.CombineXYZ(y=tick * -0.5),
        )
    label >> g.RealizeInstances() >> g.FillCurve() >> g.TransformGeometry(translation=at) >> generated.input
    if vertical:
        # MeshLine runs along x; swap onto y before instancing the marks.
        marks_points = points >> g.SetPosition(position=g.CombineXYZ(y=position.x))
    else:
        marks_points = points
    marks = g.RealizeInstances(g.InstanceOnPoints(points=marks_points, instance=mark))
    return g.JoinGeometry((generated.output, marks))


class YtImagePanel(CustomGeometryGroup):
    _name = "yt Image Panel"

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        _geometry = tree.inputs.geometry("Geometry")
        image = tree.inputs.image(
            "Image", description="A float Image holding one value per pixel in R, with empty pixels at alpha 0"
        )
        width = tree.inputs.float("Width", 0.3, description="Width of the plot, in frame widths")
        value_min = tree.inputs.float("Value Min", 0.0, description="Value at the bottom of the colormap")
        value_max = tree.inputs.float("Value Max", 1.0, description="Value at the top of the colormap")
        x_min = tree.inputs.float("X Min", 0.0, description="Axis value at the left edge")
        x_max = tree.inputs.float("X Max", 1.0, description="Axis value at the right edge")
        y_min = tree.inputs.float("Y Min", 0.0, description="Axis value at the bottom edge")
        y_max = tree.inputs.float("Y Max", 1.0, description="Axis value at the top edge")
        x_title = tree.inputs.string("X Title", "", description="Text under the x axis")
        y_title = tree.inputs.string("Y Title", "", description="Text left of the y axis")
        ticks_every = tree.inputs.float("Ticks Every", 1.0, description="Spacing of the ticks, in axis units")
        decimals = tree.inputs.integer("Decimals", 0, description="Decimal places of the tick labels")
        label_size = tree.inputs.float("Label Size", 0.015, description="Height of the labels, in frame widths")
        text_color = tree.inputs.color(
            "Text Color", (1.0, 1.0, 1.0, 1.0), description="Color of the frame, ticks and labels"
        )
        line_width = tree.inputs.float("Line Width", 0.0015, description="Width of the frame and ticks")
        geometry_1 = tree.outputs.geometry("Geometry")

        info = g.ImageInfo(image)
        height = width * info.o.height / info.o.width.max(1.0)
        position = g.Position().o.position
        texture = g.ImageTexture(
            image, g.CombineXYZ(position.x / width, position.y / height), interpolation="Closest", extension="EXTEND"
        )
        value = g.SeparateColor(texture.o.color).o.red
        color = g.ColorRamp(g.MapRange(value, value_min, value_max), items=VIRIDIS)
        bins = (
            g.Grid(size_x=width, size_y=height, vertices_x=info.o.width + 1, vertices_y=info.o.height + 1)
            >> g.TransformGeometry(translation=g.CombineXYZ(x=width * 0.5, y=height * 0.5))
            >> g.StoreNamedAttribute.face.color(name="yt_annotation_color", value=color)
            >> g.DeleteGeometry.face(selection=g.Math.less_than(texture.o.alpha, 0.5))
        )
        frame = (
            g.Grid(size_x=width, size_y=height, vertices_x=2, vertices_y=2)
            >> g.TransformGeometry(translation=g.CombineXYZ(x=width * 0.5, y=height * 0.5))
            >> g.MeshToCurve()
            >> g.CurveToMesh(profile_curve=g.CurveCircle(radius=line_width * 0.5, resolution=4))
        )
        tick = label_size * 0.5
        x_axis = _axis(x_min, x_max, ticks_every, width, tick, label_size, decimals, vertical=False)
        y_axis = _axis(y_min, y_max, ticks_every, height, tick, label_size, decimals, vertical=True)
        x_label = (
            g.StringToCurves(string=x_title, size=label_size * 1.15, align_x="Center", align_y="Top")
            >> g.RealizeInstances()
            >> g.FillCurve()
            >> g.TransformGeometry(translation=g.CombineXYZ(x=width * 0.5, y=tick * -1.0 - label_size * 1.6))
        )
        y_label = (
            g.StringToCurves(string=y_title, size=label_size * 1.15, align_x="Center", align_y="Bottom")
            >> g.RealizeInstances()
            >> g.FillCurve()
            >> g.TransformGeometry(
                translation=g.CombineXYZ(x=tick * -1.0 - label_size * 2.4, y=height * 0.5),
                rotation=(0.0, 0.0, 1.5707963),
            )
        )
        text = g.StoreNamedAttribute.point.color(
            g.JoinGeometry((frame, x_axis, y_axis, x_label, y_label)), name="yt_annotation_color", value=text_color
        )
        (
            g.JoinGeometry((text, bins))
            >> g.SetMaterial(material=bpy.data.materials.get("yt Annotation"))
            >> geometry_1
        )


ASSET = YtImagePanel

ASSET_METADATA = {
    "description": "A float Image as a colormapped plot with axes, such as a probe's phase diagram",
    "tags": ("Annotation Scene",),
}

DATABLOCK_DEPENDENCIES = {
    "fonts": ("Bfont Regular",),
    "materials": ("yt Annotation",),
}
