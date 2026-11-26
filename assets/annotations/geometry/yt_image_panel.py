# Node-group asset "yt Image Panel" (GeometryNodeTree).
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
# Hand-written: re-exporting from a .blend replaces the helpers with inline nodes.
import bpy
from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import CustomGeometryGroup

from .yt_annotation_view import YtAnnotationView

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


def _axis(low, high, every, length, tick, label_size, decimals, vertical, right=False):
    """Ticks and labels at multiples of `every` between `low` and `high`, along
    x from the origin (or up y, labels to the left, or right with `right`, when `vertical`).
    `every` 0 picks a round step; labels get at least the decimals the step needs.
    """
    step = g.Switch.float(g.Math.greater_than(every, 0.0), nice_step(high - low), every).o.output.max(1e-30)
    decimals = g.Math.maximum(decimals, (step.log(10.0).floor() * -1.0).max(0.0))
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
        side = 1.0 if right else -1.0
        label = g.StringToCurves(
            string=value.output.to_string(decimals), size=label_size, align_x="Left" if right else "Right", align_y="Middle"
        )
        at = g.CombineXYZ(x=(tick + gap) * side, y=x.output)
        mark = g.TransformGeometry(
            geometry=g.Grid(size_x=tick, size_y=tick * 0.2, vertices_x=2, vertices_y=2),
            translation=g.CombineXYZ(x=tick * 0.5 * side),
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


def nice_step(span):
    """A 1, 2 or 5 x 10^n step giving about four ticks over `span`."""
    raw = span.abs().max(1e-30) / 4.0
    power = 10.0 ** raw.log(10.0).floor()
    fraction = raw / power
    return power * (1.0 + g.Math.greater_than(fraction, 1.9999) + g.Math.greater_than(fraction, 4.9999).o.value * 3.0)


def _text(string, size, align_x, align_y, translation, rotation=(0.0, 0.0, 0.0)):
    return (
        g.StringToCurves(string=string, size=size, align_x=align_x, align_y=align_y)
        >> g.RealizeInstances()
        >> g.FillCurve()
        >> g.TransformGeometry(translation=translation, rotation=rotation)
    )


def _outline(width, height, line_width):
    """A rectangle's outline from the origin to (width, height)."""
    return (
        g.Grid(size_x=width, size_y=height, vertices_x=2, vertices_y=2)
        >> g.TransformGeometry(translation=g.CombineXYZ(x=width * 0.5, y=height * 0.5))
        >> g.MeshToCurve()
        >> g.CurveToMesh(profile_curve=g.CurveCircle(radius=line_width * 0.5, resolution=4))
    )


def _frame_and_titles(width, height, x_title, y_title, tick, label_size, line_width):
    x_label = _text(
        x_title, label_size * 1.15, "Center", "Top", g.CombineXYZ(x=width * 0.5, y=tick * -1.0 - label_size * 1.6)
    )
    y_label = _text(
        y_title, label_size * 1.15, "Center", "Bottom",
        g.CombineXYZ(x=tick * -1.0 - label_size * 2.4, y=height * 0.5), rotation=(0.0, 0.0, 1.5707963),
    )
    return _outline(width, height, line_width), x_label, y_label


def format_value(value, decimals):
    """`value` as text: fixed point, or `m.mmeN` when it's below 0.01 or 1e4 and up."""
    magnitude = value.abs()
    exponent = magnitude.max(1e-38).log(10.0).floor()
    scientific = g.BooleanMath.l_and(
        g.Math.greater_than(magnitude, 0.0),
        g.BooleanMath.l_or(g.Math.less_than(magnitude, 0.01), g.Math.greater_than(magnitude, 9999.5)),
    )
    mantissa = value / 10.0**exponent
    sci = g.JoinStrings((mantissa.to_string(decimals), g.String("e"), exponent.to_string(0)))
    return g.Switch.string(scientific, value.to_string(decimals), sci)


def _callout(units_object, probe, radius, width, height, line_width, color):
    """A ring around `probe`'s sphere where it is on the frame, and a line from
    it to the nearest point of the (0, 0)-(width, height) panel.
    """
    view = YtAnnotationView(
        **{"Units Object": units_object, "Use Point": True, "Point": g.ObjectInfo(probe).o.location}, Unit="unitary"
    )
    ring_radius = radius * view.o.frame_per_unit / view.o.scene_per_unit.max(1e-30)
    center = view.o.screen_position
    nearest = g.CombineXYZ(x=center.x.clamp(0.0, width), y=center.y.clamp(0.0, height))
    reach = (nearest - center).length()
    start = center + (nearest - center).normalize().scale(ring_radius)
    profile = g.CurveCircle(radius=line_width * 0.5, resolution=4)
    ring = (
        g.CurveCircle(radius=ring_radius, resolution=64)
        >> g.TransformGeometry(translation=center)
        >> g.CurveToMesh(profile_curve=profile)
    )
    line = g.CurveLine(start, nearest) >> g.CurveToMesh(profile_curve=profile)
    shown = g.BooleanMath.l_and(view.o.in_front, g.Math.greater_than(ring_radius, 0.0))
    callout = g.JoinGeometry((ring, g.Switch.geometry(g.Math.greater_than(reach, ring_radius), true=line)))
    return g.StoreNamedAttribute.point.color(
        g.Switch.geometry(shown, true=callout), name="yt_annotation_color", value=color
    )


def callout_inputs(tree):
    probe = tree.inputs.object("Probe", description="The probe whose results this shows")
    units_object = tree.inputs.object(
        "Units Object", description="The volume's units object (Units for Geometry Nodes), for the callout"
    )
    show = tree.inputs.boolean("Callout", True, description="Ring the probe's sphere and draw a line to the panel")
    radius = tree.inputs.float(
        "Probe Radius", 0.0, description="The probe's radius in scene units; the probe sets it when it runs"
    )
    return probe, units_object, show, radius


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
        ticks_every = tree.inputs.float("Ticks Every", 0.0, description="Spacing of the ticks, in axis units; 0 picks a round step")
        decimals = tree.inputs.integer("Decimals", 0, description="Decimal places of the tick labels")
        label_size = tree.inputs.float("Label Size", 0.015, description="Height of the labels, in frame widths")
        text_color = tree.inputs.color(
            "Text Color", (1.0, 1.0, 1.0, 1.0), description="Color of the frame, ticks and labels"
        )
        line_width = tree.inputs.float("Line Width", 0.0015, description="Width of the frame and ticks")
        show_colorbar = tree.inputs.boolean("Colorbar", True, description="Show a colorbar right of the plot")
        value_title = tree.inputs.string("Value Title", "", description="Text beside the colorbar")
        value_ticks = tree.inputs.float("Value Ticks Every", 0.0, description="Spacing of the colorbar's ticks; 0 picks a round step")
        probe, units_object, show_callout, radius = callout_inputs(tree)
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
        tick = label_size * 0.5
        frame, x_label, y_label = _frame_and_titles(width, height, x_title, y_title, tick, label_size, line_width)
        x_axis = _axis(x_min, x_max, ticks_every, width, tick, label_size, decimals, vertical=False)
        y_axis = _axis(y_min, y_max, ticks_every, height, tick, label_size, decimals, vertical=True)

        bar_x = width + label_size
        bar_width = label_size * 1.2
        bar_y = g.Position().o.position.y
        bar = (
            g.Grid(size_x=bar_width, size_y=height, vertices_x=2, vertices_y=65)
            >> g.TransformGeometry(translation=g.CombineXYZ(x=bar_x + bar_width * 0.5, y=height * 0.5))
            >> g.StoreNamedAttribute.face.color(
                name="yt_annotation_color", value=g.ColorRamp(bar_y / height, items=VIRIDIS)
            )
        )
        bar_axis = g.TransformGeometry(
            geometry=_axis(value_min, value_max, value_ticks, height, tick, label_size, decimals, vertical=True, right=True),
            translation=g.CombineXYZ(x=bar_x + bar_width),
        )
        bar_label = _text(
            value_title, label_size * 1.15, "Center", "Top",
            g.CombineXYZ(x=bar_x + bar_width + tick + label_size * 4.2, y=height * 0.5), rotation=(0.0, 0.0, 1.5707963),
        )
        bar_text = g.JoinGeometry((_outline(bar_width, height, line_width) >> g.TransformGeometry(
            translation=g.CombineXYZ(x=bar_x)), bar_axis, bar_label))
        text = g.StoreNamedAttribute.point.color(
            g.JoinGeometry((frame, x_axis, y_axis, x_label, y_label, g.Switch.geometry(show_colorbar, true=bar_text))),
            name="yt_annotation_color", value=text_color,
        )
        extent = g.Switch.float(show_colorbar, width, bar_x + bar_width)
        callout = _callout(units_object, probe, radius, extent, height, line_width, text_color)
        (
            g.JoinGeometry((text, bins, g.Switch.geometry(show_colorbar, true=bar), g.Switch.geometry(show_callout, true=callout)))
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
