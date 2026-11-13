# Node-group asset "yt Colorbar" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
import bpy
from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import CustomGeometryGroup


class YtColorbar(CustomGeometryGroup):
    _name = "yt Colorbar"

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        _geometry = tree.inputs.geometry("Geometry")
        volume = tree.inputs.object(
            "Volume",
            description="The volume whose Geometry Nodes transfer function to show",
        )
        width = tree.inputs.float(
            "Width", 0.6, description="Length of the bar, in frame widths"
        )
        height = tree.inputs.float(
            "Height", 0.03, description="Thickness of the bar, in frame widths"
        )
        ticks_every = tree.inputs.float(
            "Ticks Every",
            1.0,
            description="Spacing of the ticks, in decades (log10 units)",
        )
        decimals = tree.inputs.integer(
            "Decimals", 0, description="Decimal places of the tick labels"
        )
        label_size = tree.inputs.float(
            "Label Size", 0.02, description="Height of the tick labels, in frame widths"
        )
        title = tree.inputs.string(
            "Title", "", description="Text under the tick labels"
        )
        text_color = tree.inputs.color(
            "Text Color",
            (1.0, 1.0, 1.0, 1.0),
            description="Color of the ticks, labels, title and opacity line",
        )
        show_opacity = tree.inputs.boolean(
            "Show Opacity", True, description="Draw the opacity above the bar"
        )
        opacity_height = tree.inputs.float(
            "Opacity Height",
            0.05,
            description="Height of the opacity line's full scale",
        )
        opacity_decades = tree.inputs.float(
            "Opacity Decades",
            0.0,
            description="Plot opacity over this many decades; 0 plots it linearly",
        )
        line_width = tree.inputs.float(
            "Line Width", 0.002, description="Width of the ticks and the opacity line"
        )
        geometry_1 = tree.outputs.geometry("Geometry")

        index = g.Index()
        math_1 = ticks_every.max(0.001)
        math_2 = height * 0.4
        math_3 = math_2 * -1.0 - label_size * 0.3
        instances_to_points = g.InstancesToPoints(g.ObjectInfo(volume).o.geometry)
        domain_size = g.DomainSize(geometry=instances_to_points, component="POINTCLOUD")
        math_4 = domain_size.o.point_count - 1.0
        sample_index = g.SampleIndex.point.float(
            instances_to_points,
            g.NamedAttribute.float("yt_tf_alpha").o.attribute,
            index,
        )
        math_5 = (
            sample_index.o.value.log(10.0) / opacity_decades + 1.0
        ) * g.Math.greater_than(sample_index, 0.0)
        math_6 = sample_index.o.value + g.Math.greater_than(
            opacity_decades, 0.0
        ).o.value * (math_5 - sample_index)
        combine_xyz = g.CombineXYZ(
            x=index.o.index / math_4 * width,
            y=height.mul_add(1.25, math_6.clamp() * opacity_height),
        )
        sample_index_1 = g.SampleIndex.point.float(
            instances_to_points, g.NamedAttribute.float("yt_tf_log10_value").o.attribute
        )
        math_7 = (sample_index_1.o.value / math_1).ceil() * math_1
        sample_index_2 = g.SampleIndex.point.float(
            instances_to_points,
            g.NamedAttribute.float("yt_tf_log10_value").o.attribute,
            math_4,
        )
        math_8 = sample_index_2.o.value - sample_index_1
        math_9 = (
            ((sample_index_2.o.value - math_7) / math_1 + 0.0001).floor() + 1.0
        ).max(0.0) * g.Math.greater_than(math_8, 0.0)
        position = g.Position().o.position
        sample_index_3 = g.SampleIndex.point.color(
            instances_to_points,
            g.NamedAttribute.color("yt_tf_color").o.attribute,
            (position.x / width * math_4).round(),
        )
        curve_to_mesh = (
            g.MeshLine(count=domain_size.o.point_count, offset=(1.0, 0.0, 0.0))
            >> g.SetPosition(position=combine_xyz)
            >> g.MeshToCurve()
            >> g.CurveToMesh(
                profile_curve=g.CurveCircle(radius=line_width * 0.5, resolution=6)
            )
        )
        set_position = (
            g.MeshLine(
                count=math_9,
                start_location=g.CombineXYZ(math_7),
                offset=g.CombineXYZ(math_1),
            )
            >> g.StoreNamedAttribute.point.float(name="yt_tick_value", value=position.x)
            >> g.SetPosition(
                position=g.CombineXYZ((position.x - sample_index_1) / math_8 * width)
            )
        )
        for_each = g.ForEachGeometryElementZone(geometry=set_position)
        value = for_each.items.float(
            g.NamedAttribute.float("yt_tick_value").o.attribute, "Value"
        )
        x = for_each.items.float(position.x, "X")
        geometry_2 = for_each.generated_items.geometry(name="Geometry")
        store_named_attribute = (
            g.Grid(
                size_x=width,
                size_y=height,
                vertices_x=domain_size.o.point_count,
                vertices_y=2,
            )
            >> g.TransformGeometry(
                translation=g.CombineXYZ(x=width * 0.5, y=height * 0.5)
            )
            >> g.StoreNamedAttribute.point.color(
                name="yt_annotation_color", value=sample_index_3
            )
        )
        transform_geometry = g.TransformGeometry(
            geometry=g.Grid(
                size_x=line_width, size_y=math_2, vertices_x=2, vertices_y=2
            ),
            translation=g.CombineXYZ(y=math_2 * -0.5),
        )
        transform_geometry_1 = (
            g.StringToCurves(
                string=title, size=label_size * 1.15, align_x="Center", align_y="Top"
            )
            >> g.RealizeInstances()
            >> g.FillCurve()
            >> g.TransformGeometry(
                translation=g.CombineXYZ(x=width * 0.5, y=math_3 - label_size * 1.6)
            )
        )
        transform_geometry_2 = (
            g.StringToCurves(
                string=value.output.to_string(decimals),
                size=label_size,
                align_x="Center",
                align_y="Top",
            )
            >> g.RealizeInstances()
            >> g.FillCurve()
            >> g.TransformGeometry(translation=g.CombineXYZ(x=x.output, y=math_3))
        )
        transform_geometry_2 >> geometry_2.input
        join_geometry = g.JoinGeometry(
            (
                transform_geometry_1,
                geometry_2.output,
                g.RealizeInstances(
                    g.InstanceOnPoints(points=set_position, instance=transform_geometry)
                ),
                show_opacity.switch.geometry(true=curve_to_mesh),
            )
        )
        store_named_attribute_1 = g.StoreNamedAttribute.point.color(
            join_geometry, name="yt_annotation_color", value=text_color
        )
        (
            g.JoinGeometry((store_named_attribute_1, store_named_attribute))
            >> g.SetMaterial(material=bpy.data.materials.get("yt Annotation"))
            >> geometry_1
        )


ASSET = YtColorbar

DATABLOCK_DEPENDENCIES = {
    "fonts": ("Bfont Regular",),
    "materials": ("yt Annotation",),
}
