# Node-group asset "Sphere Annotation" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
import math

import bpy
from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import CustomGeometryGroup


class SphereAnnotation(CustomGeometryGroup):
    _name = "Sphere Annotation"
    _tree_properties = {"is_modifier": True}

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        _geometry = tree.inputs.geometry("Geometry")
        unit = tree.inputs.string("Unit", "")
        units_object = tree.inputs.object("Units Object", optional_label=True)
        unit_fraction = tree.inputs.float(
            "Unit Fraction", 0.1, min_value=-10_000.0, max_value=10_000.0
        )
        annotation_color = tree.inputs.color("Annotation Color", (0.8, 0.8, 0.8, 1.0))
        annotation_brightness = tree.inputs.float(
            "Annotation Brightness", 1.0, min_value=-10_000.0, max_value=10_000.0
        )
        with tree.inputs.panel("Text Annotation"):
            _text_annotation = tree.inputs.boolean(
                "Text Annotation", False, is_panel_toggle=True
            )
            face_camera = tree.inputs.boolean("Face Camera", False, optional_label=True)
            text_color = tree.inputs.color("Text Color", (0.8, 0.8, 0.8, 1.0))
            text_strength = tree.inputs.float("Text Strength", 0.0)
            padding = tree.inputs.float(
                "Padding", 0.1, min_value=-10_000.0, max_value=10_000.0
            )
            text_size = tree.inputs.float(
                "Text Size", 1.0, min_value=0.0, subtype="DISTANCE"
            )
        with tree.inputs.panel("Disappearing"):
            disappearing = tree.inputs.boolean(
                "Disappearing", False, is_panel_toggle=True
            )
            start_frame = tree.inputs.integer("Start Frame", 0)
            end_frame = tree.inputs.integer("End Frame", 0)
        geometry_1 = tree.outputs.geometry("Geometry")

        float_curve = g.FloatCurve(
            value=g.SceneTime()
            .props(hide_outputs=["Seconds"])
            .o.frame.map_range(start_frame, end_frame),
            items=((0.0, 0.0, "VECTOR"), (0.1, 1.0), (0.9, 1.0), (1.0, 0.0, "VECTOR")),
        )
        float_curve.node.mapping.update()
        float_curve.node.mapping.curves[0].points[2].select = True
        location = g.ObjectInfo(
            object=g.ActiveCamera(), transform_space="RELATIVE"
        ).o.location
        math_1 = g.ImplicitConversion.float(face_camera).o.value * (
            location.y.atan2(location.x) + math.pi / 2
        )
        sample_index = g.SampleIndex.point.float(
            g.ObjectInfo(units_object).o.geometry,
            g.NamedAttribute.float(
                g.FormatString("scene_per_{s}", items={"s": unit.lowercase()})
            ).o.attribute,
        )
        math_2 = sample_index.o.value * unit_fraction
        curve_to_mesh = g.CurveToMesh(
            curve=g.CurveCircle(radius=math_2, resolution=64),
            profile_curve=g.CurveCircle(resolution=8, radius=0.01),
        )
        geometry_to_instance = g.GeometryToInstance(curve_to_mesh)
        store_named_attribute = (
            g.MeshBoolean(
                mesh_2=(
                    geometry_to_instance,
                    g.RotateInstances(
                        instances=geometry_to_instance, rotation=(0.0, math.pi / 2, 0.0)
                    ),
                    g.RotateInstances(
                        instances=geometry_to_instance, rotation=(math.pi / 2, 0.0, 0.0)
                    ),
                ),
                operation="UNION",
                solver="MANIFOLD",
            )
            >> g.SetMaterial(material=bpy.data.materials.get("Sphere Material"))
            >> g.SetShadeSmooth.face()
            >> g.RealizeInstances()
            >> g.StoreNamedAttribute.point.float(
                name="emission_strength",
                value=disappearing.switch.float(1.0, float_curve)
                * annotation_brightness,
            )
            >> g.StoreNamedAttribute.point.color(
                name="emission_color", value=annotation_color
            )
        )
        set_shade_smooth = (
            g.StringToCurves(
                string=g.FormatString("{u} {v}", items={"u": unit_fraction, "v": unit}),
                size=text_size,
                align_x="Center",
            )
            >> g.TranslateInstances(
                translation=g.CombineXYZ(z=(1.0 + padding) * math_2)
            )
            >> g.RotateInstances(rotation=(math.pi / 2, 0.0, 0.0))
            >> g.RotateInstances(
                rotation=g.AxisAngleToRotation(angle=math_1), local_space=False
            )
            >> g.FillCurve()
            >> g.StoreNamedAttribute.point.color(
                name="emission_color", value=text_color
            )
            >> g.StoreNamedAttribute.point.float(
                name="emission_strength", value=text_strength
            )
            >> g.SetMaterial(material=bpy.data.materials.get("Sphere Material"))
            >> g.SetShadeSmooth.face()
        )
        join_geometry = g.JoinGeometry((set_shade_smooth, store_named_attribute))

        join_geometry >> geometry_1


ASSET = SphereAnnotation

ASSET_METADATA = {
    "description": "Add a gyroscope-like sphere to a scene at specified size.",
    "tags": ("3D Scene",),
}

DATABLOCK_DEPENDENCIES = {
    "fonts": ("Bfont Regular",),
    "materials": ("Sphere Material",),
}
