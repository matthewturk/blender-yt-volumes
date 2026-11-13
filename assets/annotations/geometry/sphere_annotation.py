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
        with tree.inputs.panel("Disappearing"):
            disappearing = tree.inputs.boolean(
                "Disappearing", False, is_panel_toggle=True
            )
            start_frame = tree.inputs.integer("Start Frame", 0)
            end_frame = tree.inputs.integer("End Frame", 0)
        geometry_1 = tree.outputs.geometry("Geometry")

        closure_zone = g.ClosureZone()
        geometry_2 = closure_zone.inputs.geometry("Geometry")
        geometry_3 = closure_zone.outputs.geometry("Geometry")
        closure_zone_1 = g.ClosureZone()
        geometry_4 = closure_zone_1.inputs.geometry("Geometry")
        geometry_5 = closure_zone_1.outputs.geometry("Geometry")
        geometry_4.output >> geometry_5.input
        sample_index = g.SampleIndex.point.float(
            g.ObjectInfo(units_object).o.geometry,
            g.NamedAttribute.float(
                g.FormatString("scene_per_{s}", items={"s": unit})
            ).o.attribute,
        )
        curve_to_mesh = g.CurveToMesh(
            curve=g.CurveCircle(
                radius=sample_index.o.value * unit_fraction, resolution=64
            ),
            profile_curve=g.CurveCircle(resolution=8, radius=0.01),
        )
        geometry_to_instance = g.GeometryToInstance(curve_to_mesh)
        realize_instances = (
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
        )
        store_named_attribute = (
            geometry_2.output
            >> g.StoreNamedAttribute.point.float(name="start_frame", value=start_frame)
            >> g.StoreNamedAttribute.point.float(name="end_frame", value=end_frame)
        )
        store_named_attribute >> geometry_3.input
        evaluate_closure = g.EvaluateClosure(
            disappearing.switch.closure(closure_zone_1.closure, closure_zone.closure)
        )
        evaluate_closure.inputs.geometry(realize_instances, "Geometry")
        geometry_6 = evaluate_closure.outputs.geometry("Geometry")

        geometry_6.output >> geometry_1


ASSET = SphereAnnotation

ASSET_METADATA = {
    "description": "Add a gyroscope-like sphere to a scene at specified size.",
    "tags": ("3D Scene",),
}

DATABLOCK_DEPENDENCIES = {
    "materials": ("Sphere Material",),
}
