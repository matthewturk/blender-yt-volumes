# Node-group asset "yt Annotation View" (GeometryNodeTree), dumped by nodebpy.assets.dump_library.
# Rebuild the library with nodebpy.assets.build_library (nodebpy build).
from bpy.types import GeometryNodeTree
from nodebpy import TreeBuilder
from nodebpy import geometry as g
from nodebpy.builder import CustomGeometryGroup


class YtAnnotationView(CustomGeometryGroup):
    _name = "yt Annotation View"
    _tree_properties = {
        "description": "Where a main-scene point lands on the annotation frame, and how long one Unit is there"
    }

    def _build_group(self, tree: TreeBuilder[GeometryNodeTree]) -> None:
        units_object = tree.inputs.object(
            "Units Object",
            description="The volume's units object (Units for Geometry Nodes)",
        )
        unit = tree.inputs.string(
            "Unit",
            "unitary",
            description="A unit listed in the volume's Data Units, or unitary",
        )
        use_point = tree.inputs.boolean(
            "Use Point",
            False,
            description="Ask about Point instead of the volume's zoom target or middle",
        )
        point = tree.inputs.vector(
            "Point",
            (0.0, 0.0, 0.0),
            description="A position in the main scene, e.g. an Object Info's Location",
        )
        screen_position = tree.outputs.vector(
            "Screen Position",
            description="Where the point is on the frame, in the space of the object these nodes run on",
        )
        frame_per_unit = tree.outputs.float(
            "Frame per Unit",
            description="On-frame length of one Unit at the point's depth; multiply a length in Unit by it",
        )
        frame_width = tree.outputs.float(
            "Frame Width", description="The frame's width in Unit at the point's depth"
        )
        depth = tree.outputs.float(
            "Depth",
            description="The point's distance along the camera's view axis, in scene units",
        )
        scene_per_unit = tree.outputs.float(
            "Scene per Unit", description="Scene units per Unit"
        )
        in_front = tree.outputs.boolean(
            "In Front", description="Whether the point is in front of the camera"
        )

        object_info = g.ObjectInfo(units_object)
        sample_index = g.SampleIndex.point.matrix(
            object_info.o.geometry,
            g.NamedAttribute.matrix("camera_to_world").o.attribute,
        )
        sample_index_1 = g.SampleIndex.point.vector(
            object_info.o.geometry, g.NamedAttribute.vector("view_center").o.attribute
        )
        sample_index_2 = g.SampleIndex.point.float(
            object_info.o.geometry,
            g.NamedAttribute.float("frame_width_per_depth").o.attribute,
        )
        sample_index_3 = g.SampleIndex.point.float(
            object_info.o.geometry,
            g.NamedAttribute.float("frame_width_ortho").o.attribute,
        )
        sample_index_4 = g.SampleIndex.point.vector(
            object_info.o.geometry, g.NamedAttribute.vector("frame_shift").o.attribute
        )
        sample_index_5 = g.SampleIndex.point.float(
            object_info.o.geometry, g.NamedAttribute.float("frame_height").o.attribute
        )
        sample_index_6 = g.SampleIndex.point.float(
            object_info.o.geometry,
            g.NamedAttribute.float(
                g.JoinStrings((g.String("scene_per_"), unit))
            ).o.attribute,
        )
        vector_math = use_point.switch.vector(sample_index_1, point) - g.TransformPoint(
            transform=sample_index
        )
        vector_math_1 = g.TransformDirection(
            transform=sample_index, direction=(0.0, 0.0, -1.0)
        ).o.direction.normalize()
        vector_math_2 = vector_math.dot(vector_math_1)
        math_1 = vector_math_2.mul_add(sample_index_2, sample_index_3)
        sample_index_6.o.value / math_1 >> frame_per_unit
        math_1 / sample_index_6 >> frame_width
        math_2 = g.Math.greater_than(vector_math_2, 0.0)
        vector_math_3 = g.TransformDirection(
            transform=sample_index, direction=(1.0, 0.0, 0.0)
        ).o.direction.normalize()
        vector_math_4 = g.TransformDirection(
            transform=sample_index, direction=(0.0, 1.0, 0.0)
        ).o.direction.normalize()
        math_3 = (
            sample_index_5.o.value.mul_add(
                addend=vector_math.dot(vector_math_4) / math_1
            )
            - sample_index_4.o.value.y
        )
        combine_xyz = g.CombineXYZ(
            x=vector_math.dot(vector_math_3) / math_1 + 0.5 - sample_index_4.o.value.x,
            y=math_3,
        )
        (
            combine_xyz.o.vector.transform(
                g.ObjectInfo(g.SelfObject()).o.transform.invert()
            )
            >> screen_position
        )

        vector_math_2 >> depth
        sample_index_6 >> scene_per_unit
        math_2 >> in_front


ASSET = YtAnnotationView

ASSET_METADATA = {
    "description": "Where a main-scene point lands on the annotation frame, and how long one Unit is there",
}
