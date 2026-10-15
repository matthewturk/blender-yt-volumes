"""A transfer function drawn in Geometry Nodes on the volume Empty itself.

A Geometry Nodes modifier on the volume object outputs points, in order from
the low to the high end of the transfer function's value range, carrying a
`yt_tf_color` color attribute and optionally a `yt_tf_alpha` float attribute
(else the color's alpha). Any node setup that produces them works;
`add_transfer_function_modifier` makes a starter group with a Color Ramp and
an opacity Float Curve.
"""

import bpy
import numpy as np

COLOR_ATTRIBUTE = "yt_tf_color"
ALPHA_ATTRIBUTE = "yt_tf_alpha"
COLOR_NODE_NAME = "TF Color"
ALPHA_NODE_NAME = "TF Opacity"
MODIFIER_NAME = "yt Transfer Function"
NBINS = 256

# Starter ramp: inferno, sRGB.
_STARTER_STOPS = (
    (0.0, (0.001, 0.0, 0.014)),
    (0.25, (0.341, 0.062, 0.429)),
    (0.5, (0.735, 0.216, 0.330)),
    (0.75, (0.978, 0.557, 0.035)),
    (1.0, (0.988, 0.998, 0.645)),
)


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def build_node_group(name=MODIFIER_NAME, nbins=NBINS):
    """A new Geometry Nodes group: `nbins` vertices along x in [0, 1], their x
    through a Color Ramp and a Float Curve into the two attributes.
    """
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, link = tree.nodes, tree.links.new

    line = nodes.new("GeometryNodeMeshLine")
    line.inputs["Count"].default_value = nbins
    line.inputs["Offset"].default_value = (1.0 / (nbins - 1), 0.0, 0.0)
    # Vertices only: edges would draw as a line through the volume box.
    no_edges = nodes.new("GeometryNodeDeleteGeometry")
    no_edges.domain, no_edges.mode = "EDGE", "EDGE_FACE"
    position = nodes.new("GeometryNodeInputPosition")
    value = nodes.new("ShaderNodeSeparateXYZ")
    value.label = "Value (0-1, log)"

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = ramp.label = COLOR_NODE_NAME
    elements = ramp.color_ramp.elements
    while len(elements) < len(_STARTER_STOPS):
        elements.new(0.5)
    # Ramp colors are scene-linear; the UI shows them display-encoded.
    for element, (stop, srgb) in zip(elements, _STARTER_STOPS):
        element.position = stop
        element.color = (*_srgb_to_linear(srgb), 1.0)

    curve = nodes.new("ShaderNodeFloatCurve")
    curve.name = curve.label = ALPHA_NODE_NAME
    points = curve.mapping.curves[0].points
    points[0].location = (0.0, 0.0)
    points[1].location = (1.0, 0.8)
    points.new(0.6, 0.0)  # transparent over the low 60%, so the box doesn't fill with fog
    curve.mapping.update()

    store_color = nodes.new("GeometryNodeStoreNamedAttribute")
    store_color.data_type, store_color.domain = "FLOAT_COLOR", "POINT"
    store_color.inputs["Name"].default_value = COLOR_ATTRIBUTE
    store_alpha = nodes.new("GeometryNodeStoreNamedAttribute")
    store_alpha.data_type, store_alpha.domain = "FLOAT", "POINT"
    store_alpha.inputs["Name"].default_value = ALPHA_ATTRIBUTE
    output = nodes.new("NodeGroupOutput")

    link(line.outputs["Mesh"], no_edges.inputs["Geometry"])
    link(position.outputs["Position"], value.inputs["Vector"])
    link(value.outputs["X"], ramp.inputs["Fac"])
    link(value.outputs["X"], curve.inputs["Value"])
    link(no_edges.outputs["Geometry"], store_color.inputs["Geometry"])
    link(ramp.outputs["Color"], store_color.inputs["Value"])
    link(store_color.outputs["Geometry"], store_alpha.inputs["Geometry"])
    link(curve.outputs["Value"], store_alpha.inputs["Value"])
    link(store_alpha.outputs["Geometry"], output.inputs["Geometry"])

    for column, node in enumerate((line, no_edges, store_color, store_alpha, output)):
        node.location = (200 * column, 0)
    for node, (x, y) in ((position, (-400, -250)), (value, (-200, -250)), (ramp, (0, -250)), (curve, (300, -450))):
        node.location = (x, y)
    return tree


def add_transfer_function_modifier(obj):
    """Add a Geometry Nodes modifier with its own starter group to `obj`."""
    modifier = obj.modifiers.new(MODIFIER_NAME, "NODES")
    modifier.node_group = build_node_group()
    return modifier


def find_nodes(obj):
    """`(color_ramp_node, opacity_curve_node)` from `obj`'s Geometry Nodes
    modifiers, either None if absent.
    """
    ramp = curve = None
    for modifier in obj.modifiers:
        tree = getattr(modifier, "node_group", None) if modifier.type == "NODES" else None
        if tree is None:
            continue
        ramp = ramp or tree.nodes.get(COLOR_NODE_NAME)
        curve = curve or tree.nodes.get(ALPHA_NODE_NAME)
    return ramp, curve


def _read_attribute(attributes, name, components):
    attribute = attributes.get(name)
    if attribute is None:
        return None
    values = np.empty(len(attribute.data) * components, dtype=np.float32)
    attribute.data.foreach_get("color" if components == 4 else "value", values)
    return values.reshape(-1, components) if components > 1 else values


def sample(obj, depsgraph, nbins=NBINS):
    """`(nbins, 4)` RGBA in [0, 1], color sRGB-encoded, from `obj`'s evaluated
    geometry, resampled by point order.

    Raises `ValueError` if the geometry has no `yt_tf_color` attribute.
    """
    try:
        geometry = obj.evaluated_get(depsgraph).evaluated_geometry()
        components = (geometry.mesh, geometry.pointcloud)
    except TypeError:  # an Empty with no geometry-producing modifier
        components = ()
    for component in components:
        if component is not None and COLOR_ATTRIBUTE in component.attributes:
            attributes = component.attributes
            break
    else:
        raise ValueError(
            f"{obj.name!r} has no {COLOR_ATTRIBUTE!r} attribute: add a transfer function "
            "in the Volume Override panel, or store one with Geometry Nodes"
        )
    rgba = _read_attribute(attributes, COLOR_ATTRIBUTE, 4).astype(np.float64)
    if len(rgba) < 2:
        raise ValueError(f"{obj.name!r}: {COLOR_ATTRIBUTE!r} needs at least 2 points")
    alpha = _read_attribute(attributes, ALPHA_ATTRIBUTE, 1)
    if alpha is not None:
        rgba[:, 3] = alpha
    rgba[:, :3] = _linear_to_srgb(rgba[:, :3])
    rgba[:, 3] = np.clip(rgba[:, 3], 0.0, 1.0)
    src = np.linspace(0.0, 1.0, len(rgba))
    dst = np.linspace(0.0, 1.0, nbins)
    return np.stack([np.interp(dst, src, rgba[:, c]) for c in range(4)], axis=1)


class VOLUMEOVERRIDE_OT_add_transfer_function(bpy.types.Operator):
    """Add a Geometry Nodes transfer function (Color Ramp and opacity curve) to this volume"""

    bl_idname = "volume_override.add_transfer_function"
    bl_label = "Add Transfer Function"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        add_transfer_function_modifier(context.object)
        return {"FINISHED"}


CLASSES = (VOLUMEOVERRIDE_OT_add_transfer_function,)
