"""The annotation scene: a flat scene laid over the finished frame, holding
colorbars that read volumes' transfer functions, scale bars, and anything
built with the "yt Annotation View" group, all through Geometry Nodes.

Its orthographic camera puts x from 0 to 1 across the frame and y from 0 to
height/width up it, origin at the bottom left; build annotations on z = 0.
Pass 2 renders it with Cycles through a Render Layers node, at the main
scene's frame and resolution, and lays it over the volumes.
"""

import bpy

from . import units
from .node_layout import arrange

SCENE_PROPERTY = "yt_annotation_scene"
COLORBAR_GROUP = "yt Colorbar"
FOV_GROUP = "yt Field of View"
VIEW_GROUP = "yt Annotation View"
TEMPLATE_GROUP = "yt Annotation"
MATERIAL_NAME = "yt Annotation"
COLOR_ATTRIBUTE = "yt_annotation_color"  # scene-linear; the material emits it
TICK_ATTRIBUTE = "yt_tick_value"
CAMERA_HEIGHT = 10.0


def aspect(scene):
    """Frame height over width, in the annotation scene's y units."""
    render = scene.render
    return (render.resolution_y * render.pixel_aspect_y) / (render.resolution_x * render.pixel_aspect_x)


def annotation_scene(scene):
    return getattr(scene, SCENE_PROPERTY, None)


def _assign(owner, name, value):
    # Same-value assignments still tag the scene for re-sync.
    if getattr(owner, name) != value:
        setattr(owner, name, value)


def sync(scene, annotations):
    """Match `annotations`' frame, resolution and camera framing to `scene`'s."""
    for name in ("resolution_x", "resolution_y", "resolution_percentage", "pixel_aspect_x", "pixel_aspect_y"):
        _assign(annotations.render, name, getattr(scene.render, name))
    _assign(annotations, "frame_current", scene.frame_current)
    camera = annotations.camera
    if camera is not None:
        location = (0.5, aspect(scene) / 2.0, CAMERA_HEIGHT)
        if tuple(camera.location) != location:
            camera.location = location


def add_annotation_scene(scene):
    """`scene`'s annotation scene, made if it has none."""
    view_node_group()  # in Add > Group for building annotations
    existing = annotation_scene(scene)
    if existing is not None:
        return existing
    annotations = bpy.data.scenes.new(f"{scene.name} Annotations")
    annotations.render.engine = "CYCLES"
    annotations.cycles.samples = 16
    annotations.cycles.use_denoising = False
    annotations.cycles.max_bounces = 0
    annotations.render.film_transparent = True
    annotations.view_settings.view_transform = "Standard"
    data = bpy.data.cameras.new(annotations.name)
    data.type, data.ortho_scale, data.sensor_fit = "ORTHO", 1.0, "HORIZONTAL"
    data.clip_start, data.clip_end = 0.1, 2.0 * CAMERA_HEIGHT
    camera = bpy.data.objects.new(annotations.name, data)
    annotations.collection.objects.link(camera)
    annotations.camera = camera
    setattr(scene, SCENE_PROPERTY, annotations)
    sync(scene, annotations)
    return annotations


def annotation_material():
    """An emission material showing each point's `yt_annotation_color`."""
    material = bpy.data.materials.get(MATERIAL_NAME)
    if material is not None:
        return material
    material = bpy.data.materials.new(MATERIAL_NAME)
    nodes, link = material.node_tree.nodes, material.node_tree.links.new
    nodes.clear()
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = COLOR_ATTRIBUTE
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    link(attribute.outputs["Color"], emission.inputs["Color"])
    link(emission.outputs["Emission"], output.inputs["Surface"])
    arrange(material.node_tree)
    return material


class _Builder:
    """Small helpers for wiring a Geometry Nodes tree."""

    def __init__(self, tree):
        self.tree, self.nodes = tree, tree.nodes

    def link(self, source, target):
        self.tree.links.new(source, target)

    def node(self, kind, inputs=None, **settings):
        node = self.nodes.new(kind)
        for name, value in settings.items():
            setattr(node, name, value)
        for key, source in (inputs or {}).items():
            socket = node.inputs[key]
            if isinstance(source, bpy.types.NodeSocket):
                self.link(source, socket)
            elif source is not None:
                socket.default_value = source
        return node

    def math(self, operation, *inputs):
        return self.node("ShaderNodeMath", dict(enumerate(inputs)), operation=operation).outputs[0]

    def xyz(self, x=0.0, y=0.0, z=0.0):
        return self.node("ShaderNodeCombineXYZ", {"X": x, "Y": y, "Z": z}).outputs[0]

    def sample(self, geometry, attribute, data_type, index):
        named = self.node("GeometryNodeInputNamedAttribute", {"Name": attribute}, data_type=data_type)
        sample = self.node("GeometryNodeSampleIndex", {"Geometry": geometry, "Index": index}, data_type=data_type, domain="POINT")
        self.link(named.outputs["Attribute"], sample.inputs["Value"])
        return sample.outputs["Value"]

    def store(self, geometry, name, data_type, value):
        return self.node(
            "GeometryNodeStoreNamedAttribute", {"Geometry": geometry, "Name": name, "Value": value},
            data_type=data_type, domain="POINT",
        ).outputs["Geometry"]

    def translate(self, geometry, offset):
        return self.node("GeometryNodeTransform", {"Geometry": geometry, "Translation": offset}).outputs["Geometry"]

    def text(self, string, size, offset):
        """Filled, centered text hanging below `offset`."""
        curves = self.node(
            "GeometryNodeStringToCurves", {"String": string, "Size": size, "Align X": "Center", "Align Y": "Top"},
        )
        real = self.node("GeometryNodeRealizeInstances", {"Geometry": curves.outputs["Curve Instances"]})
        fill = self.node("GeometryNodeFillCurve", {"Curve": real.outputs["Geometry"]})
        return self.translate(fill.outputs["Mesh"], offset)

    def join(self, *geometries):
        join = self.node("GeometryNodeJoinGeometry")
        for geometry in geometries:
            self.link(geometry, join.inputs["Geometry"])
        return join.outputs["Geometry"]


_INPUTS = (
    # name, socket type, default, description
    ("Volume", "NodeSocketObject", None, "The volume whose Geometry Nodes transfer function to show"),
    ("Width", "NodeSocketFloat", 0.6, "Length of the bar, in frame widths"),
    ("Height", "NodeSocketFloat", 0.03, "Thickness of the bar, in frame widths"),
    ("Ticks Every", "NodeSocketFloat", 1.0, "Spacing of the ticks, in decades (log10 units)"),
    ("Decimals", "NodeSocketInt", 0, "Decimal places of the tick labels"),
    ("Label Size", "NodeSocketFloat", 0.02, "Height of the tick labels, in frame widths"),
    ("Title", "NodeSocketString", "", "Text under the tick labels"),
    ("Text Color", "NodeSocketColor", (1.0, 1.0, 1.0, 1.0), "Color of the ticks, labels, title and opacity line"),
    ("Show Opacity", "NodeSocketBool", True, "Draw the opacity above the bar"),
    ("Opacity Height", "NodeSocketFloat", 0.05, "Height of the opacity line's full scale"),
    ("Opacity Decades", "NodeSocketFloat", 0.0, "Plot opacity over this many decades; 0 plots it linearly"),
    ("Line Width", "NodeSocketFloat", 0.002, "Width of the ticks and the opacity line"),
)


def _new_group(name, inputs):
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for socket_name, socket_type, default, description in inputs:
        item = tree.interface.new_socket(socket_name, in_out="INPUT", socket_type=socket_type)
        item.description = description
        if default is not None:
            item.default_value = default
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return tree


def colorbar_node_group():
    """The "yt Colorbar" group, made if the file has none: a bar of the
    volume's transfer function colors from the object's origin to +x, ticks
    and log10 labels under it, a title, and its opacity above it.
    """
    tree = bpy.data.node_groups.get(COLORBAR_GROUP)
    if tree is not None:
        return tree
    tree = _new_group(COLORBAR_GROUP, _INPUTS)
    b = _Builder(tree)
    inputs = b.node("NodeGroupInput").outputs
    width, height, size = inputs["Width"], inputs["Height"], inputs["Label Size"]

    info = b.node("GeometryNodeObjectInfo", {"Object": inputs["Volume"]}, transform_space="ORIGINAL")
    table = b.node("GeometryNodeInstancesToPoints", {"Instances": info.outputs["Geometry"]}).outputs["Points"]
    count = b.node("GeometryNodeAttributeDomainSize", {"Geometry": table}, component="POINTCLOUD").outputs["Point Count"]
    last = b.math("SUBTRACT", count, 1.0)
    position = b.node("GeometryNodeInputPosition").outputs["Position"]
    x = b.node("ShaderNodeSeparateXYZ", {"Vector": position}).outputs["X"]
    index = b.node("GeometryNodeInputIndex").outputs["Index"]

    # The bar: one grid column per table entry.
    grid = b.node("GeometryNodeMeshGrid", {"Size X": width, "Size Y": height, "Vertices X": count, "Vertices Y": 2})
    bar = b.translate(grid.outputs["Mesh"], b.xyz(b.math("MULTIPLY", width, 0.5), b.math("MULTIPLY", height, 0.5)))
    column = b.math("ROUND", b.math("MULTIPLY", b.math("DIVIDE", x, width), last))
    bar = b.store(bar, COLOR_ATTRIBUTE, "FLOAT_COLOR", b.sample(table, "yt_tf_color", "FLOAT_COLOR", column))

    # Opacity above the bar, linear or over Opacity Decades.
    alpha = b.sample(table, "yt_tf_alpha", "FLOAT", index)
    decades = inputs["Opacity Decades"]
    logarithmic = b.math("MULTIPLY", b.math("ADD", b.math("DIVIDE", b.math("LOGARITHM", alpha, 10.0), decades), 1.0),
                         b.math("GREATER_THAN", alpha, 0.0))
    is_log = b.math("GREATER_THAN", decades, 0.0)
    level = b.math("ADD", alpha, b.math("MULTIPLY", is_log, b.math("SUBTRACT", logarithmic, alpha)))
    level = b.node("ShaderNodeClamp", {"Value": level}).outputs[0]
    line = b.node("GeometryNodeMeshLine", {"Count": count, "Offset": (1.0, 0.0, 0.0)}).outputs["Mesh"]
    above = b.math("MULTIPLY_ADD", height, 1.25, b.math("MULTIPLY", level, inputs["Opacity Height"]))
    line = b.node("GeometryNodeSetPosition", {
        "Geometry": line, "Position": b.xyz(b.math("MULTIPLY", b.math("DIVIDE", index, last), width), above),
    }).outputs["Geometry"]
    profile = b.node("GeometryNodeCurvePrimitiveCircle", {"Resolution": 6, "Radius": b.math("MULTIPLY", inputs["Line Width"], 0.5)})
    curve = b.node("GeometryNodeMeshToCurve", {"Mesh": line}).outputs["Curve"]
    tube = b.node("GeometryNodeCurveToMesh", {"Curve": curve, "Profile Curve": profile.outputs["Curve"]}).outputs["Mesh"]
    opacity = b.node("GeometryNodeSwitch", {"Switch": inputs["Show Opacity"], "True": tube}, input_type="GEOMETRY").outputs[0]

    # Ticks at multiples of Ticks Every within the table's log10 range.
    low = b.sample(table, "yt_tf_log10_value", "FLOAT", 0)
    high = b.sample(table, "yt_tf_log10_value", "FLOAT", last)
    step = b.math("MAXIMUM", inputs["Ticks Every"], 1e-3)
    span = b.math("SUBTRACT", high, low)
    first = b.math("MULTIPLY", b.math("CEIL", b.math("DIVIDE", low, step)), step)
    ticks = b.math("ADD", b.math("FLOOR", b.math("ADD", b.math("DIVIDE", b.math("SUBTRACT", high, first), step), 1e-4)), 1.0)
    ticks = b.math("MULTIPLY", b.math("MAXIMUM", ticks, 0.0), b.math("GREATER_THAN", span, 0.0))
    points = b.node("GeometryNodeMeshLine", {"Count": ticks, "Start Location": b.xyz(first), "Offset": b.xyz(step)}).outputs["Mesh"]
    points = b.store(points, TICK_ATTRIBUTE, "FLOAT", x)
    points = b.node("GeometryNodeSetPosition", {
        "Geometry": points, "Position": b.xyz(b.math("MULTIPLY", b.math("DIVIDE", b.math("SUBTRACT", x, low), span), width)),
    }).outputs["Geometry"]
    tick_length = b.math("MULTIPLY", height, 0.4)
    mark = b.node("GeometryNodeMeshGrid", {"Size X": inputs["Line Width"], "Size Y": tick_length, "Vertices X": 2, "Vertices Y": 2})
    mark = b.translate(mark.outputs["Mesh"], b.xyz(0.0, b.math("MULTIPLY", tick_length, -0.5)))
    marks = b.node("GeometryNodeInstanceOnPoints", {"Points": points, "Instance": mark}).outputs["Instances"]
    marks = b.node("GeometryNodeRealizeInstances", {"Geometry": marks}).outputs["Geometry"]

    # A label under each tick.
    zone_in = b.node("GeometryNodeForeachGeometryElementInput", {"Geometry": points})
    zone_out = b.node("GeometryNodeForeachGeometryElementOutput")
    zone_in.pair_with_output(zone_out)
    zone_out.domain = "POINT"
    zone_out.input_items.new("FLOAT", "Value")
    zone_out.input_items.new("FLOAT", "X")
    named = b.node("GeometryNodeInputNamedAttribute", {"Name": TICK_ATTRIBUTE}, data_type="FLOAT")
    b.link(named.outputs["Attribute"], zone_in.inputs["Value"])
    b.link(x, zone_in.inputs["X"])
    label = b.node("FunctionNodeValueToString", {"Value": zone_in.outputs["Value"], "Decimals": inputs["Decimals"]}).outputs[0]
    below = b.math("SUBTRACT", b.math("MULTIPLY", tick_length, -1.0), b.math("MULTIPLY", size, 0.3))
    b.link(b.text(label, size, b.xyz(zone_in.outputs["X"], below)), zone_out.inputs["Geometry"])
    labels = zone_out.outputs[2]  # the generated geometry

    title_y = b.math("SUBTRACT", below, b.math("MULTIPLY", size, 1.6))
    title = b.text(inputs["Title"], b.math("MULTIPLY", size, 1.15), b.xyz(b.math("MULTIPLY", width, 0.5), title_y))

    text = b.store(b.join(opacity, marks, labels, title), COLOR_ATTRIBUTE, "FLOAT_COLOR", inputs["Text Color"])
    material = b.node("GeometryNodeSetMaterial", {"Geometry": b.join(bar, text)})
    material.inputs["Material"].default_value = annotation_material()
    b.link(material.outputs["Geometry"], b.node("NodeGroupOutput").inputs["Geometry"])
    arrange(tree)
    return tree


_VIEW_INPUTS = (
    ("Units Object", "NodeSocketObject", None, "The volume's units object (Units for Geometry Nodes)"),
    ("Unit", "NodeSocketString", "unitary", "A unit listed in the volume's Data Units, or unitary"),
    ("Use Point", "NodeSocketBool", False, "Ask about Point instead of the volume's zoom target or middle"),
    ("Point", "NodeSocketVector", (0.0, 0.0, 0.0), "A position in the main scene, e.g. an Object Info's Location"),
)
_VIEW_OUTPUTS = (
    ("Screen Position", "NodeSocketVector", "Where the point is on the frame, in the space of the object these nodes run on"),
    ("Frame per Unit", "NodeSocketFloat", "On-frame length of one Unit at the point's depth; multiply a length in Unit by it"),
    ("Frame Width", "NodeSocketFloat", "The frame's width in Unit at the point's depth"),
    ("Depth", "NodeSocketFloat", "The point's distance along the camera's view axis, in scene units"),
    ("Scene per Unit", "NodeSocketFloat", "Scene units per Unit"),
    ("In Front", "NodeSocketBool", "Whether the point is in front of the camera"),
)


def _vector(b, operation, *values):
    node = b.node("ShaderNodeVectorMath", dict(enumerate(values)), operation=operation)
    return node.outputs["Value" if operation == "DOT_PRODUCT" else "Vector"]


def _joined(b, *strings):
    join = b.node("GeometryNodeStringJoin")
    for string in reversed(strings):  # a multi-input lists its newest link first
        if isinstance(string, str):
            string = b.node("FunctionNodeInputString", string=string).outputs[0]
        b.link(string, join.inputs["Strings"])
    return join.outputs[0]


def view_node_group():
    """The "yt Annotation View" group, made if the file has none: where a
    main-scene point lands on the frame, and how long one Unit is there.
    """
    tree = bpy.data.node_groups.get(VIEW_GROUP)
    if tree is not None:
        return tree
    tree = bpy.data.node_groups.new(VIEW_GROUP, "GeometryNodeTree")
    tree.description = "Where a main-scene point lands on the annotation frame, and how long one Unit is there"
    for name, socket_type, default, description in _VIEW_INPUTS:
        item = tree.interface.new_socket(name, in_out="INPUT", socket_type=socket_type)
        item.description = description
        if default is not None:
            item.default_value = default
    for name, socket_type, description in _VIEW_OUTPUTS:
        tree.interface.new_socket(name, in_out="OUTPUT", socket_type=socket_type).description = description
    b = _Builder(tree)
    inputs = b.node("NodeGroupInput").outputs
    outputs = b.node("NodeGroupOutput").inputs

    geometry = b.node("GeometryNodeObjectInfo", {"Object": inputs["Units Object"]}, transform_space="ORIGINAL").outputs["Geometry"]

    def attribute(name, data_type="FLOAT"):
        return b.sample(geometry, name, data_type, 0)

    camera = attribute(units.CAMERA_TO_WORLD, "FLOAT4X4")

    def axis(direction):
        node = b.node("FunctionNodeTransformDirection", {"Direction": direction, "Transform": camera})
        return _vector(b, "NORMALIZE", node.outputs[0])

    point = b.node("GeometryNodeSwitch", {
        "Switch": inputs["Use Point"], "False": attribute(units.VIEW_CENTER, "FLOAT_VECTOR"), "True": inputs["Point"],
    }, input_type="VECTOR").outputs[0]
    eye = b.node("FunctionNodeTransformPoint", {"Vector": (0.0, 0.0, 0.0), "Transform": camera}).outputs[0]
    relative = _vector(b, "SUBTRACT", point, eye)
    depth = _vector(b, "DOT_PRODUCT", relative, axis((0.0, 0.0, -1.0)))
    span = b.math("MULTIPLY_ADD", depth, attribute(units.WIDTH_PER_DEPTH), attribute(units.WIDTH_ORTHO))  # frame width, scene units
    per_unit = attribute(_joined(b, "scene_per_", inputs["Unit"]))

    # Frame coordinates: x 0 to 1 across, y 0 to frame_height up; shifting the camera moves the frame, not the point.
    shift = b.node("ShaderNodeSeparateXYZ", {"Vector": attribute(units.FRAME_SHIFT, "FLOAT_VECTOR")}).outputs
    x = b.math("SUBTRACT", b.math("ADD", b.math("DIVIDE", _vector(b, "DOT_PRODUCT", relative, axis((1.0, 0.0, 0.0))), span), 0.5), shift["X"])
    y = b.math("SUBTRACT", b.math("MULTIPLY_ADD", attribute(units.FRAME_HEIGHT), 0.5, b.math("DIVIDE", _vector(b, "DOT_PRODUCT", relative, axis((0.0, 1.0, 0.0))), span)),
               shift["Y"])
    self_info = b.node("GeometryNodeObjectInfo", {"Object": b.node("GeometryNodeSelfObject").outputs[0]})
    to_local = b.node("FunctionNodeInvertMatrix", {"Matrix": self_info.outputs["Transform"]}).outputs["Matrix"]
    screen = b.node("FunctionNodeTransformPoint", {"Vector": b.xyz(x, y), "Transform": to_local}).outputs[0]

    b.link(screen, outputs["Screen Position"])
    b.link(b.math("DIVIDE", per_unit, span), outputs["Frame per Unit"])
    b.link(b.math("DIVIDE", span, per_unit), outputs["Frame Width"])
    b.link(depth, outputs["Depth"])
    b.link(per_unit, outputs["Scene per Unit"])
    b.link(b.math("GREATER_THAN", depth, 0.0), outputs["In Front"])
    arrange(tree)
    return tree


_FOV_INPUTS = (
    ("Units Object", "NodeSocketObject", None, "The volume's units object (Units for Geometry Nodes)"),
    ("Unit", "NodeSocketString", "unitary", "A unit listed in the volume's Data Units, or unitary"),
    ("Length", "NodeSocketFloat", 0.0, "Length in Unit, which can be keyframed; 0 picks a round length up to Auto Max"),
    ("Auto Max", "NodeSocketFloat", 0.2, "With Length 0: the largest 1, 2 or 5 x 10^n at most this many frame widths"),
    ("Line Width", "NodeSocketFloat", 0.004, "Thickness of the bar, in frame widths"),
    ("Label Size", "NodeSocketFloat", 0.02, "Height of the label, in frame widths"),
    ("Text Color", "NodeSocketColor", (1.0, 1.0, 1.0, 1.0), "Color of the bar and label"),
)


def fov_node_group():
    """The "yt Field of View" group, made if the file has none: a labeled
    scale bar from the object's origin to +x, sized at the volume's zoom
    target or middle; also an example of "yt Annotation View".
    """
    tree = bpy.data.node_groups.get(FOV_GROUP)
    if tree is not None:
        return tree
    tree = _new_group(FOV_GROUP, _FOV_INPUTS)
    b = _Builder(tree)
    inputs = b.node("NodeGroupInput").outputs
    line_width, size, unit = inputs["Line Width"], inputs["Label Size"], inputs["Unit"]
    view = b.node("GeometryNodeGroup", {}, node_tree=view_node_group())
    b.link(inputs["Units Object"], view.inputs["Units Object"])
    b.link(unit, view.inputs["Unit"])
    view = view.outputs

    def log10(value):
        return b.math("FLOOR", b.math("LOGARITHM", value, 10.0))

    # Length, or the largest 1, 2 or 5 x 10^n at most Auto Max of the frame.
    longest = b.math("MULTIPLY", view["Frame Width"], inputs["Auto Max"])
    exponent = log10(longest)
    power = b.math("POWER", 10.0, exponent)
    mantissa = b.math("DIVIDE", longest, power)
    step = b.math("ADD", b.math("ADD", 1.0, b.math("GREATER_THAN", mantissa, 1.9999)), b.math("MULTIPLY", b.math("GREATER_THAN", mantissa, 4.9999), 3.0))
    auto = b.math("LESS_THAN", inputs["Length"], 1e-30)
    value = b.node("GeometryNodeSwitch", {"Switch": auto, "False": inputs["Length"], "True": b.math("MULTIPLY", step, power)}, input_type="FLOAT").outputs[0]
    length = b.math("MULTIPLY", value, view["Frame per Unit"])

    grid = b.node("GeometryNodeMeshGrid", {"Size X": length, "Size Y": line_width, "Vertices X": 2, "Vertices Y": 2})
    bar = b.translate(grid.outputs["Mesh"], b.xyz(b.math("MULTIPLY", length, 0.5), b.math("MULTIPLY", line_width, 0.5)))

    # A round length exactly, a keyframed one to 2 figures.
    decimals = b.node("GeometryNodeSwitch", {
        "Switch": auto, "False": b.math("MAXIMUM", b.math("SUBTRACT", 1.0, log10(value)), 0.0), "True": b.math("MAXIMUM", b.math("MULTIPLY", exponent, -1.0), 0.0),
    }, input_type="FLOAT").outputs[0]
    number = b.node("FunctionNodeValueToString", {"Value": value, "Decimals": decimals}).outputs[0]
    label = b.text(_joined(b, number, " ", unit), size, b.xyz(b.math("MULTIPLY", length, 0.5), b.math("MULTIPLY", size, -0.3)))

    # Nothing when the depth or units are unknown.
    valid = b.math("GREATER_THAN", view["Frame per Unit"], 0.0)
    shown = b.node("GeometryNodeSwitch", {"Switch": valid, "True": b.join(bar, label)}, input_type="GEOMETRY").outputs[0]
    shown = b.store(shown, COLOR_ATTRIBUTE, "FLOAT_COLOR", inputs["Text Color"])
    material = b.node("GeometryNodeSetMaterial", {"Geometry": shown})
    material.inputs["Material"].default_value = annotation_material()
    b.link(material.outputs["Geometry"], b.node("NodeGroupOutput").inputs["Geometry"])
    arrange(tree)
    return tree


_TEMPLATE_INPUTS = (
    ("Units Object", "NodeSocketObject", None, "The volume's units object (Units for Geometry Nodes)"),
    ("Unit", "NodeSocketString", "unitary", "A unit listed in the volume's Data Units, or unitary"),
    ("Size", "NodeSocketFloat", 1.0, "Diameter of the dot, in Unit"),
    ("Color", "NodeSocketColor", (1.0, 1.0, 1.0, 1.0), "Scene-linear color"),
)


def annotation_template():
    """A new Geometry Nodes tree to build an annotation from: a dot Size
    across at the volume's zoom target or middle, through "yt Annotation View".
    """
    tree = _new_group(TEMPLATE_GROUP, _TEMPLATE_INPUTS)
    b = _Builder(tree)
    inputs = b.node("NodeGroupInput").outputs
    view = b.node("GeometryNodeGroup", {}, node_tree=view_node_group())
    b.link(inputs["Units Object"], view.inputs["Units Object"])
    b.link(inputs["Unit"], view.inputs["Unit"])
    radius = b.math("MULTIPLY", b.math("MULTIPLY", inputs["Size"], view.outputs["Frame per Unit"]), 0.5)
    dot = b.node("GeometryNodeMeshCircle", {"Vertices": 48, "Radius": radius}, fill_type="NGON").outputs["Mesh"]
    dot = b.translate(dot, view.outputs["Screen Position"])
    shown = b.node("GeometryNodeSwitch", {"Switch": view.outputs["In Front"], "True": dot}, input_type="GEOMETRY").outputs[0]
    shown = b.store(shown, COLOR_ATTRIBUTE, "FLOAT_COLOR", inputs["Color"])
    material = b.node("GeometryNodeSetMaterial", {"Geometry": shown})
    material.inputs["Material"].default_value = annotation_material()
    b.link(material.outputs["Geometry"], b.node("NodeGroupOutput").inputs["Geometry"])
    arrange(tree)
    return tree


def _set_input(modifier, name, value):
    identifier = modifier.node_group.interface.items_tree[name].identifier
    getattr(modifier.properties.inputs, identifier).value = value


def default_title(volume):
    """`log₁₀ <field> (<units>)` for a volume, as far as it's known."""
    from . import yt_idv_renderer

    params = volume.volume_override.to_custom_params()
    field = params.get("yt_idv_field") or ""
    field = field.split(",")[-1].strip() if field else "field"
    fit = yt_idv_renderer.gn_tf_fit(volume.volume_override.dataset_uuid)
    units = f" ({fit['units']})" if fit and fit.get("units") else ""
    return f"log₁₀ {field}{units}"


def _annotations_with(annotations, group_name):
    return [o for o in annotations.objects if any(
        m.type == "NODES" and m.node_group is not None and m.node_group.name.startswith(group_name) for m in o.modifiers
    )]


def colorbars(annotations):
    return _annotations_with(annotations, COLORBAR_GROUP)


def add_colorbar(scene, volume):
    """A colorbar of `volume`'s transfer function in `scene`'s annotation
    scene (made if needed), below any others there.
    """
    annotations = add_annotation_scene(scene)
    mesh = bpy.data.meshes.new(f"{volume.name} Colorbar")
    obj = bpy.data.objects.new(mesh.name, mesh)
    obj.location = (0.2, 0.08 + 0.15 * len(colorbars(annotations)), 0.0)
    annotations.collection.objects.link(obj)
    modifier = obj.modifiers.new("yt Colorbar", "NODES")
    modifier.node_group = colorbar_node_group()
    _set_input(modifier, "Volume", volume)
    _set_input(modifier, "Title", default_title(volume))
    return obj


def default_unit(volume):
    """The first of `volume`'s Data Units its units object knows, else unitary."""
    group = volume.get(units.KEY, {})
    return next((u for u in units.data_units(volume) if group.get(f"scene_per_{u}")), "unitary")


def add_field_of_view(scene, volume):
    """A field-of-view scale bar for `volume` in `scene`'s annotation scene
    (made if needed), at the bottom right. `volume` needs a units object.
    """
    units_object = volume.volume_override.units_object
    if units_object is None:
        raise ValueError(f"{volume.name!r} has no units object; add one with Units for Geometry Nodes")
    annotations = add_annotation_scene(scene)
    mesh = bpy.data.meshes.new(f"{volume.name} Field of View")
    obj = bpy.data.objects.new(mesh.name, mesh)
    obj.location = (0.7, 0.1 + 0.12 * len(_annotations_with(annotations, FOV_GROUP)), 0.0)
    annotations.collection.objects.link(obj)
    modifier = obj.modifiers.new(FOV_GROUP, "NODES")
    modifier.node_group = fov_node_group()
    _set_input(modifier, "Units Object", units_object)
    _set_input(modifier, "Unit", default_unit(volume))
    return obj


def add_annotation(scene, volume):
    """An annotation object in `scene`'s annotation scene (made if needed),
    with its own copy of the starter tree to edit. `volume` needs a units object.
    """
    units_object = volume.volume_override.units_object
    if units_object is None:
        raise ValueError(f"{volume.name!r} has no units object; add one with Units for Geometry Nodes")
    annotations = add_annotation_scene(scene)
    obj = bpy.data.objects.new(f"{volume.name} Annotation", bpy.data.meshes.new(f"{volume.name} Annotation"))
    annotations.collection.objects.link(obj)
    modifier = obj.modifiers.new(TEMPLATE_GROUP, "NODES")
    modifier.node_group = annotation_template()
    _set_input(modifier, "Units Object", units_object)
    _set_input(modifier, "Unit", default_unit(volume))
    return obj


class VOLUMEOVERRIDE_OT_add_colorbar(bpy.types.Operator):
    """Add a colorbar of this volume's transfer function to the scene's annotation scene"""

    bl_idname = "volume_override.add_colorbar"
    bl_label = "Add Colorbar"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and getattr(obj, "volume_override", None) is not None and obj.volume_override.is_volume_override

    def execute(self, context):
        obj = add_colorbar(context.scene, context.object)
        self.report({"INFO"}, f"Added {obj.name!r} to {annotation_scene(context.scene).name!r}; open that scene to place it")
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_add_field_of_view(bpy.types.Operator):
    """Add a scale bar estimating this volume's field of view to the scene's annotation scene"""

    bl_idname = "volume_override.add_field_of_view"
    bl_label = "Add Field of View"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        props = getattr(obj, "volume_override", None) if obj is not None else None
        if props is None or not props.is_volume_override:
            return False
        if props.units_object is None:
            cls.poll_message_set("Add a units object first (Units for Geometry Nodes)")
            return False
        return True

    def execute(self, context):
        obj = add_field_of_view(context.scene, context.object)
        self.report({"INFO"}, f"Added {obj.name!r} to {annotation_scene(context.scene).name!r}; open that scene to place it")
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_add_annotation(bpy.types.Operator):
    """Add an annotation with its own Geometry Nodes tree to edit, wired to this volume's units object"""

    bl_idname = "volume_override.add_annotation"
    bl_label = "Add Annotation"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return VOLUMEOVERRIDE_OT_add_field_of_view.poll(context)

    def execute(self, context):
        obj = add_annotation(context.scene, context.object)
        self.report({"INFO"}, f"Added {obj.name!r} to {annotation_scene(context.scene).name!r}; edit its yt Annotation nodes there")
        return {"FINISHED"}


_CLASSES = (VOLUMEOVERRIDE_OT_add_colorbar, VOLUMEOVERRIDE_OT_add_field_of_view, VOLUMEOVERRIDE_OT_add_annotation)


def register():
    bpy.types.Scene.yt_annotation_scene = bpy.props.PointerProperty(
        type=bpy.types.Scene, name="Annotation Scene",
        description="A flat scene laid over the finished frame (x 0-1 across it); holds colorbars and scale bars",
    )
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, SCENE_PROPERTY):
        del bpy.types.Scene.yt_annotation_scene
