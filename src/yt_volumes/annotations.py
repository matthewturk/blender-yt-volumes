"""The annotation scene: a flat scene laid over the finished frame, holding
colorbars that read volumes' transfer functions through Geometry Nodes.

Its orthographic camera puts x from 0 to 1 across the frame and y from 0 to
height/width up it, origin at the bottom left; build annotations on z = 0.
Pass 2 renders it with Cycles through a Render Layers node, at the main
scene's frame and resolution, and lays it over the volumes.
"""

import bpy

from .node_layout import arrange

SCENE_PROPERTY = "yt_annotation_scene"
COLORBAR_GROUP = "yt Colorbar"
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
            else:
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


def colorbar_node_group():
    """The "yt Colorbar" group, made if the file has none: a bar of the
    volume's transfer function colors from the object's origin to +x, ticks
    and log10 labels under it, a title, and its opacity above it.
    """
    tree = bpy.data.node_groups.get(COLORBAR_GROUP)
    if tree is not None:
        return tree
    tree = bpy.data.node_groups.new(COLORBAR_GROUP, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    for name, socket_type, default, description in _INPUTS:
        item = tree.interface.new_socket(name, in_out="INPUT", socket_type=socket_type)
        item.description = description
        if default is not None:
            item.default_value = default
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
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


def colorbars(annotations):
    return [o for o in annotations.objects if any(
        m.type == "NODES" and m.node_group is not None and m.node_group.name.startswith(COLORBAR_GROUP) for m in o.modifiers
    )]


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


def register():
    bpy.types.Scene.yt_annotation_scene = bpy.props.PointerProperty(
        type=bpy.types.Scene, name="Annotation Scene",
        description="A flat scene laid over the finished frame (x 0-1 across it); holds colorbars",
    )
    bpy.utils.register_class(VOLUMEOVERRIDE_OT_add_colorbar)


def unregister():
    if VOLUMEOVERRIDE_OT_add_colorbar.is_registered:
        bpy.utils.unregister_class(VOLUMEOVERRIDE_OT_add_colorbar)
    if hasattr(bpy.types.Scene, SCENE_PROPERTY):
        del bpy.types.Scene.yt_annotation_scene
