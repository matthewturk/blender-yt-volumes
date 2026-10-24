"""A transfer function drawn in Geometry Nodes on the volume Empty itself.

A Geometry Nodes modifier on the volume object outputs, alongside its input
geometry, instances in order from the low to the high end of the transfer
function's value range, carrying a `yt_tf_color` color attribute and
optionally a `yt_tf_alpha` float attribute (else the color's alpha). They are
instances of nothing, so nothing draws or renders. Points or vertices with
the attributes work too. `add_transfer_function_modifier` makes a starter
group with a Color Ramp and an opacity Float Curve (linear, or log over a
number of decades), plus a faceless outline
of the Empty's cube or sphere: with geometry from a modifier, the viewport no
longer draws the Empty itself, and edges alone don't render. Its points also
carry the data's distribution, for viewer node groups to plot:
`yt_tf_volume_fraction` and `yt_tf_mass_fraction` (the share of the volume and
of the mass in each point's bin) and `yt_tf_log10_value` (the field there, in
its own units). `update_distribution` writes them, after a render, into an
image the group samples.
"""

import bpy
import numpy as np

COLOR_ATTRIBUTE = "yt_tf_color"
ALPHA_ATTRIBUTE = "yt_tf_alpha"
COLOR_NODE_NAME = "TF Color"
ALPHA_NODE_NAME = "TF Opacity"
DECADES_NODE_NAME = "TF Opacity Decades"
DISTRIBUTION_NODE_NAME = "TF Distribution"
# Image channels -> point attributes.
DISTRIBUTION_ATTRIBUTES = (("X", "yt_tf_volume_fraction"), ("Y", "yt_tf_mass_fraction"), ("Z", "yt_tf_log10_value"))
MODIFIER_NAME = "yt Transfer Function"
NBINS = 4096  # points the ramp and curve are evaluated at: fine enough for thin layers
SIZE_INPUT = "Display Size"
TYPE_INPUT = "Display Type"
_SPHERE = 6  # Object.empty_display_type's stored value for 'SPHERE'
_CIRCLE_VERTICES = 32
MAX_RAMP_STOPS = 32  # Blender's Color Ramp limit
LOG_OPACITY_FLOOR = 1e-3  # curve heights at or below this are fully transparent in log mode

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
    """A new Geometry Nodes group: its input geometry, joined with `nbins`
    empty instances whose x in [0, 1] runs through a Color Ramp and a Float
    Curve into the two attributes.
    """
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    # Driven from the Empty by add_transfer_function_modifier.
    for socket_name, socket_type in ((SIZE_INPUT, "NodeSocketFloat"), (TYPE_INPUT, "NodeSocketInt")):
        tree.interface.new_socket(socket_name, in_out="INPUT", socket_type=socket_type).hide_in_modifier = True
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, link = tree.nodes, tree.links.new

    group_input = nodes.new("NodeGroupInput")
    line = nodes.new("GeometryNodeMeshLine")
    line.inputs["Count"].default_value = nbins
    line.inputs["Offset"].default_value = (1.0 / (nbins - 1), 0.0, 0.0)
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
    points.new(0.3, 0.0)  # transparent over the sparsest 30%, so the voids stay clear
    curve.mapping.update()

    store_color = nodes.new("GeometryNodeStoreNamedAttribute")
    store_color.data_type, store_color.domain = "FLOAT_COLOR", "POINT"
    store_color.inputs["Name"].default_value = COLOR_ATTRIBUTE
    store_alpha = nodes.new("GeometryNodeStoreNamedAttribute")
    store_alpha.data_type, store_alpha.domain = "FLOAT", "POINT"
    store_alpha.inputs["Name"].default_value = ALPHA_ATTRIBUTE
    # No Instance input: instances of nothing, so the viewport shows no vertices.
    instances = nodes.new("GeometryNodeInstanceOnPoints")
    join = nodes.new("GeometryNodeJoinGeometry")
    output = nodes.new("NodeGroupOutput")

    link(position.outputs["Position"], value.inputs["Vector"])
    link(value.outputs["X"], ramp.inputs["Fac"])
    link(value.outputs["X"], curve.inputs["Value"])
    link(line.outputs["Mesh"], store_color.inputs["Geometry"])
    link(ramp.outputs["Color"], store_color.inputs["Value"])
    link(store_color.outputs["Geometry"], store_alpha.inputs["Geometry"])
    link(curve.outputs["Value"], store_alpha.inputs["Value"])
    add_opacity_scale(tree)
    distribution = _add_distribution(tree, value, name, nbins)
    link(store_alpha.outputs["Geometry"], distribution[0].inputs["Geometry"])
    link(distribution[-1].outputs["Geometry"], instances.inputs["Points"])
    outline = _add_outline(tree, group_input)
    link(group_input.outputs["Geometry"], join.inputs["Geometry"])
    link(outline.outputs["Output"], join.inputs["Geometry"])
    link(instances.outputs["Instances"], join.inputs["Geometry"])
    link(join.outputs["Geometry"], output.inputs["Geometry"])

    group_input.location = (800, 200)
    for column, node in enumerate((line, store_color, store_alpha, instances, join, output)):
        node.location = (200 * column, 0)
    for node, (x, y) in ((position, (-400, -250)), (value, (-200, -250)), (ramp, (0, -250)), (curve, (300, -450))):
        node.location = (x, y)
    return tree


def add_opacity_scale(tree):
    """Insert, after `tree`'s opacity curve, a Value node of decades: 0 keeps
    the curve's height as opacity, D > 0 makes height y opacity 10^(D (y - 1)).

    Returns the Value node; a no-op returning the existing one if present.
    """
    nodes, link = tree.nodes, tree.links.new
    if DECADES_NODE_NAME in nodes:
        return nodes[DECADES_NODE_NAME]
    curve = nodes[ALPHA_NODE_NAME]
    targets = [l.to_socket for l in tree.links if l.from_node == curve]
    decades = nodes.new("ShaderNodeValue")
    decades.name = decades.label = DECADES_NODE_NAME
    decades.outputs[0].default_value = 0.0

    def math(operation, a, b):
        node = nodes.new("ShaderNodeMath")
        node.operation = operation
        for socket, source in zip(node.inputs, (a, b)):
            if isinstance(source, float):
                socket.default_value = source
            else:
                link(source, socket)
        return node

    height = curve.outputs["Value"]
    exponent = math("MULTIPLY", math("SUBTRACT", height, 1.0).outputs[0], decades.outputs[0])
    power = math("POWER", 10.0, exponent.outputs[0])
    # A curve resting on 0 would otherwise give 10^-D, not transparent.
    visible = math("GREATER_THAN", height, LOG_OPACITY_FLOOR)
    log_alpha = math("MULTIPLY", power.outputs[0], visible.outputs[0])
    is_log = math("GREATER_THAN", decades.outputs[0], 0.0)
    switch = nodes.new("GeometryNodeSwitch")
    switch.input_type = "FLOAT"
    link(is_log.outputs[0], switch.inputs["Switch"])
    link(height, switch.inputs["False"])
    link(log_alpha.outputs[0], switch.inputs["True"])
    for socket in targets:
        link(switch.outputs[0], socket)

    x, y = curve.location
    decades.location = (x, y - 300)
    for column, node in enumerate((exponent.inputs[0].links[0].from_node, exponent, power, log_alpha, switch)):
        node.location = (x + 250 + 180 * column, y - 150)
    visible.location, is_log.location = (x + 610, y - 350), (x + 790, y - 450)
    return decades


def _height_to_opacity(y, decades):
    if decades <= 0:
        return y
    return np.where(y > LOG_OPACITY_FLOOR, 10.0 ** (decades * (y - 1.0)), 0.0)


def _opacity_to_height(alpha, decades):
    if decades <= 0:
        return alpha
    with np.errstate(divide="ignore"):
        return np.where(alpha > 0, np.clip(1.0 + np.log10(alpha) / decades, 0.0, 1.0), 0.0)


def set_opacity_decades(tree, decades):
    """Change `tree`'s opacity scale to `decades` (0 is linear), moving the
    curve's points so each keeps its opacity; opacities below 10^-decades
    become transparent.
    """
    node = add_opacity_scale(tree)
    old = node.outputs[0].default_value
    curve = tree.nodes[ALPHA_NODE_NAME]
    for point in curve.mapping.curves[0].points:
        alpha = _height_to_opacity(np.float64(point.location[1]), old)
        point.location = (point.location[0], float(_opacity_to_height(alpha, decades)))
    curve.mapping.update()
    node.outputs[0].default_value = decades
    tree.update_tag()


def load_colormap(ramp, name):
    """Fill Color Ramp node `ramp` from matplotlib colormap `name`: evenly
    sampled stops, or constant steps for a listed colormap of up to 32 colors.
    """
    import matplotlib

    try:
        import cmyt  # noqa: F401  registers yt's colormaps
    except ImportError:
        pass
    cmap = matplotlib.colormaps[name]
    color_ramp = ramp.color_ramp
    stepped = isinstance(cmap, matplotlib.colors.ListedColormap) and cmap.N <= MAX_RAMP_STOPS
    if stepped:
        positions = np.arange(cmap.N) / cmap.N
        srgb = np.asarray(cmap.colors, dtype=np.float64)[:, :3]
    else:
        positions = np.linspace(0.0, 1.0, MAX_RAMP_STOPS)
        srgb = cmap(positions)[:, :3]
    color_ramp.interpolation = "CONSTANT" if stepped else "LINEAR"
    elements = color_ramp.elements
    while len(elements) > len(positions):
        elements.remove(elements[-1])
    while len(elements) < len(positions):
        elements.new(1.0)
    # Set positions in increasing order: Blender keeps the elements sorted.
    for element, position, color in zip(elements, positions, _srgb_to_linear(srgb)):
        element.position = position
        element.color = (*color, 1.0)
    ramp.id_data.update_tag()


def colormap_names():
    import matplotlib

    try:
        import cmyt  # noqa: F401
    except ImportError:
        pass
    return sorted(matplotlib.colormaps, key=str.lower)


def _add_outline(tree, group_input):
    """Nodes for an edges-only cube (half-width Display Size) or, when Display
    Type is Sphere, three circles; returns the Switch node choosing between them.
    """
    nodes, link = tree.nodes, tree.links.new
    double = nodes.new("ShaderNodeMath")
    double.operation = "MULTIPLY"
    double.inputs[1].default_value = 2.0
    cube = nodes.new("GeometryNodeMeshCube")
    no_faces = nodes.new("GeometryNodeDeleteGeometry")
    no_faces.domain, no_faces.mode = "FACE", "ONLY_FACE"
    link(group_input.outputs[SIZE_INPUT], double.inputs[0])
    link(double.outputs["Value"], cube.inputs["Size"])
    link(cube.outputs["Mesh"], no_faces.inputs["Geometry"])

    circle = nodes.new("GeometryNodeMeshCircle")
    circle.fill_type = "NONE"
    circle.inputs["Vertices"].default_value = _CIRCLE_VERTICES
    link(group_input.outputs[SIZE_INPUT], circle.inputs["Radius"])
    circles = nodes.new("GeometryNodeJoinGeometry")
    for rotation in ((0.0, 0.0, 0.0), (1.5707963, 0.0, 0.0), (0.0, 1.5707963, 0.0)):
        turn = nodes.new("GeometryNodeTransform")
        turn.inputs["Rotation"].default_value = rotation
        link(circle.outputs["Mesh"], turn.inputs["Geometry"])
        link(turn.outputs["Geometry"], circles.inputs["Geometry"])

    is_sphere = nodes.new("FunctionNodeCompare")
    is_sphere.data_type, is_sphere.operation = "INT", "EQUAL"
    link(group_input.outputs[TYPE_INPUT], is_sphere.inputs["A"])
    is_sphere.inputs["B"].default_value = _SPHERE
    switch = nodes.new("GeometryNodeSwitch")
    switch.input_type = "GEOMETRY"
    link(is_sphere.outputs["Result"], switch.inputs["Switch"])
    link(no_faces.outputs["Geometry"], switch.inputs["False"])
    link(circles.outputs["Geometry"], switch.inputs["True"])
    for column, node in enumerate((double, cube, no_faces)):
        node.location = (900 + 200 * column, 500)
    circle.location, circles.location, is_sphere.location, switch.location = (1100, 750), (1300, 750), (1300, 950), (1500, 600)
    return switch


def _add_distribution(tree, value, name, nbins):
    """Store Named Attribute nodes, chained, putting the distribution image's
    channels on the points; returns them in order.
    """
    nodes, link = tree.nodes, tree.links.new
    image = bpy.data.images.new(f"{name} Distribution", nbins, 1, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    texture = nodes.new("GeometryNodeImageTexture")
    texture.name = texture.label = DISTRIBUTION_NODE_NAME
    texture.inputs["Image"].default_value = image
    texture.interpolation, texture.extension = "Closest", "EXTEND"
    # Point i sits at x = i / (nbins - 1); sample its pixel's center, not an edge
    # float32 can round onto the next pixel.
    center = nodes.new("ShaderNodeMath")
    center.operation = "MULTIPLY_ADD"
    center.inputs[1].default_value = (nbins - 1) / nbins
    center.inputs[2].default_value = 0.5 / nbins
    link(value.outputs["X"], center.inputs[0])
    at_x = nodes.new("ShaderNodeCombineXYZ")
    at_x.inputs["Y"].default_value = 0.5
    link(center.outputs["Value"], at_x.inputs["X"])
    link(at_x.outputs["Vector"], texture.inputs["Vector"])
    channels = nodes.new("ShaderNodeSeparateXYZ")
    link(texture.outputs["Color"], channels.inputs["Vector"])
    stores = []
    for channel, attribute in DISTRIBUTION_ATTRIBUTES:
        store = nodes.new("GeometryNodeStoreNamedAttribute")
        store.data_type, store.domain = "FLOAT", "POINT"
        store.inputs["Name"].default_value = attribute
        link(channels.outputs[channel], store.inputs["Value"])
        if stores:
            link(stores[-1].outputs["Geometry"], store.inputs["Geometry"])
        stores.append(store)
    center.location, at_x.location, texture.location, channels.location = (100, -700), (300, -700), (500, -700), (700, -700)
    for column, store in enumerate(stores):
        store.location = (900 + 200 * column, -500)
    return stores


def update_distribution(obj, points):
    """Write `points` (`yt_idv_renderer.gn_tf_fit`'s) into the image `obj`'s
    starter group samples; a no-op without one or if unchanged.
    """
    image = None
    for modifier in obj.modifiers:
        tree = getattr(modifier, "node_group", None) if modifier.type == "NODES" else None
        node = tree.nodes.get(DISTRIBUTION_NODE_NAME) if tree is not None else None
        if node is not None:
            image = node.inputs["Image"].default_value
            break
    if image is None:
        return
    width = image.size[0]
    pixels = np.ones((width, 4), dtype=np.float32)
    for column, (_, attribute) in enumerate(DISTRIBUTION_ATTRIBUTES):
        values = np.asarray(points[attribute.removeprefix("yt_tf_")], dtype=np.float64)
        pixels[:, column] = np.interp(np.linspace(0, 1, width), np.linspace(0, 1, len(values)), values)
    current = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(current)
    if current.shape == pixels.ravel().shape and np.array_equal(current, pixels.ravel()):
        return
    image.pixels.foreach_set(pixels.ravel())
    # Generated images lose edited pixels on reload unless packed; PNG would clamp them.
    image.file_format = "OPEN_EXR"
    image.pack()
    # New pixels alone don't re-evaluate the modifier.
    image.update()
    obj.update_tag()


def _drive_from(obj, modifier, socket_name, data_path):
    """Drive `modifier`'s input `socket_name` from `obj.<data_path>`, with a
    plain property driver (no Python, so it runs without script auto-run).
    """
    identifier = modifier.node_group.interface.items_tree[socket_name].identifier
    curve = obj.driver_add(getattr(modifier.properties.inputs, identifier).path_from_id("value"))
    curve.driver.type = "AVERAGE"
    variable = curve.driver.variables.new()
    variable.type = "SINGLE_PROP"
    variable.targets[0].id_type = "OBJECT"
    variable.targets[0].id = obj
    variable.targets[0].data_path = data_path


def add_transfer_function_modifier(obj):
    """Add a Geometry Nodes modifier with its own starter group to `obj`."""
    modifier = obj.modifiers.new(MODIFIER_NAME, "NODES")
    modifier.node_group = build_node_group()
    _drive_from(obj, modifier, SIZE_INPUT, "empty_display_size")
    _drive_from(obj, modifier, TYPE_INPUT, "empty_display_type")
    return modifier


def find_node(obj, name):
    """The node called `name` in `obj`'s Geometry Nodes modifiers, or None."""
    for modifier in obj.modifiers:
        tree = getattr(modifier, "node_group", None) if modifier.type == "NODES" else None
        node = tree.nodes.get(name) if tree is not None else None
        if node is not None:
            return node
    return None


def find_nodes(obj):
    """`(color_ramp_node, opacity_curve_node)` from `obj`'s Geometry Nodes
    modifiers, either None if absent.
    """
    return find_node(obj, COLOR_NODE_NAME), find_node(obj, ALPHA_NODE_NAME)


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
        components = (geometry.instances_pointcloud(), geometry.mesh, geometry.pointcloud)
    except TypeError:  # an Empty with no geometry-producing modifier
        components = ()
    for component in components:  # instances, mesh, point cloud
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


_colormap_items = []  # Blender needs the enum strings kept alive


def _colormap_enum_items(self, context):
    if not _colormap_items:
        try:
            _colormap_items.extend((name, name, "") for name in colormap_names())
        except ImportError:
            _colormap_items.append(("inferno", "inferno", ""))
    return _colormap_items


class VOLUMEOVERRIDE_OT_load_tf_colormap(bpy.types.Operator):
    """Replace the transfer function's Color Ramp with a matplotlib/yt colormap"""

    bl_idname = "volume_override.load_tf_colormap"
    bl_label = "Load Colormap"
    bl_options = {"REGISTER", "UNDO"}
    bl_property = "colormap"

    colormap: bpy.props.EnumProperty(name="Colormap", items=_colormap_enum_items)

    @classmethod
    def poll(cls, context):
        return context.object is not None and find_node(context.object, COLOR_NODE_NAME) is not None

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            load_colormap(find_node(context.object, COLOR_NODE_NAME), self.colormap)
        except (ImportError, KeyError) as error:
            self.report({"ERROR"}, f"Can't load colormap {self.colormap!r}: {error}")
            return {"CANCELLED"}
        context.object.update_tag()
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_set_opacity_curve_scale(bpy.types.Operator):
    """Make the opacity curve linear or log over some decades, keeping its points' opacities"""

    bl_idname = "volume_override.set_opacity_curve_scale"
    bl_label = "Opacity Curve Scale"
    bl_options = {"REGISTER", "UNDO"}

    decades: bpy.props.FloatProperty(
        name="Decades",
        description="Decades of opacity the curve's height spans: 0 is linear, 4 runs from 1e-4 to 1",
        min=0.0,
        soft_max=10.0,
    )

    @classmethod
    def poll(cls, context):
        return context.object is not None and find_node(context.object, ALPHA_NODE_NAME) is not None

    def invoke(self, context, event):
        node = find_node(context.object, DECADES_NODE_NAME)
        self.decades = node.outputs[0].default_value if node is not None else 0.0
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        set_opacity_decades(find_node(context.object, ALPHA_NODE_NAME).id_data, self.decades)
        context.object.update_tag()
        return {"FINISHED"}


CLASSES = (VOLUMEOVERRIDE_OT_add_transfer_function, VOLUMEOVERRIDE_OT_load_tf_colormap, VOLUMEOVERRIDE_OT_set_opacity_curve_scale)
