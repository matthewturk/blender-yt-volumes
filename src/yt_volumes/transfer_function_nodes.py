"""A transfer function drawn in Geometry Nodes on the volume Empty itself.

A Geometry Nodes modifier on the volume object outputs, alongside its input
geometry, instances in order from the low to the high end of the transfer
function's value range, carrying a `yt_tf_color` color attribute and
optionally a `yt_tf_alpha` float attribute (else the color's alpha). They are
instances of nothing, so nothing draws or renders. Points or vertices with
the attributes work too. `add_transfer_function_modifier` makes a starter
group of slots, each a nested group with a Color Ramp and an opacity Float
Curve (linear, or log over a number of decades); the modifier's keyframeable
Blend input cross-fades between neighbouring slots. It also adds a faceless outline
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

from .node_layout import arrange

COLOR_ATTRIBUTE = "yt_tf_color"
ALPHA_ATTRIBUTE = "yt_tf_alpha"
COLOR_NODE_NAME = "TF Color"
ALPHA_NODE_NAME = "TF Opacity"
DECADES_NODE_NAME = "TF Opacity Decades"
SLOT_PREFIX = "TF Slot "  # main-tree group nodes "TF Slot 1", "TF Slot 2", ... in blend order
BLEND_INPUT = "Blend"  # modifier input: 1 is the first slot, 2 the second, ...
_OLD_BLEND_NODE_NAME = "TF Blend"  # a Value node before Blend was a modifier input
COLORMAP_PROPERTY = "yt_colormap"  # on a slot's group: the colormap its ramp was loaded from
VALUE_NODE_NAME = "TF Value"
MIX_PREFIX = "TF Mix "  # generated nodes, rebuilt when slots change
MIX_EPSILON = 1e-6  # where every blended slot is transparent, color falls back to the plain cross-fade
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
COLORMAP_SAMPLES = 256
COLORMAP_TOLERANCE = 0.004  # sRGB, about 1/255: stop adding stops once this close
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


def _set_ramp(color_ramp, positions, linear_rgb):
    """Give `color_ramp` exactly these stops (positions increasing, colors scene-linear)."""
    elements = color_ramp.elements
    while len(elements) > len(positions):
        elements.remove(elements[-1])
    while len(elements) < len(positions):
        elements.new(0.0)
    # Blender re-sorts elements as positions change: park them all at 0, then
    # place them from the top down so none passes another.
    for element in elements:
        element.position = 0.0
    for element, position in zip(reversed(elements), reversed(list(positions))):
        element.position = position
    for element, color in zip(elements, linear_rgb):
        element.color = (*color[:3], 1.0)


def _new_slot_tree(name):
    """A slot: a group taking Value (0-1) to Color and Opacity through the
    starter Color Ramp, opacity curve and opacity scale.
    """
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Value", in_out="INPUT", socket_type="NodeSocketFloat")
    tree.interface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    tree.interface.new_socket("Opacity", in_out="OUTPUT", socket_type="NodeSocketFloat")
    tree.interface.new_socket("Decades", in_out="OUTPUT", socket_type="NodeSocketFloat")
    nodes, link = tree.nodes, tree.links.new
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = ramp.label = COLOR_NODE_NAME
    # Ramp colors are scene-linear; the UI shows them display-encoded.
    _set_ramp(ramp.color_ramp, [stop for stop, _ in _STARTER_STOPS], _srgb_to_linear([c for _, c in _STARTER_STOPS]))
    tree[COLORMAP_PROPERTY] = "Starter"

    curve = nodes.new("ShaderNodeFloatCurve")
    curve.name = curve.label = ALPHA_NODE_NAME
    points = curve.mapping.curves[0].points
    points[0].location = (0.0, 0.0)
    points[1].location = (1.0, 0.8)
    points.new(0.3, 0.0)  # transparent over the sparsest 30%, so the voids stay clear
    curve.mapping.update()

    link(group_input.outputs["Value"], ramp.inputs["Fac"])
    link(group_input.outputs["Value"], curve.inputs["Value"])
    link(ramp.outputs["Color"], group_output.inputs["Color"])
    link(curve.outputs["Value"], group_output.inputs["Opacity"])
    link(add_opacity_scale(tree).outputs[0], group_output.inputs["Decades"])
    arrange(tree)
    return tree


def build_node_group(name=MODIFIER_NAME, nbins=NBINS):
    """A new Geometry Nodes group: its input geometry, joined with `nbins`
    empty instances whose x in [0, 1] runs through the blended slots into the
    two attributes. It starts with one slot.
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
    value.name = VALUE_NODE_NAME
    value.label = "Value (0-1, log)"

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
    link(line.outputs["Mesh"], store_color.inputs["Geometry"])
    link(store_color.outputs["Geometry"], store_alpha.inputs["Geometry"])
    distribution = _add_distribution(tree, value, name, nbins)
    link(store_alpha.outputs["Geometry"], distribution[0].inputs["Geometry"])
    link(distribution[-1].outputs["Geometry"], instances.inputs["Points"])
    outline = _add_outline(tree, group_input)
    link(group_input.outputs["Geometry"], join.inputs["Geometry"])
    link(outline.outputs["Output"], join.inputs["Geometry"])
    link(instances.outputs["Instances"], join.inputs["Geometry"])
    link(join.outputs["Geometry"], output.inputs["Geometry"])

    _add_slot_node(tree, _new_slot_tree(f"{name} Slot"), "")
    rebuild_mix(tree)
    return tree


def _math(tree, operation, *inputs, name=None, clamp=False):
    """A Math node; each input is a constant or a socket to link from."""
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    node.use_clamp = clamp
    if name:
        node.name = name
    for socket, source in zip(node.inputs, inputs):
        if isinstance(source, (int, float)):
            socket.default_value = source
        else:
            tree.links.new(source, socket)
    return node


def _store_node(tree, attribute):
    for node in tree.nodes:
        if node.bl_idname == "GeometryNodeStoreNamedAttribute" and node.inputs["Name"].default_value == attribute:
            return node
    raise KeyError(attribute)


def _users(tree):
    """The Geometry Nodes modifiers using `tree`."""
    return [m for o in bpy.data.objects for m in o.modifiers if m.type == "NODES" and m.node_group == tree]


def _update_blend_input(tree, count):
    """Give `tree` a Blend input ranging over its `count` slots."""
    item = tree.interface.items_tree.get(BLEND_INPUT)
    if item is None:
        item = tree.interface.new_socket(BLEND_INPUT, in_out="INPUT", socket_type="NodeSocketFloat")
        item.default_value = 1.0
        old = tree.nodes.get(_OLD_BLEND_NODE_NAME)
        start = old.outputs[0].default_value if old is not None else 1.0
        if old is not None:
            tree.nodes.remove(old)
        identifier = item.identifier
        for modifier in _users(tree):
            getattr(modifier.properties.inputs, identifier).value = start
    item.description = "Which transfer function slot renders: 1 is the first, 1.5 halfway to the second"
    # Changing the interface invalidates Python handles to modifier inputs; refetch after.
    item.min_value, item.max_value = 1.0, float(max(count, 1))


def slot_nodes(tree):
    """`tree`'s slot group nodes, in blend order."""
    return sorted((n for n in tree.nodes if n.name.startswith(SLOT_PREFIX)), key=lambda n: int(n.name[len(SLOT_PREFIX):]))


def _add_slot_node(tree, slot_tree, label):
    node = tree.nodes.new("GeometryNodeGroup")
    node.node_tree = slot_tree
    node.name = f"{SLOT_PREFIX}{len(slot_nodes(tree)) + 1}"
    node.label = label
    return node


def rebuild_mix(tree):
    """Rebuild the nodes cross-fading `tree`'s slots into the two attributes.

    Blend b (clamped to the slots) weights slot i by max(0, 1 - |b - i|).
    Opacity blends linearly if every weighted slot is linear, else as curve
    heights over the largest of their decades. Color is weighted by opacity
    too, so a transparent slot doesn't tint the other.
    """
    nodes, link = tree.nodes, tree.links.new
    for node in [n for n in nodes if n.name.startswith(MIX_PREFIX)]:
        nodes.remove(node)
    slots = slot_nodes(tree)
    count = iter(range(1, 1_000_000))

    def math(operation, *inputs, clamp=False):
        return _math(tree, operation, *inputs, name=f"{MIX_PREFIX}{next(count)}", clamp=clamp).outputs[0]

    def vector(operation, a, b):
        node = nodes.new("ShaderNodeVectorMath")
        node.name = f"{MIX_PREFIX}{next(count)}"
        node.operation = operation
        link(a, node.inputs[0])
        link(b, node.inputs["Scale" if operation == "SCALE" else 1])
        return node.outputs["Vector"]

    def total(terms, add):
        result = terms[0]
        for term in terms[1:]:
            result = add(result, term)
        return result

    _update_blend_input(tree, len(slots))
    group_input = nodes.new("NodeGroupInput")
    group_input.name = f"{MIX_PREFIX}input"
    value = nodes[VALUE_NODE_NAME].outputs["X"]
    position = math("MINIMUM", math("MAXIMUM", group_input.outputs[BLEND_INPUT], 1.0), float(len(slots)))
    weights, alphas, colors, decades = [], [], [], []
    for i, slot in enumerate(slots, 1):
        link(value, slot.inputs["Value"])
        weight = math("MAXIMUM", math("SUBTRACT", 1.0, math("ABSOLUTE", math("SUBTRACT", position, float(i)))), 0.0)
        opacity = slot.outputs["Opacity"]
        weighted_alpha = math("MULTIPLY", weight, opacity)
        color_weight = math("MULTIPLY_ADD", weight, MIX_EPSILON, weighted_alpha)
        weights.append((weight, opacity))
        alphas.append(weighted_alpha)
        colors.append(vector("SCALE", slot.outputs["Color"], color_weight))
        decades.append(math("MULTIPLY", slot.outputs["Decades"], math("GREATER_THAN", weight, 0.0)))
    linear = total(alphas, lambda a, b: math("ADD", a, b))
    color = total(colors, lambda a, b: vector("ADD", a, b))
    color = vector("SCALE", color, math("DIVIDE", 1.0, math("ADD", linear, MIX_EPSILON)))

    # Log blend: each opacity as a height 1 + log10(a) / D (0 where a is 0).
    mix_decades = total(decades, lambda a, b: math("MAXIMUM", a, b))
    heights = []
    for weight, opacity in weights:
        height = math("ADD", math("DIVIDE", math("LOGARITHM", opacity, 10.0), mix_decades), 1.0, clamp=True)
        heights.append(math("MULTIPLY", weight, math("MULTIPLY", height, math("GREATER_THAN", opacity, 0.0))))
    height = total(heights, lambda a, b: math("ADD", a, b))
    logarithmic = math(
        "MULTIPLY",
        math("POWER", 10.0, math("MULTIPLY", mix_decades, math("SUBTRACT", height, 1.0))),
        math("GREATER_THAN", height, LOG_OPACITY_FLOOR),
    )
    is_log = math("GREATER_THAN", mix_decades, 0.0)
    alpha = math("ADD", linear, math("MULTIPLY", is_log, math("SUBTRACT", logarithmic, linear)))

    link(color, _store_node(tree, COLOR_ATTRIBUTE).inputs["Value"])
    link(alpha, _store_node(tree, ALPHA_ATTRIBUTE).inputs["Value"])
    arrange(tree)
    tree.update_tag()


def _copy_curve(source, target):
    src, dst = source.mapping.curves[0].points, target.mapping.curves[0].points
    while len(dst) > len(src):
        dst.remove(dst[-1])
    while len(dst) < len(src):
        dst.new(1.0, 0.0)
    for s_point, d_point in zip(src, dst):
        d_point.location, d_point.handle_type = tuple(s_point.location), s_point.handle_type
    target.mapping.update()


def _convert_to_slots(tree):
    """Move a pre-slot group's Color Ramp, opacity curve and opacity scale into
    a first slot; a no-op if `tree` already has slots.
    """
    if slot_nodes(tree) or COLOR_NODE_NAME not in tree.nodes:
        return
    nodes = tree.nodes
    ramp, curve = nodes[COLOR_NODE_NAME], nodes[ALPHA_NODE_NAME]
    ramp.inputs["Fac"].links[0].from_node.name = VALUE_NODE_NAME
    slot_tree = _new_slot_tree(f"{tree.name} Slot")
    new_ramp, new_curve = slot_tree.nodes[COLOR_NODE_NAME], slot_tree.nodes[ALPHA_NODE_NAME]
    old = ramp.color_ramp
    new_ramp.color_ramp.interpolation, new_ramp.color_ramp.color_mode = old.interpolation, old.color_mode
    _set_ramp(new_ramp.color_ramp, [e.position for e in old.elements], [tuple(e.color) for e in old.elements])
    _copy_curve(curve, new_curve)
    slot_tree[COLORMAP_PROPERTY] = "Custom"
    decades = nodes.get(DECADES_NODE_NAME)
    if decades is not None:
        slot_tree.nodes[DECADES_NODE_NAME].outputs[0].default_value = decades.outputs[0].default_value
    # The ramp, curve, and the opacity scale nodes they feed.
    doomed, frontier = set(), [ramp, curve] + ([decades] if decades is not None else [])
    stores = {_store_node(tree, COLOR_ATTRIBUTE), _store_node(tree, ALPHA_ATTRIBUTE)}
    while frontier:
        node = frontier.pop()
        if node in doomed or node in stores:
            continue
        doomed.add(node)
        frontier.extend(l.to_node for output in node.outputs for l in output.links)
    for node in doomed:
        nodes.remove(node)
    _add_slot_node(tree, slot_tree, "")
    rebuild_mix(tree)


def add_slot(tree, copy_index=None):
    """Append a slot to `tree`, a copy of slot `copy_index` (0-based) or the
    starter; returns the new slot's 0-based index.
    """
    _convert_to_slots(tree)
    slots = slot_nodes(tree)
    if copy_index is None:
        slot_tree = _new_slot_tree(f"{tree.name} Slot")
        label = ""
    else:
        source = slots[copy_index]
        slot_tree = source.node_tree.copy()
        label = f"{source.label} copy" if source.label else ""
    _add_slot_node(tree, slot_tree, label)
    rebuild_mix(tree)
    return len(slots)


def remove_slot(tree, index):
    """Remove slot `index` (0-based) from `tree`; the last one stays."""
    slots = slot_nodes(tree)
    if len(slots) < 2:
        raise ValueError("a transfer function needs at least one slot")
    slot_tree = slots[index].node_tree
    tree.nodes.remove(slots[index])
    if slot_tree is not None and slot_tree.users == 0:
        bpy.data.node_groups.remove(slot_tree)
    remaining = slot_nodes(tree)
    for i, node in enumerate(remaining):
        node.name = f"{SLOT_PREFIX}tmp{i}"
    for i, node in enumerate(remaining, 1):
        node.name = f"{SLOT_PREFIX}{i}"
    rebuild_mix(tree)


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
        return _math(tree, operation, a, b)

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

    arrange(tree)
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


def _fit_stops(srgb, max_stops, tolerance=COLORMAP_TOLERANCE):
    """Indices into `srgb` (n x 3, evenly spaced) for at most `max_stops` ramp
    stops, each added where the ramp's linear-RGB interpolation is furthest off.
    """
    x = np.linspace(0.0, 1.0, len(srgb))
    linear = _srgb_to_linear(srgb)
    chosen = [0, len(srgb) - 1]
    while len(chosen) < max_stops:
        approx = np.stack([np.interp(x, x[chosen], linear[chosen, c]) for c in range(3)], axis=1)
        error = np.abs(_linear_to_srgb(approx) - srgb).max(axis=1)
        if error.max() <= tolerance:
            break
        chosen = sorted(chosen + [int(error.argmax())])
    return chosen


def load_colormap(ramp, name, max_stops=MAX_RAMP_STOPS):
    """Fill Color Ramp node `ramp` from matplotlib colormap `name`: at most
    `max_stops` stops placed where they matter, or constant steps for a listed
    colormap of up to 32 colors.
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
        samples = np.linspace(0.0, 1.0, COLORMAP_SAMPLES)
        chosen = _fit_stops(cmap(samples)[:, :3], min(max(max_stops, 2), MAX_RAMP_STOPS))
        positions, srgb = samples[chosen], cmap(samples[chosen])[:, :3]
    color_ramp.interpolation = "CONSTANT" if stepped else "LINEAR"
    _set_ramp(color_ramp, positions, _srgb_to_linear(srgb))
    ramp.id_data[COLORMAP_PROPERTY] = name
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


def transfer_function_tree(obj):
    """The Geometry Nodes group of `obj`'s transfer function (with slots, or a
    Color Ramp from before them), or None.
    """
    for modifier in obj.modifiers:
        tree = getattr(modifier, "node_group", None) if modifier.type == "NODES" else None
        if tree is not None and (slot_nodes(tree) or COLOR_NODE_NAME in tree.nodes):
            return tree
    return None


def active_slot(obj):
    """`obj`'s slot group node being edited (its volume_override.tf_active_slot), or None."""
    tree = transfer_function_tree(obj)
    slots = slot_nodes(tree) if tree is not None else []
    if not slots:
        return None
    index = getattr(getattr(obj, "volume_override", None), "tf_active_slot", 0)
    return slots[min(max(index, 0), len(slots) - 1)]


def find_node(obj, name):
    """The node called `name` in `obj`'s transfer function: in the active
    slot if it has slots, else in its group; None if absent.
    """
    slot = active_slot(obj)
    if slot is not None and slot.node_tree is not None and name in slot.node_tree.nodes:
        return slot.node_tree.nodes[name]
    tree = transfer_function_tree(obj)
    return tree.nodes.get(name) if tree is not None else None


def find_nodes(obj):
    """`(color_ramp_node, opacity_curve_node)` from `obj`'s Geometry Nodes
    modifiers, either None if absent.
    """
    return find_node(obj, COLOR_NODE_NAME), find_node(obj, ALPHA_NODE_NAME)


def colormap_name(obj):
    """The colormap the active slot's ramp was loaded from, or None."""
    ramp = find_node(obj, COLOR_NODE_NAME)
    return ramp.id_data.get(COLORMAP_PROPERTY) if ramp is not None else None


def _blend_input(obj):
    """`(modifier input struct, its data path from obj)` for `obj`'s Blend, or None."""
    tree = transfer_function_tree(obj)
    item = tree.interface.items_tree.get(BLEND_INPUT) if tree is not None else None
    if item is None:
        return None
    for modifier in obj.modifiers:
        if modifier.type == "NODES" and modifier.node_group == tree:
            struct = getattr(modifier.properties.inputs, item.identifier)
            return struct, struct.path_from_id("value")
    return None


def _blend_channelbag(obj):
    """`(channelbag, Blend's data path)` of `obj`'s action, or None."""
    found = _blend_input(obj)
    data = obj.animation_data
    if found is None or data is None or data.action is None:
        return None
    from bpy_extras import anim_utils

    bag = anim_utils.action_get_channelbag_for_slot(data.action, data.action_slot)
    return (bag, found[1]) if bag is not None else None


def blend_fcurve(obj):
    """The F-Curve keyframing `obj`'s Blend, or None."""
    found = _blend_channelbag(obj)
    return found[0].fcurves.find(found[1]) if found is not None else None


def blend_is_animated(obj):
    found = _blend_input(obj)
    data = obj.animation_data
    if found is None or data is None:
        return False
    return data.drivers.find(found[1]) is not None or blend_fcurve(obj) is not None


def blend_value(obj):
    found = _blend_input(obj)
    return found[0].value if found is not None else 1.0


BLEND_EASINGS = (("LINEAR", "Linear", "Change at a steady rate"), ("BEZIER", "Smooth", "Ease in and out"), ("CONSTANT", "Cut", "Switch at each key"))


def set_blend(obj, value, keyframe=False):
    """Set `obj`'s Blend input, keyframing it at the current frame if asked."""
    found = _blend_input(obj)
    if found is None:
        return
    struct, path = found
    struct.value = value
    if keyframe:
        obj.keyframe_insert(path)
    obj.update_tag()


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


_slot_items = []  # Blender needs the enum strings kept alive


def slot_enum_items(props, context):
    tree = transfer_function_tree(props.id_data)
    slots = slot_nodes(tree) if tree is not None else []
    _slot_items[:] = [(str(i), f"{i + 1}: {node.label or 'Slot'}", "", i) for i, node in enumerate(slots)]
    return _slot_items or [("0", "1: Slot", "", 0)]


def select_slot(obj, index):
    """Edit slot `index` (0-based), also showing it unless Blend is animated."""
    obj.volume_override.tf_active_slot = index
    if not blend_is_animated(obj):
        set_blend(obj, index + 1.0)


class VOLUMEOVERRIDE_OT_key_tf_slot(bpy.types.Operator):
    """Keyframe Blend here to show the slot being edited"""

    bl_idname = "volume_override.key_tf_slot"
    bl_label = "Keyframe This Slot"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and _blend_input(context.object) is not None

    def execute(self, context):
        obj = context.object
        fcurve = blend_fcurve(obj)
        # New keys follow the easing the existing ones use.
        easing = fcurve.keyframe_points[0].interpolation if fcurve is not None and fcurve.keyframe_points else None
        set_blend(obj, slot_nodes(transfer_function_tree(obj)).index(active_slot(obj)) + 1.0, keyframe=True)
        if easing is not None:
            for point in blend_fcurve(obj).keyframe_points:
                point.interpolation = easing
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_remove_tf_slot_key(bpy.types.Operator):
    """Remove this Blend keyframe"""

    bl_idname = "volume_override.remove_tf_slot_key"
    bl_label = "Remove Slot Keyframe"
    bl_options = {"REGISTER", "UNDO"}

    frame: bpy.props.FloatProperty()

    def execute(self, context):
        obj = context.object
        fcurve = blend_fcurve(obj)
        if fcurve is None:
            return {"CANCELLED"}
        for point in fcurve.keyframe_points:
            if abs(point.co[0] - self.frame) < 1e-3:
                fcurve.keyframe_points.remove(point)
                break
        if not fcurve.keyframe_points:
            _blend_channelbag(obj)[0].fcurves.remove(fcurve)
        obj.update_tag()
        context.scene.frame_set(context.scene.frame_current)
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_goto_tf_slot_key(bpy.types.Operator):
    """Go to this Blend keyframe's frame"""

    bl_idname = "volume_override.goto_tf_slot_key"
    bl_label = "Go to Slot Keyframe"

    frame: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.frame_set(self.frame)
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_set_tf_blend_easing(bpy.types.Operator):
    """How Blend moves between its keyframes"""

    bl_idname = "volume_override.set_tf_blend_easing"
    bl_label = "Blend Easing"
    bl_options = {"REGISTER", "UNDO"}

    easing: bpy.props.EnumProperty(items=BLEND_EASINGS)

    def execute(self, context):
        fcurve = blend_fcurve(context.object)
        if fcurve is None:
            return {"CANCELLED"}
        for point in fcurve.keyframe_points:
            point.interpolation = self.easing
            point.easing = "AUTO"
        fcurve.update()
        context.object.update_tag()
        context.scene.frame_set(context.scene.frame_current)
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_add_tf_slot(bpy.types.Operator):
    """Add a transfer function slot, a copy of the one shown, to blend to"""

    bl_idname = "volume_override.add_tf_slot"
    bl_label = "Add Transfer Function Slot"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and transfer_function_tree(context.object) is not None

    def execute(self, context):
        obj = context.object
        tree = transfer_function_tree(obj)
        _convert_to_slots(tree)
        current = slot_nodes(tree).index(active_slot(obj))
        select_slot(obj, add_slot(tree, copy_index=current))
        obj.update_tag()
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_remove_tf_slot(bpy.types.Operator):
    """Remove the transfer function slot shown"""

    bl_idname = "volume_override.remove_tf_slot"
    bl_label = "Remove Transfer Function Slot"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        tree = transfer_function_tree(context.object) if context.object is not None else None
        return tree is not None and len(slot_nodes(tree)) > 1

    def execute(self, context):
        obj = context.object
        tree = transfer_function_tree(obj)
        index = slot_nodes(tree).index(active_slot(obj))
        remove_slot(tree, index)
        select_slot(obj, max(index - 1, 0))
        obj.update_tag()
        return {"FINISHED"}


CLASSES = (
    VOLUMEOVERRIDE_OT_add_transfer_function,
    VOLUMEOVERRIDE_OT_load_tf_colormap,
    VOLUMEOVERRIDE_OT_set_opacity_curve_scale,
    VOLUMEOVERRIDE_OT_add_tf_slot,
    VOLUMEOVERRIDE_OT_remove_tf_slot,
    VOLUMEOVERRIDE_OT_key_tf_slot,
    VOLUMEOVERRIDE_OT_remove_tf_slot_key,
    VOLUMEOVERRIDE_OT_goto_tf_slot_key,
    VOLUMEOVERRIDE_OT_set_tf_blend_easing,
)
