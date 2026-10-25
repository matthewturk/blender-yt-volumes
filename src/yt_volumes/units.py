"""Scene units per data unit, kept on each volume as `obj["yt_units"]`.

The group holds `scene_per_unitary` (scene units per unitary length right now,
following the volume's transform, its Zoom Level or Deep Zoom), and for each of
the volume's Data Units `unitary_in_<unit>` and `scene_per_<unit>`. A driver
reads them with a Single Property variable, e.g. `["yt_units"]["scene_per_kpc"]`.
Geometry Nodes can't read custom properties, so a volume's units object (Units
for Geometry Nodes) holds the same values as point attributes of its one
vertex: Object Info, then Sample Index 0 of a Named Attribute.

The units object also carries where unitary positions are in the scene now:
scene = `unitary_to_world` (a 4x4 matrix) applied to unitary - origin, with
the float64 origin split into float32 `unitary_origin_high` + `_low`. A
position kept the same way (high + low) loses no precision in Geometry Nodes
at any zoom; the "yt Unitary to Scene" node group does the arithmetic.
The data side comes from yt's unit registry, recorded when the dataset loads;
a yt_idv scene file carries it as a `dataset` entry in scene.json, or as a
`dataset.json` member added by `add_units_to_scene_file`.
"""

import json
import zipfile

import bpy
import numpy as np
from bpy.app.handlers import persistent

from .node_layout import arrange

KEY = "yt_units"
_DATASET_MEMBER = "dataset.json"
_conversions = {}  # (registry hash, unit) -> unitary length in that unit, or None
ORIGIN_HIGH, ORIGIN_LOW, TO_WORLD = "unitary_origin_high", "unitary_origin_low", "unitary_to_world"
NODE_GROUP_NAME = "yt Unitary to Scene"


def dataset_info(ds):
    """What a scene file needs to convert its lengths without the dataset."""
    return {
        "unit_registry": ds.unit_registry.to_json(),
        "current_redshift": float(getattr(ds, "current_redshift", 0.0) or 0.0),
        "current_time": str(getattr(ds, "current_time", "")),
    }


def scene_file_units(path):
    """The unit registry JSON a yt_idv scene file carries, or None."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if _DATASET_MEMBER in names:
            return json.loads(zf.read(_DATASET_MEMBER)).get("unit_registry")
        state = json.loads(zf.read("scene.json"))
    return state.get("dataset", {}).get("unit_registry")


def scene_file_box(path):
    """`(unitary origin corner, width)` of the volume cube for a scene file:
    yt_idv scales a subset so its saved bounding box's longest side spans the
    cube from its low corner. None if the file has no bounding box.
    """
    with zipfile.ZipFile(path) as zf:
        state = json.loads(zf.read("scene.json"))
        objects = state.get("data_objects") or [{}]
        key = objects[0].get("attributes", {}).get("_bbox", {}).get("__array__")
        if key is None:
            return None
        with zf.open(key) as f:
            low, high = np.lib.format.read_array(f)
    low = np.asarray(low, dtype=np.float64)
    return tuple(low), float((np.asarray(high) - low).max())


def scene_file_cube(path):
    """The volume cube's width in unitary for a scene file, or None."""
    box = scene_file_box(path)
    return None if box is None else box[1]


def add_units_to_scene_file(path, ds):
    """Append `ds`'s unit information to a scene file saved without it; the
    archive isn't rewritten.
    """
    with zipfile.ZipFile(path) as zf:
        if _DATASET_MEMBER in zf.namelist():
            raise ValueError(f"{path} already has {_DATASET_MEMBER}")
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr(_DATASET_MEMBER, json.dumps(dataset_info(ds)))


def unitary_in(registry_json, unit):
    """The unitary length in `unit` under a yt unit registry, or None if the
    unit isn't a length it knows.
    """
    key = (hash(registry_json), unit)
    if key not in _conversions:
        from unyt import UnitRegistry, unyt_quantity

        try:
            registry = UnitRegistry.from_json(registry_json)
            _conversions[key] = float(unyt_quantity(1.0, "unitary", registry=registry).to(unit).v)
        except Exception:
            _conversions[key] = None
    return _conversions[key]


def scene_per_unitary(obj, scene):
    """Scene units per unitary length for volume `obj` now."""
    from . import operators
    from .yt_idv_renderer import _bbox_local_min_extent

    deep_zoom = operators._deep_zoom_params(scene)
    if deep_zoom:
        return deep_zoom["yt_idv_shot_space"] * 10.0 ** deep_zoom["yt_idv_zoom_level"]
    linear = np.array(obj.matrix_world, dtype=np.float64)[:3, :3]
    _, extent = _bbox_local_min_extent(operators._empty_cube_bounding_box_local(obj))
    cube_width = float(np.mean(np.linalg.norm(linear, axis=0) * extent))
    group = obj.get(KEY, {})
    per_unitary = cube_width / group.get("cube_in_unitary", 1.0)
    params = obj.volume_override.to_custom_params()
    if params.get("yt_idv_zoom_target"):
        per_unitary *= 10.0 ** params["yt_idv_zoom_level"]
    return per_unitary


def split(values):
    """float64 `values` as float32 `(high, low)` with high + low == values to
    about 1e-14 relative.
    """
    values = np.asarray(values, dtype=np.float64)
    high = values.astype(np.float32)
    return high, (values - high.astype(np.float64)).astype(np.float32)


def _zoom_target(spec, group):
    if not spec:
        return None
    if spec.strip().lower() == "max":
        found = group.get("zoom_max_unitary")
        return None if found is None else np.array(found, dtype=np.float64)
    values = spec.replace(",", " ").split()
    return np.array([float(v) for v in values], dtype=np.float64) if len(values) == 3 else None


def unitary_transform(obj, scene):
    """`(origin, linear, world_origin)` placing unitary `u` in the scene now at
    `linear @ (u - origin) + world_origin`, `origin` float64 (the zoom target
    when zoomed), following the volume's transform, its Zoom Level or Deep
    Zoom. None for spherical volumes, or a "max" zoom target no render has
    found yet.
    """
    from . import operators
    from .yt_idv_renderer import _bbox_local_min_extent

    group = obj.get(KEY, {})
    deep_zoom = operators._deep_zoom_params(scene)
    params = obj.volume_override.to_custom_params()
    spec = (deep_zoom or params).get("yt_idv_zoom_target")
    target = _zoom_target(spec, group)
    if spec and target is None:
        return None
    if deep_zoom:
        return target, np.eye(3) * scene_per_unitary(obj, scene), np.zeros(3)
    if obj.empty_display_type == "SPHERE":
        return None
    ratio = group.get("cube_in_unitary", 1.0)
    offset = np.array(group.get("cube_origin_unitary", (0.0, 0.0, 0.0)), dtype=np.float64)
    matrix = np.array(obj.matrix_world, dtype=np.float64)
    local_min, extent = _bbox_local_min_extent(operators._empty_cube_bounding_box_local(obj))
    # unitary u -> cube (u - offset) / ratio -> local local_min + cube * extent -> world
    linear = matrix[:3, :3] @ np.diag(extent / ratio)
    origin = offset if target is None else target
    world_origin = matrix[:3, :3] @ (local_min + (origin - offset) / ratio * extent) + matrix[:3, 3]
    if target is not None:
        linear = linear * 10.0 ** params["yt_idv_zoom_level"]
    return origin, linear, world_origin


def data_units(obj):
    return [u.strip() for u in obj.volume_override.data_units.split(",") if u.strip()]


def update(obj, scene):
    """Refresh `obj["yt_units"]`; tags `obj` for drivers only if it changed."""
    if KEY not in obj:
        obj[KEY] = {}
    group = obj[KEY]
    values = {"scene_per_unitary": scene_per_unitary(obj, scene)}
    registry = group.get("registry")
    for unit in data_units(obj):
        length = unitary_in(registry, unit) if registry else group.get(f"unitary_in_{unit}")
        if length:
            values[f"unitary_in_{unit}"] = length
            values[f"scene_per_{unit}"] = values["scene_per_unitary"] / length
    changed = False
    for name, value in values.items():
        if group.get(name) != value:
            group[name] = value
            changed = True
    if changed:
        obj.update_tag()
    units_object = obj.volume_override.units_object
    if units_object is not None and units_object.type == "MESH":
        _write_attributes(units_object.data, values)
        _write_transform(units_object.data, unitary_transform(obj, scene))


def _write_attributes(mesh, values):
    """Set the units mesh's float point attributes to `values`, dropping
    units no longer listed; only touches the mesh if something changed.
    """
    attributes = mesh.attributes
    changed = False
    for name in [a.name for a in attributes if a.name.startswith(("scene_per_", "unitary_in_")) and a.name not in values]:
        attributes.remove(attributes[name])
        changed = True
    for name, value in values.items():
        attribute = attributes.get(name) or attributes.new(name, "FLOAT", "POINT")
        current = np.empty(len(attribute.data), dtype=np.float32)
        attribute.data.foreach_get("value", current)
        if not np.all(current == np.float32(value)):
            attribute.data.foreach_set("value", np.full(len(current), value, dtype=np.float32))
            changed = True
    if changed:
        mesh.update()


def _set_attribute(mesh, name, kind, values):
    """Set point attribute `name` everywhere to `values`; True if it changed."""
    attribute = mesh.attributes.get(name)
    if attribute is not None and attribute.data_type != kind:
        mesh.attributes.remove(attribute)
        attribute = None
    attribute = attribute or mesh.attributes.new(name, kind, "POINT")
    prop = "vector" if kind == "FLOAT_VECTOR" else "value"
    wanted = np.tile(np.asarray(values, dtype=np.float32).ravel(), len(attribute.data))
    current = np.empty(len(wanted), dtype=np.float32)
    attribute.data.foreach_get(prop, current)
    if np.array_equal(current, wanted):
        return False
    attribute.data.foreach_set(prop, wanted)
    return True


def _write_transform(mesh, transform):
    """Write `unitary_transform`'s result, or drop the attributes if None."""
    if transform is None:
        names = [n for n in (ORIGIN_HIGH, ORIGIN_LOW, TO_WORLD) if n in mesh.attributes]
        for name in names:
            mesh.attributes.remove(mesh.attributes[name])
        if names:
            mesh.update()
        return
    origin, linear, world_origin = transform
    high, low = split(origin)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = linear, world_origin
    changed = _set_attribute(mesh, ORIGIN_HIGH, "FLOAT_VECTOR", high)
    changed |= _set_attribute(mesh, ORIGIN_LOW, "FLOAT_VECTOR", low)
    # Matrix attributes are stored column by column.
    changed |= _set_attribute(mesh, TO_WORLD, "FLOAT4X4", matrix.T)
    if changed:
        mesh.update()


def store_unitary_positions(mesh, positions, name="unitary"):
    """Store float64 unitary `positions` `(n, 3)`, one per vertex, as point
    attributes `name` and `name + "_low"`, for "yt Unitary to Scene".
    """
    high, low = split(np.asarray(positions, dtype=np.float64).reshape(-1, 3))
    if len(high) != len(mesh.vertices):
        raise ValueError(f"{len(high)} positions for {len(mesh.vertices)} vertices")
    for attribute_name, values in ((name, high), (f"{name}_low", low)):
        attribute = mesh.attributes.get(attribute_name) or mesh.attributes.new(attribute_name, "FLOAT_VECTOR", "POINT")
        attribute.data.foreach_set("vector", values.ravel())
    mesh.update()


def unitary_node_group():
    """The "yt Unitary to Scene" group, made if the file has none: Unitary (+
    Unitary Low) through a Units Object's transform into the space of the
    object it runs on; also outputs Scene per Unitary.
    """
    tree = bpy.data.node_groups.get(NODE_GROUP_NAME)
    if tree is not None:
        return tree
    tree = bpy.data.node_groups.new(NODE_GROUP_NAME, "GeometryNodeTree")
    interface = tree.interface
    interface.new_socket("Units Object", in_out="INPUT", socket_type="NodeSocketObject")
    interface.new_socket("Unitary", in_out="INPUT", socket_type="NodeSocketVector")
    low_socket = interface.new_socket("Unitary Low", in_out="INPUT", socket_type="NodeSocketVector")
    low_socket.description = "The float64 position's remainder below float32 (store_unitary_positions writes it)"
    interface.new_socket("Position", in_out="OUTPUT", socket_type="NodeSocketVector")
    interface.new_socket("Scene per Unitary", in_out="OUTPUT", socket_type="NodeSocketFloat")
    nodes, link = tree.nodes, tree.links.new
    group_in, group_out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    info = nodes.new("GeometryNodeObjectInfo")
    link(group_in.outputs["Units Object"], info.inputs["Object"])

    def sampled(name, data_type):
        named = nodes.new("GeometryNodeInputNamedAttribute")
        named.data_type = data_type
        named.inputs["Name"].default_value = name
        sample = nodes.new("GeometryNodeSampleIndex")
        sample.data_type, sample.domain = data_type, "POINT"
        link(info.outputs["Geometry"], sample.inputs["Geometry"])
        link(named.outputs["Attribute"], sample.inputs["Value"])
        return sample.outputs["Value"]

    def vector(operation, a, b):
        node = nodes.new("ShaderNodeVectorMath")
        node.operation = operation
        link(a, node.inputs[0])
        link(b, node.inputs[1])
        return node.outputs["Vector"]

    # (u_high - o_high) cancels exactly near the origin; the lows carry the rest.
    relative = vector(
        "ADD",
        vector("SUBTRACT", group_in.outputs["Unitary"], sampled(ORIGIN_HIGH, "FLOAT_VECTOR")),
        vector("SUBTRACT", group_in.outputs["Unitary Low"], sampled(ORIGIN_LOW, "FLOAT_VECTOR")),
    )
    to_world = nodes.new("FunctionNodeTransformPoint")
    link(relative, to_world.inputs["Vector"])
    link(sampled(TO_WORLD, "FLOAT4X4"), to_world.inputs["Transform"])
    self_info = nodes.new("GeometryNodeObjectInfo")
    link(nodes.new("GeometryNodeSelfObject").outputs[0], self_info.inputs["Object"])
    invert = nodes.new("FunctionNodeInvertMatrix")
    link(self_info.outputs["Transform"], invert.inputs["Matrix"])
    to_local = nodes.new("FunctionNodeTransformPoint")
    link(to_world.outputs["Vector"], to_local.inputs["Vector"])
    link(invert.outputs["Matrix"], to_local.inputs["Transform"])
    link(to_local.outputs["Vector"], group_out.inputs["Position"])
    link(sampled("scene_per_unitary", "FLOAT"), group_out.inputs["Scene per Unitary"])
    arrange(tree)
    return tree


def add_units_object(obj, scene):
    """The volume's units object, made (one vertex at its origin) if it has none."""
    props = obj.volume_override
    if props.units_object is None:
        mesh = bpy.data.meshes.new(f"{obj.name} yt Units")
        mesh.vertices.add(1)
        units_object = bpy.data.objects.new(mesh.name, mesh)
        units_object.hide_render = True
        for collection in obj.users_collection or [scene.collection]:
            collection.objects.link(units_object)
        props.units_object = units_object
    update(obj, scene)
    unitary_node_group()
    return props.units_object


def record(obj, dataset_id, scene):
    """Take the unit registry and cube size from the loaded dataset `dataset_id`."""
    from . import yt_idv_renderer

    registry, cube_in_unitary = yt_idv_renderer.dataset_units(dataset_id)
    origin, maximum = yt_idv_renderer.dataset_frame(dataset_id)
    if KEY not in obj:
        obj[KEY] = {}
    group = obj[KEY]
    for name, value in (("cube_origin_unitary", origin), ("zoom_max_unitary", maximum)):
        if value is not None and tuple(group.get(name, ())) != tuple(value):
            group[name] = list(value)
    if registry and group.get("registry") != registry:
        group["registry"] = registry
    if cube_in_unitary and group.get("cube_in_unitary") != cube_in_unitary:
        group["cube_in_unitary"] = cube_in_unitary
    update(obj, scene)


def read(obj, scene):
    """Read the unit registry from `obj`'s Dataset Path without loading its data."""
    path = bpy.path.abspath(obj.volume_override.dataset_path)
    if not path:
        raise ValueError("no Dataset Path; render or preview the sample dataset to record its units")
    box = None
    if path.endswith(".zip") or obj.volume_override.dataset_kind == "yt_idv_scene":
        registry = scene_file_units(path)
        if registry is None:
            raise ValueError("the scene file has no unit information; add it with units.add_units_to_scene_file")
        box = scene_file_box(path)
    else:
        import yt

        registry = yt.load(path).unit_registry.to_json()
    if KEY not in obj:
        obj[KEY] = {}
    obj[KEY]["registry"] = registry
    if box is not None:
        obj[KEY]["cube_origin_unitary"], obj[KEY]["cube_in_unitary"] = list(box[0]), box[1]
    update(obj, scene)


@persistent
def _on_update(scene, depsgraph=None):
    from . import properties

    for obj in properties.flagged_objects(scene):
        update(obj, scene)


class VOLUMEOVERRIDE_OT_add_units_object(bpy.types.Operator):
    """Make a one-vertex mesh holding this volume's units and unitary-to-scene transform for Geometry Nodes, and the yt Unitary to Scene node group"""

    bl_idname = "volume_override.add_units_object"
    bl_label = "Units for Geometry Nodes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        units_object = add_units_object(context.object, context.scene)
        self.report({"INFO"}, f"Read the units from {units_object.name!r}: Object Info, then Sample Index 0 of a Named Attribute")
        return {"FINISHED"}


class VOLUMEOVERRIDE_OT_read_units(bpy.types.Operator):
    """Read the dataset's units (its header only, or the scene file's) for yt_units"""

    bl_idname = "volume_override.read_units"
    bl_label = "Read Units"

    def execute(self, context):
        try:
            read(context.object, context.scene)
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:200])
            return {"CANCELLED"}
        return {"FINISHED"}


def _active_unitary_node(context):
    space = context.space_data
    tree = getattr(space, "edit_tree", None) if space is not None and space.type == "NODE_EDITOR" else None
    node = tree.nodes.active if tree is not None else None
    if node is not None and node.bl_idname == "GeometryNodeGroup" and node.node_tree is not None and node.node_tree.name.startswith(NODE_GROUP_NAME):
        return node
    return None


def set_unitary_inputs(node, position):
    """Set a "yt Unitary to Scene" node's Unitary and Unitary Low to float64 `position`."""
    high, low = split(position)
    node.inputs["Unitary"].default_value = high
    node.inputs["Unitary Low"].default_value = low


class VOLUMEOVERRIDE_OT_set_unitary_position(bpy.types.Operator):
    """Set the active yt Unitary to Scene node's position to full float64 precision"""

    bl_idname = "volume_override.set_unitary_position"
    bl_label = "Set Unitary Position"
    bl_options = {"REGISTER", "UNDO"}

    position: bpy.props.StringProperty(name="Unitary x, y, z", description="Unitary coordinates, with as many digits as needed")

    @classmethod
    def poll(cls, context):
        return _active_unitary_node(context) is not None

    def invoke(self, context, event):
        node = _active_unitary_node(context)
        value = np.array(node.inputs["Unitary"].default_value, dtype=np.float64) + np.array(node.inputs["Unitary Low"].default_value, dtype=np.float64)
        self.position = ", ".join(repr(float(v)) for v in value)
        return context.window_manager.invoke_props_dialog(self, width=400)

    def execute(self, context):
        values = self.position.replace(",", " ").split()
        try:
            position = [float(v) for v in values]
        except ValueError:
            position = []
        if len(position) != 3:
            self.report({"ERROR"}, f"{self.position!r} is not 'x, y, z'")
            return {"CANCELLED"}
        set_unitary_inputs(_active_unitary_node(context), position)
        return {"FINISHED"}


class VOLUMEOVERRIDE_PT_unitary_position(bpy.types.Panel):
    bl_label = "yt Unitary Position"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "yt"

    @classmethod
    def poll(cls, context):
        return _active_unitary_node(context) is not None

    def draw(self, context):
        node = _active_unitary_node(context)
        value = np.array(node.inputs["Unitary"].default_value, dtype=np.float64) + np.array(node.inputs["Unitary Low"].default_value, dtype=np.float64)
        col = self.layout.column(align=True)
        for axis, v in zip("xyz", value):
            col.label(text=f"{axis}: {v:.15g}")
        self.layout.operator(VOLUMEOVERRIDE_OT_set_unitary_position.bl_idname, icon="GREASEPENCIL")


_CLASSES = (VOLUMEOVERRIDE_OT_read_units, VOLUMEOVERRIDE_OT_add_units_object, VOLUMEOVERRIDE_OT_set_unitary_position, VOLUMEOVERRIDE_PT_unitary_position)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.app.handlers.frame_change_post.append(_on_update)
    bpy.app.handlers.depsgraph_update_post.append(_on_update)


def unregister():
    for handlers in (bpy.app.handlers.frame_change_post, bpy.app.handlers.depsgraph_update_post):
        if _on_update in handlers:
            handlers.remove(_on_update)
    for cls in reversed(_CLASSES):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
