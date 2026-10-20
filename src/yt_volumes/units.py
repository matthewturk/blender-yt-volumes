"""Scene units per data unit, kept on each volume as `obj["yt_units"]`.

The group holds `scene_per_unitary` (scene units per unitary length right now,
following the volume's transform, its Zoom Level or Deep Zoom), and for each of
the volume's Data Units `unitary_in_<unit>` and `scene_per_<unit>`. A driver
reads them with a Single Property variable, e.g. `["yt_units"]["scene_per_kpc"]`.
Geometry Nodes can't read custom properties, so a volume's units object (Units
for Geometry Nodes) holds the same values as point attributes of its one
vertex: Object Info, then Sample Index 0 of a Named Attribute.
The data side comes from yt's unit registry, recorded when the dataset loads;
a yt_idv scene file carries it as a `dataset` entry in scene.json, or as a
`dataset.json` member added by `add_units_to_scene_file`.
"""

import json
import zipfile

import bpy
import numpy as np
from bpy.app.handlers import persistent

KEY = "yt_units"
_DATASET_MEMBER = "dataset.json"
_conversions = {}  # (registry hash, unit) -> unitary length in that unit, or None


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


def scene_file_cube(path):
    """The volume cube's width in unitary for a scene file: yt_idv scales a
    subset so its saved bounding box's longest side spans the cube. None if
    the file has no bounding box.
    """
    with zipfile.ZipFile(path) as zf:
        state = json.loads(zf.read("scene.json"))
        objects = state.get("data_objects") or [{}]
        key = objects[0].get("attributes", {}).get("_bbox", {}).get("__array__")
        if key is None:
            return None
        with zf.open(key) as f:
            low, high = np.lib.format.read_array(f)
    return float((np.asarray(high) - np.asarray(low)).max())


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
    return props.units_object


def record(obj, dataset_id, scene):
    """Take the unit registry and cube size from the loaded dataset `dataset_id`."""
    from . import yt_idv_renderer

    registry, cube_in_unitary = yt_idv_renderer.dataset_units(dataset_id)
    if KEY not in obj:
        obj[KEY] = {}
    group = obj[KEY]
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
    cube_in_unitary = None
    if path.endswith(".zip") or obj.volume_override.dataset_kind == "yt_idv_scene":
        registry = scene_file_units(path)
        if registry is None:
            raise ValueError("the scene file has no unit information; add it with units.add_units_to_scene_file")
        cube_in_unitary = scene_file_cube(path)
    else:
        import yt

        registry = yt.load(path).unit_registry.to_json()
    if KEY not in obj:
        obj[KEY] = {}
    obj[KEY]["registry"] = registry
    if cube_in_unitary:
        obj[KEY]["cube_in_unitary"] = cube_in_unitary
    update(obj, scene)


@persistent
def _on_update(scene, depsgraph=None):
    from . import properties

    for obj in properties.flagged_objects(scene):
        update(obj, scene)


class VOLUMEOVERRIDE_OT_add_units_object(bpy.types.Operator):
    """Make a one-vertex mesh whose point attributes are this volume's yt_units, for Geometry Nodes (Object Info, Sample Index 0)"""

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


def register():
    bpy.utils.register_class(VOLUMEOVERRIDE_OT_read_units)
    bpy.utils.register_class(VOLUMEOVERRIDE_OT_add_units_object)
    bpy.app.handlers.frame_change_post.append(_on_update)
    bpy.app.handlers.depsgraph_update_post.append(_on_update)


def unregister():
    for handlers in (bpy.app.handlers.frame_change_post, bpy.app.handlers.depsgraph_update_post):
        if _on_update in handlers:
            handlers.remove(_on_update)
    for cls in (VOLUMEOVERRIDE_OT_add_units_object, VOLUMEOVERRIDE_OT_read_units):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
