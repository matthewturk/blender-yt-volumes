"""Content-hash bookkeeping behind `external_render.VersionInfo`.

See that dataclass's docstring for the full contract (hash = authoritative
cache key, `changed` = best-effort convenience). This module is just the
Blender-side half: compute a stable hash for some value, remember it per
(object, category), and report whether it differs from last time.

Deliberately process-local, in-memory, not persisted to the .blend file --
"since I last rendered" naturally means "since this Blender session/add-on
instance last ran," which is the right scope for a convenience flag whose
whole caveat is "doesn't survive restarts." Don't be tempted to persist this
dict to make `changed` survive a restart -- that would make it MORE likely
to be trusted as authoritative when it still isn't (Blender's memory of what
it sent proves nothing about whether the renderer's own cache is still
warm).
"""

import hashlib
import json

from .external_render import VersionInfo

# {object_name: {category: last_hash}}
_last_hashes = {}


def _stable_hash(value) -> str:
    """A content hash stable across process runs (unlike Python's salted
    `hash()`, which varies per-process and isn't suitable as a cache key at
    all). `json.dumps(..., sort_keys=True)` handles the dict/list/tuple/
    scalar shapes everything here is built from; pass plain Python types in
    (convert numpy arrays etc. first) -- `default=str` is a fallback for
    anything else, not a substitute for doing that.
    """
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]  # truncated: a cache key, not a security digest


def version_for(cache_key, category, value) -> VersionInfo:
    """`cache_key` should be a stable per-object id -- e.g.
    `properties.ensure_dataset_uuid(obj)`, NOT `obj.name` (renaming the
    object would then look like a brand new one with no history, so
    `changed` would spuriously report True forever after). `category`
    namespaces independent things tracked for the same object (e.g.
    `"params"` vs `"dataset"`) so they don't collide in the same bucket.
    """
    h = _stable_hash(value)
    bucket = _last_hashes.setdefault(cache_key, {})
    changed = bucket.get(category) != h
    bucket[category] = h
    return VersionInfo(hash=h, changed=changed)


def reset():
    """Clear all tracked hashes. Called from `__init__.register()` so a
    fresh add-on registration (Blender startup, or a dev reload) doesn't
    silently inherit stale in-memory state from a previous instance -- the
    first call for every object after a reset reports `changed=True`, which
    is the correct "I don't actually know, so say yes" behavior for a
    convenience flag that just lost its memory.

    Also exposed as a manual operator (`VOLUMEOVERRIDE_OT_reset_change_tracking`)
    for forcing every flagged object to report changed=True on the next
    render -- e.g. if you know your renderer's own cache was cleared
    independently and want Blender's convenience flag to line back up with
    reality rather than wait for something to actually change.
    """
    _last_hashes.clear()
