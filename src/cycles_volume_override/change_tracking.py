"""Content-hash bookkeeping behind `external_render.VersionInfo`.

In-memory and per-session on purpose: `changed` is only a convenience and
shouldn't look authoritative across restarts.
"""

import hashlib
import json

from .external_render import VersionInfo

# {object_name: {category: last_hash}}
_last_hashes = {}


def _stable_hash(value) -> str:
    """Content hash stable across processes (unlike `hash()`). Pass plain
    Python types; `default=str` is only a fallback.
    """
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]  # truncated: a cache key, not a security digest


def version_for(cache_key, category, value) -> VersionInfo:
    """Hash `value` and report whether it changed since the last call.

    `cache_key` should be a stable per-object id (e.g.
    `properties.ensure_dataset_uuid(obj)`), not `obj.name`. `category`
    separates independently tracked values, e.g. "params" vs "dataset".
    """
    h = _stable_hash(value)
    bucket = _last_hashes.setdefault(cache_key, {})
    changed = bucket.get(category) != h
    bucket[category] = h
    return VersionInfo(hash=h, changed=changed)


def reset():
    """Clear all tracked hashes, so every object next reports `changed=True`.

    Also exposed as the `VOLUMEOVERRIDE_OT_reset_change_tracking` operator.
    """
    _last_hashes.clear()
