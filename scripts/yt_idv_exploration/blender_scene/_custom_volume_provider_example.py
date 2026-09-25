"""A real `yt_idv_volume_provider` function, in its own module -- imported
normally by `custom_volume_provider_demo.py` (both directly, and indirectly
via `yt_idv_renderer._resolve_volume_provider("_custom_volume_provider_
example:setup_volume")`, exactly the mechanism a real add-on user would
configure through `obj.volume_override.yt_idv_volume_provider`). Kept in its
own file rather than inside the demo script itself: a script executed as
`__main__` and separately `importlib.import_module()`d by its filename
becomes TWO DISTINCT module objects with independent globals -- this
function's own module-level `calls` list would silently split into two
never-reconciled copies if it lived in the `__main__` script instead.
"""

calls = []  # (previous_scene is not None) per call -- see custom_volume_provider_demo.py's own assertions


def setup_volume(ctx):
    """Explicitly loads the exact same IsolatedGalaxy sample
    `yt_idv_renderer._default_volume_provider` hardcodes, through real
    `yt`/`yt_idv` calls a real user would write for their OWN dataset.
    Deliberately does NOT use `yt_idv_renderer._load_sample_source`'s own
    cache -- the whole point here is a provider that manages its own state.
    """
    calls.append(ctx.previous_scene is not None)
    if ctx.previous_scene is not None:
        from cycles_volume_override.yt_idv_renderer import VolumeSetupResult
        return VolumeSetupResult(scene=ctx.previous_scene, ds=ctx.previous_ds, changed=False)

    import yt
    from cycles_volume_override.yt_idv_renderer import VolumeSetupResult

    ds = yt.load_sample("IsolatedGalaxy")
    data_source = ds.all_data()
    scene = ctx.render_context.add_scene(data_source, "density", no_ghost=True)
    return VolumeSetupResult(scene=scene, ds=ds, changed=True)
