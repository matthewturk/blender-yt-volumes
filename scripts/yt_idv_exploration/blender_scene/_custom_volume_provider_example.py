"""Example `yt_idv_volume_provider`, used by `custom_volume_provider_demo.py`.

Kept in its own module: if it lived in the `__main__` demo script, importing
it by name would create a second module copy with a separate `calls` list.
"""

calls = []  # (previous_scene is not None) per call, checked by the demo


def setup_volume(ctx):
    """Load IsolatedGalaxy directly and reuse the scene on later calls."""
    calls.append(ctx.previous_scene is not None)
    if ctx.previous_scene is not None:
        from yt_volumes.yt_idv_renderer import VolumeSetupResult
        return VolumeSetupResult(scene=ctx.previous_scene, ds=ctx.previous_ds, changed=False)

    import yt
    from yt_volumes.yt_idv_renderer import VolumeSetupResult

    ds = yt.load_sample("IsolatedGalaxy")
    data_source = ds.all_data()
    scene = ctx.render_context.add_scene(data_source, "density", no_ghost=True)
    return VolumeSetupResult(scene=scene, ds=ds, changed=True)
