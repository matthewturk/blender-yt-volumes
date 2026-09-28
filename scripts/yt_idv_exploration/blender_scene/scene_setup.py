"""Your yt_idv scene, in one place.

Edit the constants and `setup_volume()` below to change what gets
rendered; `render_headless.py` imports this file by name and re-applies it
every time you re-run that script. Nothing else needs to change for a
routine "different dataset/field/look" edit.
"""

# --- What to load ------------------------------------------------------
DATASET_NAME = "Enzo_64"  # yt.load_sample(...) name
DATASET_PATH = None  # set an absolute path to use yt.load(...) instead of a sample
FIELD = "density"  # any field yt can resolve on this dataset, e.g. ("gas", "temperature")

# --- How it should look -------------------------------------------------
#RENDERER_MODE = "YT_IDV_MAX_INTENSITY"  # see operators.RENDERER_MODES for the full list
RENDERER_MODE = "YT_IDV_PROJECTION"
COLORMAP = "octarine"  # any yt/matplotlib colormap name -- applied directly to
#                         the real component in setup_volume() below for
#                         YT_IDV_MAX_INTENSITY/YT_IDV_PROJECTION (see there for
#                         why); routed through configure_preferences() instead
#                         for the other three modes, which build their OWN
#                         colormap texture from a Preferences field rather
#                         than yt_idv's native named-colormap mechanism


def configure_preferences(prefs):
    """Called once, right after the add-on is enabled. `renderer_mode` --
    and the yt_idv-specific knobs the OTHER three modes need (colormap,
    value ranges, ...) -- lives on the add-on's Preferences, not in the
    .blend file or in custom_params directly -- see yt_idv_renderer.py's
    module docstring, "Preferences panel / experimentation". Add more
    `prefs.yt_idv.*` assignments here as you need them (tf_value_min/max,
    fog_value_min/max, mip_alpha_max, ...).
    """
    prefs.renderer_mode = RENDERER_MODE
    if RENDERER_MODE == "YT_IDV_TRANSFER_FUNCTION":
        prefs.yt_idv.tf_colormap = COLORMAP
    elif RENDERER_MODE == "YT_IDV_THRESHOLD_FOG":
        prefs.yt_idv.fog_colormap = COLORMAP
    elif RENDERER_MODE == "YT_IDV_MAX_INTENSITY_ALPHA":
        prefs.yt_idv.mip_alpha_colormap = COLORMAP


def setup_volume(ctx):
    """The yt_idv_volume_provider itself -- called once per object per
    frame (see yt_idv_renderer.py's "Custom volume setup"). Reuses its own
    scene on every call after the first; only rebuilds if you restart the
    render (a fresh process has no memory of the previous one -- that's
    what makes editing this file and re-running `render_headless.py` a
    clean way to try something different).

    This is the only function that talks to yt/yt_idv directly -- add a cut
    region, a derived field, anything real yt can do, right here. You
    already have the REAL yt_idv scene/component -- no need to route
    something like the colormap through the add-on's Preferences at all
    unless you actually want it exposed there too.
    """
    from cycles_volume_override.yt_idv_renderer import VolumeSetupResult

    if ctx.previous_scene is not None:
        return VolumeSetupResult(scene=ctx.previous_scene, ds=ctx.previous_ds, changed=False)

    import yt

    ds = yt.load(DATASET_PATH) if DATASET_PATH else yt.load_sample(DATASET_NAME)
    data_source = ds.all_data()  # swap for e.g. ds.sphere(...)/ds.cut_region(...) for a subset
    scene = ctx.render_context.add_scene(data_source, FIELD, no_ghost=True)

    # YT_IDV_MAX_INTENSITY/YT_IDV_PROJECTION never set their own colormap
    # (see yt_idv_renderer._configure_max_intensity/_configure_projection --
    # they only ever touch render_method, called AFTER this function, every
    # frame) -- so this is the one place to set it for those two modes.
    # Harmless no-op for the other three modes, which overwrite
    # component.colormap.data wholesale instead of using colormap_name.
    for component in scene.components:
        component.colormap.colormap_name = COLORMAP

    return VolumeSetupResult(scene=scene, ds=ds, changed=True)
