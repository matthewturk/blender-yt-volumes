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
RENDERER_MODE = "YT_IDV_MAX_INTENSITY"  # see operators.RENDERER_MODES for the full list
#RENDERER_MODE = "YT_IDV_PROJECTION"
COLORMAP = "octarine"  # any yt/matplotlib colormap name -- applied directly to
#                         the real component in setup_volume() below for
#                         YT_IDV_MAX_INTENSITY/YT_IDV_PROJECTION (see there for
#                         why); routed through configure_preferences() instead
#                         for the other three modes, which build their OWN
#                         colormap texture from a Preferences field rather
#                         than yt_idv's native named-colormap mechanism

# Pinned from a real frame-1 auto-scan of this exact dataset/field/camera
# (the "Computed new cmap values ..." line yt_idv prints the first time a
# component renders). Explicit, not left to "whatever frame happens to run
# first", because Cycles/OptiX was found to leak GPU memory across ~180
# consecutive bpy.ops.render.render() calls in one long-lived process (a
# real, reproducible crash: "System is out of GPU memory" at frame 183 of a
# real 250-frame HD run) -- the robust fix is restarting the render process
# periodically (see render_headless.py's RENDER_START_FRAME/RENDER_END_FRAME),
# and a fresh process's own frame 1 is a DIFFERENT camera view than the
# animation's real frame 1, which would silently re-anchor the color mapping
# partway through if left to auto-scan again. Sourced from this dataset's
# real frame 1 in a run that got to frame 182 before hitting the leak.
FIXED_CMAP_MIN = 4.552790869638557e-06
FIXED_CMAP_MAX = 0.784118115901947

# YT_IDV_MAX_INTENSITY/YT_IDV_PROJECTION only auto-scale the colormap's own
# min/max ONCE (component creation, i.e. frame 1) -- confirmed directly,
# yt_idv has no camera-move observer for this at all, so a moving camera
# keeps using frame 1's own range for every later frame regardless of what's
# actually in view. True re-fits it every frame instead (best local contrast
# per frame, at the cost of the same density value mapping to a different
# color from frame to frame -- can look like auto-exposure "pumping" in an
# animation). False (current setting) keeps yt_idv's own frame-1 estimate
# fixed for the whole animation -- temporally consistent color mapping,
# chosen because frame 1's own camera already sees a representative amount
# of the volume (not a degenerate/edge-on view), so its auto-scanned bounds
# are a reasonable estimate for the rest of the animation too.
RESET_CMAP_EACH_FRAME = False


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
        if RESET_CMAP_EACH_FRAME:
            for component in ctx.previous_scene.components:
                component._cmap_bounds_invalid = True  # force a fresh auto-scale this frame
        return VolumeSetupResult(scene=ctx.previous_scene, ds=ctx.previous_ds, changed=False)

    import yt

    ds = yt.load(DATASET_PATH) if DATASET_PATH else yt.load_sample(DATASET_NAME)
    data_source = ds.all_data()  # swap for e.g. ds.sphere(...)/ds.cut_region(...) for a subset
    # no_ghost=False: ghost zones ON -- ghost zones fill each AMR grid patch's
    # boundary cells with data from its neighbors, so interpolation/sampling
    # near a patch edge doesn't sample stale/zeroed edge cells. Costs more at
    # scene-build time (one extra data fetch per grid, at the point add_scene
    # builds the block collection) but does NOT touch per-frame render cost --
    # ghost zones are baked into the block collection once, not recomputed
    # per camera move/per rc.run() call.
    scene = ctx.render_context.add_scene(ds, FIELD, no_ghost=False)

    # YT_IDV_MAX_INTENSITY/YT_IDV_PROJECTION never set their own colormap
    # (see yt_idv_renderer._configure_max_intensity/_configure_projection --
    # they only ever touch render_method, called AFTER this function, every
    # frame) -- so this is the one place to set it for those two modes.
    # Harmless no-op for the other three modes, which overwrite
    # component.colormap.data wholesale instead of using colormap_name.
    for component in scene.components:
        component.colormap.colormap_name = COLORMAP
        # Already True on a freshly-created component, but explicit here too
        # since it's the one flag RESET_CMAP_EACH_FRAME cares about above.
        component._cmap_bounds_invalid = True
        if not RESET_CMAP_EACH_FRAME:
            # Pin to FIXED_CMAP_MIN/MAX (see its own comment above) instead of
            # trusting whichever frame happens to be first in THIS process --
            # base_component._reset_cmap_bounds() applies these AFTER its own
            # auto-scan, overriding it, so this is a real override, not just
            # a hint.
            component.fixed_cmap_min = FIXED_CMAP_MIN
            component.fixed_cmap_max = FIXED_CMAP_MAX

    return VolumeSetupResult(scene=scene, ds=ds, changed=True)
