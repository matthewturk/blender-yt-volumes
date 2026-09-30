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
COLORMAP = "octarine"  # any yt/matplotlib colormap name

# Pin the colormap range so colors stay consistent across partial-range runs
# (None = auto-scale from the first frame rendered). These values come from
# the "Computed new cmap values ..." line yt_idv logs on frame 1.
FIXED_CMAP_MIN = 4.552790869638557e-06
FIXED_CMAP_MAX = 0.784118115901947

# True re-fits the colormap range every frame (ignores FIXED_CMAP_*; best
# per-frame contrast, but colors can "pump" in an animation). False keeps
# one range for the whole run.
RESET_CMAP_EACH_FRAME = False


def configure_preferences(prefs):
    """Set add-on Preferences; called once after the add-on is enabled.

    Add more `prefs.yt_idv.*` assignments here as needed (tf_value_min/max,
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
    """yt_idv_volume_provider: build the yt_idv scene; called per object per frame.

    The scene is built on the first call and reused afterwards. Add cut
    regions, derived fields, etc. here.
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
    # no_ghost=False: ghost zones ON. More accurate at grid edges, slower
    # scene build; no per-frame cost.
    scene = ctx.render_context.add_scene(ds, FIELD, no_ghost=False)

    # MAX_INTENSITY/PROJECTION modes take their colormap from here; the other
    # modes set it via configure_preferences() and ignore colormap_name.
    for component in scene.components:
        component.colormap.colormap_name = COLORMAP
        component._cmap_bounds_invalid = True
        if not RESET_CMAP_EACH_FRAME:
            # Applied after yt_idv's auto-scan, so these override it.
            component.fixed_cmap_min = FIXED_CMAP_MIN
            component.fixed_cmap_max = FIXED_CMAP_MAX

    return VolumeSetupResult(scene=scene, ds=ds, changed=True)
