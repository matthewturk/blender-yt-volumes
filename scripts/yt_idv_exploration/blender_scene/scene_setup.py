"""Your yt_idv scene, in one place.

Edit the constants and `setup_volume()` below to change what gets
rendered; `render_headless.py` imports this file by name and re-applies it
every time you re-run that script. Nothing else needs to change for a
routine "different dataset/field/look" edit.
"""

# --- What to load ------------------------------------------------------
#DATASET_NAME = "Enzo_64"  # yt.load_sample(...) name
DATASET_NAME = "HiresIsolatedGalaxy"  # yt.load_sample(...) name
DATASET_PATH = None  # set an absolute path to use yt.load(...) instead of a sample
FIELD = "density"  # any field yt can resolve on this dataset, e.g. ("gas", "temperature")

# Render a cube of this half-width (kpc) around the density maximum, or the
# whole domain with None.
SUBSET_HALF_WIDTH_KPC = 50
# Stretch the subset so its longest axis fills the Blender volume box. False
# keeps it at its true position and size within the full domain.
SCALE_SUBSET = True

# --- How it should look -------------------------------------------------
RENDERER_MODE = "YT_IDV_MAX_INTENSITY"  # see operators.RENDERER_MODES for the full list
#RENDERER_MODE = "YT_IDV_PROJECTION"
COLORMAP = "arbre"  # any yt/matplotlib colormap name

# Pin the colormap range so colors stay consistent across partial-range runs
# (None = auto-scale from the first frame rendered). These values come from
# the "Computed new cmap values ..." line yt_idv logs on frame 1. Projection
# values scale with SCALE_SUBSET's stretch, so re-read them after changing it.
#FIXED_CMAP_MIN = 4.552790869638557e-06
#FIXED_CMAP_MAX = 0.784118115901947
FIXED_CMAP_MIN = FIXED_CMAP_MAX = None

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
    from cycles_volume_override.yt_idv_renderer import VolumeSetupResult, scale_scene_to_subset

    if ctx.previous_scene is not None:
        if RESET_CMAP_EACH_FRAME:
            for component in ctx.previous_scene.components:
                component._cmap_bounds_invalid = True  # force a fresh auto-scale this frame
        return VolumeSetupResult(scene=ctx.previous_scene, ds=ctx.previous_ds, changed=False)

    import yt

    ds = yt.load(DATASET_PATH) if DATASET_PATH else yt.load_sample(DATASET_NAME)
    if SUBSET_HALF_WIDTH_KPC is None:
        data_source = ds.all_data()
    else:
        # Any data object works here, e.g. ds.sphere(...) or ds.cut_region(...).
        _, center = ds.find_max("density")
        half_width = SUBSET_HALF_WIDTH_KPC * ds.units.kpc
        data_source = ds.region(center, center - half_width, center + half_width)
    # no_ghost=True: ghost zones OFF. Faster scene build, less accurate at grid
    # edges (False turns them on); no per-frame cost either way.
    scene = ctx.render_context.add_scene(data_source, FIELD, no_ghost=True)
    if SCALE_SUBSET:
        scale_scene_to_subset(scene)  # no-op for the full domain

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
