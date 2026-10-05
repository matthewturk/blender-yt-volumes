import types

import numpy as np
import pytest

from cycles_yt import render_volume as rv


@pytest.fixture
def blend(tmp_path):
    path = tmp_path / "scene.blend"
    path.write_bytes(b"")
    return path


def _write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def _minimal(blend_name="scene.blend"):
    return f'[data]\nsample = "IsolatedGalaxy"\n[blender]\nfile = "{blend_name}"\nvolume = "Volume 1"\n'


def test_load_config_fills_defaults_and_resolves_paths_against_the_config(tmp_path, blend):
    config = rv.load_config(_write(tmp_path, _minimal()))
    assert config["blender"]["file"] == str(blend)
    assert config["blender"]["output"] == str(tmp_path / "renders" / "frame_")
    assert config["render"]["mode"] is None  # keep the .blend's own
    assert config["data"]["subset"] is None


def test_overrides_win_over_the_file_and_create_tables(tmp_path, blend):
    overrides = [
        ("render.mode", "YT_IDV_PROJECTION"),
        ("data.subset.kind", "sphere"),
        ("data.subset.radius", 20),
    ]
    config = rv.load_config(_write(tmp_path, _minimal()), overrides)
    assert config["render"]["mode"] == "YT_IDV_PROJECTION"
    assert config["data"]["subset"] == {"kind": "sphere", "radius": 20}


def test_path_overrides_resolve_against_the_current_directory(tmp_path, blend, monkeypatch):
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config = rv.load_config(_write(tmp_path, _minimal()), [("blender.output", "out/f_")])
    assert config["blender"]["output"] == str(cwd / "out" / "f_")
    assert config["blender"]["file"] == str(blend)  # from the file: still config-relative


def test_options_alone_are_enough_without_a_config_file(blend, monkeypatch):
    monkeypatch.chdir(blend.parent)
    args = rv._parse_args(["--sample", "IsolatedGalaxy", "--blend", "scene.blend", "--volume", "V",
                           "--set", "data.subset.kind=\"region\"", "--set", "data.subset.half_width=[5, \"kpc\"]"])
    config = rv.load_config(None, rv._overrides(args))
    assert config["blender"]["file"] == str(blend)
    assert config["data"]["subset"]["half_width"] == [5, "kpc"]


@pytest.mark.parametrize("text, message", [
    ('[blender]\nfile = "scene.blend"\nvolume = "V"\n', "exactly one of data.sample, data.path, data.scene or data.loader"),
    ('[data]\nsample = "x"\npath = "y"\n[blender]\nfile = "scene.blend"\nvolume = "V"\n', "exactly one"),
    ('[data]\nsample = "x"\n[blender]\nfile = "missing.blend"\nvolume = "V"\n', "does not exist"),
    ('[data]\nsample = "x"\n[blender]\nfile = "scene.blend"\n', "blender.volume is required"),
    (_minimal() + 'frames = "9-3"\n', "ends before it starts"),
    (_minimal() + '[render]\nmode = "SAMPLE_CONSTANT"\n', "not a YT_IDV_"),
    (_minimal() + '[data.subset]\nkind = "cylinder"\n', "data.subset.kind"),
    (_minimal() + '[data.subset]\nkind = "sphere"\n', "needs radius"),
    (_minimal() + 'colour = "red"\n', "unknown config keys: blender.colour"),
])
def test_validation_errors(tmp_path, blend, text, message):
    with pytest.raises(rv.ConfigError, match=message):
        rv.load_config(_write(tmp_path, text))


@pytest.mark.parametrize("frames, expected", [(7, (7, 7)), ("3-9", (3, 9)), ("4", (4, 4)), ([2, 5], (2, 5)), (None, None)])
def test_parse_frames(frames, expected):
    assert rv.parse_frames(frames) == expected


def test_parse_value_reads_toml_and_falls_back_to_strings():
    assert rv._parse_value("false") is False
    assert rv._parse_value("[1, 2]") == [1, 2]
    assert rv._parse_value("arbre") == "arbre"


def test_main_reports_config_errors_without_a_traceback(tmp_path, capsys):
    assert rv.main([str(_write(tmp_path, "[data]\n"))]) == 2
    assert "render_volume:" in capsys.readouterr().err


class _FakeDs:
    """Records which yt data object build_data_source asks for."""

    domain_center = np.array([0.5, 0.5, 0.5])

    def quan(self, value, units):
        return ("quan", value, units)

    def arr(self, value, units):
        return np.asarray(value, dtype=float)

    def find_max(self, field):
        return 1.0, np.array([0.25, 0.25, 0.25])

    def all_data(self):
        return ("all_data",)

    def box(self, left, right):
        return ("box", left.tolist(), right.tolist())

    def sphere(self, center, radius):
        return ("sphere", center.tolist(), radius)

    def region(self, center, left, right):
        return ("region", center.tolist())


def _data(subset):
    return {**rv.DEFAULTS["data"], "subset": subset}


def test_build_data_source_whole_domain_without_subset():
    assert rv.build_data_source(_FakeDs(), _data(None)) == ("all_data",)


def test_build_data_source_sphere_at_the_field_maximum():
    got = rv.build_data_source(_FakeDs(), _data({"kind": "sphere", "radius": 20, "units": "kpc"}))
    assert got == ("sphere", [0.25, 0.25, 0.25], ("quan", 20, "kpc"))


def test_build_data_source_region_accepts_value_unit_pairs_and_domain_center():
    ds = _FakeDs()
    ds.quan = lambda value, units: 0.1  # plain number so center +/- half_width works
    got = rv.build_data_source(ds, _data({"kind": "region", "center": "c", "half_width": [3, "Mpc"]}))
    assert got == ("region", [0.5, 0.5, 0.5])


def test_build_data_source_box_uses_explicit_edges():
    got = rv.build_data_source(_FakeDs(), _data({"kind": "box", "left_edge": [0, 0, 0], "right_edge": [1, 2, 3]}))
    assert got == ("box", [0.0, 0.0, 0.0], [1.0, 2.0, 3.0])


def test_volume_provider_reuses_the_previous_scene(monkeypatch):
    monkeypatch.setattr(rv, "_active_config", rv.DEFAULTS)
    ctx = types.SimpleNamespace(previous_scene=object(), previous_ds=object(),
                                frame=types.SimpleNamespace(frame_number=1))
    result = rv.volume_provider(ctx)
    assert result.scene is ctx.previous_scene and result.changed is False


def test_gpu_device_also_denoises_on_the_gpu():
    import bpy

    scene = bpy.context.scene
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    if hasattr(scene.cycles, "denoising_use_gpu"):
        scene.cycles.denoising_use_gpu = False
    try:
        rv.configure_cycles_device(scene, "OPTIX")
    except rv.ConfigError:
        pytest.skip("no OptiX device on this machine")
    assert scene.cycles.device == "GPU"
    assert scene.cycles.denoiser == "OPTIX"


def test_cpu_device_leaves_the_denoiser_alone():
    import bpy

    scene = bpy.context.scene
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    rv.configure_cycles_device(scene, "CPU")
    assert scene.cycles.device == "CPU"
    assert scene.cycles.denoiser == "OPENIMAGEDENOISE"


def test_interpolate_rows_linear_log_and_clamped():
    keys = [[1, 1e-4, 0.2], [11, 1e-2, 0.8]]
    assert rv.interpolate_rows(keys, 6) == pytest.approx([0.00505, 0.5])
    assert rv.interpolate_rows(keys, 6, log=True) == pytest.approx([1e-3, 0.4])
    assert rv.interpolate_rows(keys, -3) == [1e-4, 0.2]
    assert rv.interpolate_rows(keys, 50) == [1e-2, 0.8]


def test_cmap_range_at_handles_auto_fixed_and_curves():
    render = dict(rv.DEFAULTS["render"])
    assert rv.cmap_range_at(render, 5) == (None, None)
    render["cmap_range"] = [0.1, 0.9]
    assert rv.cmap_range_at(render, 5) == (0.1, 0.9)
    render["cmap_range"] = [[0, 0.0, 1.0], [10, 1.0, 3.0]]
    assert rv.cmap_range_at(render, 5) == pytest.approx((0.5, 2.0))


@pytest.mark.parametrize("extra, message", [
    ('[render]\ncmap_range = [[1, 0.1, 0.5]]\n', "at least two"),
    ('[render]\ncmap_range = [[5, 0.1, 0.5], [5, 0.2, 0.6]]\n', "strictly increasing"),
    ('[render]\ncmap_range = [[1, 0.0, 0.5], [9, 0.2, 0.6]]\ncmap_interpolation = "log"\n', "positive"),
    ('[render]\ncmap_range = "first_last"\n', "cmap_range must be"),  # series-only
    ('[render]\ncmap_interpolation = "cubic"\n', "cmap_interpolation"),
])
def test_cmap_validation(tmp_path, blend, extra, message):
    with pytest.raises(rv.ConfigError, match=message):
        rv.load_config(_write(tmp_path, _minimal() + extra))


@pytest.fixture
def volume_props():
    import bpy

    from cycles_yt import properties

    bpy.ops.wm.read_factory_settings(use_empty=True)
    properties.register()
    obj = bpy.data.objects.new("Volume", None)
    yield obj.volume_override
    properties.unregister()


def test_an_empty_render_table_keeps_the_volumes_settings(volume_props):
    volume_props.renderer_mode = "YT_IDV_THRESHOLD_FOG"
    volume_props.fog_colormap = "arbre"
    volume_props.rescale_colormap = True

    rv.apply_render_settings(volume_props, rv.DEFAULTS["render"])

    assert volume_props.renderer_mode == "YT_IDV_THRESHOLD_FOG"
    assert volume_props.fog_colormap == "arbre"
    assert volume_props.rescale_colormap is True


def test_the_render_table_overrides_the_volumes_settings(volume_props):
    render = {**rv.DEFAULTS["render"], "mode": "YT_IDV_PROJECTION", "colormap": "inferno",
              "reset_cmap_each_frame": False, "yt_idv": {"backend": "osmesa"}}
    volume_props.rescale_colormap = True

    rv.apply_render_settings(volume_props, render)

    assert volume_props.renderer_mode == "YT_IDV_PROJECTION"
    assert volume_props.colormap == "inferno"
    assert volume_props.rescale_colormap is False
    assert volume_props.backend == "osmesa"
    with pytest.raises(rv.ConfigError, match="not a volume look setting"):
        rv.apply_render_settings(volume_props, {"yt_idv": {"is_volume_override": False}})


class _FakeComponent:
    fixed_cmap_min = fixed_cmap_max = None
    _cmap_bounds_invalid = False


def test_volume_provider_applies_the_curve_each_frame_and_reports_changes(monkeypatch):
    component = _FakeComponent()
    scene = types.SimpleNamespace(components=[component])
    render = {**rv.DEFAULTS["render"], "cmap_range": [[1, 0.0, 1.0], [3, 1.0, 3.0]]}
    monkeypatch.setattr(rv, "_active_config", {**rv.DEFAULTS, "render": render})
    monkeypatch.setattr(rv, "_applied_cmap_range", None)

    def at(frame):
        ctx = types.SimpleNamespace(previous_scene=scene, previous_ds=None,
                                    frame=types.SimpleNamespace(frame_number=frame))
        return rv.volume_provider(ctx)

    assert at(2).changed is True
    assert (component.fixed_cmap_min, component.fixed_cmap_max) == pytest.approx((0.5, 2.0))
    assert component._cmap_bounds_invalid is True
    assert at(2).changed is False  # same frame, same range: the cached render is still right
    assert at(3).changed is True
    assert (component.fixed_cmap_min, component.fixed_cmap_max) == (1.0, 3.0)


@pytest.mark.parametrize(
    "extra, message",
    [
        ('[zoom]\ntarget = "center"\n', "zoom.target"),
        ("[zoom]\ntarget = [0.5, 0.5]\n", "zoom.target"),
        ('[zoom]\ntarget = "max"\nlevel = [[1, 0.0]]\n', "zoom.level"),
        ('[zoom]\ntarget = "max"\nlevel = [[5, 0.0], [5, 3.0]]\n', "strictly increasing"),
    ],
)
def test_zoom_validation(tmp_path, blend, extra, message):
    with pytest.raises(rv.ConfigError, match=message):
        rv.load_config(_write(tmp_path, _minimal() + extra))


def test_zoom_level_is_a_constant_or_a_clamped_linear_curve(tmp_path, blend):
    config = rv.load_config(_write(tmp_path, _minimal() + '[zoom]\ntarget = "max"\nlevel = [[1, 0.0], [101, 10.0]]\n'))
    zoom = config["zoom"]
    assert rv.zoom_level_at(zoom, 1) == 0.0
    assert rv.zoom_level_at(zoom, 51) == pytest.approx(5.0)
    assert rv.zoom_level_at(zoom, 500) == 10.0
    assert rv.zoom_level_at({"level": 2}, 7) == 2.0


def test_zoom_target_spec_round_trips_float64():
    target = [0.49150733019824, 0.505260534, 0.1 + 0.2]
    spec = rv._zoom_target_spec(target)
    assert [float(v) for v in spec.split(",")] == target
    assert rv._zoom_target_spec("max") == "max"
    assert rv._zoom_target_spec(None) == ""


def test_saved_scene_is_a_data_source_and_resolves_against_the_config(tmp_path, blend):
    text = '[data]\nscene = "saved.zip"\n[blender]\nfile = "scene.blend"\nvolume = "V"\n'
    config = rv.load_config(_write(tmp_path, text))
    assert config["data"]["scene"] == str(tmp_path / "saved.zip")


@pytest.mark.parametrize("extra", [
    'sample = "IsolatedGalaxy"\n',
    '[data.subset]\nkind = "sphere"\nradius = 1\n',
])
def test_saved_scene_excludes_other_sources_and_subsets(tmp_path, blend, extra):
    text = '[blender]\nfile = "scene.blend"\nvolume = "V"\n[data]\nscene = "saved.zip"\n' + extra
    with pytest.raises(rv.ConfigError, match="data.s"):
        rv.load_config(_write(tmp_path, text))


def test_time_varying_needs_a_loader(tmp_path, blend):
    with pytest.raises(rv.ConfigError, match="time_varying needs data.loader"):
        rv.load_config(_write(tmp_path, _minimal()), [("data.time_varying", True)])


def test_file_loaders_resolve_against_the_config(tmp_path, blend):
    text = '[data]\nloader = "gen.py:load"\ntime_varying = true\n[blender]\nfile = "scene.blend"\nvolume = "V"\n'
    config = rv.load_config(_write(tmp_path, text))
    assert config["data"]["loader"] == f"{tmp_path / 'gen.py'}:load"


def test_yt_transfer_function_files_resolve_against_the_config(tmp_path, blend):
    text = _minimal() + '[render]\nmode = "YT_IDV_YT_TRANSFER_FUNCTION"\n[render.yt_idv]\nyt_tf = "tf.py:make"\n'
    config = rv.load_config(_write(tmp_path, text))
    assert config["render"]["yt_idv"]["yt_tf"] == f"{tmp_path / 'tf.py'}:make"
