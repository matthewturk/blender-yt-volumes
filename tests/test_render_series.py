import itertools
import json

import pytest

from cycles_volume_override import render_series as rs
from cycles_volume_override import render_volume as rv

TIME = {"by": "time", "units": "Myr", "keys": None, "pick": "nearest"}


def _outputs(*times):
    return [rs.Output(f"DD{i:04d}", t) for i, t in enumerate(times)]


def _spans(segments):
    return [(s.output_index, s.start, s.end) for s in segments]


def test_interpolate_keys_is_piecewise_linear_and_clamped():
    keys = [[1, 0.0], [11, 100.0], [21, 110.0]]
    assert rs.interpolate_keys(keys, 6) == pytest.approx(50.0)
    assert rs.interpolate_keys(keys, 16) == pytest.approx(105.0)
    assert rs.interpolate_keys(keys, -5) == 0.0
    assert rs.interpolate_keys(keys, 99) == 110.0


@pytest.mark.parametrize("target, pick, expected", [
    (0.0, "nearest", 0), (4.9, "nearest", 0), (5.1, "nearest", 1), (99.0, "nearest", 2),
    (-1.0, "previous", 0), (9.9, "previous", 0), (10.0, "previous", 1), (99.0, "previous", 2),
])
def test_pick_output(target, pick, expected):
    assert rs.pick_output([0.0, 10.0, 20.0], target, pick) == expected


def test_plan_by_time_spreads_outputs_over_frames_by_their_times():
    # The gap between outputs 1 and 2 is 3x the first, so output 2 shows up late.
    segments = rs.plan_segments(_outputs(0.0, 10.0, 40.0), (1, 9), TIME)
    # t(frame) = 5 * (frame - 1); ties go to the earlier output.
    assert _spans(segments) == [(0, 1, 2), (1, 3, 6), (2, 7, 9)]


def test_plan_by_index_gives_each_output_an_equal_share():
    segments = rs.plan_segments(_outputs(0.0, 10.0, 40.0), (1, 9), {**TIME, "by": "index"})
    lengths = [s.end - s.start + 1 for s in segments]
    assert [s.output_index for s in segments] == [0, 1, 2]
    assert max(lengths) - min(lengths) <= 2


def test_plan_keys_stretch_and_hold():
    # Frames 1-10 sweep all outputs; frames 10-20 hold on the last one.
    keys = [[1, 0.0], [10, 20.0], [20, 20.0]]
    segments = rs.plan_segments(_outputs(0.0, 10.0, 20.0), (1, 20), {**TIME, "keys": keys})
    assert segments[-1].output_index == 2 and segments[-1].end == 20
    assert segments[-1].end - segments[-1].start >= 10


def test_plan_keeps_frames_contiguous_and_complete():
    segments = rs.plan_segments(_outputs(*range(0, 50, 7)), (3, 40), TIME)
    frames = [f for s in segments for f in range(s.start, s.end + 1)]
    assert frames == list(range(3, 41))


def test_missing_runs_skips_rendered_frames(tmp_path):
    prefix = str(tmp_path / "frame_")
    for frame in (3, 4, 7):
        (tmp_path / f"frame_{frame:04d}.png").write_bytes(b"")
    segment = rs.Segment(0, "DD0000", 2, 8)
    assert rs.missing_runs(segment, prefix) == [(2, 2), (5, 6), (8, 8)]


def _series_config(tmp_path, extra=""):
    (tmp_path / "scene.blend").write_bytes(b"")
    text = ('[data]\nseries = "DD*/DD*"\n[blender]\nfile = "scene.blend"\nvolume = "V"\n'
            'frames = "1-6"\noutput = "out/frame_"\n' + extra)
    path = tmp_path / "series.toml"
    path.write_text(text)
    return rv.load_config(path, series=True)


def test_series_config_resolves_the_glob_and_requires_frames(tmp_path):
    config = _series_config(tmp_path)
    assert config["data"]["series"] == str(tmp_path / "DD*" / "DD*")
    config["blender"]["frames"] = None
    with pytest.raises(rv.ConfigError, match="frames is required"):
        rv.validate(config, series=True)


@pytest.mark.parametrize("extra, message", [
    ('[time]\nby = "redshift"\n', "time.by"),
    ('[time]\nkeys = [[5, 1.0], [5, 2.0]]\n', "strictly increasing"),
    ('[time]\nkeys = [[5, 1.0]]\n', "at least two"),
])
def test_series_time_validation(tmp_path, extra, message):
    with pytest.raises(rv.ConfigError, match=message):
        _series_config(tmp_path, extra)


def test_render_volume_refuses_a_series_config(tmp_path):
    config = _series_config(tmp_path)
    with pytest.raises(rv.ConfigError, match="render_series"):
        rv.validate(config)


def test_segment_config_is_a_valid_single_dataset_config(tmp_path):
    config = _series_config(tmp_path, '[render]\ncmap_range = "first"\n')
    seg = rs.Segment(1, str(tmp_path / "DD0001"), 3, 5)
    out = rs.segment_config(config, seg, (3, 5), [1e-6, 0.5])
    path = tmp_path / "seg.json"
    path.write_text(json.dumps(out))
    loaded = rv.load_config(path)
    assert loaded["data"]["path"] == str(tmp_path / "DD0001")
    assert loaded["blender"]["frames"] == [3, 5]
    assert loaded["render"]["cmap_range"] == [1e-6, 0.5]
    assert config["render"]["cmap_range"] == "first"  # the series config is untouched


def test_run_pins_the_first_segments_colormap_range(tmp_path, monkeypatch):
    config = _series_config(tmp_path, '[render]\ncmap_range = "first"\n')
    calls = []

    def fake_run_segment(config_path, report_path):
        with open(config_path) as f:
            calls.append(json.load(f))
        with open(report_path, "w") as f:
            json.dump({"cmap_range": [0.25, 0.75], "frames": []}, f)
        return 0

    monkeypatch.setattr(rs, "_run_segment", fake_run_segment)
    assert rs.run(config, _outputs(0.0, 10.0, 20.0)) == 0
    assert len(calls) == 3
    assert calls[0]["render"]["cmap_range"] is None
    assert all(c["render"]["cmap_range"] == [0.25, 0.75] for c in calls[1:])
    assert [c["data"]["path"] for c in calls] == ["DD0000", "DD0001", "DD0002"]


def test_run_stops_at_the_first_failed_segment(tmp_path, monkeypatch, capsys):
    config = _series_config(tmp_path)
    calls = []
    monkeypatch.setattr(rs, "_run_segment", lambda c, r: calls.append(c) or 3)
    assert rs.run(config, _outputs(0.0, 10.0, 20.0)) == 3
    assert len(calls) == 1
    assert "rerun with" in capsys.readouterr().err


def test_run_dry_run_prints_the_plan_without_rendering(tmp_path, monkeypatch, capsys):
    config = _series_config(tmp_path)
    monkeypatch.setattr(rs, "_run_segment", lambda c, r: pytest.fail("rendered during a dry run"))
    assert rs.run(config, _outputs(0.0, 10.0, 20.0), dry_run=True) == 0
    assert capsys.readouterr().out.count("[render_series] frames") == 3


def test_resumed_run_reuses_the_originally_pinned_range(tmp_path, monkeypatch):
    config = _series_config(tmp_path, '[render]\ncmap_range = "first"\n')
    ranges = itertools.chain([[0.25, 0.75]], itertools.repeat([0.1, 0.9]))  # later segments auto-scale differently
    calls = []

    def fake_run_segment(config_path, report_path):
        with open(config_path) as f:
            calls.append(json.load(f)["render"]["cmap_range"])
        with open(report_path, "w") as f:
            json.dump({"cmap_range": next(ranges), "frames": []}, f)
        return 0

    monkeypatch.setattr(rs, "_run_segment", fake_run_segment)
    outputs = _outputs(0.0, 10.0, 20.0)
    rs.run({**config, "blender": {**config["blender"], "frames": "1-2"}}, outputs)  # first run: segment 1 only
    calls.clear()
    assert rs.run(config, outputs) == 0  # later run starting elsewhere
    assert calls and all(c == [0.25, 0.75] for c in calls)


def _fake_renderer(calls, ranges):
    """_run_segment stand-in: records each config and reports the next range."""
    def fake_run_segment(config_path, report_path):
        with open(config_path) as f:
            calls.append(json.load(f))
        with open(report_path, "w") as f:
            json.dump({"cmap_range": next(ranges), "frames": []}, f)
        return 0
    return fake_run_segment


def test_first_last_probes_both_ends_then_renders_with_the_curve(tmp_path, monkeypatch):
    config = _series_config(tmp_path, '[render]\ncmap_range = "first_last"\n')
    calls = []
    monkeypatch.setattr(rs, "_run_segment", _fake_renderer(calls, iter([[0.2, 0.6], [0.1, 0.9]] + [[9, 9]] * 10)))
    assert rs.run(config, _outputs(0.0, 10.0, 20.0)) == 0

    probe_first, probe_last, *segments = calls
    assert probe_first["blender"]["frames"] == [1, 1] and probe_first["data"]["path"] == "DD0000"
    assert probe_last["blender"]["frames"] == [6, 6] and probe_last["data"]["path"] == "DD0002"
    assert probe_first["render"]["cmap_range"] is None  # probes auto-scale
    state_dir = tmp_path / "out" / ".render_series"
    assert probe_first["blender"]["output"].startswith(str(state_dir))  # not the real output
    curve = [[1, 0.2, 0.6], [6, 0.1, 0.9]]
    assert [s["render"]["cmap_range"] for s in segments] == [curve] * 3
    assert json.loads((state_dir / "cmap_range.json").read_text()) == curve


def test_first_last_resume_reuses_the_saved_curve_without_probing(tmp_path, monkeypatch):
    config = _series_config(tmp_path, '[render]\ncmap_range = "first_last"\n')
    state_dir = tmp_path / "out" / ".render_series"
    state_dir.mkdir(parents=True)
    (state_dir / "cmap_range.json").write_text(json.dumps([[1, 0.2, 0.6], [6, 0.1, 0.9]]))
    calls = []
    monkeypatch.setattr(rs, "_run_segment", _fake_renderer(calls, itertools.repeat([9, 9])))
    assert rs.run(config, _outputs(0.0, 10.0, 20.0)) == 0
    assert len(calls) == 3
    assert all(c["render"]["cmap_range"] == [[1, 0.2, 0.6], [6, 0.1, 0.9]] for c in calls)


def test_interrupting_the_driver_stops_the_running_segment(monkeypatch):
    class FakeChild:
        terminated = False

        def __init__(self, cmd):
            pass

        def wait(self, timeout=None):
            if not FakeChild.terminated:
                raise KeyboardInterrupt
            return -15

        def terminate(self):
            FakeChild.terminated = True

        def kill(self):
            pytest.fail("terminate() was enough")

    monkeypatch.setattr(rs.subprocess, "Popen", FakeChild)
    with pytest.raises(KeyboardInterrupt):
        rs._run_segment("c.json", "r.json")
    assert FakeChild.terminated
