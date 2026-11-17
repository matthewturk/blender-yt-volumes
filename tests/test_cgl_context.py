"""Backend selection and the macOS CGL context, with CGL and yt_idv faked."""

import ctypes
import types
import warnings

import pytest

from yt_volumes import cgl_context, properties
from yt_volumes import yt_idv_renderer as yir


class _FakeCGL:
    """CGL that has a pixel format only for the profiles in `profiles`."""

    def __init__(self, profiles):
        self.profiles, self.asked, self.destroyed = profiles, [], 0

    def CGLChoosePixelFormat(self, attribs, pix, npix):
        profile = attribs[1]
        self.asked.append(profile)
        if profile not in self.profiles:
            return 10002  # kCGLBadPixelFormat
        pix._obj.value = 0x1234
        return 0

    def CGLCreateContext(self, pix, share, ctx):
        ctx._obj.value = 0x5678
        return 0

    def CGLDestroyPixelFormat(self, pix):
        self.destroyed += 1

    def CGLErrorString(self, err):
        return b"invalid pixel format"


def test_cgl_falls_back_from_gl4_to_gl3_core():
    cgl = _FakeCGL(profiles={0x3200})
    ctx = cgl_context._create_cgl_context(cgl)
    assert ctx.value == 0x5678
    assert cgl.asked == [0x4100, 0x3200]
    assert cgl.destroyed == 1


def test_cgl_without_any_profile_is_unavailable():
    with pytest.raises(cgl_context.CGLUnavailable, match="invalid pixel format"):
        cgl_context._create_cgl_context(_FakeCGL(profiles=set()))


def test_cgl_is_macos_only(monkeypatch):
    monkeypatch.setattr(cgl_context.sys, "platform", "linux")
    with pytest.raises(cgl_context.CGLUnavailable, match="macOS-only"):
        cgl_context.make_cgl_rendering_context(64, 64)


def _fake_yt_idv(made):
    return types.SimpleNamespace(render_context=lambda engine, **kw: made.append(engine) or f"rc:{engine}")


@pytest.mark.parametrize("backend, engine", [("auto", "egl"), ("egl", "egl"), ("osmesa", "osmesa")])
def test_off_macos_auto_is_egl(monkeypatch, backend, engine):
    monkeypatch.setattr(yir.sys, "platform", "linux")
    made = []
    assert yir._create_render_context(_fake_yt_idv(made), backend, 64) == f"rc:{engine}"
    assert made == [engine]


def test_on_macos_auto_prefers_cgl(monkeypatch):
    monkeypatch.setattr(yir.sys, "platform", "darwin")
    monkeypatch.setattr(cgl_context, "make_cgl_rendering_context", lambda w, h: "rc:cgl")
    made = []
    assert yir._create_render_context(_fake_yt_idv(made), "auto", 64) == "rc:cgl"
    assert made == []


def test_on_macos_auto_falls_back_to_mesa_egl(monkeypatch):
    monkeypatch.setattr(yir.sys, "platform", "darwin")

    def no_cgl(w, h):
        raise cgl_context.CGLUnavailable("no accelerated OpenGL context")

    monkeypatch.setattr(cgl_context, "make_cgl_rendering_context", no_cgl)
    made = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert yir._create_render_context(_fake_yt_idv(made), "auto", 64) == "rc:egl"
    assert made == ["egl"]
    assert any("Mesa's EGL" in str(w.message) for w in caught)


def test_a_cached_cgl_context_is_made_current_again(monkeypatch):
    rc = types.SimpleNamespace(current=0)
    rc.make_current = lambda: setattr(rc, "current", rc.current + 1)
    monkeypatch.setitem(yir._render_context_cache, "cgl", rc)
    assert yir._get_render_context("cgl") is rc
    assert rc.current == 1


def test_backend_numbers_saved_in_blend_files_keep_their_meaning():
    backend = properties.VolumeOverrideProperties.__annotations__["backend"].keywords
    assert {item[0]: item[3] for item in backend["items"]} == {"egl": 0, "osmesa": 1, "auto": 2, "cgl": 3}
    assert backend["default"] == "auto"
