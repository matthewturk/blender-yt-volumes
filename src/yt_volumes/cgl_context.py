"""Offscreen yt_idv render context on macOS's own OpenGL (CGL): GPU, no window.

Blender on macOS draws with Metal, so this context has the process's OpenGL to itself.
"""

import ctypes
import os
import sys

OPENGL_FRAMEWORK = "/System/Library/Frameworks/OpenGL.framework/OpenGL"

# CGLPixelFormatAttribute values, from CGLTypes.h
_PFA_COLOR_SIZE = 8
_PFA_ALPHA_SIZE = 11
_PFA_DEPTH_SIZE = 12
_PFA_ACCELERATED = 73
_PFA_ALLOW_OFFLINE_RENDERERS = 96
_PFA_OPENGL_PROFILE = 99
_PROFILES = (0x4100, 0x3200)  # kCGLOGLPVersion_GL4_Core, then GL3_Core


class CGLUnavailable(RuntimeError):
    """No CGL context could be made (not macOS, no accelerated renderer, or PyOpenGL bound elsewhere)."""


def _framework():
    cgl = ctypes.CDLL(OPENGL_FRAMEWORK)
    vp, pvp = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    cgl.CGLChoosePixelFormat.argtypes = [ctypes.POINTER(ctypes.c_int), pvp, ctypes.POINTER(ctypes.c_int)]
    cgl.CGLCreateContext.argtypes = [vp, vp, pvp]
    cgl.CGLDestroyPixelFormat.argtypes = [vp]
    cgl.CGLSetCurrentContext.argtypes = [vp]
    cgl.CGLDestroyContext.argtypes = [vp]
    cgl.CGLErrorString.argtypes = [ctypes.c_int]
    cgl.CGLErrorString.restype = ctypes.c_char_p
    return cgl


def _create_cgl_context(cgl):
    """An accelerated core-profile CGL context, the newest profile available."""
    errors = []
    for profile in _PROFILES:
        attribs = [
            _PFA_OPENGL_PROFILE, profile, _PFA_ACCELERATED, _PFA_ALLOW_OFFLINE_RENDERERS,
            _PFA_COLOR_SIZE, 24, _PFA_ALPHA_SIZE, 8, _PFA_DEPTH_SIZE, 24, 0,
        ]
        pix, npix = ctypes.c_void_p(), ctypes.c_int(0)
        err = cgl.CGLChoosePixelFormat((ctypes.c_int * len(attribs))(*attribs), ctypes.byref(pix), ctypes.byref(npix))
        if err or not pix:
            errors.append(f"profile 0x{profile:x}: {cgl.CGLErrorString(err).decode() if err else 'no pixel format'}")
            continue
        ctx = ctypes.c_void_p()
        err = cgl.CGLCreateContext(pix, None, ctypes.byref(ctx))
        cgl.CGLDestroyPixelFormat(pix)
        if err == 0 and ctx:
            return ctx
        errors.append(f"profile 0x{profile:x}: {cgl.CGLErrorString(err).decode()}")
    raise CGLUnavailable("no accelerated OpenGL context (" + "; ".join(errors) + ")")


def _prepare_pyopengl():
    """Bind PyOpenGL to Apple's OpenGL; it picks its platform once, on first import."""
    if "OpenGL.platform" in sys.modules:
        from OpenGL import platform

        if type(platform.PLATFORM).__name__ != "DarwinPlatform":
            raise CGLUnavailable(f"PyOpenGL is already bound to {type(platform.PLATFORM).__name__}")
        return
    os.environ.pop("PYOPENGL_PLATFORM", None)


def make_cgl_rendering_context(width, height):
    """A yt_idv offscreen render context drawing with CGL; raises CGLUnavailable."""
    if sys.platform != "darwin":
        raise CGLUnavailable("CGL is macOS-only")
    from yt_idv.rendering_contexts.base_offscreen import OffscreenRenderingContext

    class CGLRenderingContext(OffscreenRenderingContext):
        # The context has no default framebuffer: everything must draw into framebuffer objects.
        def __init__(self, width, height):
            super().__init__(width, height)
            self._cgl = _framework()
            self.context = _create_cgl_context(self._cgl)
            self.make_current()

        def make_current(self):
            self._cgl.CGLSetCurrentContext(self.context)

        def close(self):
            if self.context is None:
                return
            self._cgl.CGLSetCurrentContext(None)
            self._cgl.CGLDestroyContext(self.context)
            self.context = None

    _prepare_pyopengl()
    try:
        rc = CGLRenderingContext(width, height)
    except OSError as exc:  # the framework failed to load
        raise CGLUnavailable(str(exc)) from exc
    from OpenGL import GL

    GL.glViewport(0, 0, width, height)
    GL.glClearColor(0.0, 0.0, 0.0, 0.0)
    return rc
