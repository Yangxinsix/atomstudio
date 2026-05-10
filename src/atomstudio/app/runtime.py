from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def configure_qt_runtime() -> None:
    """Configure Qt, PyOpenGL, and VisPy before any canvas is created."""
    _configure_qt_plugins()
    _configure_gl_platform()
    _configure_vispy()


def configure_preview_runtime() -> None:
    """Configure GL/VisPy for modules that are imported outside app.main."""
    _configure_qt_plugins()
    _configure_gl_platform()
    _configure_vispy()


def _configure_qt_plugins() -> None:
    try:
        import PySide6  # type: ignore
    except Exception:
        return

    plugin_root = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    platform_root = plugin_root / "platforms"
    if plugin_root.is_dir():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_root))
    if platform_root.is_dir():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platform_root))


def _configure_gl_platform() -> None:
    if "QT_QPA_PLATFORM" not in os.environ:
        if os.environ.get("DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "xcb"
        elif os.environ.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"

    os.environ.setdefault("QT_OPENGL", "desktop")
    if os.environ.get("QT_QPA_PLATFORM") == "xcb":
        os.environ["PYOPENGL_PLATFORM"] = "glx"
    elif os.environ.get("QT_QPA_PLATFORM") == "wayland":
        os.environ["PYOPENGL_PLATFORM"] = "egl"


def _configure_vispy() -> None:
    try:
        import vispy  # type: ignore

        vispy.use(app="PySide6", gl="gl+")
        _patch_vispy_glplus_framebuffer_functions()
    except Exception as exc:
        raise RuntimeError("AtomStudio app requires VisPy PySide6 + gl+ backend for instanced OpenGL preview") from exc


def _patch_vispy_glplus_framebuffer_functions() -> None:
    """Bind VisPy gl+ FBO helpers to core PyOpenGL functions.

    PyOpenGL exposes framebuffer functions through multiple paths. On WSL/GWSL
    the extension path can be unresolved even when core OpenGL functions are
    available; VisPy's pyopengl2 shim may otherwise call the unresolved path.
    """
    from OpenGL import GL  # type: ignore
    from vispy import gloo  # type: ignore

    required = (
        "glGenFramebuffers",
        "glDeleteFramebuffers",
        "glBindFramebuffer",
        "glFramebufferRenderbuffer",
        "glFramebufferTexture2D",
        "glCheckFramebufferStatus",
        "glGenRenderbuffers",
        "glDeleteRenderbuffers",
        "glBindRenderbuffer",
        "glRenderbufferStorage",
    )
    missing = [name for name in required if not _callable_gl(getattr(GL, name, None))]
    if missing:
        raise RuntimeError("PyOpenGL core framebuffer functions unavailable: " + ", ".join(missing))

    def glCreateFramebuffer():
        return GL.glGenFramebuffers(1)

    def glDeleteFramebuffer(framebuffer):
        GL.glDeleteFramebuffers(1, [framebuffer])

    def glCreateRenderbuffer():
        return GL.glGenRenderbuffers(1)

    def glDeleteRenderbuffer(renderbuffer):
        GL.glDeleteRenderbuffers(1, [renderbuffer])

    patch = {
        "glCreateFramebuffer": glCreateFramebuffer,
        "glDeleteFramebuffer": glDeleteFramebuffer,
        "glCreateRenderbuffer": glCreateRenderbuffer,
        "glDeleteRenderbuffer": glDeleteRenderbuffer,
        "glBindFramebuffer": GL.glBindFramebuffer,
        "glFramebufferRenderbuffer": GL.glFramebufferRenderbuffer,
        "glFramebufferTexture2D": GL.glFramebufferTexture2D,
        "glCheckFramebufferStatus": GL.glCheckFramebufferStatus,
        "glBindRenderbuffer": GL.glBindRenderbuffer,
        "glRenderbufferStorage": GL.glRenderbufferStorage,
    }
    backend = getattr(gloo.gl, "current_backend", None)
    for name, func in patch.items():
        setattr(gloo.gl, name, func)
        if backend is not None:
            setattr(backend, name, func)


def _callable_gl(func: Any) -> bool:
    try:
        return bool(func)
    except Exception:
        return func is not None


__all__ = ["configure_preview_runtime", "configure_qt_runtime"]
