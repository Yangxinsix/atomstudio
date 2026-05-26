from __future__ import annotations

import os
from pathlib import Path
import socket
from typing import Any


def configure_qt_runtime(*, enable_vispy: bool = True) -> None:
    """Configure Qt, PyOpenGL, and VisPy before any canvas is created."""
    _configure_qt_plugins()
    _configure_gl_platform()
    _configure_default_surface_format()
    if enable_vispy:
        _configure_vispy()


def configure_preview_runtime(*, enable_vispy: bool = True) -> None:
    """Configure GL/VisPy for modules that are imported outside app.main."""
    _configure_qt_plugins()
    _configure_gl_platform()
    _configure_default_surface_format()
    if enable_vispy:
        _configure_vispy()


def configure_native_opengl_runtime() -> None:
    """Configure only the pieces needed by the native PyOpenGL preview path."""
    _configure_qt_plugins()
    _configure_gl_platform()
    _configure_qt_library_paths()
    _configure_default_surface_format()


def _configure_qt_plugins() -> None:
    _configure_qt_plugin_path()
    _configure_qpa_plugin_path()


def _configure_qt_plugin_path() -> None:
    try:
        import PySide6  # type: ignore
    except Exception:
        return

    plugin_root = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    if plugin_root.is_dir():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_root))


def _configure_qpa_plugin_path() -> None:
    try:
        import PySide6  # type: ignore
    except Exception:
        return

    plugin_root = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    platform_root = plugin_root / "platforms"
    if platform_root.is_dir():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platform_root))


def _configure_qt_library_paths() -> None:
    try:
        from PySide6 import QtCore  # type: ignore
    except Exception:
        return
    try:
        QtCore.QCoreApplication.setLibraryPaths(
            [QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.PluginsPath)]
        )
    except Exception:
        return


def _configure_gl_platform() -> None:
    profile = _gl_profile()
    if os.environ.get("DISPLAY"):
        _repair_x11_display()
    if "QT_QPA_PLATFORM" not in os.environ:
        if profile in {"wslg", "d3d12", "nvidia"} and _is_wslg_runtime():
            os.environ["QT_QPA_PLATFORM"] = "wayland"
        elif os.environ.get("DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "xcb"
        elif os.environ.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"

    os.environ.setdefault("QT_OPENGL", "desktop")
    if os.environ.get("QT_QPA_PLATFORM") == "xcb":
        os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
    elif os.environ.get("QT_QPA_PLATFORM") == "wayland":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        if _is_wslg_runtime() and profile != "system":
            _configure_wslg_d3d12()


def _repair_x11_display() -> None:
    display = str(os.environ.get("DISPLAY") or "").strip()
    parsed = _parse_display(display)
    if parsed is None:
        return
    host, display_number, suffix = parsed
    port = 6000 + display_number
    if _can_connect_tcp(host, port, timeout=0.15):
        return
    if _can_connect_tcp("127.0.0.1", port, timeout=0.15):
        os.environ["DISPLAY"] = f"127.0.0.1:{display_number}{suffix}"


def _parse_display(display: str) -> tuple[str, int, str] | None:
    if not display:
        return None
    if display.startswith(":"):
        host = "127.0.0.1"
        rest = display[1:]
    elif ":" in display:
        host, rest = display.split(":", 1)
        host = host.strip() or "127.0.0.1"
    else:
        return None
    display_part, dot, screen = rest.partition(".")
    try:
        display_number = int(display_part)
    except ValueError:
        return None
    suffix = f".{screen}" if dot else ""
    return host, display_number, suffix


def _can_connect_tcp(host: str, port: int, *, timeout: float) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout)):
            return True
    except OSError:
        return False


def _configure_default_surface_format() -> None:
    try:
        from PySide6 import QtGui  # type: ignore
    except Exception:
        return
    fmt = QtGui.QSurfaceFormat()
    fmt.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
    fmt.setDepthBufferSize(_int_env("ATOMSTUDIO_GL_DEPTH_BITS", 24, minimum=0))
    fmt.setSamples(_int_env("ATOMSTUDIO_GL_SAMPLES", 0, minimum=0))
    fmt.setSwapInterval(_int_env("ATOMSTUDIO_GL_SWAP_INTERVAL", 0, minimum=0))
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)


def _int_env(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    try:
        value = int(str(raw).strip()) if raw is not None and str(raw).strip() else int(default)
    except ValueError:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def _is_wslg_runtime() -> bool:
    return bool(os.environ.get("WSL_DISTRO_NAME") and os.environ.get("WAYLAND_DISPLAY") and Path("/dev/dxg").exists())


def _gl_profile() -> str:
    return str(os.environ.get("ATOMSTUDIO_GL_PROFILE", "auto") or "auto").strip().lower()


def _configure_wslg_d3d12() -> None:
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "d3d12")
    dri_path = _system_dri_path()
    if dri_path:
        os.environ.setdefault("LIBGL_DRIVERS_PATH", dri_path)

    adapter = str(os.environ.get("ATOMSTUDIO_GL_ADAPTER", "NVIDIA") or "NVIDIA").strip()
    if adapter and adapter.lower() not in {"auto", "default", "system"}:
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", adapter)


def _system_dri_path() -> str | None:
    path = Path("/usr/lib/x86_64-linux-gnu/dri")
    return str(path) if (path / "d3d12_dri.so").is_file() else None


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


__all__ = ["configure_native_opengl_runtime", "configure_preview_runtime", "configure_qt_runtime"]
