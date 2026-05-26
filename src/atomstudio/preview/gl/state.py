from __future__ import annotations

import os
from typing import Any


SOFTWARE_RENDERER_MARKERS = (
    "llvmpipe",
    "softpipe",
    "software rasterizer",
    "swrast",
    "mesa offscreen",
    "microsoft basic render",
    "gdi generic",
)


def _gl():
    try:
        from OpenGL import GL  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional GUI extra
        raise RuntimeError("PyOpenGL is required for the OpenGL preview backend") from exc
    return GL


def configure_default_state(*, enable_msaa: bool = True) -> None:
    GL = _gl()
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDepthFunc(GL.GL_LEQUAL)
    GL.glDepthMask(GL.GL_TRUE)
    GL.glDisable(GL.GL_BLEND)
    if enable_msaa and hasattr(GL, "GL_MULTISAMPLE"):
        try:
            GL.glEnable(GL.GL_MULTISAMPLE)
        except Exception:
            pass


def resize_viewport(width: int, height: int) -> None:
    GL = _gl()
    GL.glViewport(0, 0, max(1, int(width)), max(1, int(height)))


def clear_frame(background: tuple[float, float, float, float]) -> None:
    GL = _gl()
    r, g, b, a = (float(value) for value in background)
    GL.glClearColor(r, g, b, a)
    GL.glClearDepth(1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)


def decode_gl_string(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    data = getattr(value, "data", None)
    if callable(data):
        try:
            raw = data()
        except Exception:
            raw = None
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
    return str(value)


def context_info() -> dict[str, Any]:
    GL = _gl()
    renderer = decode_gl_string(GL.glGetString(GL.GL_RENDERER))
    vendor = decode_gl_string(GL.glGetString(GL.GL_VENDOR))
    info: dict[str, Any] = {
        "opengl_version": decode_gl_string(GL.glGetString(GL.GL_VERSION)),
        "renderer": renderer,
        "vendor": vendor,
        "software_renderer": is_software_renderer(renderer=renderer, vendor=vendor),
        "depth_bits": None,
        "samples": None,
        "max_viewport_dims": "unavailable",
    }
    for key, constant_name in (
        ("depth_bits", "GL_DEPTH_BITS"),
        ("samples", "GL_SAMPLES"),
        ("max_viewport_dims", "GL_MAX_VIEWPORT_DIMS"),
    ):
        constant = getattr(GL, constant_name, None)
        if constant is None:
            continue
        try:
            value = GL.glGetIntegerv(constant)
            if key == "max_viewport_dims":
                info[key] = tuple(int(item) for item in value)
            else:
                info[key] = int(value)
        except Exception:
            info[key] = "unavailable" if key == "max_viewport_dims" else None
    return info


def is_software_renderer(*, renderer: str | None, vendor: str | None = None) -> bool:
    text = " ".join(part.lower() for part in (renderer or "", vendor or "") if part)
    return any(marker in text for marker in SOFTWARE_RENDERER_MARKERS)


def graphics_environment() -> dict[str, str]:
    keys = (
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XDG_SESSION_TYPE",
        "QT_QPA_PLATFORM",
        "QT_OPENGL",
        "PYOPENGL_PLATFORM",
        "GALLIUM_DRIVER",
        "MESA_D3D12_DEFAULT_ADAPTER_NAME",
        "LIBGL_ALWAYS_SOFTWARE",
        "LIBGL_DRIVERS_PATH",
        "MESA_LOADER_DRIVER_OVERRIDE",
        "ATOMSTUDIO_GL_PROFILE",
        "ATOMSTUDIO_GL_ADAPTER",
        "ATOMSTUDIO_INTERACTION_DRIVER",
        "ATOMSTUDIO_INTERACTION_TIMER_MS",
        "ATOMSTUDIO_INTERACTION_REPAINT",
        "ATOMSTUDIO_GL_DEPTH_BITS",
        "ATOMSTUDIO_GL_SAMPLES",
        "ATOMSTUDIO_GL_SWAP_INTERVAL",
        "ATOMSTUDIO_GL_VSYNC",
        "vblank_mode",
        "__GL_SYNC_TO_VBLANK",
        "WSL_DISTRO_NAME",
    )
    return {key.lower(): os.environ.get(key, "") for key in keys}


__all__ = [
    "clear_frame",
    "configure_default_state",
    "context_info",
    "decode_gl_string",
    "graphics_environment",
    "is_software_renderer",
    "resize_viewport",
]
