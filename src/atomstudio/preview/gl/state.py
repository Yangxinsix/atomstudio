from __future__ import annotations

from typing import Any


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
    info: dict[str, Any] = {
        "opengl_version": decode_gl_string(GL.glGetString(GL.GL_VERSION)),
        "renderer": decode_gl_string(GL.glGetString(GL.GL_RENDERER)),
        "vendor": decode_gl_string(GL.glGetString(GL.GL_VENDOR)),
        "depth_bits": None,
        "samples": None,
    }
    for key, constant_name in (("depth_bits", "GL_DEPTH_BITS"), ("samples", "GL_SAMPLES")):
        constant = getattr(GL, constant_name, None)
        if constant is None:
            continue
        try:
            info[key] = int(GL.glGetIntegerv(constant))
        except Exception:
            info[key] = None
    return info


__all__ = [
    "clear_frame",
    "configure_default_state",
    "context_info",
    "decode_gl_string",
    "resize_viewport",
]
