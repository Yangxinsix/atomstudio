from __future__ import annotations

from typing import Any

from .widgets import PreviewView

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = None
    QtWidgets = None


def build_preview_host(parent: Any | None = None, *, preview_backend: str = "opengl"):
    if QtWidgets is None:  # pragma: no cover - import-time fallback only
        return None
    try:
        return PreviewView(parent=parent, backend=preview_backend)
    except Exception:
        placeholder = QtWidgets.QLabel("Preview canvas unavailable")
        if QtCore is not None and hasattr(QtCore, "Qt"):
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        placeholder.setMinimumWidth(360)
        placeholder.setMinimumHeight(360)
        return placeholder


def apply_preview_scene(
    canvas: Any,
    preview_scene: Any,
    *,
    frame_index: int | None = None,
    preserve_camera: bool | None = None,
) -> None:
    if canvas is None:
        return
    for method_name in (
        "set_preview_scene",
        "update_preview_scene",
        "set_scene",
        "load_scene",
        "update_scene",
        "set_data",
    ):
        method = getattr(canvas, method_name, None)
        if callable(method):
            kwargs: dict[str, Any] = {}
            if frame_index is not None:
                kwargs["frame_index"] = frame_index
            if preserve_camera is not None:
                kwargs["preserve_camera"] = preserve_camera
            if not kwargs:
                method(preview_scene)
            else:
                try:
                    method(preview_scene, **kwargs)
                except TypeError:
                    if frame_index is None:
                        method(preview_scene)
                    else:
                        try:
                            method(preview_scene, frame_index=frame_index)
                        except TypeError:
                            method(preview_scene)
            return
    if hasattr(canvas, "scene"):
        canvas.scene = preview_scene
