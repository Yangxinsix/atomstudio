from __future__ import annotations

import os
from typing import Any

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = None
    QtWidgets = None


if QtWidgets is not None:  # pragma: no cover - exercised only when GUI deps are installed

    class PreviewView(QtWidgets.QWidget):
        def __init__(self, parent: Any | None = None, *, backend: str = "opengl") -> None:
            super().__init__(parent)
            self.backend = self._normalize_backend(backend)
            self._canvas = self._build_canvas(self, backend=self.backend)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self._canvas, 1)
            self.setLayout(layout)

        @staticmethod
        def _normalize_backend(backend: str | None) -> str:
            value = str(backend or "opengl").strip().lower()
            if value in {"opengl", "gl"}:
                return "opengl"
            if value in {"opengl-window", "gl-window"}:
                return "opengl-window"
            if value in {"opengl-widget", "gl-widget"}:
                return "opengl-widget"
            if value in {"opengl-detached", "gl-detached", "opengl-top", "gl-top"}:
                return "opengl-detached"
            return "vispy"

        def _build_canvas(self, parent: Any | None, *, backend: str):
            if backend == "opengl":
                surface = str(os.environ.get("ATOMSTUDIO_GL_SURFACE", "widget") or "widget").strip().lower()
                if surface in {"detached", "top", "top-level", "toplevel"}:
                    from atomstudio.app.preview_window import PreviewDetachedHost

                    return PreviewDetachedHost(parent=parent)
                if surface in {"window", "qopenglwindow"} and self._qt_platform() != "offscreen":
                    try:
                        from atomstudio.app.preview_window import PreviewWindowHost

                        return PreviewWindowHost(parent=parent)
                    except Exception:
                        pass
                from atomstudio.app.preview_widget import PreviewWidget

                return PreviewWidget(parent=parent)
            if backend == "opengl-window":
                try:
                    from atomstudio.app.preview_window import PreviewWindowHost

                    return PreviewWindowHost(parent=parent)
                except Exception:
                    from atomstudio.app.preview_widget import PreviewWidget

                    return PreviewWidget(parent=parent)
            if backend == "opengl-widget":
                from atomstudio.app.preview_widget import PreviewWidget

                return PreviewWidget(parent=parent)
            if backend == "opengl-detached":
                from atomstudio.app.preview_window import PreviewDetachedHost

                return PreviewDetachedHost(parent=parent)
            try:
                from atomstudio.app.preview_canvas import PreviewCanvas  # type: ignore

                try:
                    return PreviewCanvas(parent=parent)
                except TypeError:
                    return PreviewCanvas()
            except Exception:
                placeholder = QtWidgets.QLabel("Preview canvas unavailable", parent)
                if QtCore is not None and hasattr(QtCore, "Qt"):
                    placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                placeholder.setMinimumWidth(360)
                placeholder.setMinimumHeight(360)
                return placeholder

        @staticmethod
        def _qt_platform() -> str:
            if QtWidgets is None:
                return ""
            app = QtWidgets.QApplication.instance()
            if app is not None:
                try:
                    return str(app.platformName()).lower()
                except Exception:
                    pass
            return str(os.environ.get("QT_QPA_PLATFORM", "") or "").lower()

        @property
        def canvas(self):
            return self._canvas

        def __getattr__(self, name: str) -> Any:
            return getattr(self._canvas, name)

else:  # pragma: no cover - importable fallback for tests and docs

    class PreviewView:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PySide6 is required to instantiate PreviewView")
