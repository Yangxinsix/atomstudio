from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = None
    QtWidgets = None


if QtWidgets is not None:  # pragma: no cover - exercised only when GUI deps are installed

    class PreviewView(QtWidgets.QWidget):
        def __init__(self, parent: Any | None = None) -> None:
            super().__init__(parent)
            self._canvas = self._build_canvas(self)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self._canvas, 1)
            self.setLayout(layout)

        def _build_canvas(self, parent: Any | None):
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

        @property
        def canvas(self):
            return self._canvas

        def __getattr__(self, name: str) -> Any:
            return getattr(self._canvas, name)

else:  # pragma: no cover - importable fallback for tests and docs

    class PreviewView:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PySide6 is required to instantiate PreviewView")
