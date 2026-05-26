from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from atomstudio.preview.renderer import PreviewCameraState

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = QtGui = QtWidgets = None


if QtWidgets is not None:  # pragma: no cover - exercised by Qt tests

    class AxisOverlayWidget(QtWidgets.QWidget):
        ORIGIN = (60.0, 84.0)
        AXIS_LENGTH = 48.0
        MIN_PROJECTED_LENGTH = 0.18

        def __init__(self, parent: Any | None = None, *, background: tuple[float, float, float, float]) -> None:
            super().__init__(parent)
            self.setFixedSize(128, 128)
            if QtCore is not None:
                self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self._background = tuple(float(v) for v in background)
            self._camera = PreviewCameraState()

        def sync_camera(self, camera: PreviewCameraState) -> None:
            self._camera = replace(camera)
            self.update()

        def paintEvent(self, _event: Any) -> None:
            if QtGui is None or QtCore is None:
                return
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self._draw_panel(painter)
            origin = QtCore.QPointF(*self.ORIGIN)
            axes = [
                ("X", (1.0, 0.0, 0.0), QtGui.QColor(210, 54, 45)),
                ("Y", (0.0, 1.0, 0.0), QtGui.QColor(40, 150, 72)),
                ("Z", (0.0, 0.0, 1.0), QtGui.QColor(42, 102, 210)),
            ]
            for label, _axis, color, endpoint, _projected_len, _depth in sorted(
                ((label, axis, color, *self._axis_projection(origin, axis)) for label, axis, color in axes),
                key=lambda item: item[5],
            ):
                self._draw_axis(painter, origin, endpoint, label, color)

        def _draw_panel(self, painter: Any) -> None:
            alpha = 36 if self._background[3] >= 0.9 else 20
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(255, 255, 255, alpha))
            painter.drawRoundedRect(QtCore.QRectF(8.0, 8.0, 112.0, 112.0), 12.0, 12.0)

        def _draw_axis(self, painter: Any, start: Any, end: Any, label: str, color: Any) -> None:
            vector = end - start
            length = max(1.0, (vector.x() ** 2 + vector.y() ** 2) ** 0.5)
            direction = QtCore.QPointF(vector.x() / length, vector.y() / length)
            normal = QtCore.QPointF(-direction.y(), direction.x())
            arrow_a = end - direction * 9.0 + normal * 4.5
            arrow_b = end - direction * 9.0 - normal * 4.5

            pen = QtGui.QPen(color, 3.0)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(start, end)
            painter.setBrush(color)
            painter.drawPolygon(QtGui.QPolygonF([end, arrow_a, arrow_b]))

            font = painter.font()
            font.setBold(True)
            font.setPointSize(9)
            painter.setFont(font)
            painter.drawText(end + direction * 8.0 + normal * 2.0, label)

        def _project_axis(self, origin: Any, axis: tuple[float, float, float]) -> Any:
            return self._axis_projection(origin, axis)[0]

        def _axis_projection(self, origin: Any, axis: tuple[float, float, float]) -> tuple[Any, float, float]:
            right, up, forward = self._camera.basis()
            axis = self._rotated_axis(axis)
            x = self._dot(axis, right)
            y = self._dot(axis, up)
            depth = self._dot(axis, forward)
            projected_len = (x * x + y * y) ** 0.5
            if projected_len < 0.12:
                y = -self.MIN_PROJECTED_LENGTH if depth < 0.0 else self.MIN_PROJECTED_LENGTH
                x = 0.0
                projected_len = self.MIN_PROJECTED_LENGTH
            visible_len = max(self.MIN_PROJECTED_LENGTH, projected_len)
            scale = self.AXIS_LENGTH * visible_len
            endpoint = origin + QtCore.QPointF((x / visible_len) * scale, -(y / visible_len) * scale)
            return endpoint, projected_len, depth

        def _rotated_axis(self, axis: tuple[float, float, float]) -> tuple[float, float, float]:
            values = getattr(self._camera, "model_rotation", None)
            if values is None:
                return axis
            try:
                matrix = np.asarray(values, dtype=float).reshape((4, 4))
            except (TypeError, ValueError):
                return axis
            vector = matrix[:3, :3] @ np.asarray(axis, dtype=float)
            norm = float(np.linalg.norm(vector))
            if norm <= 1.0e-12:
                return axis
            return tuple(float(value) for value in vector / norm)

        @staticmethod
        def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
            return float(left[0] * right[0] + left[1] * right[1] + left[2] * right[2])


else:  # pragma: no cover

    class AxisOverlayWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PySide6 is required to instantiate AxisOverlayWidget")


__all__ = ["AxisOverlayWidget"]
