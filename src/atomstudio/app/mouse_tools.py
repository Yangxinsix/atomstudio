from __future__ import annotations

import time
from typing import Any

from atomstudio.preview.picking import point_distance_2d
from atomstudio.preview.interaction import MeasurementController, normalize_mouse_mode

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = None
    QtWidgets = None


class PreviewMouseToolController(QtCore.QObject if QtCore is not None else object):
    """Own mouse tool state and dispatch; PreviewCanvas owns rendering only."""

    def __init__(self, canvas: Any, native_widget: Any, parent: Any | None = None) -> None:
        if QtCore is not None:
            super().__init__(parent)
        self.canvas = canvas
        self.native_widget = native_widget
        self.mode = "rotate"
        self._drag_start: tuple[float, float] | None = None
        self._drag_last: tuple[float, float] | None = None
        self._middle_pan_active = False
        self._last_press: tuple[float, tuple[float, float]] | None = None
        self._measurement = MeasurementController()
        self._rubber_band = None

    def install(self) -> None:
        self.canvas.set_camera_mouse_enabled(True)

    def set_mode(self, mode: str) -> str:
        self.mode = normalize_mouse_mode(mode)
        self._drag_start = None
        self._drag_last = None
        self._middle_pan_active = False
        self._last_press = None
        self._measurement.clear()
        self._hide_selection_rect()
        self.canvas.set_camera_mouse_enabled(self.mode == "rotate")
        self.canvas.emit_interaction_message(f"Mouse mode: {self.mode.replace('_', ' ')}")
        return self.mode

    def handle_press(self, event: Any) -> None:
        if self._is_middle_button(event):
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            self._mark_event_handled(event)
            point = (float(pos[0]), float(pos[1]))
            self._drag_start = point
            self._drag_last = point
            self._middle_pan_active = True
            return
        if not self._is_left_button(event):
            return
        pos = getattr(event, "pos", None)
        if pos is None:
            return
        point = (float(pos[0]), float(pos[1]))
        if self.mode == "rotate":
            if self._is_double_press(point):
                self._mark_event_handled(event)
                self.canvas.select_object_at_screen(
                    point,
                    toggle=self._event_has_ctrl(event),
                    clear_on_miss=False,
                )
            self._last_press = (time.monotonic(), point)
            return
        self._mark_event_handled(event)
        if self.mode == "select":
            self._drag_start = point
            self._drag_last = point
            ctrl = self._event_has_ctrl(event)
            self._hide_selection_rect()
            self.canvas.select_object_at_screen(point, toggle=ctrl, clear_on_miss=not ctrl)
            return
        if self.mode == "pan":
            self._drag_start = point
            self._drag_last = point
            return
        if self.mode.startswith("measure_"):
            self._handle_measure_click(point)

    def _is_double_press(self, point: tuple[float, float]) -> bool:
        if self._last_press is None:
            return False
        last_time, last_point = self._last_press
        return (time.monotonic() - last_time) <= 0.45 and point_distance_2d(point, last_point) <= 6.0

    def handle_double_click(self, event: Any) -> None:
        pos = getattr(event, "pos", None)
        if pos is None or self.mode != "rotate":
            return
        self._mark_event_handled(event)
        self.canvas.select_object_at_screen(
            (float(pos[0]), float(pos[1])),
            toggle=self._event_has_ctrl(event),
            clear_on_miss=False,
        )

    def handle_move(self, event: Any) -> None:
        if self._middle_pan_active and self._drag_last is not None:
            self._handle_pan_drag(event)
            return
        if self.mode == "select" and self._drag_start is not None:
            pos = getattr(event, "pos", None)
            if pos is None:
                return
            self._mark_event_handled(event)
            point = (float(pos[0]), float(pos[1]))
            self._drag_last = point
            self._show_selection_rect(self._drag_start, point)
            return
        if self.mode != "pan" or self._drag_last is None:
            return
        self._handle_pan_drag(event)

    def handle_release(self, event: Any) -> None:
        if self._middle_pan_active:
            self._mark_event_handled(event)
            self._drag_start = None
            self._drag_last = None
            self._middle_pan_active = False
            return
        if self.mode == "pan":
            self._mark_event_handled(event)
            self._drag_start = None
            self._drag_last = None
            return
        if self.mode != "select" or self._drag_start is None:
            return
        pos = getattr(event, "pos", None)
        if pos is None:
            return
        self._mark_event_handled(event)
        end = (float(pos[0]), float(pos[1]))
        if abs(end[0] - self._drag_start[0]) > 4.0 or abs(end[1] - self._drag_start[1]) > 4.0:
            selected = self.canvas.select_objects_in_screen_rect(
                self._drag_start,
                end,
                append=self._event_has_ctrl(event),
            )
            self.canvas.emit_interaction_message(
                f"Selected {len(selected.get('atoms', ()))} atom(s), {len(selected.get('bonds', ()))} bond(s)"
            )
        self._hide_selection_rect()
        self._drag_start = None
        self._drag_last = None

    def _handle_pan_drag(self, event: Any) -> None:
        pos = getattr(event, "pos", None)
        if pos is None or self._drag_last is None:
            return
        self._mark_event_handled(event)
        point = (float(pos[0]), float(pos[1]))
        dx = point[0] - self._drag_last[0]
        dy = self._drag_last[1] - point[1]
        self.canvas.pan_view(dx, dy)
        self._drag_last = point

    def _handle_measure_click(self, point: tuple[float, float]) -> None:
        selection = self.canvas.atom_selection_at_screen(point)
        if selection is None or selection.kind != "atom":
            return
        atom_index = int(selection.index)
        complete, atom_indices = self._measurement.add_atom(self.mode, atom_index)
        required = self._measurement.required_count(self.mode)
        self.canvas.select_atom_indices(set(atom_indices))
        if not complete:
            self.canvas.emit_interaction_message(f"Picked atom {atom_index}; pick {required - len(atom_indices)} more")
            return
        self.canvas.emit_interaction_message(self.canvas.measurement_message(atom_indices))
        self._measurement.clear()

    def measurement_message(self, scene: Any, atom_indices: list[int]) -> str:
        return self._measurement.message(scene, atom_indices)

    def _show_selection_rect(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        if QtCore is None or QtWidgets is None or self.native_widget is None:
            return
        if abs(end[0] - start[0]) <= 4.0 and abs(end[1] - start[1]) <= 4.0:
            return
        if self._rubber_band is None:
            self._rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self.native_widget)
        rect = QtCore.QRect(
            QtCore.QPoint(int(round(start[0])), int(round(start[1]))),
            QtCore.QPoint(int(round(end[0])), int(round(end[1]))),
        ).normalized()
        self._rubber_band.setGeometry(rect)
        self._rubber_band.show()

    def _hide_selection_rect(self) -> None:
        if self._rubber_band is not None:
            self._rubber_band.hide()

    @staticmethod
    def _mark_event_handled(event: Any) -> None:
        try:
            event.handled = True
        except Exception:
            return

    @staticmethod
    def _is_left_button(event: Any) -> bool:
        button = getattr(event, "button", None)
        if callable(button):
            button = button()
        return button in {1, "left", "LeftButton", None}

    @staticmethod
    def _is_middle_button(event: Any) -> bool:
        button = getattr(event, "button", None)
        if callable(button):
            button = button()
        return button in {3, 4, "middle", "MiddleButton"}

    @staticmethod
    def _event_has_ctrl(event: Any) -> bool:
        modifiers = getattr(event, "modifiers", ()) or ()
        if QtCore is not None and hasattr(QtCore.Qt, "KeyboardModifier"):
            try:
                return bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
            except TypeError:
                pass
        try:
            return any(str(item).lower() in {"control", "ctrl"} for item in modifiers)
        except TypeError:
            return False


__all__ = ["PreviewMouseToolController"]
