from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from atomstudio.app.main import _configure_qt_runtime

_configure_qt_runtime()

from PySide6 import QtWidgets

from atomstudio.app.toolbars import build_mouse_toolbar, build_view_toolbar


def _qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _ToolbarWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, object]] = []

    def set_preview_axis_view(self, axis: str) -> None:
        self.calls.append(("axis", axis))

    def view_rotation_step_degrees(self) -> float:
        return 30.0

    def view_pan_step_pixels(self) -> float:
        return 12.0

    def rotate_preview_view(self, axis: str, direction: int, degrees: float | None = None) -> None:
        self.calls.append(("rotate", (axis, direction, degrees)))

    def pan_preview_view(self, dx: float, dy: float) -> None:
        self.calls.append(("pan", (dx, dy)))

    def zoom_preview_view(self, factor: float) -> None:
        self.calls.append(("zoom", factor))

    def fit_preview_to_structure(self) -> None:
        self.calls.append(("fit", None))

    def set_mouse_mode(self, mode: str) -> None:
        self.calls.append(("mode", mode))


def test_build_view_toolbar_registers_view_controls() -> None:
    _qt_app()
    window = _ToolbarWindow()
    handles = build_view_toolbar(window)

    assert "view_axis_a" in handles.actions
    assert list(key for key in handles.actions if key.startswith("rotate_")) == [
        "rotate_a_ccw",
        "rotate_a_cw",
        "rotate_b_ccw",
        "rotate_b_cw",
        "rotate_c_ccw",
        "rotate_c_cw",
    ]
    assert "rotate_b_ccw" in handles.actions
    assert "pan_up" in handles.actions
    assert "zoom_in" in handles.actions
    assert "fit_screen" in handles.actions
    assert "rotation_step_degrees" in handles.controls
    assert "pan_step_pixels" in handles.controls

    handles.actions["view_axis_a"].trigger()
    handles.actions["rotate_c_cw"].trigger()
    handles.actions["pan_left"].trigger()
    handles.actions["zoom_out"].trigger()
    handles.actions["fit_screen"].trigger()

    assert ("axis", "a") in window.calls
    assert ("rotate", ("c", 1, 30.0)) in window.calls
    assert ("pan", (-12.0, 0.0)) in window.calls
    assert ("fit", None) in window.calls
    assert any(name == "zoom" for name, _value in window.calls)


def test_build_mouse_toolbar_registers_exclusive_modes() -> None:
    _qt_app()
    window = _ToolbarWindow()
    handles = build_mouse_toolbar(window)

    assert list(handles.actions) == [
        "rotate",
        "select",
        "pan",
        "measure_distance",
        "measure_angle",
        "measure_dihedral",
    ]
    assert handles.actions["rotate"].isChecked()

    handles.actions["measure_angle"].trigger()

    assert ("mode", "measure_angle") in window.calls
    assert handles.actions["measure_angle"].isChecked()
