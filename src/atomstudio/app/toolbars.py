from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = None
    QtGui = None
    QtWidgets = None


@dataclass
class ToolbarHandles:
    actions: dict[str, Any] = field(default_factory=dict)
    controls: dict[str, Any] = field(default_factory=dict)


def _add_action(toolbar: Any, handles: ToolbarHandles, key: str, label: str, callback, *, tooltip: str) -> Any:
    action = QtGui.QAction(label, toolbar)
    action.setToolTip(tooltip)
    action.triggered.connect(callback)
    toolbar.addAction(action)
    handles.actions[key] = action
    return action


def build_view_toolbar(window: Any) -> ToolbarHandles:
    if QtWidgets is None or QtGui is None or QtCore is None:  # pragma: no cover
        return ToolbarHandles()

    handles = ToolbarHandles()
    toolbar = QtWidgets.QToolBar("View Controls", window)
    toolbar.setObjectName("view_controls_toolbar")
    toolbar.setMovable(False)
    toolbar.setFloatable(False)
    toolbar.setIconSize(QtCore.QSize(24, 24))

    def add(key: str, label: str, callback, tooltip: str) -> Any:
        return _add_action(toolbar, handles, key, label, callback, tooltip=tooltip)

    def add_step_control(key: str, label: str, default: float, minimum: float, maximum: float, decimals: int) -> Any:
        toolbar.addWidget(QtWidgets.QLabel(label, toolbar))
        spinbox = QtWidgets.QDoubleSpinBox(toolbar)
        spinbox.setDecimals(decimals)
        spinbox.setRange(float(minimum), float(maximum))
        spinbox.setValue(float(default))
        spinbox.setMaximumWidth(82)
        toolbar.addWidget(spinbox)
        handles.controls[key] = spinbox
        return spinbox

    add("view_axis_a", "a", lambda: window.set_preview_axis_view("a"), "View perpendicular to a / x axis")
    add("view_axis_b", "b", lambda: window.set_preview_axis_view("b"), "View perpendicular to b / y axis")
    add("view_axis_c", "c", lambda: window.set_preview_axis_view("c"), "View perpendicular to c / z axis")
    toolbar.addSeparator()

    for axis in ("a", "b", "c"):
        add(
            f"rotate_{axis}_ccw",
            f"{axis}-",
            lambda _checked=False, value=axis: window.rotate_preview_view(value, -1, window.view_rotation_step_degrees()),
            f"Rotate counterclockwise around {axis} axis",
        )
        add(
            f"rotate_{axis}_cw",
            f"{axis}+",
            lambda _checked=False, value=axis: window.rotate_preview_view(value, 1, window.view_rotation_step_degrees()),
            f"Rotate clockwise around {axis} axis",
        )
    add_step_control("rotation_step_degrees", "step (deg):", 15.0, 0.1, 360.0, 1)
    toolbar.addSeparator()

    add("pan_left", "<", lambda: window.pan_preview_view(-window.view_pan_step_pixels(), 0.0), "Pan structure left")
    add("pan_right", ">", lambda: window.pan_preview_view(window.view_pan_step_pixels(), 0.0), "Pan structure right")
    add("pan_up", "^", lambda: window.pan_preview_view(0.0, window.view_pan_step_pixels()), "Pan structure up")
    add("pan_down", "v", lambda: window.pan_preview_view(0.0, -window.view_pan_step_pixels()), "Pan structure down")
    add_step_control("pan_step_pixels", "step (px):", 24.0, 1.0, 1000.0, 0)
    toolbar.addSeparator()

    add("zoom_in", "Zoom +", lambda: window.zoom_preview_view(1.2), "Zoom in")
    add("zoom_out", "Zoom -", lambda: window.zoom_preview_view(1.0 / 1.2), "Zoom out")
    add("fit_screen", "Fit", window.fit_preview_to_structure, "Fit structure to screen")

    window.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)
    window._view_toolbar = toolbar
    return handles


def build_mouse_toolbar(window: Any) -> ToolbarHandles:
    if QtWidgets is None or QtGui is None or QtCore is None:  # pragma: no cover
        return ToolbarHandles()

    handles = ToolbarHandles()
    toolbar = QtWidgets.QToolBar("Mouse Tools", window)
    toolbar.setObjectName("mouse_tools_toolbar")
    toolbar.setMovable(False)
    toolbar.setFloatable(False)
    toolbar.setOrientation(QtCore.Qt.Orientation.Vertical)
    toolbar.setIconSize(QtCore.QSize(28, 28))
    group = QtGui.QActionGroup(window)
    group.setExclusive(True)

    specs = (
        ("rotate", "Rotate", "Rotate view with mouse drag"),
        ("select", "Select", "Box-select atoms"),
        ("pan", "Pan", "Pan view with mouse drag"),
        ("measure_distance", "Dist", "Measure distance between two atoms"),
        ("measure_angle", "Angle", "Measure angle across three atoms"),
        ("measure_dihedral", "Dihedral", "Measure dihedral across four atoms"),
    )
    for mode, label, tooltip in specs:
        action = QtGui.QAction(label, toolbar)
        action.setToolTip(tooltip)
        action.setCheckable(True)
        action.triggered.connect(lambda _checked=False, value=mode: window.set_mouse_mode(value))
        toolbar.addAction(action)
        group.addAction(action)
        handles.actions[mode] = action
    handles.actions["rotate"].setChecked(True)

    window.addToolBar(QtCore.Qt.ToolBarArea.LeftToolBarArea, toolbar)
    window._mouse_toolbar = toolbar
    return handles


__all__ = ["ToolbarHandles", "build_mouse_toolbar", "build_view_toolbar"]
