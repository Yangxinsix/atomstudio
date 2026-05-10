from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from atomstudio.app.main import _configure_qt_runtime

_configure_qt_runtime()

from PySide6 import QtWidgets

from atomstudio.app.menus import build_main_menu


def _qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _MenuWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, object]] = []

    def _record(self, name: str, value: object = None) -> None:
        self.calls.append((name, value))

    def open_structure_dialog(self) -> None:
        self._record("open")

    def reload_current_input(self) -> None:
        self._record("reload")

    def export_render_config_yaml(self) -> None:
        self._record("export")

    def copy_selection_summary(self) -> None:
        self._record("copy_summary")

    def copy_selection_json(self) -> None:
        self._record("copy_json")

    def delete_selected_objects(self) -> None:
        self._record("delete")

    def fit_preview_to_structure(self) -> None:
        self._record("fit")

    def reset_preview_camera(self) -> None:
        self._record("reset_camera")

    def set_preview_view(self, view: str) -> None:
        self._record("view", view)

    def set_axis_overlay_visible(self, visible: bool) -> None:
        self._record("axis", visible)

    def set_inspector_visible(self, visible: bool) -> None:
        self._record("inspector", visible)

    def set_log_visible(self, visible: bool) -> None:
        self._record("log", visible)

    def clear_preview_selection(self) -> None:
        self._record("clear")

    def cycle_selection(self, kind: str, step: int) -> None:
        self._record("cycle", (kind, step))

    def refresh_preview_from_menu(self) -> None:
        self._record("refresh")

    def render_final_image(self) -> None:
        self._record("render")

    def apply_style_choice(self, field_name: str, value: str) -> None:
        self._record("style", (field_name, value))

    def reset_window_layout(self) -> None:
        self._record("layout")

    def focus_preview(self) -> None:
        self._record("focus_preview")

    def focus_inspector(self) -> None:
        self._record("focus_inspector")

    def show_keyboard_shortcuts(self) -> None:
        self._record("shortcuts")

    def show_about_dialog(self) -> None:
        self._record("about")


def test_build_main_menu_registers_actions_and_wires_callbacks():
    _qt_app()
    window = _MenuWindow()
    handles = build_main_menu(window)

    assert "open_structure" in handles.actions
    assert "toggle_axis_overlay" in handles.actions
    assert "delete_selection" in handles.actions
    assert "scene_style" in handles.style_actions
    assert "glass" in handles.style_actions["material_style"]

    handles.actions["open_structure"].trigger()
    handles.actions["view_top"].trigger()
    handles.actions["next_bond"].trigger()
    handles.actions["delete_selection"].trigger()
    handles.style_actions["material_style"]["glass"].trigger()

    assert ("open", None) in window.calls
    assert ("view", "top") in window.calls
    assert ("cycle", ("bond", 1)) in window.calls
    assert ("delete", None) in window.calls
    assert ("style", ("material_style", "glass")) in window.calls
