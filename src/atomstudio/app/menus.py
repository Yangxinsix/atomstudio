from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atomstudio.style.registry import (
    color_style_choices,
    light_style_choices,
    material_style_choices,
    radius_style_choices,
    scene_style_choices,
)

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtGui, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtGui = None
    QtWidgets = None


@dataclass
class MenuHandles:
    actions: dict[str, Any] = field(default_factory=dict)
    style_actions: dict[str, dict[str, Any]] = field(default_factory=dict)


def _make_action(
    window: Any,
    label: str,
    callback,
    *,
    shortcut: str | list[str] | tuple[str, ...] | None = None,
    checkable: bool = False,
    checked: bool = False,
) -> Any:
    action = QtGui.QAction(label, window)
    if isinstance(shortcut, (list, tuple)):
        action.setShortcuts([QtGui.QKeySequence(str(item)) for item in shortcut])
    elif shortcut:
        action.setShortcut(shortcut)
    action.setCheckable(checkable)
    if checkable:
        action.setChecked(bool(checked))
    action.triggered.connect(callback)
    return action


def build_main_menu(window: Any) -> MenuHandles:
    if QtWidgets is None or QtGui is None:  # pragma: no cover
        return MenuHandles()

    handles = MenuHandles()
    menu_bar = window.menuBar()

    file_menu = menu_bar.addMenu("&File")
    edit_menu = menu_bar.addMenu("&Edit")
    view_menu = menu_bar.addMenu("&View")
    select_menu = menu_bar.addMenu("&Select")
    render_menu = menu_bar.addMenu("&Render")
    tools_menu = menu_bar.addMenu("&Tools")
    window_menu = menu_bar.addMenu("&Window")
    help_menu = menu_bar.addMenu("&Help")

    def add(menu, key: str, label: str, callback, **kwargs):
        action = _make_action(window, label, callback, **kwargs)
        menu.addAction(action)
        handles.actions[key] = action
        return action

    add(file_menu, "open_structure", "Open Structure...", window.open_structure_dialog, shortcut="Ctrl+O")
    add(file_menu, "reload_current", "Reload Current", window.reload_current_input, shortcut="Ctrl+Shift+R")
    file_menu.addSeparator()
    add(file_menu, "export_render_config", "Export Render Config YAML...", window.export_render_config_yaml, shortcut="Ctrl+E")
    add(file_menu, "export_render_script", "Export render script...", window.export_render_script)
    file_menu.addSeparator()
    add(file_menu, "quit", "Quit", window.close, shortcut="Ctrl+Q")

    add(edit_menu, "undo", "Undo", window.undo_last_change, shortcut="Ctrl+Z")
    add(edit_menu, "redo", "Redo", window.redo_last_change, shortcut=("Ctrl+Y", "Ctrl+Shift+Z"))
    edit_menu.addSeparator()
    add(edit_menu, "copy_selection_summary", "Copy Selection Summary", window.copy_selection_summary, shortcut="Ctrl+C")
    add(edit_menu, "copy_selection_json", "Copy Selection JSON", window.copy_selection_json, shortcut="Ctrl+Shift+C")
    edit_menu.addSeparator()
    add(edit_menu, "delete_selection", "Delete Selection", window.delete_selected_objects, shortcut="Delete")

    add(view_menu, "fit_to_structure", "Fit to Structure", window.fit_preview_to_structure, shortcut="F")
    add(view_menu, "reset_camera", "Reset Camera", window.reset_preview_camera, shortcut="Shift+F")
    view_menu.addSeparator()
    add(view_menu, "view_orbit", "Orbit", lambda: window.set_preview_view("orbit"), shortcut="1")
    add(view_menu, "view_top", "Top", lambda: window.set_preview_view("top"), shortcut="2")
    add(view_menu, "view_front", "Front", lambda: window.set_preview_view("front"), shortcut="3")
    add(view_menu, "view_side", "Side", lambda: window.set_preview_view("side"), shortcut="4")
    view_menu.addSeparator()
    add(
        view_menu,
        "toggle_wrap_atoms_into_cell",
        "Wrap Atoms Into Cell",
        lambda checked: window.set_wrap_atoms_into_cell(bool(checked)),
        checkable=True,
        checked=False,
    )
    add(
        view_menu,
        "toggle_axis_overlay",
        "Toggle Axis Overlay",
        lambda checked: window.set_axis_overlay_visible(bool(checked)),
        checkable=True,
        checked=True,
    )
    add(
        view_menu,
        "toggle_inspector_dock",
        "Toggle Inspector Dock",
        lambda checked: window.set_inspector_visible(bool(checked)),
        checkable=True,
        checked=True,
    )
    add(
        view_menu,
        "toggle_status_log",
        "Toggle Status Log",
        lambda checked: window.set_log_visible(bool(checked)),
        checkable=True,
        checked=True,
    )

    add(select_menu, "clear_selection", "Clear Selection", window.clear_preview_selection)
    add(select_menu, "next_atom", "Next Atom", lambda: window.cycle_selection("atom", 1), shortcut="]")
    add(select_menu, "previous_atom", "Previous Atom", lambda: window.cycle_selection("atom", -1), shortcut="[")
    add(select_menu, "next_bond", "Next Bond", lambda: window.cycle_selection("bond", 1), shortcut="}")
    add(select_menu, "previous_bond", "Previous Bond", lambda: window.cycle_selection("bond", -1), shortcut="{")

    add(render_menu, "refresh_preview", "Refresh Preview", window.refresh_preview_from_menu, shortcut="Ctrl+R")
    add(render_menu, "render_final_image", "Render Final Image", window.render_final_image, shortcut="Ctrl+Shift+P")

    style_specs = (
        ("scene_style", "Scene Style", scene_style_choices()),
        ("color_style", "Color Style", color_style_choices()),
        ("material_style", "Material Style", material_style_choices()),
        ("light_style", "Light Style", light_style_choices()),
        ("radius_style", "Radius Style", radius_style_choices()),
    )
    for field_name, label, choices in style_specs:
        submenu = tools_menu.addMenu(label)
        action_group = QtGui.QActionGroup(window)
        action_group.setExclusive(True)
        handles.style_actions[field_name] = {}
        for choice in choices:
            action = _make_action(
                window,
                str(choice),
                lambda checked=False, value=choice, field=field_name: window.apply_style_choice(field, value),
                checkable=True,
            )
            action_group.addAction(action)
            submenu.addAction(action)
            handles.style_actions[field_name][str(choice)] = action

    add(window_menu, "reset_layout", "Reset Layout", window.reset_window_layout)
    add(window_menu, "focus_preview", "Focus Preview", window.focus_preview)
    add(window_menu, "focus_inspector", "Focus Inspector", window.focus_inspector)

    add(help_menu, "keyboard_shortcuts", "Keyboard Shortcuts", window.show_keyboard_shortcuts)
    add(help_menu, "about", "About", window.show_about_dialog)
    return handles
