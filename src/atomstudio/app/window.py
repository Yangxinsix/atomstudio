from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from atomstudio.config import AtomStyleRuleConfig, BoundaryAtomsConfig, CellStyleConfig, RenderJobConfig
from atomstudio.paths import normalize_host_path
from atomstudio.preview.camera import model_rotation_euler_degrees
from atomstudio.preview.gl.shader_styles import DEFAULT_SHADER_STYLE, resolve_shader_style
from atomstudio.preview.selection import build_selection_payload as preview_build_selection_payload
from atomstudio.preview.types import PreviewSelection
from atomstudio.scene.builder import build_render_scene
from atomstudio.scene.materials.specs import as_material_spec
from atomstudio.structure.boundary import wrap_structure_into_cell
from atomstudio.structure.bonding import BondEngine
from atomstudio.structure.data import normalize_pair_key
from atomstudio.structure.selectors import AtomSelector
from atomstudio.structure.structure import Structure
from atomstudio.visual_defaults import CELL_DEFAULT_COLOR

from .control_panels import build_object_tab, build_render_tab, build_scene_tab
from .inspector import SelectionInspector
from .menus import MenuHandles, build_main_menu
from .panels import apply_preview_scene, build_preview_host
from .render_script_export import (
    build_render_script_text,
    default_batch_output_spec,
    default_render_script_path,
)
from .state import AppState, AppUndoSnapshot, LoadedFrameBundle
from .summary import build_structure_summary
from .toolbars import ToolbarHandles, build_mouse_toolbar, build_view_toolbar
from .workers import (
    AnimationRenderRequest,
    AnimationRenderWorker,
    LoadStructureRequest,
    LoadStructureWorker,
    PreviewRequest,
    PreviewWorker,
    RenderRequest,
    RenderWorker,
    build_default_render_config,
    start_background_task,
)

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = QtGui = QtWidgets = None


def _resolve_qt_align_center() -> Any:
    if QtCore is not None and hasattr(QtCore, "Qt"):
        return QtCore.Qt.AlignmentFlag.AlignCenter
    return 0


def _default_frame_selector() -> str:
    return "all"


def _build_output_path(structure) -> str:
    if structure.source_path:
        source = Path(normalize_host_path(structure.source_path)).expanduser()
        stem = source.stem or source.name or "atomstudio_render"
        return str((source.parent / f"{stem}.png").resolve())
    return str((Path.cwd() / "atomstudio_render.png").resolve())


def _build_animation_output_spec(output_path: str, structure) -> tuple[str, str]:
    raw = normalize_host_path(output_path or "")
    if not raw:
        raw = _build_output_path(structure)
    path = Path(raw).expanduser()
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    if path.suffix.lower() in image_suffixes:
        output_dir = path.parent / f"{path.stem}_frames"
        template = f"{path.stem}_{{frame:04d}}.png"
    else:
        output_dir = path
        template = "frame_{frame:04d}.png"
    return str(output_dir.resolve()), template


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {str(k): _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    return value


def _record_field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _selection_fields(selection: Any | None) -> tuple[str | None, int | None]:
    if selection is None:
        return None, None
    kind = _record_field(selection, "kind")
    index = _record_field(selection, "index")
    if kind is None or index is None:
        atom_index = _record_field(selection, "atom_index")
        if atom_index is not None:
            return "atom", int(atom_index)
        return None, None
    return str(kind).strip().lower(), int(index)


def _find_record(records: Any, *, key: str, value: int) -> Any | None:
    if not records:
        return None
    for record in records:
        try:
            if int(_record_field(record, key, -1)) == int(value):
                return record
        except Exception:
            continue
    if 0 <= int(value) < len(records):
        return records[int(value)]
    return None


def _resolve_atom_payload(scene: Any, atom_index: int) -> dict[str, Any] | None:
    atom_records = getattr(scene, "atom_records", None)
    record = _find_record(atom_records, key="index", value=atom_index)
    if record is not None:
        object_payload = {
            key: _jsonable(_record_field(record, key))
            for key in ("index", "symbol", "atomic_number", "position", "radius", "representation", "style", "tag")
        }
        metadata = _jsonable(_record_field(record, "metadata", {})) or {}
        metadata.setdefault("selection_kind", "atom")
        metadata.setdefault("selection_index", atom_index)
        return {
            "object": object_payload,
            "material": _jsonable(_record_field(record, "material", {})) or {},
            "metadata": metadata,
        }

    atoms = getattr(scene, "atoms", None)
    if atoms is None:
        return None
    indices = list(_jsonable(getattr(atoms, "atom_indices", [])) or _jsonable(getattr(atoms, "indices", [])) or [])
    positions = list(_jsonable(getattr(atoms, "positions", [])) or [])
    radii = list(_jsonable(getattr(atoms, "radii", [])) or [])
    atomic_numbers = list(_jsonable(getattr(atoms, "atomic_numbers", [])) or [])
    symbols = list(getattr(atoms, "symbols", ()) or [])
    representations = list(getattr(atoms, "representations", ()) or [])
    ordinal = indices.index(atom_index) if atom_index in indices else None
    if ordinal is None:
        return None
    return {
        "object": {
            "index": atom_index,
            "symbol": symbols[ordinal] if ordinal < len(symbols) else "X",
            "atomic_number": atomic_numbers[ordinal] if ordinal < len(atomic_numbers) else None,
            "position": positions[ordinal] if ordinal < len(positions) else None,
            "radius": radii[ordinal] if ordinal < len(radii) else None,
            "representation": representations[ordinal] if ordinal < len(representations) else None,
            "style": None,
            "tag": None,
        },
        "material": {},
        "metadata": {"selection_kind": "atom", "selection_index": atom_index},
    }


def _resolve_bond_payload(scene: Any, bond_index: int) -> dict[str, Any] | None:
    bond_records = getattr(scene, "bond_records", None)
    record = _find_record(bond_records, key="id", value=bond_index) or _find_record(bond_records, key="index", value=bond_index)
    if record is not None:
        object_payload = {
            key: _jsonable(_record_field(record, key))
            for key in ("id", "a", "b", "bond_type", "order", "distance", "split_ratio")
        }
        material_payload = {
            "uniform": _jsonable(_record_field(record, "material_uniform", None)),
            "left": _jsonable(_record_field(record, "material_left", None)),
            "right": _jsonable(_record_field(record, "material_right", None)),
        }
        metadata = _jsonable(_record_field(record, "metadata", {})) or {}
        metadata.setdefault("selection_kind", "bond")
        metadata.setdefault("selection_index", bond_index)
        return {
            "object": object_payload,
            "material": {key: value for key, value in material_payload.items() if value is not None},
            "metadata": metadata,
        }

    bonds = getattr(scene, "bonds", None)
    if bonds is None:
        return None
    bond_ids = list(_jsonable(getattr(bonds, "bond_ids", [])) or [])
    atom_indices = list(_jsonable(getattr(bonds, "atom_indices", [])) or [])
    orders = list(_jsonable(getattr(bonds, "orders", [])) or [])
    bond_types = list(getattr(bonds, "bond_types", ()) or [])
    split_ratios = list(_jsonable(getattr(bonds, "split_ratios", [])) or [])
    ordinal = bond_ids.index(bond_index) if bond_index in bond_ids else None
    if ordinal is None:
        return None
    pair = atom_indices[ordinal] if ordinal < len(atom_indices) else [None, None]
    return {
        "object": {
            "id": bond_index,
            "a": pair[0] if len(pair) > 0 else None,
            "b": pair[1] if len(pair) > 1 else None,
            "bond_type": bond_types[ordinal] if ordinal < len(bond_types) else None,
            "order": orders[ordinal] if ordinal < len(orders) else None,
            "distance": None,
            "split_ratio": split_ratios[ordinal] if ordinal < len(split_ratios) else None,
        },
        "material": {},
        "metadata": {"selection_kind": "bond", "selection_index": bond_index},
    }


def build_selection_payload(scene: Any | None, selection: Any | None) -> dict[str, Any] | None:
    return preview_build_selection_payload(scene, selection)


def _selection_summary(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "No selection"
    obj = payload.get("object") if isinstance(payload, dict) else None
    if not isinstance(obj, dict):
        return "No selection"
    if obj.get("symbol") is not None:
        return (
            f"Atom {obj.get('index')} ({obj.get('symbol')}) "
            f"at {obj.get('position')} radius={obj.get('radius')}"
        )
    return (
        f"Bond {obj.get('id')} ({obj.get('a')} - {obj.get('b')}) "
        f"order={obj.get('order')} distance={obj.get('distance')}"
    )


def _selection_atom_objects(canvas: Any, structure: Structure | None = None) -> list[dict[str, Any]]:
    model = getattr(canvas, "model", None)
    scene = getattr(model, "scene", None)
    selected = getattr(model, "selected_ordered_atoms", None) or sorted(getattr(model, "selected_atom_indices", set()) or [])
    if scene is None or not selected:
        return []
    selected_set = {int(index) for index in selected}
    atoms = [
        atom
        for atom in getattr(scene, "atoms", ())
        if int(getattr(atom, "index", -1)) in selected_set
        and not bool((getattr(getattr(atom, "record", None), "metadata", {}) or {}).get("boundary_atom"))
    ]
    order = {int(index): position for position, index in enumerate(selected)}
    atoms.sort(key=lambda atom: order.get(int(atom.index), int(atom.index)))
    return [
        _add_fractional_to_atom_payload(
            {
                "index": int(atom.index),
                "symbol": str(atom.symbol),
                "position": [float(value) for value in atom.position],
                "color": [float(value) for value in atom.color],
                "radius": float(atom.radius),
            },
            structure,
        )
        for atom in atoms
    ]


def _color_css(color: Any) -> str:
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return "#b8b8b8"
    values = [max(0, min(255, int(round(float(value) * 255.0)))) for value in color[:3]]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def _color_text(color: Any) -> str:
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return ""
    values = [float(value) for value in color[:4]]
    if len(values) == 3:
        values.append(1.0)
    return ", ".join(f"{value:.4f}" for value in values)


def _parse_float_tuple(text: str, *, length: int) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(text).replace(";", ",").split(",") if item.strip()]
    if len(values) != int(length):
        raise ValueError(f"Expected {length} comma-separated values")
    return tuple(values)


def _cell_matrix(structure: Structure | None) -> np.ndarray | None:
    raw = getattr(structure, "cell_vectors", None) if structure is not None else None
    if raw is None:
        return None
    try:
        cell = np.asarray(raw, dtype=float).reshape((3, 3))
    except Exception:
        return None
    if not np.any(np.abs(cell) > 1e-8) or abs(float(np.linalg.det(cell))) <= 1e-12:
        return None
    return cell


def _fractional_from_position(structure: Structure | None, position: Any) -> tuple[float, float, float] | None:
    cell = _cell_matrix(structure)
    if cell is None:
        return None
    try:
        cart = np.asarray(position, dtype=float).reshape((3,))
    except Exception:
        return None
    frac = cart @ np.linalg.inv(cell)
    return (float(frac[0]), float(frac[1]), float(frac[2]))


def _position_from_fractional(structure: Structure | None, fractional: Any) -> tuple[float, float, float] | None:
    cell = _cell_matrix(structure)
    if cell is None:
        return None
    try:
        frac = np.asarray(fractional, dtype=float).reshape((3,))
    except Exception:
        return None
    cart = frac @ cell
    return (float(cart[0]), float(cart[1]), float(cart[2]))


def _format_vector(values: Any, *, digits: int = 6) -> str:
    vector = [] if values is None else list(values)
    return ", ".join(f"{float(vector[idx] if len(vector) > idx else 0.0):.{digits}f}" for idx in range(3))


def _add_fractional_to_atom_payload(atom: dict[str, Any], structure: Structure | None) -> dict[str, Any]:
    out = dict(atom)
    position = out.get("position")
    fractional = _fractional_from_position(structure, (0.0, 0.0, 0.0) if position is None else position)
    if fractional is not None:
        out["fractional"] = [float(value) for value in fractional]
    return out


def _add_fractional_to_selection_payload(payload: dict[str, Any] | None, structure: Structure | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    obj = out.get("object")
    if isinstance(obj, dict) and obj.get("symbol") is not None:
        out["object"] = _add_fractional_to_atom_payload(obj, structure)
    objects = out.get("objects")
    if isinstance(objects, list):
        out["objects"] = [
            _add_fractional_to_atom_payload(item, structure) if isinstance(item, dict) and item.get("symbol") is not None else item
            for item in objects
        ]
    return out


def _bond_pair_key(structure: Structure, bond: Any) -> str | None:
    symbols = list(getattr(structure, "symbols", ()) or ())
    a = int(getattr(bond, "a", -1))
    b = int(getattr(bond, "b", -1))
    if a < 0 or b < 0 or a >= len(symbols) or b >= len(symbols):
        return None
    return normalize_pair_key(f"{symbols[a]}-{symbols[b]}")


def _changed_bond_rule_pair_keys(old_bonding: Any, new_bonding: Any) -> set[str]:
    old_distances = {normalize_pair_key(str(key)): tuple(float(v) for v in value) for key, value in old_bonding.pair_distances.items()}
    new_distances = {normalize_pair_key(str(key)): tuple(float(v) for v in value) for key, value in new_bonding.pair_distances.items()}
    old_disabled = {normalize_pair_key(str(value)) for value in old_bonding.disabled_pairs}
    new_disabled = {normalize_pair_key(str(value)) for value in new_bonding.disabled_pairs}
    old_orders = {normalize_pair_key(str(key)): int(value) for key, value in old_bonding.order_rules.items()}
    new_orders = {normalize_pair_key(str(key)): int(value) for key, value in new_bonding.order_rules.items()}

    keys = set(old_distances) | set(new_distances) | old_disabled | new_disabled | set(old_orders) | set(new_orders)
    changed: set[str] = set()
    for key in keys:
        if old_distances.get(key) != new_distances.get(key):
            changed.add(key)
            continue
        if (key in old_disabled) != (key in new_disabled):
            changed.add(key)
            continue
        if old_orders.get(key, 1) != new_orders.get(key, 1):
            changed.add(key)
    return changed


def _sort_bonds_for_display(bonds: list[Any]) -> list[Any]:
    return sorted(
        bonds,
        key=lambda bond: (
            min(int(getattr(bond, "a", 0)), int(getattr(bond, "b", 0))),
            max(int(getattr(bond, "a", 0)), int(getattr(bond, "b", 0))),
            str(getattr(bond, "bond_type", "")),
        ),
    )


_NO_ELEMENT_STYLE_CHANGE = object()


def _app_style_sheet() -> str:
    return """
AtomStudioWindow {
    font-size: 12pt;
}
QMenuBar, QMenu {
    font-size: 13pt;
}
QToolBar, QToolButton {
    font-size: 12pt;
    padding: 4px 6px;
}
QGroupBox {
    font-size: 12pt;
    font-weight: 600;
}
QTabWidget, QTabBar::tab, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox {
    font-size: 12pt;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 28px;
    max-height: 28px;
}
QRadioButton, QCheckBox {
    font-size: 12pt;
    spacing: 8px;
    min-height: 26px;
}
QPushButton#colorSwatchButton {
    min-width: 34px;
    max-width: 34px;
    min-height: 28px;
    max-height: 28px;
    padding: 0px;
}
QStatusBar QLabel {
    font-size: 11pt;
}
QPlainTextEdit#statusLogView,
QPlainTextEdit#inspectorObjectView,
QPlainTextEdit#inspectorSummaryView,
QPlainTextEdit#inspectorMetadataView {
    font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 11pt;
}
"""


if QtWidgets is not None:  # pragma: no cover - exercised only when GUI deps are installed

    class AtomStudioWindow(QtWidgets.QMainWindow):
        def __init__(self, *, state: AppState | None = None, preview_backend: str = "opengl") -> None:
            super().__init__()
            self.state = state or AppState()
            self.preview_backend = self._normalize_preview_backend(preview_backend)
            self._threads: list[Any] = []
            self._preview_canvas = None
            self._frame_count = 0
            self._menu_handles = MenuHandles()
            self._toolbar_handles = ToolbarHandles()
            self._mouse_toolbar_handles = ToolbarHandles()
            self._inspector_output_text: str | None = None
            self._render_in_progress = False
            self._preview_shader_style = DEFAULT_SHADER_STYLE
            self._syncing_preview_rotation_controls = False
            self._syncing_element_style_controls = False
            self._element_style_loaded_values: tuple[str, tuple[float, float, float, float], float] | None = None
            self._element_radius_dirty = False
            self._preview_request_serial = 0

            self._build_ui()
            self._sync_state_to_ui()

        @staticmethod
        def _normalize_preview_backend(preview_backend: str | None) -> str:
            value = str(preview_backend or "opengl").strip().lower()
            if value in {"opengl", "gl"}:
                return "opengl"
            if value in {"opengl-window", "gl-window"}:
                return "opengl-window"
            if value in {"opengl-widget", "gl-widget"}:
                return "opengl-widget"
            if value in {"opengl-detached", "gl-detached", "opengl-top", "gl-top"}:
                return "opengl-detached"
            return "vispy"

        def _build_ui(self) -> None:
            self.setObjectName("AtomStudioWindow")
            self.setStyleSheet(_app_style_sheet())
            self.setMinimumSize(1200, 800)
            self.setDockNestingEnabled(True)

            central = QtWidgets.QWidget(self)
            root = QtWidgets.QHBoxLayout(central)
            root.setContentsMargins(12, 12, 12, 12)
            root.setSpacing(12)

            self._build_hidden_file_controls()
            controls = QtWidgets.QVBoxLayout()
            controls.setSpacing(10)
            self._left_tabs = QtWidgets.QTabWidget()
            self._left_tabs.setObjectName("leftControlTabs")
            self._left_tabs.addTab(build_scene_tab(self), "Scene")
            self._left_tabs.addTab(build_render_tab(self), "Render")
            self._left_tabs.addTab(build_object_tab(self), "Object")
            controls.addWidget(self._left_tabs, 1)

            left = QtWidgets.QWidget()
            left.setLayout(controls)
            left.setMinimumWidth(430)
            left.setMaximumWidth(560)

            if self.preview_backend in {"opengl", "opengl-window", "opengl-widget", "opengl-detached"}:
                try:
                    self._preview_canvas = build_preview_host(self, preview_backend=self.preview_backend)
                except TypeError:
                    self._preview_canvas = build_preview_host(self)
            else:
                self._preview_canvas = build_preview_host(self)
            if self._preview_canvas is None:
                self._preview_canvas = QtWidgets.QLabel("Preview canvas unavailable")
                self._preview_canvas.setAlignment(_resolve_qt_align_center())
            if hasattr(self._preview_canvas, "setFocusPolicy") and QtCore is not None:
                self._preview_canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

            preview_container = QtWidgets.QVBoxLayout()
            preview_container.addWidget(self._preview_canvas, 1)

            preview_widget = QtWidgets.QWidget()
            preview_widget.setLayout(preview_container)

            root.addWidget(left, 0)
            root.addWidget(preview_widget, 1)
            self.setCentralWidget(central)

            self._log_view = QtWidgets.QPlainTextEdit()
            self._log_view.setObjectName("statusLogView")
            self._log_view.setReadOnly(True)
            self._log_view.setMaximumBlockCount(2000)
            self._log_dock = QtWidgets.QDockWidget("Status Log", self)
            self._log_dock.setWidget(self._log_view)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)

            self._inspector = SelectionInspector(self)
            self._inspector_dock = QtWidgets.QDockWidget("", self)
            inspector_title = QtWidgets.QWidget(self._inspector_dock)
            inspector_title.setFixedHeight(0)
            self._inspector_dock.setTitleBarWidget(inspector_title)
            self._inspector_dock.setWidget(self._inspector)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self._inspector_dock)
            self.tabifyDockWidget(self._log_dock, self._inspector_dock)
            self._inspector_dock.raise_()

            self._status_label = QtWidgets.QLabel("Ready")
            self.statusBar().addWidget(self._status_label, 1)
            self._dirty_label = QtWidgets.QLabel("Clean")
            self.statusBar().addPermanentWidget(self._dirty_label)
            self._selection_label = QtWidgets.QLabel("No selection")
            self.statusBar().addPermanentWidget(self._selection_label)

            self._menu_handles = build_main_menu(self)
            self._toolbar_handles = build_view_toolbar(self)
            self._mouse_toolbar_handles = build_mouse_toolbar(self)

            self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
            self._render_button.clicked.connect(self._on_render_clicked)
            self._animation_button.clicked.connect(self._on_animation_clicked)
            self._browse_output_button.clicked.connect(self._browse_output)
            self._browse_blender_button.clicked.connect(self._browse_blender)
            self._output_path_edit.editingFinished.connect(lambda: self._mark_dirty("Output path changed"))
            self._transparent_bg_radio.toggled.connect(lambda _checked: self._apply_render_settings())
            self._unicolor_bg_radio.toggled.connect(lambda _checked: self._apply_render_settings())
            self._background_color_edit.editingFinished.connect(self._apply_render_settings)
            self._log_dock.visibilityChanged.connect(self._on_log_visibility_changed)
            self._inspector_dock.visibilityChanged.connect(self._on_inspector_visibility_changed)
            self._connect_preview_selection()
            self._connect_preview_interaction_messages()
            self._connect_preview_camera_changes()

            self._update_window_title()

        def _build_hidden_file_controls(self) -> None:
            self._input_path_edit = QtWidgets.QLineEdit()
            self._frame_selector_edit = QtWidgets.QLineEdit(_default_frame_selector())

        def _browse_input(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open structure",
                str(Path.cwd()),
                "Structures (*.xyz *.cif *.traj *.db *.json *.yaml *.yml);;All files (*)",
            )
            if path:
                self._input_path_edit.setText(normalize_host_path(path))
                self._frame_selector_edit.setText(_default_frame_selector())
                self._on_load_clicked()

        def _browse_output(self) -> None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select output path",
                str(Path.cwd() / "atomstudio_render.png"),
                "Images (*.png *.jpg *.jpeg *.webp);;All files (*)",
            )
            if path:
                self._output_path_edit.setText(normalize_host_path(path))
                self._mark_dirty("Output path changed")

        def _browse_blender(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Blender executable",
                str(Path.cwd()),
                "Executables (*)",
            )
            if path:
                self._blender_path_edit.setText(normalize_host_path(path))

        def _append_log(self, level: str, message: str) -> None:
            self.state.append_log(level, message)
            self._log_view.appendPlainText(f"[{level.upper()}] {message}")

        def _track_thread(self, handle: Any | None) -> None:
            if handle is None:
                return
            self._threads.append(handle)
            finished = getattr(handle, "finished", None)
            if finished is None:
                return

            def _remove_finished() -> None:
                if handle in self._threads:
                    self._threads.remove(handle)

            finished.connect(_remove_finished)

        def _mark_dirty(self, reason: str) -> None:
            self.state.mark_dirty(reason)
            self._sync_state_to_ui()

        def _capture_undo_snapshot(self, label: str) -> AppUndoSnapshot:
            return self.state.capture_undo_snapshot(label)

        def _commit_undo_snapshot(self, snapshot: AppUndoSnapshot) -> None:
            self.state.push_undo_snapshot(snapshot)

        def undo_last_change(self) -> None:
            snapshot = self.state.undo()
            if snapshot is None:
                return
            self._restore_preview_after_history_change()
            self._append_log("info", f"Undo: {snapshot.label}")
            self._sync_state_to_ui()

        def redo_last_change(self) -> None:
            snapshot = self.state.redo()
            if snapshot is None:
                return
            self._restore_preview_after_history_change()
            self._append_log("info", f"Redo: {snapshot.label}")
            self._sync_state_to_ui()

        def _restore_preview_after_history_change(self) -> None:
            self._inspector_output_text = None
            if self.state.preview_scene is not None:
                apply_preview_scene(self._preview_canvas, self.state.preview_scene, frame_index=self.state.current_frame_index())
                self._apply_preview_shader_style_to_canvas()
            elif self.state.current_structure() is not None:
                self._refresh_preview()
            selection = self.state.selected_object
            if selection is not None:
                self._select_preview_object(selection.kind, selection.index)
                return
            method = getattr(self._preview_canvas, "clear_selection", None)
            if callable(method):
                method()

        def _connect_preview_selection(self) -> None:
            if self._preview_canvas is None:
                return
            signal = getattr(self._preview_canvas, "selection_changed", None)
            if signal is None:
                model = getattr(self._preview_canvas, "model", None)
                signal = getattr(model, "selection_changed", None)
            if signal is not None and hasattr(signal, "connect"):
                signal.connect(self._on_canvas_selection_changed)

        def _connect_preview_interaction_messages(self) -> None:
            signal = getattr(self._preview_canvas, "interaction_message_changed", None)
            if signal is not None and hasattr(signal, "connect"):
                signal.connect(self._on_canvas_interaction_message)
            for signal_name, callback in (
                ("delete_requested", self.delete_selected_objects),
                ("undo_requested", self.undo_last_change),
                ("redo_requested", self.redo_last_change),
            ):
                requested = getattr(self._preview_canvas, signal_name, None)
                if requested is not None and hasattr(requested, "connect"):
                    requested.connect(callback)

        def _connect_preview_camera_changes(self) -> None:
            signal = getattr(self._preview_canvas, "camera_changed", None)
            if signal is not None and hasattr(signal, "connect"):
                signal.connect(self._on_preview_camera_changed)
            self._sync_preview_rotation_controls()

        def _on_preview_camera_changed(self, _camera: Any = None) -> None:
            self._sync_preview_rotation_controls()

        def _set_action_checked(self, key: str, checked: bool) -> None:
            action = self._menu_handles.actions.get(key)
            if action is None or not getattr(action, "isCheckable", lambda: False)():
                return
            previous = action.blockSignals(True)
            try:
                action.setChecked(bool(checked))
            finally:
                action.blockSignals(previous)

        def _sync_style_actions(self) -> None:
            cfg = self.state.render_config
            if cfg is None:
                return
            current_values = {
                "scene_style": str(cfg.style.scene_style or "default"),
                "color_style": str(cfg.style.color_style or ""),
                "material_style": str(cfg.style.material_style or ""),
                "light_style": str(cfg.lighting.light_style or cfg.style.light_style or ""),
                "radius_style": str(cfg.style.radius_style or ""),
            }
            for field_name, actions in self._menu_handles.style_actions.items():
                selected = current_values.get(field_name, "")
                for value, action in actions.items():
                    previous = action.blockSignals(True)
                    try:
                        action.setChecked(str(value) == selected)
                    finally:
                        action.blockSignals(previous)

        def _sync_action_states(self) -> None:
            has_structure = self.state.current_structure() is not None
            has_selection = self.state.selected_payload is not None
            enable_map = {
                "reload_current": has_structure or bool(self._input_path_edit.text().strip()),
                "export_render_config": self.state.render_config is not None,
                "export_render_script": has_structure and self.state.render_config is not None,
                "copy_selection_summary": has_selection,
                "copy_selection_json": has_selection,
                "delete_selection": has_selection,
                "undo": self.state.can_undo(),
                "redo": self.state.can_redo(),
                "fit_to_structure": self.state.preview_scene is not None,
                "reset_camera": self.state.preview_scene is not None,
                "clear_selection": has_selection,
                "next_atom": bool(getattr(self.state.preview_scene, "atom_records", None)) or bool(getattr(getattr(self.state.preview_scene, "atoms", None), "count", 0)),
                "previous_atom": bool(getattr(self.state.preview_scene, "atom_records", None)) or bool(getattr(getattr(self.state.preview_scene, "atoms", None), "count", 0)),
                "next_bond": bool(getattr(self.state.preview_scene, "bond_records", None)) or bool(getattr(getattr(self.state.preview_scene, "bonds", None), "count", 0)),
                "previous_bond": bool(getattr(self.state.preview_scene, "bond_records", None)) or bool(getattr(getattr(self.state.preview_scene, "bonds", None), "count", 0)),
                "refresh_preview": has_structure,
                "render_final_image": has_structure,
                "focus_preview": self._preview_canvas is not None,
                "focus_inspector": True,
            }
            for key, enabled in enable_map.items():
                action = self._menu_handles.actions.get(key)
                if action is not None:
                    action.setEnabled(bool(enabled))

        def _sync_state_to_ui(self) -> None:
            self._dirty_label.setText("Dirty" if self.state.dirty else "Clean")
            self._status_label.setText(self.state.status)
            self._selection_label.setText(_selection_summary(self.state.selected_payload))
            self._update_window_title()
            if self.state.render_config is not None:
                output_path = normalize_host_path(self.state.render_config.output.path or "")
                if self._output_path_edit.text() != output_path:
                    self._output_path_edit.setText(output_path)
                self._sync_config_controls(self.state.render_config)
            if self.state.bundle is not None:
                current = self.state.bundle.current()
                if current is not None:
                    total = self.state.bundle.frame_count
                    self._frame_label.setText(f"Frame {self.state.bundle.selected_index + 1} of {total} (source {current.frame_index})")
                    self._sync_animation_range_controls(total)
                    step = self._animation_frame_step()
                    start, end = self._animation_frame_range()
                    animation_count = self._animation_structure_count()
                    if step <= 1 and start == 0 and end == total:
                        self._animation_scope_label.setText(f"All loaded frames ({total})")
                    else:
                        self._animation_scope_label.setText(
                            f"{animation_count} of {total} loaded frames ({start + 1}-{end}, step {step})"
                        )
                else:
                    self._frame_label.setText("No frame selected")
                    self._animation_scope_label.setText("No frames loaded")
            else:
                self._frame_label.setText("No frames loaded")
                self._animation_scope_label.setText("No frames loaded")
                self._sync_animation_range_controls(0)
            self._frame_slider.blockSignals(True)
            try:
                if self.state.bundle is None or self.state.bundle.frame_count == 0:
                    self._frame_slider.setEnabled(False)
                    self._frame_slider.setMinimum(0)
                    self._frame_slider.setMaximum(0)
                    self._frame_slider.setValue(0)
                else:
                    self._frame_slider.setEnabled(self.state.bundle.frame_count > 1)
                    self._frame_slider.setMinimum(0)
                    self._frame_slider.setMaximum(self.state.bundle.frame_count - 1)
                    self._frame_slider.setValue(self.state.bundle.selected_index)
            finally:
                self._frame_slider.blockSignals(False)
            frame_count = self.state.bundle.frame_count if self.state.bundle is not None else 0
            selected_index = self.state.bundle.selected_index if self.state.bundle is not None else 0
            self._first_frame_button.setEnabled(frame_count > 1 and selected_index > 0)
            self._previous_frame_button.setEnabled(frame_count > 1 and selected_index > 0)
            self._next_frame_button.setEnabled(frame_count > 1 and selected_index < frame_count - 1)
            self._last_frame_button.setEnabled(frame_count > 1 and selected_index < frame_count - 1)
            if self._inspector_dock.isVisible() != self.state.dock_visibility.inspector_visible:
                self._inspector_dock.setVisible(self.state.dock_visibility.inspector_visible)
            if self._log_dock.isVisible() != self.state.dock_visibility.log_visible:
                self._log_dock.setVisible(self.state.dock_visibility.log_visible)
            axis_toggle = getattr(self._preview_canvas, "set_axis_overlay_visible", None)
            if callable(axis_toggle):
                axis_toggle(self.state.dock_visibility.axis_overlay_visible)
            self._set_action_checked("toggle_inspector_dock", self.state.dock_visibility.inspector_visible)
            self._set_action_checked("toggle_status_log", self.state.dock_visibility.log_visible)
            self._set_action_checked("toggle_axis_overlay", self.state.dock_visibility.axis_overlay_visible)
            self._set_action_checked("toggle_wrap_atoms_into_cell", self.state.wrap_atoms_into_cell)
            self._sync_style_actions()
            self._sync_action_states()
            self._inspector.set_payload(self.state.selected_payload)
            if self._inspector_output_text is not None:
                self._inspector.set_output_text(self._inspector_output_text)
            self._inspector.set_summary(self._build_summary_text())
            self._set_object_editor_payload(self.state.selected_payload)
            self._sync_element_symbol_choices()

        def _sync_config_controls(self, cfg: RenderJobConfig) -> None:
            pairs = (
                (self._resolution_x_spin, int(cfg.render.resolution[0])),
                (self._resolution_y_spin, int(cfg.render.resolution[1])),
                (self._samples_spin, int(cfg.render.samples)),
                (self._sphere_segments_spin, int(cfg.structure.sphere_segments)),
                (self._sphere_rings_spin, int(cfg.structure.sphere_rings)),
                (self._bond_vertices_spin, int(cfg.structure.bond_vertices)),
            )
            for widget, value in pairs:
                previous = widget.blockSignals(True)
                try:
                    widget.setValue(value)
                finally:
                    widget.blockSignals(previous)
            previous = self._bond_radius_spin.blockSignals(True)
            try:
                self._bond_radius_spin.setValue(float(cfg.structure.bond_radius))
            finally:
                self._bond_radius_spin.blockSignals(previous)
            if hasattr(self, "_show_hydrogen_bonds_check"):
                previous_hbond = self._show_hydrogen_bonds_check.blockSignals(True)
                try:
                    self._show_hydrogen_bonds_check.setChecked(bool(cfg.structure.bonding.hbond.enabled))
                finally:
                    self._show_hydrogen_bonds_check.blockSignals(previous_hbond)
            self._set_combo_data(self._render_engine_combo, str(cfg.render.engine or "cycles"))
            self._set_combo_data(self._render_device_combo, str(cfg.render.device or "auto"))
            previous_cell_show = self._cell_show_check.blockSignals(True)
            previous_cell_radius = self._cell_radius_spin.blockSignals(True)
            previous_cell_color = self._cell_color_edit.blockSignals(True)
            previous_cell_transparent = self._cell_transparent_check.blockSignals(True)
            previous_boundary_atoms = self._boundary_atoms_check.blockSignals(True)
            previous_boundary_sigma = self._boundary_atom_sigma_spin.blockSignals(True)
            try:
                self._cell_show_check.setChecked(bool(cfg.structure.draw_cell or cfg.structure.cell_style.show))
                self._cell_radius_spin.setValue(float(cfg.structure.cell_style.radius))
                cell_color = cfg.structure.cell_style.color or CELL_DEFAULT_COLOR
                self._cell_color_edit.setText(_color_text(cell_color))
                self._set_color_button(self._cell_color_button, cell_color)
                self._cell_transparent_check.setChecked(bool(cfg.structure.cell_style.transparent))
                self._boundary_atoms_check.setChecked(bool(cfg.structure.boundary_atoms.enabled))
                self._boundary_atom_sigma_spin.setValue(float(cfg.structure.boundary_atoms.sigma))
            finally:
                self._cell_show_check.blockSignals(previous_cell_show)
                self._cell_radius_spin.blockSignals(previous_cell_radius)
                self._cell_color_edit.blockSignals(previous_cell_color)
                self._cell_transparent_check.blockSignals(previous_cell_transparent)
                self._boundary_atoms_check.blockSignals(previous_boundary_atoms)
                self._boundary_atom_sigma_spin.blockSignals(previous_boundary_sigma)
            self._set_button_group_value(self._atom_representation_buttons, str(cfg.structure.representation or "ball_stick"))
            self._set_button_group_value(self._scene_style_buttons, self._preview_shader_style)
            self._set_combo_data(self._render_style_combo, str(cfg.style.scene_style or "default"))
            self._set_button_group_value(self._camera_projection_buttons, str(cfg.camera.projection or "ORTHOGRAPHIC").upper())
            self._set_combo_data(self._bond_style_combo, self._current_bond_style())
            previous_transparent = self._transparent_bg_radio.blockSignals(True)
            previous_unicolor = self._unicolor_bg_radio.blockSignals(True)
            previous_background = self._background_color_edit.blockSignals(True)
            try:
                self._transparent_bg_radio.setChecked(bool(cfg.render.transparent_bg))
                self._unicolor_bg_radio.setChecked(not bool(cfg.render.transparent_bg))
                background = cfg.style.background or (1.0, 1.0, 1.0, 1.0)
                self._background_color_edit.setText(_color_text(background))
                self._set_color_button(self._background_color_button, background)
            finally:
                self._transparent_bg_radio.blockSignals(previous_transparent)
                self._unicolor_bg_radio.blockSignals(previous_unicolor)
                self._background_color_edit.blockSignals(previous_background)

        @staticmethod
        def _set_combo_data(combo: Any, value: str) -> None:
            target = str(value)
            for idx in range(combo.count()):
                if str(combo.itemData(idx)) == target or str(combo.itemText(idx)) == target:
                    previous = combo.blockSignals(True)
                    try:
                        combo.setCurrentIndex(idx)
                    finally:
                        combo.blockSignals(previous)
                    return

        @staticmethod
        def _combo_data(combo: Any, fallback: str) -> str:
            value = combo.currentData() if combo is not None else None
            if value is None:
                value = combo.currentText() if combo is not None else None
            return str(value or fallback)

        @staticmethod
        def _set_button_group_value(buttons: dict[str, Any], value: str) -> None:
            button = buttons.get(str(value))
            if button is None:
                return
            group = button.group()
            previous = group.blockSignals(True) if group is not None else False
            try:
                button.setChecked(True)
            finally:
                if group is not None:
                    group.blockSignals(previous)

        def _set_object_editor_payload(self, payload: dict[str, Any] | None) -> None:
            obj = payload.get("object") if isinstance(payload, dict) else None
            is_atom = isinstance(obj, dict) and obj.get("symbol") is not None
            for edit in self._object_edit_fields.values():
                edit.setEnabled(bool(is_atom))
                edit.clear()
            self._object_color_button.setEnabled(bool(is_atom))
            if not is_atom:
                self._set_color_button(self._object_color_button, None)
                return
            position = obj.get("position") or (0.0, 0.0, 0.0)
            material = payload.get("material") if isinstance(payload, dict) else None
            color = obj.get("color") or obj.get("face_color")
            if color is None and isinstance(material, dict):
                color = material.get("color")
            self._object_edit_fields["symbol"].setText(str(obj.get("symbol", "")))
            fractional = obj.get("fractional")
            self._object_edit_fields["fractional"].setEnabled(fractional is not None)
            self._object_edit_fields["fractional"].setText("" if fractional is None else _format_vector(fractional))
            self._object_edit_fields["position"].setText(", ".join(str(position[idx] if len(position) > idx else 0.0) for idx in range(3)))
            self._object_edit_fields["color"].setText(_color_text(color))
            self._object_edit_fields["radius"].setText("" if obj.get("radius") is None else str(obj.get("radius")))
            self._set_color_button(self._object_color_button, color)

        def _sync_element_symbol_choices(self) -> None:
            structures = self._loaded_frames_or_current()
            symbols = sorted({str(atom.symbol) for structure in structures for atom in structure.atoms})
            current = self._selected_element_symbol()
            previous = self._element_symbol_combo.blockSignals(True)
            self._syncing_element_style_controls = True
            try:
                self._element_symbol_combo.clear()
                for symbol in symbols:
                    self._element_symbol_combo.addItem(symbol, symbol)
                if current in symbols:
                    self._set_combo_data(self._element_symbol_combo, current)
                elif symbols:
                    self._element_symbol_combo.setCurrentIndex(0)
                else:
                    self._element_symbol_combo.setCurrentIndex(-1)
            finally:
                self._element_symbol_combo.blockSignals(previous)
                self._syncing_element_style_controls = False
            self._populate_element_style_controls()

        def _selected_element_symbol(self) -> str:
            data = self._element_symbol_combo.currentData()
            if data is not None:
                return str(data).strip()
            return self._element_symbol_combo.currentText().strip()

        def _populate_element_style_controls(self) -> None:
            symbol = self._selected_element_symbol()
            color, radius = self._resolved_element_default_style(symbol)
            previous_color = self._element_color_edit.blockSignals(True)
            previous_radius = self._element_radius_spin.blockSignals(True)
            was_syncing = self._syncing_element_style_controls
            self._syncing_element_style_controls = True
            try:
                self._element_color_edit.setText(_color_text(color))
                self._element_radius_spin.setValue(float(radius))
            finally:
                self._element_color_edit.blockSignals(previous_color)
                self._element_radius_spin.blockSignals(previous_radius)
                self._syncing_element_style_controls = was_syncing
            self._set_color_button(self._element_color_button, color)
            self._element_style_loaded_values = (symbol, tuple(float(value) for value in color), float(radius))
            self._element_radius_dirty = False

        def _resolved_element_default_style(self, symbol: str) -> tuple[tuple[float, float, float, float], float]:
            structure = self._reference_structure_for_symbol(symbol)
            cfg = self.state.render_config
            if structure is None or cfg is None:
                return (0.7, 0.7, 0.7, 1.0), 1.0
            try:
                default_structure = Structure.from_dict(structure.to_dict())
                for atom in default_structure.atoms:
                    if str(atom.symbol) != str(symbol):
                        continue
                    atom.color = None
                    atom.radius = None
                    atom.material = None
                    atom.representation = None
                    atom.style = None
                scene = build_render_scene(default_structure, cfg)
                for atom in scene.atoms:
                    if str(atom.symbol) == str(symbol):
                        return tuple(float(value) for value in as_material_spec(atom.material).color), float(atom.radius)
            except Exception:
                return (0.7, 0.7, 0.7, 1.0), 1.0
            return (0.7, 0.7, 0.7, 1.0), 1.0

        def _mark_element_radius_dirty(self) -> None:
            if not self._syncing_element_style_controls:
                self._element_radius_dirty = True

        @staticmethod
        def _is_element_default_rule(rule: AtomStyleRuleConfig, symbol: str) -> bool:
            selector = rule.selector
            return (
                selector.symbol == str(symbol)
                and not selector.symbols
                and not selector.indices
                and selector.index_range is None
                and selector.z_range is None
                and not selector.tags
            )

        def _config_with_element_default_style(
            self,
            cfg: RenderJobConfig,
            symbol: str,
            *,
            color: Any = _NO_ELEMENT_STYLE_CHANGE,
            radius: Any = _NO_ELEMENT_STYLE_CHANGE,
        ) -> RenderJobConfig:
            rules = list(cfg.style.atom_style_rules)
            for index, rule in enumerate(rules):
                if not self._is_element_default_rule(rule, symbol):
                    continue
                rules[index] = replace(
                    rule,
                    color=rule.color if color is _NO_ELEMENT_STYLE_CHANGE else tuple(float(value) for value in color),
                    radius=rule.radius if radius is _NO_ELEMENT_STYLE_CHANGE else float(radius),
                )
                break
            else:
                rules.append(
                    AtomStyleRuleConfig(
                        selector=AtomSelector(symbol=str(symbol)),
                        color=None if color is _NO_ELEMENT_STYLE_CHANGE else tuple(float(value) for value in color),
                        radius=None if radius is _NO_ELEMENT_STYLE_CHANGE else float(radius),
                    )
                )
            return replace(cfg, style=replace(cfg.style, atom_style_rules=rules))

        @staticmethod
        def _clear_element_overrides_that_shadow_default(
            atoms: list[Any],
            *,
            old_color: tuple[float, float, float, float] | None,
            new_color: tuple[float, float, float, float] | None,
            color_changed: bool,
            old_radius: float | None,
            new_radius: float | None,
            radius_changed: bool,
        ) -> None:
            for atom in atoms:
                if color_changed:
                    atom.color = None
                if radius_changed:
                    atom.radius = None
                atom.sync_color_to_material()

        @staticmethod
        def _checked_button_value(group: Any, fallback: str) -> str:
            button = group.checkedButton() if group is not None else None
            if button is None:
                return str(fallback)
            value = button.property("value")
            return str(value if value is not None else fallback)

        def _current_bond_style(self) -> str:
            structure = self.state.current_structure()
            bonds = getattr(structure, "bonds", ()) or ()
            for bond in bonds:
                if str(getattr(bond, "bond_type", "covalent")) != "hydrogen":
                    return str(getattr(bond, "metadata", {}).get("preview_bond_style", "bicolor"))
            if bonds:
                return str(getattr(bonds[0], "metadata", {}).get("preview_bond_style", "bicolor"))
            return "bicolor"

        def _build_summary_text(self) -> str:
            graphics = {}
            graphics_method = getattr(self._preview_canvas, "graphics_info", None)
            if callable(graphics_method):
                graphics = graphics_method()
            source_path = self.state.bundle.source_path if self.state.bundle is not None else ""
            return build_structure_summary(
                structure=self.state.current_structure(),
                preview_scene=self.state.preview_scene,
                source_path=source_path,
                graphics=graphics,
            )

        def _loaded_frames_or_current(self) -> list[Structure]:
            if self.state.bundle is not None:
                cached = self.state.bundle.cached_structures()
                if cached:
                    return cached
            current = self.state.current_structure()
            return [] if current is None else [current]

        def _animation_frame_step(self) -> int:
            control = getattr(self, "_animation_frame_step_spin", None)
            value = control.value() if control is not None and hasattr(control, "value") else 1
            return max(1, int(value))

        def _sync_animation_range_controls(self, total: int) -> None:
            start_control = getattr(self, "_animation_start_spin", None)
            end_control = getattr(self, "_animation_end_spin", None)
            if start_control is None or end_control is None:
                return
            maximum = max(1, int(total))
            for control in (start_control, end_control):
                previous = control.blockSignals(True)
                try:
                    control.setRange(1, maximum)
                finally:
                    control.blockSignals(previous)
            previous_start = start_control.blockSignals(True)
            previous_end = end_control.blockSignals(True)
            try:
                start_value = max(1, min(int(start_control.value()), maximum))
                end_value = int(end_control.value())
                if total > 0 and (end_value <= 1 or end_value > maximum):
                    end_value = maximum
                end_value = max(1, min(end_value, maximum))
                if start_value > end_value:
                    end_value = start_value
                start_control.setValue(start_value)
                end_control.setValue(end_value)
            finally:
                start_control.blockSignals(previous_start)
                end_control.blockSignals(previous_end)

        def _animation_frame_range(self) -> tuple[int, int]:
            total = self.state.bundle.frame_count if self.state.bundle is not None else 0
            if total <= 0:
                return 0, 0
            start_control = getattr(self, "_animation_start_spin", None)
            end_control = getattr(self, "_animation_end_spin", None)
            start_value = start_control.value() if start_control is not None and hasattr(start_control, "value") else 1
            end_value = end_control.value() if end_control is not None and hasattr(end_control, "value") else total
            start = max(1, min(int(start_value), total))
            end = max(start, min(int(end_value), total))
            return start - 1, end

        def _animation_structure_count(self) -> int:
            start, end = self._animation_frame_range()
            if end <= start:
                return 0
            step = self._animation_frame_step()
            return ((end - start - 1) // step) + 1

        def _animation_structures(self) -> tuple[Structure, ...]:
            if self.state.bundle is None:
                return ()
            start, end = self._animation_frame_range()
            step = self._animation_frame_step()
            return tuple(self.state.bundle.structures_in_range(start, end, step))

        def _reference_structure_for_symbol(self, symbol: str) -> Structure | None:
            for structure in self._loaded_frames_or_current():
                if any(str(atom.symbol) == str(symbol) for atom in structure.atoms):
                    return structure
            return self.state.current_structure()

        def _atoms_with_symbol_across_loaded_frames(self, symbol: str) -> list[Any]:
            return [
                atom
                for structure in self._loaded_frames_or_current()
                for atom in structure.atoms
                if str(atom.symbol) == str(symbol)
            ]

        def _update_window_title(self) -> None:
            title = "AtomStudio"
            if self.state.bundle is not None:
                source = Path(self.state.bundle.source_path).name or self.state.bundle.source_path
                title = f"{title} - {source}"
                current = self.state.bundle.current()
                if current is not None:
                    title = f"{title} [frame {current.frame_index}]"
            if self.state.dirty:
                title = f"{title} *"
            self.setWindowTitle(title)

        def _step_frame(self, step: int) -> None:
            if self.state.bundle is None:
                return
            self._select_loaded_frame(self.state.bundle.selected_index + int(step))

        def _first_frame(self) -> None:
            if self.state.bundle is not None:
                self._select_loaded_frame(0)

        def _last_frame(self) -> None:
            if self.state.bundle is not None:
                self._select_loaded_frame(max(0, self.state.bundle.frame_count - 1))

        def _on_frame_slider_changed(self, value: int) -> None:
            self._select_loaded_frame(int(value))

        def _select_loaded_frame(self, value: int) -> None:
            if self.state.bundle is None:
                return
            previous_index = int(self.state.bundle.selected_index)
            current = self.state.select_frame(int(value))
            if current is None:
                return
            if int(self.state.bundle.selected_index) == previous_index and int(value) == previous_index:
                self._sync_state_to_ui()
                return
            self.state.clear_selection()
            self._refresh_preview(preserve_camera=True)
            self._sync_state_to_ui()

        def _resolve_load_request(self) -> LoadStructureRequest:
            return LoadStructureRequest(
                input_path=normalize_host_path(self._input_path_edit.text().strip()),
                frame_selector=self._frame_selector_edit.text().strip() or _default_frame_selector(),
            )

        def _on_load_clicked(self) -> None:
            request = self._resolve_load_request()
            if not request.input_path:
                self._append_log("error", "Input path is required")
                self.state.last_error = "Input path is required"
                self._sync_state_to_ui()
                return

            self.state.set_status("Loading structure...")
            self._sync_state_to_ui()
            worker = LoadStructureWorker(request)
            handle = start_background_task(
                worker.run,
                on_result=self._handle_load_result,
                on_error=self._handle_task_error,
                parent=self,
                label="load_structure",
            )
            self._track_thread(handle)

        def _handle_load_result(self, bundle: LoadedFrameBundle) -> None:
            current = bundle.current()
            render_config = self.state.render_config
            if current is not None:
                if render_config is None:
                    render_config = build_default_render_config(current, _build_output_path(current))
                elif not render_config.output.path:
                    render_config = render_config.with_output_path(_build_output_path(current))
            self.state.set_loaded_frames(bundle, render_config=render_config, status="Structure loaded")
            self._append_log("info", f"Loaded {bundle.frame_count} frame(s)")
            self._sync_state_to_ui()
            self._refresh_preview()

        def _build_preview_request(self) -> PreviewRequest | None:
            structure = self.state.current_structure()
            if structure is None:
                return None
            if self.state.wrap_atoms_into_cell:
                structure = wrap_structure_into_cell(structure)
            render_config = self._ensure_render_config(structure=structure)
            render_config = self._render_config_from_ui(render_config, include_preview_camera=False)
            self.state.set_render_config(render_config, mark_dirty=self.state.dirty)
            return PreviewRequest(structure=structure, render_config=render_config, preview_settings=None)

        def _refresh_preview(self, *, preserve_camera: bool | None = None) -> None:
            request = self._build_preview_request()
            if request is None:
                return
            self._preview_request_serial += 1
            request_serial = int(self._preview_request_serial)
            request_frame_index = int(request.structure.frame_index)
            self.state.set_status("Updating preview...")
            self._sync_state_to_ui()
            worker = PreviewWorker(request)
            handle = start_background_task(
                worker.run,
                on_result=lambda preview_scene: self._handle_preview_result(
                    preview_scene,
                    preserve_camera=preserve_camera,
                    request_serial=request_serial,
                    request_frame_index=request_frame_index,
                ),
                on_error=self._handle_task_error,
                parent=self,
                label="preview_update",
            )
            self._track_thread(handle)

        def _handle_preview_result(
            self,
            preview_scene: Any,
            *,
            preserve_camera: bool | None = None,
            request_serial: int | None = None,
            request_frame_index: int | None = None,
        ) -> None:
            if request_serial is not None and int(request_serial) != int(self._preview_request_serial):
                return
            frame_index = self.state.current_frame_index() if request_frame_index is None else int(request_frame_index)
            current_frame_index = self.state.current_frame_index()
            if current_frame_index is not None and frame_index != int(current_frame_index):
                return
            self.state.set_preview_scene(preview_scene)
            apply_preview_scene(
                self._preview_canvas,
                preview_scene,
                frame_index=frame_index,
                preserve_camera=preserve_camera,
            )
            self._apply_preview_shader_style_to_canvas()
            self._sync_preview_rotation_controls()
            report = getattr(preview_scene, "report", None)
            renderer = report.get("preview_renderer") if isinstance(report, dict) else None
            if renderer == "unavailable":
                self._append_log("error", "Preview mesh renderer unavailable")
            self.state.set_status("Preview updated")
            self._append_log("info", "Preview updated")
            self._sync_state_to_ui()

        def _on_canvas_selection_changed(self, selection: PreviewSelection | None) -> None:
            self._inspector_output_text = None
            structure = self.state.current_structure()
            payload = _add_fractional_to_selection_payload(build_selection_payload(self.state.preview_scene, selection), structure)
            atom_objects = _selection_atom_objects(self._preview_canvas, structure)
            if atom_objects:
                payload = dict(payload or {})
                payload["objects"] = atom_objects
            self.state.set_selection(selection, payload=payload)
            self._sync_state_to_ui()

        def _on_canvas_interaction_message(self, message: str) -> None:
            text = str(message)
            self._status_label.setText(text)
            if text.startswith(("Distance", "Angle", "Dihedral", "Selected", "Updated")):
                self._append_log("info", text)

        def _on_inspector_object_edit(self, updates: dict[str, Any]) -> None:
            self._inspector_output_text = None
            target_label = f"Updated atom {updates.get('index')}" if updates.get("index") is not None else "Updated atom"
            undo_snapshot = self._capture_undo_snapshot(target_label)
            method = getattr(self._preview_canvas, "update_selected_atom_properties", None)
            if not callable(method) or not method(dict(updates)):
                return
            self._commit_undo_snapshot(undo_snapshot)
            self._apply_atom_updates_to_current_structure(updates)
            selection = getattr(getattr(self._preview_canvas, "model", None), "selection", None)
            structure = self.state.current_structure()
            payload = _add_fractional_to_selection_payload(
                build_selection_payload(getattr(getattr(self._preview_canvas, "model", None), "scene", None), selection),
                structure,
            )
            atom_objects = _selection_atom_objects(self._preview_canvas, structure)
            if atom_objects:
                payload = dict(payload or {})
                payload["objects"] = atom_objects
            self.state.set_selection(selection, payload=payload)
            self._sync_state_to_ui()

        def _apply_atom_updates_to_current_structure(self, updates: dict[str, Any]) -> None:
            structure = self.state.current_structure()
            if structure is None or updates.get("index") is None:
                return
            target = int(updates["index"])
            for atom in structure.atoms:
                if int(atom.index) != target:
                    continue
                if updates.get("symbol") is not None:
                    atom.symbol = str(updates["symbol"])
                if updates.get("atomic_number") is not None:
                    atom.atomic_number = int(updates["atomic_number"])
                if updates.get("position") is not None:
                    atom.position = tuple(float(v) for v in updates["position"])
                if updates.get("color") is not None:
                    color = tuple(float(v) for v in updates["color"])
                    atom.color = color if len(color) == 4 else (*color[:3], 1.0)
                if updates.get("radius") is not None:
                    atom.radius = float(updates["radius"])
                atom.sync_color_to_material()
                self.state.mark_dirty(f"Updated atom {target}")
                return

        def _emit_selected_object_edit(self, *, position_source: str = "position") -> None:
            payload = self.state.selected_payload or {}
            obj = payload.get("object") if isinstance(payload, dict) else None
            if not isinstance(obj, dict) or obj.get("index") is None:
                return
            try:
                position = _parse_float_tuple(self._object_edit_fields["position"].text(), length=3)
                if str(position_source) == "fractional":
                    converted = _position_from_fractional(
                        self.state.current_structure(),
                        _parse_float_tuple(self._object_edit_fields["fractional"].text(), length=3),
                    )
                    if converted is None:
                        raise ValueError("Fractional coordinates require a non-singular cell")
                    position = converted
                updates = {
                    "index": int(obj["index"]),
                    "symbol": self._object_edit_fields["symbol"].text().strip() or obj.get("symbol"),
                    "position": position,
                    "color": _parse_float_tuple(self._object_edit_fields["color"].text(), length=4),
                    "radius": float(self._object_edit_fields["radius"].text()),
                }
            except (TypeError, ValueError):
                self._append_log("error", "Invalid selected atom edit values")
                return
            self._on_inspector_object_edit(updates)

        def _choose_color_for_edit(self, edit: Any, button: Any, callback=None) -> None:
            if QtGui is None:
                return
            initial = self._color_from_text(edit.text())
            color = QtWidgets.QColorDialog.getColor(initial, self, "Select color", QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
            if not color.isValid():
                return
            rgba = (color.redF(), color.greenF(), color.blueF(), color.alphaF())
            edit.setText(_color_text(rgba))
            self._set_color_button(button, rgba)
            if callable(callback):
                callback()

        @staticmethod
        def _color_from_text(text: str) -> Any:
            if QtGui is None:
                return None
            try:
                rgba = _parse_float_tuple(text, length=4)
                return QtGui.QColor.fromRgbF(
                    max(0.0, min(1.0, rgba[0])),
                    max(0.0, min(1.0, rgba[1])),
                    max(0.0, min(1.0, rgba[2])),
                    max(0.0, min(1.0, rgba[3])),
                )
            except Exception:
                return QtGui.QColor("#b8b8b8")

        @staticmethod
        def _set_color_button(button: Any, color: Any) -> None:
            button.setStyleSheet(f"background-color: {_color_css(color)}; border: 1px solid #606060;")

        def _build_render_request(self) -> RenderRequest | None:
            structure = self.state.current_structure()
            if structure is None:
                return None
            render_config = self._ensure_render_config(structure=structure)
            render_config = self._render_config_from_ui(render_config, include_preview_camera=True)
            output_path = normalize_host_path(self._output_path_edit.text().strip() or render_config.output.path or _build_output_path(structure))
            self._output_path_edit.setText(output_path)
            render_config = render_config.with_output_path(output_path)
            self.state.set_render_config(render_config, mark_dirty=self.state.dirty)
            return RenderRequest(
                structure=structure,
                render_config=render_config,
                blender_path=self._blender_path_edit.text().strip() or None,
            )

        def _build_animation_request(self) -> AnimationRenderRequest | None:
            if self.state.bundle is None or self.state.bundle.frame_count <= 0:
                return None
            structures = self._animation_structures()
            if not structures:
                return None
            structure = self.state.current_structure() or structures[0]
            render_config = self._ensure_render_config(structure=structure)
            render_config = self._render_config_from_ui(render_config, include_preview_camera=True)
            output_text = self._output_path_edit.text().strip() or render_config.output.path or _build_output_path(structure)
            output_dir, filename_template = _build_animation_output_spec(output_text, structure)
            static_output_path = normalize_host_path(output_text)
            self._output_path_edit.setText(static_output_path)
            render_config = render_config.with_output_path(static_output_path)
            self.state.set_render_config(render_config, mark_dirty=self.state.dirty)
            animation_config = replace(
                render_config,
                output=replace(
                    render_config.output,
                    path=None,
                    dir=output_dir,
                    filename_template=filename_template,
                ),
            )
            return AnimationRenderRequest(
                structures=structures,
                render_config=animation_config,
                output_dir=output_dir,
                filename_template=filename_template,
                blender_path=self._blender_path_edit.text().strip() or None,
            )

        def _on_render_clicked(self) -> None:
            request = self._build_render_request()
            if request is None:
                self._append_log("error", "Load a structure before rendering")
                self.state.last_error = "Load a structure before rendering"
                self._set_inspector_output("Render failed\n\nLoad a structure before rendering")
                self._sync_state_to_ui()
                return

            self.state.set_status("Rendering with Blender...")
            self._render_in_progress = True
            self._set_inspector_output(
                "\n".join(
                    [
                        "Render started",
                        "",
                        f"Output path: {request.render_config.output.path}",
                        f"Blender path: {request.blender_path or 'auto'}",
                        f"Frame: {request.structure.frame_index}",
                    ]
                )
            )
            self._sync_state_to_ui()
            worker = RenderWorker(request)
            handle = start_background_task(
                worker.run,
                on_result=self._handle_render_result,
                on_error=self._handle_task_error,
                parent=self,
                label="render",
            )
            self._track_thread(handle)

        def _on_animation_clicked(self) -> None:
            request = self._build_animation_request()
            if request is None:
                self._append_log("error", "Load a trajectory before rendering animation")
                self.state.last_error = "Load a trajectory before rendering animation"
                self._set_inspector_output("Animation render failed\n\nLoad a trajectory before rendering animation")
                self._sync_state_to_ui()
                return

            self.state.set_status("Rendering animation with Blender...")
            self._render_in_progress = True
            self._set_inspector_output(
                "\n".join(
                    [
                        "Animation render started",
                        "",
                        f"Output directory: {request.output_dir}",
                        f"Filename template: {request.filename_template}",
                        f"Frames: {len(request.structures)}",
                        f"Blender path: {request.blender_path or 'auto'}",
                    ]
                )
            )
            self._sync_state_to_ui()
            worker = AnimationRenderWorker(request)
            handle = start_background_task(
                worker.run,
                on_result=self._handle_animation_result,
                on_error=self._handle_task_error,
                parent=self,
                label="render_animation",
            )
            self._track_thread(handle)

        def _handle_render_result(self, result) -> None:
            self._render_in_progress = False
            output_path = normalize_host_path(str(getattr(result, "output_path", "") or ""))
            success = bool(getattr(result, "success", False))
            if success and (not output_path or not Path(output_path).expanduser().is_file()):
                success = False
                message = f"Render completed but output file was not created: {output_path or '<empty output path>'}"
            else:
                message = str(getattr(result, "message", "") or "")
            if not success:
                self.state.last_render_output = None
                self.state.last_error = message or "Render failed"
                self.state.set_status(f"Error: {self.state.last_error}")
                self._append_log("error", self.state.last_error)
                self._set_inspector_output(
                    "\n".join(
                        [
                            "Render failed",
                            "",
                            f"Output path: {output_path or '<empty output path>'}",
                            f"Message: {self.state.last_error}",
                        ]
                    )
                )
                self._sync_state_to_ui()
                return
            self.state.last_render_output = output_path
            self.state.last_error = None
            self.state.mark_clean()
            self.state.set_status(f"Rendered {output_path}")
            self._append_log("info", f"Rendered {output_path}")
            self._set_inspector_output(
                "\n".join(
                    [
                        "Render succeeded",
                        "",
                        f"Output path: {output_path}",
                        f"Frame: {getattr(result, 'frame_index', '')}",
                        f"Elapsed seconds: {float(getattr(result, 'elapsed_seconds', 0.0) or 0.0):.3f}",
                        f"Message: {str(getattr(result, 'message', '') or 'ok')}",
                    ]
                )
            )
            self._sync_state_to_ui()

        def _handle_animation_result(self, result) -> None:
            self._render_in_progress = False
            output_dir = normalize_host_path(str(getattr(result, "output_dir", "") or ""))
            outputs = [normalize_host_path(str(path)) for path in getattr(result, "outputs", []) or []]
            failed_frames = [int(frame) for frame in getattr(result, "failed_frames", []) or []]
            existing_outputs = [path for path in outputs if path and Path(path).expanduser().is_file()]
            success = bool(getattr(result, "success", False)) and len(existing_outputs) == len(outputs) and not failed_frames
            message = str(getattr(result, "message", "") or "")
            if not success and not message:
                message = "Animation render failed"
            if not success:
                self.state.last_render_output = None
                self.state.last_error = message
                self.state.set_status(f"Error: {message}")
                self._append_log("error", message)
                self._set_inspector_output(
                    "\n".join(
                        [
                            "Animation render failed",
                            "",
                            f"Output directory: {output_dir or '<empty output dir>'}",
                            f"Rendered frames: {len(existing_outputs)}",
                            f"Failed frames: {failed_frames}",
                            f"Message: {message}",
                        ]
                    )
                )
                self._sync_state_to_ui()
                return
            self.state.last_render_output = output_dir
            self.state.last_error = None
            self.state.mark_clean()
            self.state.set_status(f"Rendered animation {output_dir}")
            self._append_log("info", f"Rendered animation {output_dir}")
            preview_outputs = existing_outputs[:8]
            suffix = [] if len(existing_outputs) <= 8 else [f"... {len(existing_outputs) - 8} more frame(s)"]
            self._set_inspector_output(
                "\n".join(
                    [
                        "Animation render succeeded",
                        "",
                        f"Output directory: {output_dir}",
                        f"Frames: {len(existing_outputs)}",
                        f"Elapsed seconds: {float(getattr(result, 'elapsed_seconds', 0.0) or 0.0):.3f}",
                        f"Message: {message or 'ok'}",
                        "",
                        "Outputs:",
                        *preview_outputs,
                        *suffix,
                    ]
                )
            )
            self._sync_state_to_ui()

        def _handle_task_error(self, error: BaseException) -> None:
            message = str(error)
            self.state.last_error = message
            self.state.set_status(f"Error: {message}")
            self._append_log("error", message)
            if self._render_in_progress:
                self._render_in_progress = False
                self._set_inspector_output(f"Render failed\n\nMessage: {message}")
            self._sync_state_to_ui()

        def _set_inspector_output(self, text: str) -> None:
            self._inspector_output_text = str(text)
            if hasattr(self, "_inspector"):
                self._inspector.set_output_text(self._inspector_output_text)
            if hasattr(self, "_inspector_dock"):
                self._inspector_dock.setVisible(True)
                self._inspector_dock.raise_()

        def _ensure_render_config(self, *, structure=None) -> RenderJobConfig:
            render_config = self.state.render_config
            if render_config is None:
                active_structure = structure or self.state.current_structure()
                if active_structure is None:
                    raise RuntimeError("No structure is loaded")
                render_config = build_default_render_config(active_structure, _build_output_path(active_structure))
                self.state.set_render_config(render_config, mark_dirty=False)
            return render_config

        def _render_config_from_ui(self, cfg: RenderJobConfig, *, include_preview_camera: bool = False) -> RenderJobConfig:
            resolution = (int(self._resolution_x_spin.value()), int(self._resolution_y_spin.value()))
            scene_style = self._combo_data(self._render_style_combo, cfg.style.scene_style or "default")
            representation = self._checked_button_value(self._atom_representation_group, cfg.structure.representation or "ball_stick")
            try:
                background = _parse_float_tuple(self._background_color_edit.text(), length=4)
            except (TypeError, ValueError):
                background = cfg.style.background or (1.0, 1.0, 1.0, 1.0)
            try:
                cell_color = _parse_float_tuple(self._cell_color_edit.text(), length=4)
            except (TypeError, ValueError):
                cell_color = cfg.structure.cell_style.color or CELL_DEFAULT_COLOR
            camera = replace(
                cfg.camera,
                projection=self._checked_button_value(self._camera_projection_group, cfg.camera.projection or "ORTHOGRAPHIC").upper(),
            )
            if include_preview_camera:
                camera = self._camera_config_from_preview(camera)
            return replace(
                cfg,
                style=replace(
                    cfg.style,
                    scene_style=scene_style,
                    background=tuple(float(v) for v in background),
                ),
                structure=replace(
                    cfg.structure,
                    representation=representation,
                    sphere_segments=int(self._sphere_segments_spin.value()),
                    sphere_rings=int(self._sphere_rings_spin.value()),
                    bond_vertices=int(self._bond_vertices_spin.value()),
                    bond_radius=float(self._bond_radius_spin.value()),
                    draw_cell=bool(self._cell_show_check.isChecked()),
                    cell_style=CellStyleConfig(
                        show=bool(self._cell_show_check.isChecked()),
                        radius=float(self._cell_radius_spin.value()),
                        material=cfg.structure.cell_style.material,
                        color=tuple(float(v) for v in cell_color),
                        transparent=bool(self._cell_transparent_check.isChecked()),
                    ),
                    boundary_atoms=BoundaryAtomsConfig(
                        enabled=bool(self._boundary_atoms_check.isChecked()),
                        sigma=float(self._boundary_atom_sigma_spin.value()),
                    ),
                ),
                camera=camera,
                render=replace(
                    cfg.render,
                    engine=self._combo_data(self._render_engine_combo, cfg.render.engine or "cycles"),
                    device=self._combo_data(self._render_device_combo, cfg.render.device or "auto"),
                    samples=int(self._samples_spin.value()),
                    resolution=resolution,
                    transparent_bg=bool(self._transparent_bg_radio.isChecked()),
                ),
            )

        def _camera_config_from_preview(self, camera_cfg):
            method = getattr(self._preview_canvas, "current_camera_state", None)
            model = getattr(self._preview_canvas, "model", None)
            camera = method() if callable(method) else getattr(model, "camera", None)
            if camera is None:
                return camera_cfg
            scale_factor = max(1e-6, float(getattr(camera, "scale_factor", 1.0)))
            model_rotation = getattr(camera, "model_rotation", None)
            model_rotation_tuple = (
                tuple(float(v) for v in model_rotation)
                if isinstance(model_rotation, (list, tuple)) and len(model_rotation) == 16
                else None
            )
            model_translation = getattr(camera, "model_translation", None)
            model_translation_tuple = (
                tuple(float(v) for v in model_translation)
                if isinstance(model_translation, (list, tuple)) and len(model_translation) == 3
                else None
            )
            return replace(
                camera_cfg,
                center=tuple(float(v) for v in getattr(camera, "center", (0.0, 0.0, 0.0))),
                right=tuple(float(v) for v in getattr(camera, "right", (1.0, 0.0, 0.0))),
                up=tuple(float(v) for v in getattr(camera, "up", (0.0, 1.0, 0.0))),
                forward=tuple(float(v) for v in getattr(camera, "forward", (0.0, 0.0, -1.0))),
                ortho_scale=scale_factor,
                distance=max(3.0, scale_factor),
                model_rotation=model_rotation_tuple,
                model_translation=model_translation_tuple,
                view=str(getattr(camera, "view", camera_cfg.view) or camera_cfg.view),
                fit_mode="gui",
            )

        def _apply_scene_tab_settings(self) -> None:
            structure = self.state.current_structure()
            if structure is None:
                return
            self._preview_shader_style = resolve_shader_style(
                self._checked_button_value(self._scene_style_group, self._preview_shader_style)
            ).name
            self._apply_preview_shader_style_to_canvas()
            cfg = self._render_config_from_ui(self._ensure_render_config(structure=structure), include_preview_camera=False)
            undo_snapshot = self._capture_undo_snapshot("Updated scene settings")
            self.state.set_render_config(cfg, mark_dirty=True)
            self._commit_undo_snapshot(undo_snapshot)
            self._refresh_preview()

        def _apply_preview_shader_style_to_canvas(self) -> None:
            for method_name in ("set_preview_shader_style", "set_shader_style"):
                method = getattr(self._preview_canvas, method_name, None)
                if callable(method):
                    method(self._preview_shader_style)
                    return

        def _apply_render_style_choice(self) -> None:
            style_name = self._combo_data(self._render_style_combo, "default")
            structure = self.state.current_structure()
            cfg = self._ensure_render_config(structure=structure) if structure is not None else self.state.render_config
            if cfg is None:
                return
            structure_cfg = cfg.structure
            next_cfg = replace(cfg, style=replace(cfg.style, scene_style=style_name), structure=structure_cfg)
            undo_snapshot = self._capture_undo_snapshot(f"Changed render style to {style_name}")
            self.state.set_render_config(next_cfg, mark_dirty=True)
            self._commit_undo_snapshot(undo_snapshot)
            self._append_log("info", f"render style -> {style_name}")
            self._sync_state_to_ui()
            if structure is not None:
                self._refresh_preview()

        def _apply_render_settings(self) -> None:
            cfg = self.state.render_config
            if cfg is None:
                structure = self.state.current_structure()
                if structure is None:
                    return
                cfg = self._ensure_render_config(structure=structure)
            next_cfg = self._render_config_from_ui(cfg, include_preview_camera=False)
            undo_snapshot = self._capture_undo_snapshot("Updated render settings")
            self.state.set_render_config(next_cfg, mark_dirty=True)
            self._commit_undo_snapshot(undo_snapshot)
            self._sync_state_to_ui()

        def _apply_element_style(self) -> None:
            if self._syncing_element_style_controls:
                return
            structure = self.state.current_structure()
            if structure is None:
                return
            cfg = self._ensure_render_config(structure=structure)
            symbol = self._selected_element_symbol()
            if not symbol:
                return
            matching_atoms = self._atoms_with_symbol_across_loaded_frames(symbol)
            if not matching_atoms:
                return
            try:
                color = _parse_float_tuple(self._element_color_edit.text(), length=4) if self._element_color_edit.text().strip() else None
                radius = float(self._element_radius_spin.value())
            except (TypeError, ValueError):
                self._append_log("error", "Invalid element style values")
                return
            baseline = self._element_style_loaded_values
            if baseline is None or baseline[0] != symbol:
                self._populate_element_style_controls()
                return
            baseline_color = baseline[1] if baseline is not None and baseline[0] == symbol else None
            baseline_radius = baseline[2] if baseline is not None and baseline[0] == symbol else None
            color_changed = color is not None and (
                baseline_color is None
                or any(abs(float(current) - float(previous)) > 1.0e-6 for current, previous in zip(color, baseline_color, strict=True))
            )
            radius_changed = self._element_radius_dirty and (
                baseline_radius is None or abs(float(radius) - float(baseline_radius)) > 1.0e-6
            )
            if not color_changed and not radius_changed:
                return
            undo_snapshot = self._capture_undo_snapshot(f"Updated {symbol} atom style")
            next_cfg = self._config_with_element_default_style(
                cfg,
                symbol,
                color=color if color_changed and color is not None else _NO_ELEMENT_STYLE_CHANGE,
                radius=radius if radius_changed else _NO_ELEMENT_STYLE_CHANGE,
            )
            self._clear_element_overrides_that_shadow_default(
                matching_atoms,
                old_color=baseline_color,
                new_color=tuple(float(value) for value in color) if color_changed and color is not None else None,
                color_changed=color_changed,
                old_radius=float(baseline_radius) if baseline_radius is not None else None,
                new_radius=float(radius) if radius_changed else None,
                radius_changed=radius_changed,
            )
            self.state.set_render_config(next_cfg, mark_dirty=True)
            self._commit_undo_snapshot(undo_snapshot)
            self.state.mark_dirty(f"Updated {symbol} atom default style")
            next_color, next_radius = self._resolved_element_default_style(symbol)
            self._element_style_loaded_values = (symbol, tuple(float(value) for value in next_color), float(next_radius))
            self._element_radius_dirty = False
            self._populate_element_style_controls()
            self._refresh_preview(preserve_camera=True)

        def _apply_bond_style(self) -> None:
            structure = self.state.current_structure()
            if structure is None:
                return
            try:
                color = _parse_float_tuple(self._bond_color_edit.text(), length=4) if self._bond_color_edit.text().strip() else None
            except (TypeError, ValueError):
                self._append_log("error", "Invalid bond color")
                return
            style = self._combo_data(self._bond_style_combo, "bicolor")
            undo_snapshot = self._capture_undo_snapshot("Updated bond style")
            self._apply_bond_style_to_loaded_frames(style=style, color=color)
            cfg = self._render_config_from_ui(self._ensure_render_config(structure=structure), include_preview_camera=False)
            self.state.set_render_config(cfg, mark_dirty=True)
            self._commit_undo_snapshot(undo_snapshot)
            self._refresh_preview(preserve_camera=True)

        def _apply_hydrogen_bond_visibility(self) -> None:
            structure = self.state.current_structure()
            if structure is None:
                return
            cfg = self._render_config_from_ui(self._ensure_render_config(structure=structure), include_preview_camera=False)
            enabled = bool(self._show_hydrogen_bonds_check.isChecked())
            next_cfg = replace(
                cfg,
                structure=replace(
                    cfg.structure,
                    bonding=replace(
                        cfg.structure.bonding,
                        hbond=replace(cfg.structure.bonding.hbond, enabled=enabled),
                    ),
                ),
            )
            undo_snapshot = self._capture_undo_snapshot("Updated hydrogen bond visibility")
            self._update_loaded_hydrogen_bonds(next_cfg)
            self.state.set_render_config(next_cfg, mark_dirty=True)
            self._commit_undo_snapshot(undo_snapshot)
            self.state.mark_dirty("Updated hydrogen bond visibility")
            self._refresh_preview(preserve_camera=True)

        def _apply_bond_style_to_loaded_frames(self, *, style: str, color: Any | None) -> None:
            for frame in self._loaded_frames_or_current():
                for bond in frame.bonds:
                    if str(getattr(bond, "bond_type", "covalent")) == "hydrogen":
                        bond.metadata["preview_bond_style"] = "dashed"
                        continue
                    bond.metadata["preview_bond_style"] = style
                    if style == "bicolor":
                        bond.color = None
                        bond.color_a = None
                        bond.color_b = None
                    elif color is not None:
                        bond.color = tuple(float(value) for value in color)

        def _open_bond_search_dialog(self) -> None:
            structure = self.state.current_structure()
            if structure is None:
                self._append_log("error", "No structure is loaded")
                return
            cfg = self._render_config_from_ui(self._ensure_render_config(structure=structure), include_preview_camera=False)
            self.state.set_render_config(cfg, mark_dirty=self.state.dirty)
            from .bond_editor import BondSearchDialog

            dialog = BondSearchDialog(
                structure=structure,
                config=cfg,
                on_apply=lambda rules: self._apply_bond_search_rules(rules, label="Updated bond search rules"),
                parent=self,
            )
            dialog.exec()

        def _apply_bond_search_rules(self, rules: Any, *, label: str) -> None:
            structure = self.state.current_structure()
            if structure is None:
                return
            cfg = self._render_config_from_ui(self._ensure_render_config(structure=structure), include_preview_camera=False)
            pair_distances = {
                normalize_pair_key(str(key)): (float(value[0]), float(value[1]))
                for key, value in dict(getattr(rules, "pair_distances", {}) or {}).items()
            }
            disabled_pairs = [normalize_pair_key(str(value)) for value in getattr(rules, "disabled_pairs", []) or []]
            order_rules = {
                normalize_pair_key(str(key)): int(value)
                for key, value in dict(getattr(rules, "order_rules", {}) or {}).items()
                if int(value) in {1, 2, 3}
            }
            next_cfg = replace(
                cfg,
                structure=replace(
                    cfg.structure,
                    bonding=replace(
                        cfg.structure.bonding,
                        pair_distances=dict(sorted(pair_distances.items())),
                        disabled_pairs=sorted(disabled_pairs),
                        order_rules=dict(sorted(order_rules.items())),
                    ),
                ),
            )
            undo_snapshot = self._capture_undo_snapshot(label)
            self._update_loaded_bonds_for_pair_rule_changes(cfg.structure.bonding, next_cfg.structure.bonding, next_cfg)
            self.state.set_render_config(next_cfg, mark_dirty=True)
            self._commit_undo_snapshot(undo_snapshot)
            self.state.mark_dirty(label)
            self._refresh_preview(preserve_camera=True)

        def _recompute_loaded_bonds(self, cfg: RenderJobConfig) -> None:
            engine = BondEngine()
            style = self._combo_data(self._bond_style_combo, "bicolor") if hasattr(self, "_bond_style_combo") else "bicolor"
            try:
                color = _parse_float_tuple(self._bond_color_edit.text(), length=4) if self._bond_color_edit.text().strip() else None
            except (TypeError, ValueError):
                color = None
            frames = self._loaded_frames_or_current()
            for frame in frames:
                frame.bonds = engine.compute(frame, cfg.structure.bonding)
            self._apply_bond_style_to_loaded_frames(style=style, color=color)

        def _update_loaded_hydrogen_bonds(self, cfg: RenderJobConfig) -> None:
            engine = BondEngine()
            for frame in self._loaded_frames_or_current():
                covalent_bonds = [
                    bond for bond in frame.bonds if str(getattr(bond, "bond_type", "covalent")) != "hydrogen"
                ]
                if not covalent_bonds:
                    covalent_bonds = engine.compute_covalent(frame, cfg.structure.bonding)
                bonds = list(covalent_bonds)
                bonds.extend(engine.compute_hydrogen_bonds(frame, cfg.structure.bonding, covalent_bonds=covalent_bonds))
                frame.bonds = BondEngine.assign_ids(_sort_bonds_for_display(bonds))
            self._restyle_loaded_bonds()

        def _update_loaded_bonds_for_pair_rule_changes(self, old_bonding: Any, new_bonding: Any, cfg: RenderJobConfig) -> None:
            changed_pairs = _changed_bond_rule_pair_keys(old_bonding, new_bonding)
            if not changed_pairs:
                self._restyle_loaded_bonds()
                return
            engine = BondEngine()
            for frame in self._loaded_frames_or_current():
                existing_covalent = [
                    bond for bond in frame.bonds if str(getattr(bond, "bond_type", "covalent")) != "hydrogen"
                ]
                if not existing_covalent:
                    frame.bonds = engine.compute(frame, cfg.structure.bonding)
                    continue
                kept_covalent = [
                    bond
                    for bond in existing_covalent
                    if (_bond_pair_key(frame, bond) is not None and _bond_pair_key(frame, bond) not in changed_pairs)
                ]
                updated_covalent = engine.compute_covalent_pairs(frame, cfg.structure.bonding, changed_pairs)
                covalent_bonds = _sort_bonds_for_display(kept_covalent + updated_covalent)
                bonds = list(covalent_bonds)
                bonds.extend(engine.compute_hydrogen_bonds(frame, cfg.structure.bonding, covalent_bonds=covalent_bonds))
                frame.bonds = BondEngine.assign_ids(_sort_bonds_for_display(bonds))
            self._restyle_loaded_bonds()

        def _restyle_loaded_bonds(self) -> None:
            style = self._combo_data(self._bond_style_combo, "bicolor") if hasattr(self, "_bond_style_combo") else "bicolor"
            try:
                color = _parse_float_tuple(self._bond_color_edit.text(), length=4) if self._bond_color_edit.text().strip() else None
            except (TypeError, ValueError):
                color = None
            self._apply_bond_style_to_loaded_frames(style=style, color=color)

        def _preview_selection_records(self, kind: str) -> list[int]:
            scene = self.state.preview_scene
            if scene is None:
                return []
            if kind == "atom":
                records = getattr(scene, "atom_records", None)
                if records:
                    return [int(_record_field(record, "index", idx)) for idx, record in enumerate(records)]
                atoms = getattr(scene, "atoms", None)
                raw = getattr(atoms, "atom_indices", None) if atoms is not None else None
                return [int(value) for value in (_jsonable(raw) if raw is not None else [])]
            if kind == "bond":
                records = getattr(scene, "bond_records", None)
                if records:
                    return [int(_record_field(record, "id", _record_field(record, "index", idx))) for idx, record in enumerate(records)]
                bonds = getattr(scene, "bonds", None)
                raw = getattr(bonds, "bond_ids", None) if bonds is not None else None
                return [int(value) for value in (_jsonable(raw) if raw is not None else [])]
            return []

        def _select_preview_object(self, kind: str, index: int | None) -> None:
            if self._preview_canvas is None:
                return
            if index is None:
                for method_name in ("clear_selection", "select_preview_object", "select_selection"):
                    method = getattr(self._preview_canvas, method_name, None)
                    if callable(method):
                        if method_name == "clear_selection":
                            method()
                        else:
                            method(None)
                        return
                return

            for method_name in ("select_preview_object", "select_selection"):
                method = getattr(self._preview_canvas, method_name, None)
                if callable(method):
                    method(PreviewSelection(kind=kind, index=int(index)))
                    return
            if kind == "atom":
                method = getattr(self._preview_canvas, "select_atom", None)
                if callable(method):
                    method(int(index))
                    return
            if kind == "bond":
                method = getattr(self._preview_canvas, "select_bond", None)
                if callable(method):
                    method(int(index))

        def open_structure_dialog(self) -> None:
            self._browse_input()

        def reload_current_input(self) -> None:
            input_path = normalize_host_path(self._input_path_edit.text().strip())
            if not input_path and self.state.bundle is not None:
                input_path = self.state.bundle.source_path
            if not input_path:
                self._append_log("warning", "No input path to reload")
                self._sync_state_to_ui()
                return
            self.load_input(input_path, self._frame_selector_edit.text().strip() or _default_frame_selector())

        def export_render_config_yaml(self) -> None:
            cfg = self.state.render_config
            if cfg is None:
                self._append_log("warning", "No render config available to export")
                self._sync_state_to_ui()
                return
            source_name = Path(self.state.bundle.source_path).stem if self.state.bundle is not None else "atomstudio"
            default_path = Path.cwd() / f"{source_name}_app.yaml"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export render config",
                str(default_path),
                "YAML (*.yaml *.yml);;All files (*)",
            )
            if not path:
                return
            path = normalize_host_path(path)
            payload = {"version": 2, "jobs": [cfg.to_dict()]}
            with Path(path).expanduser().open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
            self._append_log("info", f"Exported render config to {path}")
            self.state.mark_clean()
            self._sync_state_to_ui()

        def export_render_script(self) -> None:
            structure = self.state.current_structure()
            if structure is None:
                self._append_log("warning", "No structure available to export render script")
                self._sync_state_to_ui()
                return
            cfg = self.state.render_config
            if cfg is None:
                self._append_log("warning", "No render config available to export render script")
                self._sync_state_to_ui()
                return

            cfg = self._render_config_from_ui(self._ensure_render_config(structure=structure), include_preview_camera=True)
            output_path = normalize_host_path(self._output_path_edit.text().strip() or cfg.output.path or _build_output_path(structure))
            self._output_path_edit.setText(output_path)
            cfg = cfg.with_output_path(output_path)
            self.state.set_render_config(cfg, mark_dirty=self.state.dirty)

            source_path = (
                self.state.bundle.source_path
                if self.state.bundle is not None and self.state.bundle.source_path
                else structure.source_path or cfg.input.path
            )
            output_dir, filename_template = default_batch_output_spec(output_path, source_path)
            default_path = default_render_script_path(source_path)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export render script",
                str(default_path),
                "Python scripts (*.py);;All files (*)",
            )
            if not path:
                return
            path = normalize_host_path(path)
            script = build_render_script_text(
                render_config=cfg,
                reference_structure=structure,
                default_input_path=source_path,
                default_frames="all",
                default_out_dir=output_dir,
                default_filename_template=filename_template,
            )
            with Path(path).expanduser().open("w", encoding="utf-8") as handle:
                handle.write(script)
            self._append_log("info", f"Exported render script to {path}")
            self.state.mark_clean()
            self._sync_state_to_ui()

        def copy_selection_summary(self) -> None:
            text = _selection_summary(self.state.selected_payload)
            QtWidgets.QApplication.clipboard().setText(text)
            self._append_log("info", "Copied selection summary")

        def copy_selection_json(self) -> None:
            payload = self.state.selected_payload or {}
            QtWidgets.QApplication.clipboard().setText(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False))
            self._append_log("info", "Copied selection JSON")

        def fit_preview_to_structure(self) -> None:
            method = getattr(self._preview_canvas, "fit_to_structure", None)
            if callable(method):
                method()
            self._sync_preview_rotation_controls()

        def reset_preview_camera(self) -> None:
            self.fit_preview_to_structure()
            self.set_preview_axis_view("c")

        def set_preview_view(self, view: str) -> None:
            method = getattr(self._preview_canvas, "set_view_preset", None)
            if callable(method):
                method(str(view))
            self._sync_preview_rotation_controls()

        def set_preview_axis_view(self, axis: str) -> None:
            method = getattr(self._preview_canvas, "set_axis_view", None)
            if callable(method):
                method(str(axis))
            self._sync_preview_rotation_controls()

        def set_wrap_atoms_into_cell(self, enabled: bool) -> None:
            enabled = bool(enabled)
            if self.state.wrap_atoms_into_cell == enabled:
                self._sync_state_to_ui()
                return
            self.state.wrap_atoms_into_cell = enabled
            self.state.clear_selection()
            self._append_log("info", f"Wrap atoms into cell {'enabled' if enabled else 'disabled'}")
            if self.state.current_structure() is not None:
                self._refresh_preview(preserve_camera=True)
            self._sync_state_to_ui()

        def view_rotation_step_degrees(self) -> float:
            control = self._toolbar_handles.controls.get("rotation_step_degrees")
            value = control.value() if control is not None and hasattr(control, "value") else 15.0
            return max(0.1, float(value))

        def view_pan_step_pixels(self) -> float:
            control = self._toolbar_handles.controls.get("pan_step_pixels")
            value = control.value() if control is not None and hasattr(control, "value") else 24.0
            return max(1.0, float(value))

        def rotate_preview_view(self, axis: str, direction: int, degrees: float | None = None) -> None:
            method = getattr(self._preview_canvas, "rotate_view", None)
            if callable(method):
                method(str(axis), int(direction), self.view_rotation_step_degrees() if degrees is None else float(degrees))
            self._sync_preview_rotation_controls()

        def set_preview_model_rotation_from_toolbar(self) -> None:
            if self._syncing_preview_rotation_controls:
                return
            controls = self._toolbar_handles.controls
            x_control = controls.get("rotation_x_degrees")
            y_control = controls.get("rotation_y_degrees")
            z_control = controls.get("rotation_z_degrees")
            if x_control is None or y_control is None or z_control is None:
                return
            method = getattr(self._preview_canvas, "set_model_rotation_angles", None)
            if callable(method):
                method(float(x_control.value()), float(y_control.value()), float(z_control.value()))

        def _preview_model_rotation_angles(self) -> tuple[float, float, float] | None:
            method = getattr(self._preview_canvas, "model_rotation_angles", None)
            if callable(method):
                values = method()
                return tuple(float(value) for value in values[:3])
            camera_method = getattr(self._preview_canvas, "current_camera_state", None)
            camera = camera_method() if callable(camera_method) else getattr(getattr(self._preview_canvas, "model", None), "camera", None)
            if camera is None:
                return None
            return model_rotation_euler_degrees(camera)

        def _sync_preview_rotation_controls(self) -> None:
            controls = getattr(self, "_toolbar_handles", ToolbarHandles()).controls
            angle_controls = (
                controls.get("rotation_x_degrees"),
                controls.get("rotation_y_degrees"),
                controls.get("rotation_z_degrees"),
            )
            if any(control is None for control in angle_controls):
                return
            angles = self._preview_model_rotation_angles()
            if angles is None:
                return
            self._syncing_preview_rotation_controls = True
            try:
                for control, value in zip(angle_controls, angles, strict=True):
                    previous = control.blockSignals(True) if hasattr(control, "blockSignals") else False
                    try:
                        control.setValue(float(value))
                    finally:
                        if hasattr(control, "blockSignals"):
                            control.blockSignals(previous)
            finally:
                self._syncing_preview_rotation_controls = False

        def pan_preview_view(self, dx: float, dy: float) -> None:
            method = getattr(self._preview_canvas, "pan_view", None)
            if callable(method):
                method(float(dx), float(dy))

        def zoom_preview_view(self, factor: float) -> None:
            method = getattr(self._preview_canvas, "zoom_view", None)
            if callable(method):
                method(float(factor))
            self._sync_preview_rotation_controls()

        def set_mouse_mode(self, mode: str) -> None:
            method = getattr(self._preview_canvas, "set_mouse_mode", None)
            if callable(method):
                resolved = method(str(mode))
            else:
                resolved = str(mode)
            for key, action in self._mouse_toolbar_handles.actions.items():
                if hasattr(action, "setChecked"):
                    action.setChecked(str(key) == str(resolved))
            self._status_label.setText(f"Mouse mode: {str(resolved).replace('_', ' ')}")

        def set_axis_overlay_visible(self, visible: bool) -> None:
            method = getattr(self._preview_canvas, "set_axis_overlay_visible", None)
            if callable(method):
                method(bool(visible))
            self.state.set_dock_visibility(axis_overlay_visible=bool(visible))
            self._sync_state_to_ui()

        def set_inspector_visible(self, visible: bool) -> None:
            self._inspector_dock.setVisible(bool(visible))
            self.state.set_dock_visibility(inspector_visible=bool(visible))
            self._sync_state_to_ui()

        def set_log_visible(self, visible: bool) -> None:
            self._log_dock.setVisible(bool(visible))
            self.state.set_dock_visibility(log_visible=bool(visible))
            self._sync_state_to_ui()

        def _on_log_visibility_changed(self, visible: bool) -> None:
            self.state.set_dock_visibility(log_visible=bool(visible))
            self._set_action_checked("toggle_status_log", bool(visible))

        def _on_inspector_visibility_changed(self, visible: bool) -> None:
            self.state.set_dock_visibility(inspector_visible=bool(visible))
            self._set_action_checked("toggle_inspector_dock", bool(visible))

        def clear_preview_selection(self) -> None:
            self._select_preview_object("atom", None)
            self.state.clear_selection()
            self._sync_state_to_ui()

        def delete_selected_objects(self) -> None:
            model = getattr(self._preview_canvas, "model", None)
            atom_ids = {int(index) for index in getattr(model, "selected_atom_indices", set()) or set()}
            bond_ids = {int(index) for index in getattr(model, "selected_bond_indices", set()) or set()}
            selection = getattr(model, "selection", None) or self.state.selected_object
            if selection is not None and getattr(selection, "index", None) is not None:
                if selection.kind == "atom":
                    atom_ids.add(int(selection.index))
                elif selection.kind == "bond":
                    bond_ids.add(int(selection.index))
            if not atom_ids and not bond_ids:
                return

            undo_snapshot = self._capture_undo_snapshot("Deleted selected objects")
            deleted_from_structure = self._delete_from_current_structure(atom_ids, bond_ids)
            method = getattr(self._preview_canvas, "delete_selected_objects", None)
            deleted = method() if callable(method) else {"atoms": len(atom_ids), "bonds": len(bond_ids)}
            if deleted_from_structure["atoms"] or deleted_from_structure["bonds"] or deleted.get("atoms") or deleted.get("bonds"):
                self._commit_undo_snapshot(undo_snapshot)
            self.state.clear_selection()
            self.state.mark_dirty("Deleted selected objects")
            self._append_log(
                "info",
                f"Deleted {deleted_from_structure['atoms']} atom(s), {deleted_from_structure['bonds']} bond(s)",
            )
            if not self._refresh_preview_now(preserve_camera=True):
                preview_scene = getattr(model, "scene", None)
                if preview_scene is not None:
                    self.state.set_preview_scene(preview_scene)
                self._sync_state_to_ui()

        def _delete_from_current_structure(self, atom_ids: set[int], bond_ids: set[int]) -> dict[str, int]:
            structure = self.state.current_structure()
            if structure is None:
                return {"atoms": 0, "bonds": 0}
            atom_ids = {int(index) for index in atom_ids}
            bond_ids = {int(index) for index in bond_ids}
            old_atoms = list(structure.atoms)
            old_bonds = list(structure.bonds)
            kept_atoms = [atom for atom in old_atoms if int(atom.index) not in atom_ids]
            remap = {int(atom.index): new_index for new_index, atom in enumerate(kept_atoms)}
            structure.atoms = kept_atoms
            for new_index, atom in enumerate(structure.atoms):
                atom.index = int(new_index)
            kept_bonds = []
            for bond in old_bonds:
                if int(bond.id) in bond_ids or int(bond.a) in atom_ids or int(bond.b) in atom_ids:
                    continue
                if int(bond.a) not in remap or int(bond.b) not in remap:
                    continue
                bond.a = int(remap[int(bond.a)])
                bond.b = int(remap[int(bond.b)])
                kept_bonds.append(bond)
            structure.bonds = kept_bonds
            for new_id, bond in enumerate(structure.bonds):
                bond.id = int(new_id)
            structure.polyhedra = [
                poly
                for poly in structure.polyhedra
                if int(poly.center) not in atom_ids and not (set(int(index) for index in poly.neighbor_indices) & atom_ids)
            ]
            for poly in structure.polyhedra:
                poly.center = int(remap.get(int(poly.center), int(poly.center)))
                poly.neighbor_indices = [int(remap[index]) for index in poly.neighbor_indices if int(index) in remap]
            return {
                "atoms": len(old_atoms) - len(structure.atoms),
                "bonds": len(old_bonds) - len(structure.bonds),
            }

        def _refresh_preview_now(self, *, preserve_camera: bool | None = None) -> bool:
            request = self._build_preview_request()
            if request is None:
                return False
            self._preview_request_serial += 1
            try:
                preview_scene = PreviewWorker(request).run()
            except Exception as exc:
                self._handle_task_error(exc)
                return False
            self._handle_preview_result(preview_scene, preserve_camera=preserve_camera)
            return True

        def cycle_selection(self, kind: str, step: int) -> None:
            ids = self._preview_selection_records(kind)
            if not ids:
                return
            current_kind, current_index = _selection_fields(self.state.selected_object)
            if current_kind != kind or current_index not in ids:
                target = ids[0 if step >= 0 else -1]
            else:
                current_pos = ids.index(current_index)
                target = ids[(current_pos + int(step)) % len(ids)]
            self._select_preview_object(kind, target)

        def refresh_preview_from_menu(self) -> None:
            self._refresh_preview()

        def render_final_image(self) -> None:
            self._on_render_clicked()

        def apply_style_choice(self, field_name: str, value: str) -> None:
            structure = self.state.current_structure()
            if structure is None:
                return
            cfg = self._ensure_render_config(structure=structure)
            style = cfg.style
            lighting = cfg.lighting
            if field_name == "light_style":
                style = replace(style, light_style=str(value))
                lighting = replace(lighting, light_style=str(value))
            elif hasattr(style, field_name):
                style = replace(style, **{field_name: str(value)})
            else:
                return
            next_cfg = replace(cfg, style=style, lighting=lighting)
            undo_snapshot = self._capture_undo_snapshot(f"Changed {field_name} to {value}")
            self.state.set_render_config(next_cfg, mark_dirty=True)
            self._commit_undo_snapshot(undo_snapshot)
            self._append_log("info", f"{field_name} -> {value}")
            self._sync_state_to_ui()
            self._refresh_preview()

        def reset_window_layout(self) -> None:
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self._inspector_dock)
            self.tabifyDockWidget(self._log_dock, self._inspector_dock)
            self._inspector_dock.raise_()
            self.set_log_visible(True)
            self.set_inspector_visible(True)
            self.set_axis_overlay_visible(True)

        def focus_preview(self) -> None:
            if hasattr(self._preview_canvas, "setFocus"):
                self._preview_canvas.setFocus()

        def focus_inspector(self) -> None:
            self._inspector_dock.raise_()
            self._inspector.setFocus()

        def show_keyboard_shortcuts(self) -> None:
            QtWidgets.QMessageBox.information(
                self,
                "Keyboard Shortcuts",
                "\n".join(
                    [
                        "Ctrl+O: Open structure",
                        "Ctrl+Z: Undo",
                        "Ctrl+Y / Ctrl+Shift+Z: Redo",
                        "Ctrl+R: Refresh preview",
                        "Ctrl+Shift+P: Render final image",
                        "F: Fit to structure",
                        "1/2/3/4: Orbit/Top/Front/Side views",
                        "Esc: Clear selection",
                        "Delete: Delete selected atoms/bonds",
                        "[ and ]: Previous/Next atom",
                        "{ and }: Previous/Next bond",
                    ]
                ),
            )

        def show_about_dialog(self) -> None:
            QtWidgets.QMessageBox.about(
                self,
                "About AtomStudio",
                "AtomStudio desktop preview app\nOpenGL preview + Blender final rendering",
            )

        def load_input(self, input_path: str, frame_selector: str = "all") -> None:
            input_path = normalize_host_path(input_path)
            self._input_path_edit.setText(input_path)
            self._frame_selector_edit.setText(frame_selector)
            self._on_load_clicked()

        def set_render_config(self, cfg: RenderJobConfig) -> None:
            if cfg.output.path:
                cfg = cfg.with_output_path(normalize_host_path(cfg.output.path))
            undo_snapshot = self._capture_undo_snapshot("Loaded render config")
            self.state.set_render_config(cfg, mark_dirty=True)
            self._commit_undo_snapshot(undo_snapshot)
            self._append_log("info", f"Loaded render config {cfg.id}")
            self._output_path_edit.setText(cfg.output.path or "")
            self._sync_state_to_ui()

else:  # pragma: no cover - importable fallback for tests and documentation builds

    class AtomStudioWindow:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PySide6 is required to instantiate AtomStudioWindow")
