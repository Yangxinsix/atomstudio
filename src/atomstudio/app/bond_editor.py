from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Any, Callable

from atomstudio.config import RenderJobConfig
from atomstudio.structure.data import VESTA_PAIR_BOND_DISTANCE_RANGES, normalize_pair_key
from atomstudio.structure.structure import Structure

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = QtWidgets = None


@dataclass(frozen=True)
class BondSearchRules:
    pair_distances: dict[str, tuple[float, float]]
    disabled_pairs: list[str]
    order_rules: dict[str, int] = field(default_factory=dict)


if QtWidgets is not None:  # pragma: no cover - exercised only when GUI deps are installed

    class BondSearchDialog(QtWidgets.QDialog):
        def __init__(
            self,
            *,
            structure: Structure,
            config: RenderJobConfig,
            on_apply: Callable[[BondSearchRules], None] | None = None,
            parent: Any | None = None,
        ) -> None:
            super().__init__(parent)
            self.structure = structure
            self.config = config
            self.on_apply = on_apply
            self._disabled_pairs = set(str(value) for value in config.structure.bonding.disabled_pairs)
            self._order_rules = _normalized_order_rules(config.structure.bonding.order_rules)
            self._result_rules: BondSearchRules | None = None
            self._syncing_editor = False
            self.setWindowTitle("Bond Search")
            self.resize(880, 560)
            self._build_ui()
            self._load_rules()

        @property
        def result_rules(self) -> BondSearchRules | None:
            return self._result_rules

        def _build_ui(self) -> None:
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(12, 12, 12, 12)
            root.setSpacing(10)

            controls_box = QtWidgets.QGroupBox("Search bonds and atoms")
            controls = QtWidgets.QVBoxLayout(controls_box)
            controls.setSpacing(8)

            self._mode_group = QtWidgets.QButtonGroup(controls_box)
            mode_box = QtWidgets.QGroupBox("Search mode")
            mode_layout = QtWidgets.QVBoxLayout(mode_box)
            for label in ("Search A2 bonded to A1", "Search atoms bonded to A1", "Search molecules"):
                radio = QtWidgets.QRadioButton(label)
                radio.setEnabled(label == "Search A2 bonded to A1")
                radio.setChecked(label == "Search A2 bonded to A1")
                self._mode_group.addButton(radio)
                mode_layout.addWidget(radio)
            mode_row = QtWidgets.QHBoxLayout()
            mode_row.setSpacing(8)

            boundary_box = QtWidgets.QGroupBox("Boundary mode")
            boundary_layout = QtWidgets.QVBoxLayout(boundary_box)
            for label in (
                "Do not search atoms beyond the boundary",
                "Search additional atoms if A1 is included in the boundary",
                "Search additional atoms recursively if either A1 or A2 is visible",
            ):
                radio = QtWidgets.QRadioButton(label)
                radio.setChecked(label.startswith("Search additional atoms if"))
                boundary_layout.addWidget(radio)
            mode_row.addWidget(mode_box, 2)
            mode_row.addWidget(boundary_box, 5)
            controls.addLayout(mode_row)

            self._a1_combo = QtWidgets.QComboBox()
            self._a2_combo = QtWidgets.QComboBox()
            self._a1_combo.setEditable(True)
            self._a2_combo.setEditable(True)
            for symbol in self._structure_symbols():
                self._a1_combo.addItem(symbol, symbol)
                self._a2_combo.addItem(symbol, symbol)
            self._min_spin = QtWidgets.QDoubleSpinBox()
            self._min_spin.setRange(0.0, 20.0)
            self._min_spin.setDecimals(4)
            self._min_spin.setSingleStep(0.01)
            self._max_spin = QtWidgets.QDoubleSpinBox()
            self._max_spin.setRange(0.0, 20.0)
            self._max_spin.setDecimals(4)
            self._max_spin.setSingleStep(0.01)
            self._order_spin = QtWidgets.QSpinBox()
            self._order_spin.setRange(1, 3)
            self._order_spin.setSingleStep(1)
            self._poly_check = QtWidgets.QCheckBox("Poly.")
            self._poly_check.setChecked(False)
            self._show_poly_check = QtWidgets.QCheckBox("Show polyhedra")
            self._show_poly_check.setChecked(True)
            self._search_by_label_check = QtWidgets.QCheckBox("Search by label")

            checks = QtWidgets.QHBoxLayout()
            checks.setSpacing(12)
            checks.addWidget(self._search_by_label_check)
            checks.addWidget(self._show_poly_check)
            checks.addStretch(1)
            controls.addLayout(checks)

            editor = QtWidgets.QHBoxLayout()
            editor.setSpacing(6)
            for combo in (self._a1_combo, self._a2_combo):
                combo.setMinimumWidth(72)
                combo.setMaximumWidth(110)
            for spin in (self._min_spin, self._max_spin):
                spin.setMinimumWidth(92)
                spin.setMaximumWidth(110)
            self._order_spin.setMinimumWidth(56)
            self._order_spin.setMaximumWidth(64)
            editor.addWidget(QtWidgets.QLabel("A1:"))
            editor.addWidget(self._a1_combo)
            editor.addSpacing(8)
            editor.addWidget(QtWidgets.QLabel("A2:"))
            editor.addWidget(self._a2_combo)
            editor.addSpacing(12)
            editor.addWidget(QtWidgets.QLabel("Min. length:"))
            editor.addWidget(self._min_spin)
            editor.addSpacing(8)
            editor.addWidget(QtWidgets.QLabel("Max. length:"))
            editor.addWidget(self._max_spin)
            editor.addSpacing(12)
            editor.addWidget(QtWidgets.QLabel("Order:"))
            editor.addWidget(self._order_spin)
            editor.addSpacing(8)
            editor.addWidget(self._poly_check)
            editor.addStretch(1)
            controls.addLayout(editor)
            self._a1_combo.currentTextChanged.connect(lambda _text: self._populate_editor_default_range())
            self._a2_combo.currentTextChanged.connect(lambda _text: self._populate_editor_default_range())
            root.addWidget(controls_box)

            table_row = QtWidgets.QHBoxLayout()
            self._table = QtWidgets.QTableWidget(0, 8)
            self._table.setHorizontalHeaderLabels(("No.", "Atom 1", "Atom 2", "Min. (Å)", "Max. (Å)", "Order", "Bound.", "Poly."))
            header = self._table.horizontalHeader()
            header.setStretchLastSection(False)
            for column in range(8):
                header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)
            self._table.setColumnWidth(0, 54)
            self._table.setColumnWidth(1, 78)
            self._table.setColumnWidth(2, 78)
            self._table.setColumnWidth(3, 98)
            self._table.setColumnWidth(4, 98)
            self._table.setColumnWidth(5, 54)
            self._table.setColumnWidth(6, 56)
            self._table.setColumnWidth(7, 58)
            self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self._table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            self._table.itemSelectionChanged.connect(self._sync_editor_from_selection)
            table_row.addWidget(self._table, 1)

            button_col = QtWidgets.QVBoxLayout()
            for label, callback in (
                ("New", self._new_rule),
                ("Delete", self._delete_selected_rule),
                ("Clear", self._clear_rules),
                ("Up", lambda: self._move_selected_rule(-1)),
                ("Down", lambda: self._move_selected_rule(1)),
            ):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(callback)
                button_col.addWidget(button)
            button_col.addStretch(1)
            table_row.addLayout(button_col)
            root.addLayout(table_row, 1)

            bottom = QtWidgets.QHBoxLayout()
            bottom.addStretch(1)
            for label, callback in (("OK", self._accept), ("Cancel", self.reject), ("Apply", self._apply)):
                button = QtWidgets.QPushButton(label)
                button.clicked.connect(callback)
                bottom.addWidget(button)
            root.addLayout(bottom)

        def _structure_symbols(self) -> list[str]:
            symbols = sorted({str(symbol) for symbol in getattr(self.structure, "symbols", []) if str(symbol)})
            return symbols or ["X"]

        def _load_rules(self) -> None:
            active_pairs = set(_structure_pair_keys(self.structure))
            active_pairs.update(self.config.structure.bonding.pair_distances)
            active_pairs.difference_update(self._disabled_pairs)
            for pair_key in sorted(active_pairs):
                distance_range = self._distance_range_for_pair(pair_key)
                if distance_range is None:
                    continue
                self._add_rule_row(pair_key, distance_range, order=self._order_for_pair(pair_key))
            self._renumber_rows()
            if self._table.rowCount() > 0:
                self._table.selectRow(0)
            else:
                self._populate_editor_default_range()

        def _distance_range_for_pair(self, pair_key: str) -> tuple[float, float] | None:
            normalized = normalize_pair_key(pair_key)
            overrides = self.config.structure.bonding.pair_distances
            if normalized in overrides:
                return (float(overrides[normalized][0]), float(overrides[normalized][1]))
            default = VESTA_PAIR_BOND_DISTANCE_RANGES.get(normalized)
            if default is None:
                return None
            return (float(default[0]), float(default[1]))

        def _add_rule_row(
            self,
            pair_key: str,
            distance_range: tuple[float, float],
            *,
            order: int = 1,
            poly: bool = False,
        ) -> None:
            row = self._table.rowCount()
            left, right = normalize_pair_key(pair_key).split("-", 1)
            self._table.insertRow(row)
            values = (str(row + 1), left, right, f"{float(distance_range[0]):.4f}", f"{float(distance_range[1]):.4f}")
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if column == 0:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, column, item)
            order_item = QtWidgets.QTableWidgetItem(str(int(order)))
            self._table.setItem(row, 5, order_item)
            bound_item = QtWidgets.QTableWidgetItem("2")
            bound_item.setFlags(bound_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 6, bound_item)
            poly_item = QtWidgets.QTableWidgetItem("")
            poly_item.setFlags(poly_item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            poly_item.setCheckState(QtCore.Qt.CheckState.Checked if poly else QtCore.Qt.CheckState.Unchecked)
            self._table.setItem(row, 7, poly_item)

        def _sync_editor_from_selection(self) -> None:
            row = self._selected_row()
            if row is None:
                return
            self._syncing_editor = True
            left = self._cell_text(row, 1)
            right = self._cell_text(row, 2)
            try:
                self._set_combo_text(self._a1_combo, left)
                self._set_combo_text(self._a2_combo, right)
                self._min_spin.setValue(self._cell_float(row, 3))
                self._max_spin.setValue(self._cell_float(row, 4))
                self._order_spin.setValue(self._row_order(row))
                self._poly_check.setChecked(self._table.item(row, 7).checkState() == QtCore.Qt.CheckState.Checked)
            finally:
                self._syncing_editor = False

        def _new_rule(self) -> None:
            try:
                pair_key = normalize_pair_key(f"{self._a1_combo.currentText()}-{self._a2_combo.currentText()}")
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Invalid Bond Pair", "A1 and A2 must be valid element symbols.")
                return
            min_distance = float(self._min_spin.value())
            max_distance = float(self._max_spin.value())
            if max_distance <= min_distance:
                QtWidgets.QMessageBox.warning(self, "Invalid Bond Range", "Max. length must be greater than Min. length.")
                return
            row = self._find_pair_row(pair_key)
            if row is not None:
                self._table.selectRow(row)
                QtWidgets.QMessageBox.information(
                    self,
                    "Bond Search",
                    f"{pair_key} is already listed. Edit the table row directly, then click Apply.",
                )
                return
            self._add_rule_row(
                pair_key,
                (min_distance, max_distance),
                order=int(self._order_spin.value()),
                poly=bool(self._poly_check.isChecked()),
            )
            self._disabled_pairs.discard(pair_key)
            self._renumber_rows()
            self._table.selectRow(self._table.rowCount() - 1)

        def _populate_editor_default_range(self) -> None:
            if self._syncing_editor:
                return
            try:
                pair_key = normalize_pair_key(f"{self._a1_combo.currentText()}-{self._a2_combo.currentText()}")
            except ValueError:
                return
            row = self._find_pair_row(pair_key)
            if row is not None:
                self._min_spin.setValue(self._cell_float(row, 3))
                self._max_spin.setValue(self._cell_float(row, 4))
                self._order_spin.setValue(self._row_order(row))
                return
            distance_range = self._distance_range_for_pair(pair_key) or (0.0, 1.0)
            self._min_spin.setValue(float(distance_range[0]))
            self._max_spin.setValue(float(distance_range[1]))
            self._order_spin.setValue(self._order_for_pair(pair_key))

        def _delete_selected_rule(self) -> None:
            row = self._selected_row()
            if row is None:
                return
            pair_key = self._row_pair_key(row)
            self._disabled_pairs.add(pair_key)
            self._table.removeRow(row)
            self._renumber_rows()

        def _clear_rules(self) -> None:
            for row in range(self._table.rowCount()):
                self._disabled_pairs.add(self._row_pair_key(row))
            self._table.setRowCount(0)

        def _move_selected_rule(self, direction: int) -> None:
            row = self._selected_row()
            if row is None:
                return
            target = row + int(direction)
            if target < 0 or target >= self._table.rowCount():
                return
            values = {column: self._cell_text(row, column) for column in (0, 1, 2, 3, 4, 5, 6)}
            poly_state = self._table.item(row, 7).checkState()
            self._table.removeRow(row)
            self._table.insertRow(target)
            for column, value in values.items():
                item = QtWidgets.QTableWidgetItem(value)
                if column == 0:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if column == 6:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(target, column, item)
            poly_item = QtWidgets.QTableWidgetItem("")
            poly_item.setFlags(poly_item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            poly_item.setCheckState(poly_state)
            self._table.setItem(target, 7, poly_item)
            self._renumber_rows()
            self._table.selectRow(target)

        def _rules_from_table(self) -> BondSearchRules:
            pair_distances: dict[str, tuple[float, float]] = {}
            order_rules: dict[str, int] = {}
            active_pairs: set[str] = set()
            for row in range(self._table.rowCount()):
                pair_key = self._row_pair_key(row)
                active_pairs.add(pair_key)
                distance_range = (self._cell_float(row, 3), self._cell_float(row, 4))
                if distance_range[1] <= distance_range[0]:
                    raise ValueError(f"{pair_key}: Max. length must be greater than Min. length.")
                default = VESTA_PAIR_BOND_DISTANCE_RANGES.get(pair_key)
                if default is None or abs(distance_range[0] - default[0]) > 1e-8 or abs(distance_range[1] - default[1]) > 1e-8:
                    pair_distances[pair_key] = distance_range
                order = self._row_order(row)
                if order not in {1, 2, 3}:
                    raise ValueError(f"{pair_key}: Order must be 1, 2, or 3.")
                if order != 1:
                    order_rules[pair_key] = order
            disabled_pairs = sorted(pair for pair in self._disabled_pairs if pair not in active_pairs)
            return BondSearchRules(
                pair_distances=dict(sorted(pair_distances.items())),
                disabled_pairs=disabled_pairs,
                order_rules=dict(sorted(order_rules.items())),
            )

        def _apply(self) -> bool:
            try:
                rules = self._rules_from_table()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid Bond Rules", str(exc))
                return False
            self._result_rules = rules
            if self.on_apply is not None:
                self.on_apply(rules)
            return True

        def _accept(self) -> None:
            if self._apply():
                self.accept()

        def _selected_row(self) -> int | None:
            rows = self._table.selectionModel().selectedRows()
            if not rows:
                return None
            return int(rows[0].row())

        def _find_pair_row(self, pair_key: str) -> int | None:
            normalized = normalize_pair_key(pair_key)
            for row in range(self._table.rowCount()):
                if self._row_pair_key(row) == normalized:
                    return row
            return None

        def _row_pair_key(self, row: int) -> str:
            return normalize_pair_key(f"{self._cell_text(row, 1)}-{self._cell_text(row, 2)}")

        def _row_order(self, row: int) -> int:
            try:
                order = int(self._cell_text(row, 5))
            except ValueError:
                return 1
            return order

        def _cell_text(self, row: int, column: int) -> str:
            item = self._table.item(row, column)
            return "" if item is None else str(item.text()).strip()

        def _cell_float(self, row: int, column: int) -> float:
            try:
                return float(self._cell_text(row, column))
            except ValueError:
                return 0.0

        def _renumber_rows(self) -> None:
            for row in range(self._table.rowCount()):
                item = self._table.item(row, 0)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                    self._table.setItem(row, 0, item)
                item.setText(str(row + 1))

        @staticmethod
        def _set_combo_text(combo: Any, value: str) -> None:
            index = combo.findText(str(value))
            if index >= 0:
                combo.setCurrentIndex(index)
            elif combo.isEditable():
                combo.setEditText(str(value))

        def _order_for_pair(self, pair_key: str) -> int:
            return int(self._order_rules.get(normalize_pair_key(pair_key), 1))


else:  # pragma: no cover

    class BondSearchDialog:  # type: ignore[no-redef]
        pass


def _structure_pair_keys(structure: Structure) -> list[str]:
    symbols = sorted({str(symbol) for symbol in getattr(structure, "symbols", []) if str(symbol)})
    return [normalize_pair_key(f"{left}-{right}") for left, right in combinations_with_replacement(symbols, 2)]


def _normalized_order_rules(order_rules: Any) -> dict[str, int]:
    if not isinstance(order_rules, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in order_rules.items():
        try:
            order = int(value.get("order") if isinstance(value, dict) else value)
            pair_key = normalize_pair_key(str(key))
        except (TypeError, ValueError):
            continue
        if order in {1, 2, 3}:
            out[pair_key] = order
    return out


__all__ = ["BondSearchDialog", "BondSearchRules"]
