from __future__ import annotations

from typing import Any

from atomstudio.preview.gl.shader_styles import shader_style_choices
from atomstudio.style.registry import scene_style_choices
from atomstudio.visual_defaults import CELL_DEFAULT_COLOR

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtWidgets  # type: ignore
except Exception:  # pragma: no cover
    QtCore = QtWidgets = None


def compact_layout(layout: Any, *, spacing: int = 4) -> Any:
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(int(spacing))
    return layout


def panel_box_layout(layout: Any) -> Any:
    layout.setContentsMargins(8, 10, 8, 8)
    layout.setSpacing(6)
    return layout


def panel_form_layout(layout: Any) -> Any:
    layout.setContentsMargins(8, 10, 8, 8)
    layout.setHorizontalSpacing(8)
    layout.setVerticalSpacing(4)
    return layout


def build_scene_tab(owner: Any) -> Any:
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    layout.setSpacing(10)
    layout.addWidget(build_frame_panel(owner))
    layout.addWidget(build_atom_style_panel(owner))
    layout.addWidget(build_scene_style_panel(owner))
    layout.addWidget(build_cell_style_panel(owner))
    layout.addWidget(build_camera_panel(owner))
    layout.addStretch(1)
    return page


def build_render_tab(owner: Any) -> Any:
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    layout.setSpacing(10)
    layout.addWidget(build_render_path_panel(owner))
    layout.addWidget(build_render_style_panel(owner))
    layout.addWidget(build_render_panel(owner))
    layout.addWidget(build_animation_panel(owner))
    layout.addStretch(1)
    return page


def build_object_tab(owner: Any) -> Any:
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    layout.setSpacing(10)
    layout.addWidget(build_element_style_panel(owner))
    layout.addWidget(build_bond_style_panel(owner))
    layout.addWidget(build_selected_object_panel(owner))
    layout.addStretch(1)
    layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setWidget(page)
    return scroll


def build_frame_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Frames")
    layout = panel_box_layout(QtWidgets.QVBoxLayout(box))

    owner._frame_label = QtWidgets.QLabel("No frames loaded")
    layout.addWidget(owner._frame_label)

    owner._frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    owner._frame_slider.setEnabled(False)
    owner._frame_slider.setMinimum(0)
    owner._frame_slider.setMaximum(0)
    layout.addWidget(owner._frame_slider)

    nav_row = compact_layout(QtWidgets.QHBoxLayout(), spacing=6)
    owner._first_frame_button = QtWidgets.QPushButton("First")
    owner._previous_frame_button = QtWidgets.QPushButton("Prev")
    owner._next_frame_button = QtWidgets.QPushButton("Next")
    owner._last_frame_button = QtWidgets.QPushButton("Last")
    nav_row.addWidget(owner._first_frame_button)
    nav_row.addWidget(owner._previous_frame_button)
    nav_row.addWidget(owner._next_frame_button)
    nav_row.addWidget(owner._last_frame_button)
    nav_widget = QtWidgets.QWidget()
    nav_widget.setLayout(nav_row)
    layout.addWidget(nav_widget)

    owner._first_frame_button.clicked.connect(owner._first_frame)
    owner._previous_frame_button.clicked.connect(lambda: owner._step_frame(-1))
    owner._next_frame_button.clicked.connect(lambda: owner._step_frame(1))
    owner._last_frame_button.clicked.connect(owner._last_frame)
    return box


def build_atom_style_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Representation")
    layout = panel_box_layout(QtWidgets.QVBoxLayout(box))
    owner._atom_representation_group = QtWidgets.QButtonGroup(box)
    owner._atom_representation_buttons: dict[str, Any] = {}
    for label, value in (
        ("Ball-stick", "ball_stick"),
        ("Space-filling", "space_filling"),
        ("Polyhedra", "polyhedra"),
        ("Stick", "stick"),
    ):
        radio = QtWidgets.QRadioButton(label)
        radio.setProperty("value", value)
        owner._atom_representation_group.addButton(radio)
        owner._atom_representation_buttons[value] = radio
        layout.addWidget(radio)
    owner._atom_representation_group.buttonToggled.connect(
        lambda _button, checked: owner._apply_scene_tab_settings() if checked else None
    )
    return box


def build_scene_style_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Scene Style")
    layout = panel_box_layout(QtWidgets.QVBoxLayout(box))
    owner._scene_style_group = QtWidgets.QButtonGroup(box)
    owner._scene_style_buttons: dict[str, Any] = {}
    for value in shader_style_choices():
        label = str(value).replace("_", " ").title()
        radio = QtWidgets.QRadioButton(label)
        radio.setProperty("value", value)
        owner._scene_style_group.addButton(radio)
        owner._scene_style_buttons[value] = radio
        layout.addWidget(radio)
    owner._scene_style_group.buttonToggled.connect(
        lambda _button, checked: owner._apply_scene_tab_settings() if checked else None
    )
    return box


def build_cell_style_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Cell")
    layout = panel_form_layout(QtWidgets.QFormLayout(box))
    owner._cell_show_check = QtWidgets.QCheckBox("Show cell")
    owner._cell_show_check.setChecked(True)
    layout.addRow(owner._cell_show_check)

    owner._cell_radius_spin = QtWidgets.QDoubleSpinBox()
    owner._cell_radius_spin.setRange(0.001, 1.0)
    owner._cell_radius_spin.setDecimals(3)
    owner._cell_radius_spin.setSingleStep(0.01)
    owner._cell_radius_spin.setValue(0.01)
    layout.addRow("Line radius", owner._cell_radius_spin)

    owner._cell_color_edit = QtWidgets.QLineEdit()
    owner._cell_color_edit.setText("0.4500, 0.4500, 0.4500, 1.0000")
    owner._cell_color_button = QtWidgets.QPushButton()
    owner._cell_color_button.setObjectName("colorSwatchButton")
    owner._cell_color_button.setFixedSize(34, 28)
    owner._set_color_button(owner._cell_color_button, CELL_DEFAULT_COLOR)
    owner._cell_color_button.clicked.connect(
        lambda: owner._choose_color_for_edit(
            owner._cell_color_edit,
            owner._cell_color_button,
            owner._apply_scene_tab_settings,
        )
    )
    color_row = compact_layout(QtWidgets.QHBoxLayout(), spacing=6)
    color_row.addWidget(owner._cell_color_edit, 1)
    color_row.addWidget(owner._cell_color_button)
    color_widget = QtWidgets.QWidget()
    color_widget.setLayout(color_row)
    layout.addRow("Color", color_widget)

    owner._cell_transparent_check = QtWidgets.QCheckBox("Transparent")
    layout.addRow(owner._cell_transparent_check)

    owner._boundary_atoms_check = QtWidgets.QCheckBox("Show boundary atoms")
    owner._boundary_atoms_check.setChecked(True)
    layout.addRow(owner._boundary_atoms_check)

    owner._boundary_atom_sigma_spin = QtWidgets.QDoubleSpinBox()
    owner._boundary_atom_sigma_spin.setRange(0.0, 0.49)
    owner._boundary_atom_sigma_spin.setDecimals(4)
    owner._boundary_atom_sigma_spin.setSingleStep(0.005)
    owner._boundary_atom_sigma_spin.setValue(0.03)
    layout.addRow("Boundary sigma", owner._boundary_atom_sigma_spin)

    owner._cell_show_check.toggled.connect(lambda _checked: owner._apply_scene_tab_settings())
    owner._cell_radius_spin.valueChanged.connect(lambda _value: owner._apply_scene_tab_settings())
    owner._cell_color_edit.editingFinished.connect(owner._apply_scene_tab_settings)
    owner._cell_transparent_check.toggled.connect(lambda _checked: owner._apply_scene_tab_settings())
    owner._boundary_atoms_check.toggled.connect(lambda _checked: owner._apply_scene_tab_settings())
    owner._boundary_atom_sigma_spin.valueChanged.connect(lambda _value: owner._apply_scene_tab_settings())
    return box


def build_camera_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Camera")
    layout = panel_box_layout(QtWidgets.QVBoxLayout(box))
    owner._camera_projection_group = QtWidgets.QButtonGroup(box)
    owner._camera_projection_buttons: dict[str, Any] = {}
    for label, value in (("Orthogonal", "ORTHOGRAPHIC"), ("Perspective", "PERSPECTIVE")):
        radio = QtWidgets.QRadioButton(label)
        radio.setProperty("value", value)
        owner._camera_projection_group.addButton(radio)
        owner._camera_projection_buttons[value] = radio
        layout.addWidget(radio)
    owner._camera_projection_group.buttonToggled.connect(
        lambda _button, checked: owner._apply_scene_tab_settings() if checked else None
    )
    return box


def build_render_style_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Style")
    layout = panel_form_layout(QtWidgets.QFormLayout(box))
    owner._render_style_combo = QtWidgets.QComboBox()
    for style_name in scene_style_choices():
        owner._render_style_combo.addItem(str(style_name).replace("_", " ").title(), str(style_name))
    layout.addRow("Render style", owner._render_style_combo)
    owner._render_style_combo.currentIndexChanged.connect(lambda _index: owner._apply_render_style_choice())
    return box


def build_render_path_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Path")
    layout = panel_form_layout(QtWidgets.QFormLayout(box))

    output_row = compact_layout(QtWidgets.QHBoxLayout(), spacing=6)
    owner._output_path_edit = QtWidgets.QLineEdit()
    owner._browse_output_button = QtWidgets.QPushButton("Browse")
    output_row.addWidget(owner._output_path_edit, 1)
    output_row.addWidget(owner._browse_output_button)
    output_widget = QtWidgets.QWidget()
    output_widget.setLayout(output_row)
    layout.addRow("Output path", output_widget)

    blender_row = compact_layout(QtWidgets.QHBoxLayout(), spacing=6)
    owner._blender_path_edit = QtWidgets.QLineEdit()
    owner._browse_blender_button = QtWidgets.QPushButton("Browse")
    blender_row.addWidget(owner._blender_path_edit, 1)
    blender_row.addWidget(owner._browse_blender_button)
    blender_widget = QtWidgets.QWidget()
    blender_widget.setLayout(blender_row)
    layout.addRow("Blender path", blender_widget)
    return box


def build_render_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Settings")
    layout = panel_form_layout(QtWidgets.QFormLayout(box))

    owner._render_engine_combo = QtWidgets.QComboBox()
    for label, value in (
        ("Cycles", "cycles"),
        ("Eevee", "eevee"),
    ):
        owner._render_engine_combo.addItem(label, value)
    layout.addRow("Engine", owner._render_engine_combo)

    owner._render_device_combo = QtWidgets.QComboBox()
    for label, value in (
        ("Auto", "auto"),
        ("GPU", "gpu"),
        ("CPU", "cpu"),
    ):
        owner._render_device_combo.addItem(label, value)
    layout.addRow("Device", owner._render_device_combo)

    owner._resolution_x_spin = QtWidgets.QSpinBox()
    owner._resolution_y_spin = QtWidgets.QSpinBox()
    for spin in (owner._resolution_x_spin, owner._resolution_y_spin):
        spin.setRange(64, 32768)
        spin.setSingleStep(64)
    resolution_row = compact_layout(QtWidgets.QHBoxLayout(), spacing=6)
    resolution_row.addWidget(owner._resolution_x_spin)
    resolution_row.addWidget(QtWidgets.QLabel("x"))
    resolution_row.addWidget(owner._resolution_y_spin)
    resolution_widget = QtWidgets.QWidget()
    resolution_widget.setLayout(resolution_row)
    layout.addRow("Resolution", resolution_widget)

    owner._samples_spin = QtWidgets.QSpinBox()
    owner._samples_spin.setRange(1, 4096)
    owner._samples_spin.setSingleStep(16)
    layout.addRow("Samples", owner._samples_spin)

    owner._sphere_segments_spin = QtWidgets.QSpinBox()
    owner._sphere_segments_spin.setRange(8, 256)
    owner._sphere_segments_spin.setSingleStep(4)
    layout.addRow("Sphere segments", owner._sphere_segments_spin)

    owner._sphere_rings_spin = QtWidgets.QSpinBox()
    owner._sphere_rings_spin.setRange(4, 128)
    owner._sphere_rings_spin.setSingleStep(2)
    layout.addRow("Sphere rings", owner._sphere_rings_spin)

    owner._bond_vertices_spin = QtWidgets.QSpinBox()
    owner._bond_vertices_spin.setRange(6, 128)
    owner._bond_vertices_spin.setSingleStep(2)
    layout.addRow("Bond vertices", owner._bond_vertices_spin)

    owner._background_mode_group = QtWidgets.QButtonGroup(box)
    owner._transparent_bg_radio = QtWidgets.QRadioButton("Transparent")
    owner._unicolor_bg_radio = QtWidgets.QRadioButton("Unicolor")
    owner._background_color_edit = QtWidgets.QLineEdit()
    owner._background_color_edit.setText("1.0000, 1.0000, 1.0000, 1.0000")
    owner._background_color_button = QtWidgets.QPushButton()
    owner._background_color_button.setObjectName("colorSwatchButton")
    owner._background_color_button.setFixedSize(34, 28)
    owner._set_color_button(owner._background_color_button, (1.0, 1.0, 1.0, 1.0))
    owner._background_color_button.clicked.connect(
        lambda: owner._choose_color_for_edit(
            owner._background_color_edit,
            owner._background_color_button,
            owner._apply_render_settings,
        )
    )
    background_row = compact_layout(QtWidgets.QHBoxLayout(), spacing=6)
    background_row.addWidget(owner._transparent_bg_radio)
    background_row.addWidget(owner._unicolor_bg_radio)
    background_row.addWidget(owner._background_color_edit, 1)
    background_row.addWidget(owner._background_color_button)
    owner._background_mode_group.addButton(owner._transparent_bg_radio)
    owner._background_mode_group.addButton(owner._unicolor_bg_radio)
    owner._unicolor_bg_radio.setChecked(True)
    background_widget = QtWidgets.QWidget()
    background_widget.setLayout(background_row)
    layout.addRow("Background", background_widget)

    owner._render_button = QtWidgets.QPushButton("Render final image")
    layout.addRow(owner._render_button)
    owner._render_engine_combo.currentIndexChanged.connect(lambda _index: owner._apply_render_settings())
    owner._render_device_combo.currentIndexChanged.connect(lambda _index: owner._apply_render_settings())
    return box


def build_animation_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Animation")
    layout = panel_form_layout(QtWidgets.QFormLayout(box))
    owner._animation_scope_label = QtWidgets.QLabel("All loaded frames")
    layout.addRow("Frames", owner._animation_scope_label)
    owner._animation_start_spin = QtWidgets.QSpinBox()
    owner._animation_end_spin = QtWidgets.QSpinBox()
    for spin in (owner._animation_start_spin, owner._animation_end_spin):
        spin.setRange(1, 100000)
        spin.setSingleStep(1)
        spin.setValue(1)
    range_row = compact_layout(QtWidgets.QHBoxLayout(), spacing=6)
    range_row.addWidget(owner._animation_start_spin)
    range_row.addWidget(QtWidgets.QLabel("to"))
    range_row.addWidget(owner._animation_end_spin)
    range_widget = QtWidgets.QWidget()
    range_widget.setLayout(range_row)
    layout.addRow("Range", range_widget)
    owner._animation_frame_step_spin = QtWidgets.QSpinBox()
    owner._animation_frame_step_spin.setRange(1, 100000)
    owner._animation_frame_step_spin.setSingleStep(1)
    owner._animation_frame_step_spin.setValue(1)
    layout.addRow("Frame step", owner._animation_frame_step_spin)
    owner._animation_button = QtWidgets.QPushButton("Render animation")
    owner._animation_start_spin.valueChanged.connect(lambda _value: owner._sync_state_to_ui())
    owner._animation_end_spin.valueChanged.connect(lambda _value: owner._sync_state_to_ui())
    owner._animation_frame_step_spin.valueChanged.connect(lambda _value: owner._sync_state_to_ui())
    layout.addRow(owner._animation_button)
    return box


def build_element_style_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Atoms")
    layout = panel_form_layout(QtWidgets.QFormLayout(box))
    owner._element_symbol_combo = QtWidgets.QComboBox()
    owner._element_symbol_combo.setEditable(False)
    layout.addRow("Element", owner._element_symbol_combo)
    owner._element_color_edit = QtWidgets.QLineEdit()
    owner._element_color_edit.setText("0.7000, 0.7000, 0.7000, 1.0000")
    owner._element_color_button = QtWidgets.QPushButton()
    owner._element_color_button.setObjectName("colorSwatchButton")
    owner._element_color_button.setFixedSize(34, 28)
    owner._set_color_button(owner._element_color_button, (0.7, 0.7, 0.7, 1.0))
    owner._element_color_button.clicked.connect(
        lambda: owner._choose_color_for_edit(
            owner._element_color_edit,
            owner._element_color_button,
            owner._apply_element_style,
        )
    )
    color_row = compact_layout(QtWidgets.QHBoxLayout(), spacing=6)
    color_row.addWidget(owner._element_color_edit, 1)
    color_row.addWidget(owner._element_color_button)
    color_widget = QtWidgets.QWidget()
    color_widget.setLayout(color_row)
    layout.addRow("Color", color_widget)
    owner._element_radius_spin = QtWidgets.QDoubleSpinBox()
    owner._element_radius_spin.setRange(0.01, 10.0)
    owner._element_radius_spin.setDecimals(3)
    owner._element_radius_spin.setSingleStep(0.05)
    owner._element_radius_spin.setKeyboardTracking(False)
    owner._element_radius_spin.setValue(1.0)
    layout.addRow("Radius", owner._element_radius_spin)
    owner._element_symbol_combo.currentTextChanged.connect(lambda _text: owner._populate_element_style_controls())
    owner._element_radius_spin.valueChanged.connect(lambda _value: owner._mark_element_radius_dirty())
    owner._element_color_edit.editingFinished.connect(owner._apply_element_style)
    owner._element_radius_spin.editingFinished.connect(owner._apply_element_style)
    return box


def build_bond_style_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Bonds")
    layout = panel_form_layout(QtWidgets.QFormLayout(box))
    owner._bond_style_combo = QtWidgets.QComboBox()
    for label, value in (
        ("Unicolor", "unicolor"),
        ("Bicolor", "bicolor"),
        ("Dashed", "dashed"),
        ("Dotted", "dotted"),
        ("Gradient", "gradient"),
    ):
        owner._bond_style_combo.addItem(label, value)
    layout.addRow("Style", owner._bond_style_combo)
    owner._bond_color_edit = QtWidgets.QLineEdit()
    owner._bond_color_edit.setText("0.6500, 0.6500, 0.6500, 1.0000")
    owner._bond_color_button = QtWidgets.QPushButton()
    owner._bond_color_button.setObjectName("colorSwatchButton")
    owner._bond_color_button.setFixedSize(34, 28)
    owner._set_color_button(owner._bond_color_button, (0.65, 0.65, 0.65, 1.0))
    owner._bond_color_button.clicked.connect(
        lambda: owner._choose_color_for_edit(
            owner._bond_color_edit,
            owner._bond_color_button,
            owner._apply_bond_style,
        )
    )
    color_row = compact_layout(QtWidgets.QHBoxLayout(), spacing=6)
    color_row.addWidget(owner._bond_color_edit, 1)
    color_row.addWidget(owner._bond_color_button)
    color_widget = QtWidgets.QWidget()
    color_widget.setLayout(color_row)
    layout.addRow("Color", color_widget)
    owner._bond_radius_spin = QtWidgets.QDoubleSpinBox()
    owner._bond_radius_spin.setRange(0.005, 2.0)
    owner._bond_radius_spin.setDecimals(3)
    owner._bond_radius_spin.setSingleStep(0.01)
    owner._bond_radius_spin.setValue(0.08)
    layout.addRow("Radius", owner._bond_radius_spin)
    owner._show_hydrogen_bonds_check = QtWidgets.QCheckBox("Show hydrogen bonds")
    owner._show_hydrogen_bonds_check.setChecked(True)
    layout.addRow(owner._show_hydrogen_bonds_check)
    owner._bond_search_button = QtWidgets.QPushButton("Edit bond search...")
    layout.addRow("", owner._bond_search_button)
    owner._bond_style_combo.setCurrentIndex(1)
    owner._bond_style_combo.currentIndexChanged.connect(lambda _index: owner._apply_bond_style())
    owner._bond_color_edit.editingFinished.connect(owner._apply_bond_style)
    owner._bond_radius_spin.valueChanged.connect(lambda _value: owner._apply_bond_style())
    owner._show_hydrogen_bonds_check.toggled.connect(lambda _checked: owner._apply_hydrogen_bond_visibility())
    owner._bond_search_button.clicked.connect(owner._open_bond_search_dialog)
    return box


def build_selected_object_panel(owner: Any) -> Any:
    box = QtWidgets.QGroupBox("Selected Objects")
    layout = panel_box_layout(QtWidgets.QVBoxLayout(box))
    fields = QtWidgets.QGridLayout()
    fields.setContentsMargins(0, 0, 0, 0)
    fields.setHorizontalSpacing(8)
    fields.setVerticalSpacing(4)
    owner._object_edit_fields: dict[str, Any] = {}
    for row_index, (key, label) in enumerate(
        (
            ("symbol", "Symbol"),
            ("fractional", "Fractional"),
            ("position", "Position"),
            ("color", "Color"),
            ("radius", "Radius"),
        )
    ):
        label_widget = QtWidgets.QLabel(label)
        label_widget.setFixedHeight(30)
        label_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
        edit = QtWidgets.QLineEdit()
        owner._object_edit_fields[key] = edit
        fields.addWidget(label_widget, row_index, 0)
        if key == "color":
            owner._object_color_button = QtWidgets.QPushButton()
            owner._object_color_button.setObjectName("colorSwatchButton")
            owner._object_color_button.setFixedSize(34, 30)
            owner._object_color_button.clicked.connect(
                lambda: owner._choose_color_for_edit(
                    owner._object_edit_fields["color"],
                    owner._object_color_button,
                    owner._emit_selected_object_edit,
                )
            )
            row = compact_layout(QtWidgets.QHBoxLayout(), spacing=6)
            row.addWidget(edit, 1)
            row.addWidget(owner._object_color_button)
            widget = QtWidgets.QWidget()
            widget.setFixedHeight(30)
            widget.setLayout(row)
            fields.addWidget(widget, row_index, 1)
        else:
            fields.addWidget(edit, row_index, 1)
        if key == "fractional":
            edit.editingFinished.connect(lambda: owner._emit_selected_object_edit(position_source="fractional"))
        else:
            edit.editingFinished.connect(owner._emit_selected_object_edit)
    fields.setColumnStretch(1, 1)
    layout.addLayout(fields)
    return box
