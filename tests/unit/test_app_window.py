from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from atomstudio.app.main import _configure_qt_runtime

_configure_qt_runtime()

from PySide6 import QtWidgets

from atomstudio.app.state import AppState, LoadedFrameBundle
from atomstudio.app.window import AtomStudioWindow
from atomstudio.config import RenderJobConfig
from atomstudio.preview.renderer import PreviewCameraState
from atomstudio.preview.builder import build_preview_scene
from atomstudio.preview.types import PreviewSelection, PreviewSettings
from atomstudio.preview.renderer import build_preview_scene as build_render_preview_scene
from atomstudio.render.results import RenderResult
from atomstudio.structure.atom import Atom
from atomstudio.structure.bond import Bond
from atomstudio.structure.structure import Structure


def _qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _Signal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, value) -> None:
        for callback in list(self._callbacks):
            callback(value)


class _FakePreviewCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.selection_changed = _Signal()
        self.interaction_message_changed = _Signal()
        self.axis_overlay_visible = True
        self.last_selection = None
        self.last_atom_update = None
        self.deleted = False
        self.fit_calls = 0
        self.view_presets: list[str] = []
        self.model = None
        self.camera = PreviewCameraState(
            center=(1.0, 2.0, 3.0),
            scale_factor=6.5,
            right=(0.0, 1.0, 0.0),
            up=(0.0, 0.0, 1.0),
            forward=(-1.0, 0.0, 0.0),
        )

    def set_preview_scene(self, preview_scene, frame_index=None):
        self.preview_scene = preview_scene
        return preview_scene

    def fit_to_structure(self, padding=None):
        self.fit_calls += 1
        return None

    def set_view_preset(self, view: str):
        self.view_presets.append(view)
        return None

    def set_axis_overlay_visible(self, visible: bool) -> None:
        self.axis_overlay_visible = bool(visible)

    def clear_selection(self) -> None:
        self.last_selection = None

    def select_preview_object(self, selection):
        self.last_selection = selection
        self.selection_changed.emit(selection)
        return selection

    def select_atom(self, atom_index: int | None):
        selection = None if atom_index is None else PreviewSelection(kind="atom", index=atom_index)
        return self.select_preview_object(selection)

    def select_bond(self, bond_index: int | None):
        selection = None if bond_index is None else PreviewSelection(kind="bond", index=bond_index)
        return self.select_preview_object(selection)

    def update_selected_atom_properties(self, updates):
        self.last_atom_update = dict(updates)
        return True

    def current_camera_state(self):
        return self.camera

    def delete_selected_objects(self):
        self.deleted = True
        return {"atoms": 1, "bonds": 1}


def _render_cfg() -> RenderJobConfig:
    return RenderJobConfig.from_dict(
        {
            "id": "app",
            "input": {"path": "example.xyz", "frames": "last"},
            "output": {"path": "/tmp/app.png"},
            "style": {"scene_style": "default"},
        }
    )


def _preview_scene():
    structure = Structure(atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))], frame_index=0)
    return build_preview_scene(structure, _render_cfg(), PreviewSettings())


def _render_preview_scene():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(0.5, 0.0, 0.0)),
        ],
        frame_index=0,
    )
    return build_render_preview_scene(structure, _render_cfg())


def test_window_builds_inspector_and_updates_selection(monkeypatch):
    _qt_app()
    fake_preview = _FakePreviewCanvas()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: fake_preview)

    state = AppState()
    state.set_loaded_frames(
        LoadedFrameBundle(source_path="/tmp/example.xyz", frames=[Structure(frame_index=0, source_path="/tmp/example.xyz")]),
        render_config=_render_cfg(),
    )
    state.set_preview_scene(_preview_scene())

    window = AtomStudioWindow(state=state)

    assert window._inspector_dock.windowTitle() == ""
    assert [window._inspector.tabText(i) for i in range(window._inspector.count())] == ["Output", "Summary", "Metadata"]
    assert [window._left_tabs.tabText(i) for i in range(window._left_tabs.count())] == ["Scene", "Render", "Object"]
    assert "OpenGL version:" in window._inspector._summary_view.toPlainText()
    assert "/tmp/example.xyz" in window._inspector._summary_view.toPlainText()
    assert window._log_dock.windowTitle() == "Status Log"
    assert "toggle_axis_overlay" in window._menu_handles.actions

    fake_preview.selection_changed.emit(PreviewSelection(kind="atom", index=0))

    assert window.state.selected_object is not None
    assert window.state.selected_object.kind == "atom"
    assert window.state.selected_payload is not None
    assert window.state.selected_payload["object"]["symbol"] == "O"
    assert '"symbol": "O"' in window._inspector._object_view.toPlainText()
    assert '"position": "0.00000, 0.00000, 0.00000"' in window._inspector._object_view.toPlainText()

    window.set_axis_overlay_visible(False)
    assert fake_preview.axis_overlay_visible is False


def test_window_inspector_shows_multiple_selected_atom_objects(monkeypatch):
    _qt_app()
    fake_preview = _FakePreviewCanvas()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: fake_preview)

    state = AppState()
    state.set_loaded_frames(
        LoadedFrameBundle(source_path="/tmp/example.xyz", frames=[Structure(frame_index=0, source_path="/tmp/example.xyz")]),
        render_config=_render_cfg(),
    )
    render_scene = _render_preview_scene()
    state.set_preview_scene(render_scene)
    fake_preview.model = type(
        "Model",
        (),
        {
            "scene": render_scene,
            "selected_atom_indices": {0, 1},
            "selected_ordered_atoms": [0, 1],
        },
    )()

    window = AtomStudioWindow(state=state)
    fake_preview.selection_changed.emit(PreviewSelection(kind="atom", index=0))

    text = window._inspector._object_view.toPlainText()
    assert '"symbol": "O"' in text
    assert '"symbol": "H"' in text
    assert text.count('"color"') == 2
    assert window._left_tabs.tabText(2) == "Object"


def test_window_applies_inspector_atom_edits(monkeypatch):
    _qt_app()
    fake_preview = _FakePreviewCanvas()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: fake_preview)

    state = AppState()
    state.set_loaded_frames(
        LoadedFrameBundle(source_path="/tmp/example.xyz", frames=[Structure(frame_index=0, source_path="/tmp/example.xyz")]),
        render_config=_render_cfg(),
    )
    state.set_preview_scene(_preview_scene())
    window = AtomStudioWindow(state=state)
    fake_preview.select_atom(0)

    window._on_inspector_object_edit(
        {
            "index": 0,
            "symbol": "N",
            "position": (0.1, 0.2, 0.3),
            "color": (0.2, 0.3, 0.4, 1.0),
            "radius": 0.8,
        }
    )

    assert fake_preview.last_atom_update["symbol"] == "N"
    assert fake_preview.last_atom_update["color"] == (0.2, 0.3, 0.4, 1.0)


def test_object_tab_atom_editor_uses_compact_fields(monkeypatch):
    _qt_app()
    fake_preview = _FakePreviewCanvas()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: fake_preview)
    state = AppState()
    state.set_loaded_frames(
        LoadedFrameBundle(source_path="/tmp/example.xyz", frames=[Structure(frame_index=0, source_path="/tmp/example.xyz")]),
        render_config=_render_cfg(),
    )
    state.set_preview_scene(_preview_scene())
    window = AtomStudioWindow(state=state)
    fake_preview.selection_changed.emit(PreviewSelection(kind="atom", index=0))

    assert set(window._object_edit_fields) == {"symbol", "position", "color", "radius"}
    assert window._object_edit_fields["position"].text() == "0.0, 0.0, 0.0"
    assert window._object_edit_fields["color"].text()

    window._object_edit_fields["symbol"].setText("N")
    window._object_edit_fields["position"].setText("0.1, 0.2, 0.3")
    window._object_edit_fields["color"].setText("0.2, 0.3, 0.4, 1.0")
    window._object_edit_fields["radius"].setText("0.8")
    window._emit_selected_object_edit()

    assert fake_preview.last_atom_update["symbol"] == "N"
    assert fake_preview.last_atom_update["position"] == (0.1, 0.2, 0.3)
    assert fake_preview.last_atom_update["color"] == (0.2, 0.3, 0.4, 1.0)


def test_window_render_result_requires_existing_output_file(monkeypatch, tmp_path: Path):
    _qt_app()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: _FakePreviewCanvas())
    window = AtomStudioWindow()
    missing = tmp_path / "missing.png"

    window._handle_render_result(RenderResult(success=True, output_path=str(missing), frame_index=0, message="ok"))

    assert window.state.last_render_output is None
    assert window.state.last_error is not None
    assert "output file was not created" in window.state.last_error
    assert "Error:" in window.state.status
    assert "Render failed" in window._inspector._object_view.toPlainText()
    assert str(missing) in window._inspector._object_view.toPlainText()


def test_window_render_result_accepts_existing_output_file(monkeypatch, tmp_path: Path):
    _qt_app()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: _FakePreviewCanvas())
    window = AtomStudioWindow()
    output = tmp_path / "render.png"
    output.write_bytes(b"fake image")

    window._handle_render_result(RenderResult(success=True, output_path=str(output), frame_index=0, message="ok"))

    assert window.state.last_render_output == str(output)
    assert window.state.last_error is None
    assert window.state.status == f"Rendered {output}"
    assert "Render succeeded" in window._inspector._object_view.toPlainText()
    assert str(output) in window._inspector._object_view.toPlainText()


def test_window_render_request_normalizes_wsl_unc_output_path(monkeypatch):
    _qt_app()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: _FakePreviewCanvas())
    state = AppState()
    structure = Structure(frame_index=0, source_path=r"\\wsl.localhost\Ubuntu\home\xinyang\water.xyz")
    state.set_loaded_frames(LoadedFrameBundle(source_path=structure.source_path, frames=[structure]), render_config=_render_cfg())
    window = AtomStudioWindow(state=state)
    window._output_path_edit.setText(r"\\wsl.localhost\Ubuntu\home\xinyang\work\atomstudio\water.png")

    request = window._build_render_request()

    assert request is not None
    assert request.render_config.output.path == "/home/xinyang/work/atomstudio/water.png"
    assert window._output_path_edit.text() == "/home/xinyang/work/atomstudio/water.png"


def test_render_tab_exposes_scene_style_combo_settings_and_animation(monkeypatch):
    _qt_app()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: _FakePreviewCanvas())
    window = AtomStudioWindow()

    assert window._render_style_combo.currentData() == "default"
    assert window._render_style_combo.findData("handdrawn") >= 0
    assert window._render_style_combo.findData("handdrawn_v2") >= 0
    groups = {box.title() for box in window.findChildren(QtWidgets.QGroupBox)}
    assert "Path" in groups
    assert "Style" in groups
    assert "Settings" in groups
    assert "Animation" in groups
    assert "Render" not in groups
    assert window._output_path_edit is not None
    assert window._blender_path_edit is not None
    assert window._transparent_bg_radio.text() == "Transparent"
    assert window._unicolor_bg_radio.text() == "Unicolor"
    assert window._unicolor_bg_radio.isChecked()
    assert window._background_color_edit is not None
    assert window._animation_button.text() == "Render animation"


def test_render_style_combo_updates_render_config(monkeypatch):
    _qt_app()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: _FakePreviewCanvas())
    state = AppState()
    structure = Structure(frame_index=0, source_path="/tmp/example.xyz")
    state.set_loaded_frames(LoadedFrameBundle(source_path="/tmp/example.xyz", frames=[structure]), render_config=_render_cfg())
    window = AtomStudioWindow(state=state)
    window._refresh_preview = lambda: None

    window._render_style_combo.setCurrentIndex(window._render_style_combo.findData("handdrawn"))

    assert window.state.render_config is not None
    assert window.state.render_config.style.scene_style == "handdrawn"


def test_render_request_uses_current_preview_camera(monkeypatch):
    _qt_app()
    fake_preview = _FakePreviewCanvas()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: fake_preview)
    state = AppState()
    structure = Structure(frame_index=0, source_path="/tmp/example.xyz")
    state.set_loaded_frames(LoadedFrameBundle(source_path="/tmp/example.xyz", frames=[structure]), render_config=_render_cfg())
    window = AtomStudioWindow(state=state)

    request = window._build_render_request()

    assert request is not None
    assert request.render_config.camera.center == (1.0, 2.0, 3.0)
    assert request.render_config.camera.right == (0.0, 1.0, 0.0)
    assert request.render_config.camera.up == (0.0, 0.0, 1.0)
    assert request.render_config.camera.forward == (-1.0, 0.0, 0.0)
    assert request.render_config.camera.ortho_scale == 6.5


def test_render_request_applies_background_settings(monkeypatch):
    _qt_app()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: _FakePreviewCanvas())
    state = AppState()
    structure = Structure(frame_index=0, source_path="/tmp/example.xyz")
    state.set_loaded_frames(LoadedFrameBundle(source_path="/tmp/example.xyz", frames=[structure]), render_config=_render_cfg())
    window = AtomStudioWindow(state=state)
    window._unicolor_bg_radio.setChecked(True)
    window._background_color_edit.setText("0.1, 0.2, 0.3, 1.0")

    request = window._build_render_request()

    assert request is not None
    assert request.render_config.render.transparent_bg is False
    assert request.render_config.style.background == (0.1, 0.2, 0.3, 1.0)


def test_render_request_applies_transparent_background(monkeypatch):
    _qt_app()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: _FakePreviewCanvas())
    state = AppState()
    structure = Structure(frame_index=0, source_path="/tmp/example.xyz")
    state.set_loaded_frames(LoadedFrameBundle(source_path="/tmp/example.xyz", frames=[structure]), render_config=_render_cfg())
    window = AtomStudioWindow(state=state)
    window._transparent_bg_radio.setChecked(True)

    request = window._build_render_request()

    assert request is not None
    assert request.render_config.render.transparent_bg is True


def test_animation_request_uses_all_loaded_frames_and_output_dir(monkeypatch, tmp_path: Path):
    _qt_app()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: _FakePreviewCanvas())
    frames = [
        Structure(frame_index=4, source_path="/tmp/example.xyz"),
        Structure(frame_index=9, source_path="/tmp/example.xyz"),
    ]
    state = AppState()
    state.set_loaded_frames(LoadedFrameBundle(source_path="/tmp/example.xyz", frames=frames), render_config=_render_cfg())
    window = AtomStudioWindow(state=state)
    window._output_path_edit.setText(str(tmp_path / "movie.png"))

    request = window._build_animation_request()

    assert request is not None
    assert len(request.structures) == 2
    assert request.output_dir == str((tmp_path / "movie_frames").resolve())
    assert request.filename_template == "movie_{frame:04d}.png"
    assert request.render_config.output.dir == request.output_dir
    assert request.render_config.output.filename_template == request.filename_template


def test_selected_atom_edit_updates_current_structure_for_render(monkeypatch):
    _qt_app()
    fake_preview = _FakePreviewCanvas()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: fake_preview)
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        source_path="/tmp/example.xyz",
    )
    state = AppState()
    state.set_loaded_frames(LoadedFrameBundle(source_path="/tmp/example.xyz", frames=[structure]), render_config=_render_cfg())
    state.set_preview_scene(_preview_scene())
    window = AtomStudioWindow(state=state)

    window._on_inspector_object_edit(
        {
            "index": 0,
            "symbol": "N",
            "position": (0.1, 0.2, 0.3),
            "color": (0.2, 0.3, 0.4, 1.0),
            "radius": 0.8,
        }
    )

    assert structure.atoms[0].symbol == "N"
    assert structure.atoms[0].position == (0.1, 0.2, 0.3)
    assert structure.atoms[0].color == (0.2, 0.3, 0.4, 1.0)
    assert structure.atoms[0].radius == 0.8


def test_window_delete_selection_updates_current_structure(monkeypatch):
    _qt_app()
    fake_preview = _FakePreviewCanvas()
    monkeypatch.setattr("atomstudio.app.window.build_preview_host", lambda parent=None: fake_preview)
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0)),
        ],
        bonds=[Bond(id=0, a=0, b=1)],
        source_path="/tmp/example.xyz",
    )
    state = AppState()
    state.set_loaded_frames(LoadedFrameBundle(source_path="/tmp/example.xyz", frames=[structure]), render_config=_render_cfg())
    state.set_preview_scene(_preview_scene())
    fake_preview.model = type(
        "Model",
        (),
        {
            "selected_atom_indices": {0},
            "selected_bond_indices": {0},
            "selection": PreviewSelection(kind="atom", index=0),
            "scene": state.preview_scene,
        },
    )()
    window = AtomStudioWindow(state=state)

    window.delete_selected_objects()

    assert fake_preview.deleted is True
    assert [atom.index for atom in structure.atoms] == [1]
    assert structure.bonds == []
    assert window.state.dirty is True
