from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from atomstudio.app.main import _configure_qt_runtime

_configure_qt_runtime()

from PySide6 import QtCore, QtTest, QtWidgets

from atomstudio.app.preview_canvas import PreviewCanvas
from atomstudio.config import RenderJobConfig
from atomstudio.structure.atom import Atom
from atomstudio.structure.bond import Bond
from atomstudio.structure.structure import Structure


class _MouseEvent:
    def __init__(self, pos: tuple[float, float], button: int = 1, modifiers=()) -> None:
        self.pos = pos
        self.button = button
        self.modifiers = modifiers
        self.handled = False


def _qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture(autouse=True)
def _force_instanced_capability_for_offscreen_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PreviewCanvas,
        "_detect_instancing_status",
        staticmethod(lambda: {"supported": True, "backend": "test-glplus", "reason": "available"}),
    )


def _cfg() -> RenderJobConfig:
    return RenderJobConfig.from_dict(
        {
            "id": "preview",
            "input": {"path": "memory.xyz", "frames": "last"},
            "output": {"path": "/tmp/preview.png"},
            "structure": {"representation": "ball_stick", "draw_bonds": True},
            "style": {"scene_style": "default"},
        }
    )


def _perspective_cfg() -> RenderJobConfig:
    return RenderJobConfig.from_dict(
        {
            "id": "preview",
            "input": {"path": "memory.xyz", "frames": "last"},
            "output": {"path": "/tmp/preview.png"},
            "structure": {"representation": "ball_stick", "draw_bonds": True},
            "camera": {"projection": "PERSPECTIVE"},
            "style": {"scene_style": "default"},
        }
    )


def _structure() -> Structure:
    return Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0)),
            Atom(index=2, atomic_number=1, symbol="H", position=(0.0, 1.0, 0.0)),
            Atom(index=3, atomic_number=6, symbol="C", position=(0.0, 0.0, 1.0)),
        ],
        bonds=[Bond(id=0, a=0, b=1, order=1, distance=1.0)],
    )


def test_axis_overlay_tracks_camera_rotation_without_scene() -> None:
    app = _qt_app()
    canvas = PreviewCanvas()
    origin = QtCore.QPointF(*canvas._axis_overlay.ORIGIN)

    canvas._view.camera.azimuth = 0.0
    canvas._view.camera.elevation = 0.0
    canvas._on_camera_transform()
    first = canvas._axis_overlay._project_axis(origin, (1.0, 0.0, 0.0))

    canvas._view.camera.azimuth = 90.0
    canvas._schedule_camera_sync()
    app.processEvents()
    second = canvas._axis_overlay._project_axis(origin, (1.0, 0.0, 0.0))

    assert (round(first.x(), 1), round(first.y(), 1)) == (108.0, 84.0)
    assert (round(second.x(), 1), round(second.y(), 1)) == (60.0, 92.6)


def test_camera_sync_updates_screen_space_preview_lighting() -> None:
    app = _qt_app()
    canvas = PreviewCanvas()
    calls = 0

    def count_lighting() -> None:
        nonlocal calls
        calls += 1

    canvas._apply_preview_lighting = count_lighting
    canvas._view.camera.azimuth = 90.0
    canvas._schedule_camera_sync()
    app.processEvents()

    assert calls == 1


def test_camera_transform_does_not_rebuild_selection_shell() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.select_atom(1)
    calls = 0

    def count_selection_update() -> None:
        nonlocal calls
        calls += 1

    canvas._update_selection_visuals = count_selection_update
    canvas._view.camera.azimuth += 10.0
    canvas._on_camera_transform()

    assert calls == 0


def test_graphics_info_capture_is_one_shot_for_draw_loop() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    calls = 0

    def count_capture(_event=None) -> None:
        nonlocal calls
        calls += 1

    canvas._capture_graphics_info = count_capture
    canvas._capture_graphics_info_once()
    canvas._capture_graphics_info_once()

    assert calls == 1


def test_preview_canvas_toolbar_view_controls_without_scene() -> None:
    _qt_app()
    canvas = PreviewCanvas()

    canvas.set_axis_view("c")
    assert canvas.model.camera.view == "top"

    scale_before = canvas.model.camera.scale_factor
    canvas.zoom_view(1.2)
    assert canvas.model.camera.scale_factor < scale_before

    center_before = canvas.model.camera.center
    canvas.pan_view(1.0, 0.0)
    assert canvas.model.camera.center != center_before

    azimuth_before = canvas.model.camera.azimuth
    canvas.rotate_view("c", 1)
    assert canvas.model.camera.azimuth != azimuth_before

    canvas.model.camera.elevation = 85.0
    canvas.rotate_view("a", 1, degrees=20.0)
    assert canvas.model.camera.elevation > 90.0


def test_preview_canvas_camera_projection_follows_render_config() -> None:
    _qt_app()
    canvas = PreviewCanvas()

    canvas.set_scene(_structure(), _cfg())
    assert float(canvas._view.camera.fov) == pytest.approx(0.0)

    canvas.set_scene(_structure(), _perspective_cfg())
    assert float(canvas._view.camera.fov) == pytest.approx(45.0)


def test_preview_canvas_requires_instanced_renderer() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    scene = canvas.set_scene(_structure(), _cfg())

    assert canvas.render_mode == "instanced"
    assert scene.report is not None
    assert scene.report["preview_renderer"] == "instanced"
    assert canvas._atom_instance_visual is not None
    assert canvas._atom_instance_visual.instance_positions.shape[0] == len(scene.atoms)


def test_preview_canvas_zoom_preserves_world_geometry() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    atoms_before = tuple(atom.position for atom in canvas.model.scene.atoms)
    bonds_before = tuple((segment.start, segment.end) for bond in canvas.model.scene.bonds for segment in bond.segments)
    bounds_before = (canvas.model.scene.bounds_min, canvas.model.scene.bounds_max)
    scale_before = canvas.model.camera.scale_factor

    canvas.zoom_view(2.0)

    assert canvas.model.camera.scale_factor < scale_before
    assert tuple(atom.position for atom in canvas.model.scene.atoms) == atoms_before
    assert tuple((segment.start, segment.end) for bond in canvas.model.scene.bonds for segment in bond.segments) == bonds_before
    assert (canvas.model.scene.bounds_min, canvas.model.scene.bounds_max) == bounds_before


def test_preview_canvas_mouse_modes_select_and_measure_atoms() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()

    projected = canvas.atom_screen_positions()
    x0, y0, _ = projected[0]
    x1, y1, _ = projected[1]
    selected = canvas.select_atoms_in_rect((min(x0, x1) - 5.0, min(y0, y1) - 5.0), (max(x0, x1) + 5.0, max(y0, y1) + 5.0))

    assert set(selected) >= {0, 1}
    assert canvas.model.selected_atom_indices >= {0, 1}

    message = canvas.measurement_message([0, 1])
    assert message.startswith("Distance 0-1:")


def test_preview_canvas_select_mode_click_selects_single_atom_immediately() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()
    canvas.set_mouse_mode("select")

    x, y, _ = canvas.atom_screen_positions()[1]
    event = _MouseEvent((x, y))
    canvas._mouse_tools.handle_press(event)

    assert event.handled is True
    assert canvas.model.selection is not None
    assert canvas.model.selection.kind == "atom"
    assert canvas.model.selection.index == 1
    assert canvas.model.selected_atom_indices == {1}
    assert any(item["index"] == 1 and item["highlighted"] for item in canvas.model.atom_draw_data())


def test_preview_canvas_middle_drag_pans_in_rotate_mode() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()

    press = _MouseEvent((100.0, 100.0), button=3)
    move = _MouseEvent((130.0, 80.0), button=3)
    release = _MouseEvent((130.0, 80.0), button=3)
    canvas._mouse_tools.handle_press(press)
    canvas._mouse_tools.handle_move(move)
    canvas._mouse_tools.handle_release(release)

    assert press.handled is True
    assert move.handled is True
    assert release.handled is True
    assert canvas.mouse_mode == "rotate"
    assert canvas.model.camera.center != (0.0, 0.0, 0.0)
    assert canvas._mouse_tools._middle_pan_active is False


def test_preview_canvas_qt_mouse_click_selects_atom_without_custom_cursor() -> None:
    app = _qt_app()
    canvas = PreviewCanvas()
    canvas.resize(800, 600)
    canvas.show()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()
    app.processEvents()
    canvas.set_mouse_mode("select")

    x, y, _ = canvas.atom_screen_positions()[1]
    QtTest.QTest.mouseClick(
        canvas._canvas.native,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.QPoint(int(x), int(y)),
    )
    app.processEvents()

    assert getattr(canvas._view.camera, "interactive", None) is False
    assert canvas.model.selection is not None
    assert canvas.model.selection.kind == "atom"
    assert canvas.model.selection.index == 1


def test_preview_canvas_rotate_mode_double_click_selects_atom() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()

    x, y, _ = canvas.atom_screen_positions()[1]
    event = _MouseEvent((x, y))
    canvas._mouse_tools.handle_double_click(event)

    assert event.handled is True
    assert canvas.model.selection is not None
    assert canvas.model.selection.index == 1
    assert canvas.model.selected_atom_indices == {1}


def test_preview_canvas_selected_atom_has_visible_shell_marker() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()

    canvas.select_atom(1)
    payload = canvas._selection_shell_payload()

    assert len(payload["vertices"]) > 0
    assert canvas._selection_shell_visual.visible is True

    canvas.clear_selection()
    assert len(canvas._selection_shell_payload()["vertices"]) == 0
    assert canvas._selection_shell_visual.visible is False


def test_preview_canvas_rotate_mode_two_presses_select_atom() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()

    x, y, _ = canvas.atom_screen_positions()[1]
    first = _MouseEvent((x, y))
    second = _MouseEvent((x + 2.0, y + 1.0))
    canvas._mouse_tools.handle_press(first)
    canvas._mouse_tools.handle_press(second)

    assert first.handled is False
    assert second.handled is True
    assert canvas.model.selection is not None
    assert canvas.model.selection.index == 1
    assert canvas.model.selected_atom_indices == {1}


def test_preview_canvas_qt_rotate_double_click_selects_atom() -> None:
    app = _qt_app()
    canvas = PreviewCanvas()
    canvas.resize(800, 600)
    canvas.show()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()
    app.processEvents()

    x, y, _ = canvas.atom_screen_positions()[1]
    QtTest.QTest.mouseDClick(
        canvas._canvas.native,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.QPoint(int(x), int(y)),
    )
    app.processEvents()

    assert canvas.model.selection is not None
    assert canvas.model.selection.kind == "atom"
    assert canvas.model.selection.index == 1


def test_preview_canvas_ctrl_click_toggles_atom_selection() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()
    canvas.set_mouse_mode("select")

    projected = canvas.atom_screen_positions()
    x0, y0, _ = projected[0]
    x1, y1, _ = projected[1]
    canvas._mouse_tools.handle_press(_MouseEvent((x0, y0)))
    canvas._mouse_tools.handle_press(_MouseEvent((x1, y1), modifiers=("Control",)))

    assert canvas.model.selected_atom_indices == {0, 1}


def test_preview_canvas_selects_bond_when_no_atom_is_hit() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()

    projected = canvas.atom_screen_positions()
    x0, y0, _ = projected[0]
    x1, y1, _ = projected[1]
    selection = canvas.select_object_at_screen(((x0 + x1) * 0.5, (y0 + y1) * 0.5))

    assert selection is not None
    assert selection.kind == "bond"
    assert selection.index == 0
    assert canvas.model.selected_atom_indices == set()
    assert canvas.model.bond_draw_data()[0]["highlighted"] is True


def test_preview_canvas_box_selects_atoms_and_bonds() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()

    projected = canvas.atom_screen_positions()
    x0, y0, _ = projected[0]
    x1, y1, _ = projected[1]
    selected = canvas.select_objects_in_screen_rect(
        (min(x0, x1) - 8.0, min(y0, y1) - 8.0),
        (max(x0, x1) + 8.0, max(y0, y1) + 8.0),
    )

    assert set(selected["atoms"]) >= {0, 1}
    assert selected["bonds"] == (0,)
    assert canvas.model.selected_atom_indices >= {0, 1}
    assert canvas.model.selected_bond_indices == {0}
    assert canvas.model.bond_draw_data()[0]["highlighted"] is True


def test_preview_canvas_box_selects_bond_body_without_atoms() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.set_view_preset("front")
    canvas.model.camera.scale_factor = 4.0
    canvas.model.camera.center = (0.0, 0.0, 0.0)
    canvas._sync_camera()

    projected = canvas.atom_screen_positions()
    x0, y0, _ = projected[0]
    x1, y1, _ = projected[1]
    mx = (x0 + x1) * 0.5
    my = (y0 + y1) * 0.5
    selected = canvas.select_objects_in_screen_rect((mx - 3.0, my - 3.0), (mx + 3.0, my + 3.0))

    assert selected["atoms"] == ()
    assert selected["bonds"] == (0,)
    assert canvas.model.selected_bond_indices == {0}


def test_preview_canvas_delete_selected_objects_removes_atoms_and_bonds() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.model.select_objects({0}, {0})

    deleted = canvas.delete_selected_objects()

    assert deleted == {"atoms": 1, "bonds": 1}
    assert [atom.index for atom in canvas.model.scene.atoms] == [1, 2, 3]
    assert canvas.model.scene.bonds == ()
    assert canvas.model.selected_atom_indices == set()
    assert canvas.model.selected_bond_indices == set()


def test_preview_canvas_select_drag_shows_rubber_band() -> None:
    app = _qt_app()
    canvas = PreviewCanvas()
    canvas.resize(800, 600)
    canvas.show()
    canvas.set_mouse_mode("select")
    app.processEvents()

    canvas._mouse_tools.handle_press(_MouseEvent((10.0, 20.0)))
    canvas._mouse_tools.handle_move(_MouseEvent((80.0, 95.0)))

    assert canvas._mouse_tools._rubber_band is not None
    assert canvas._mouse_tools._rubber_band.geometry().width() > 0
    assert canvas._mouse_tools._rubber_band.geometry().height() > 0

    canvas._mouse_tools.handle_release(_MouseEvent((80.0, 95.0)))
    assert canvas._mouse_tools._rubber_band.isVisible() is False


def test_preview_canvas_updates_selected_atom_properties() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.select_atom(0)

    assert canvas.update_selected_atom_properties(
        {
            "symbol": "N",
            "atomic_number": 7,
            "position": (0.25, 0.5, 0.75),
            "radius": 0.9,
            "representation": "space_filling",
        }
    )

    atom = next(item for item in canvas.model.scene.atoms if item.index == 0)
    assert atom.symbol == "N"
    assert atom.position == pytest.approx((0.25, 0.5, 0.75))
    assert atom.radius == pytest.approx(0.9)
    assert atom.record is not None
    assert atom.record.atomic_number == 7
    assert canvas.model.selected_atom_indices == {0}


def test_preview_canvas_atom_color_updates_connected_bond_color() -> None:
    _qt_app()
    canvas = PreviewCanvas()
    canvas.set_scene(_structure(), _cfg())
    canvas.select_atom(0)

    assert canvas.update_selected_atom_properties({"color": (0.15, 0.25, 0.85, 1.0)})

    bond = canvas.model.scene.bonds[0]
    assert bond.material_left is not None
    assert bond.material_left.color == pytest.approx((0.15, 0.25, 0.85, 1.0))
    assert bond.segments[0].color == pytest.approx((0.15, 0.25, 0.85, 1.0))
