from __future__ import annotations

import math

import pytest

from atomstudio.app.preview_canvas import PreviewCanvasModel, PreviewSelection, PreviewSettings, build_preview_scene
from atomstudio.config import RenderJobConfig
from atomstudio.preview.picking import rotation_basis
from atomstudio.structure.atom import Atom
from atomstudio.structure.bond import Bond
from atomstudio.structure.cell import Cell
from atomstudio.structure.polyhedron import Polyhedron
from atomstudio.structure.structure import Structure


def _cfg(**structure_overrides) -> RenderJobConfig:
    payload = {
        "id": "preview",
        "input": {"path": "memory.xyz", "frames": "last"},
        "output": {"path": "/tmp/preview.png"},
        "structure": {
            "representation": "ball_stick",
            "draw_bonds": True,
            "draw_cell": True,
            "cell_style": {"show": True, "radius": 0.05},
            **structure_overrides,
        },
        "style": {"scene_style": "default"},
    }
    return RenderJobConfig.from_dict(payload)


def _structure() -> Structure:
    return Structure(
        atoms=[
            Atom(index=0, atomic_number=6, symbol="C", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=8, symbol="O", position=(1.3, 0.0, 0.0)),
            Atom(index=2, atomic_number=1, symbol="H", position=(0.0, 1.4, 0.0)),
            Atom(index=3, atomic_number=1, symbol="H", position=(0.0, 0.0, 1.4)),
        ],
        bonds=[Bond(id=0, a=0, b=1, order=2, distance=1.3)],
        polyhedra=[
            Polyhedron(
                id=0,
                center=0,
                center_symbol="C",
                vertex_positions=[
                    (-0.3, -0.3, -0.3),
                    (0.9, -0.3, -0.3),
                    (-0.3, 0.9, -0.3),
                    (-0.3, -0.3, 0.9),
                ],
                show_edges=True,
            )
        ],
        cell=Cell(vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], pbc=(True, True, True), show=True),
    )


def _depth_overlap_structure() -> Structure:
    return Structure(
        atoms=[
            Atom(index=0, atomic_number=6, symbol="C", position=(0.0, -2.0, 0.0)),
            Atom(index=1, atomic_number=8, symbol="O", position=(0.0, 2.0, 0.0)),
        ],
        bonds=[],
        cell=Cell(),
    )


def test_build_preview_scene_creates_all_geometry_layers():
    scene = build_preview_scene(_structure(), _cfg())

    assert len(scene.atoms) == 4
    assert len(scene.bonds) == 1
    assert len(scene.bonds[0].segments) == 2
    assert len(scene.cell_edges) == 12
    assert len(scene.polyhedra) == 1
    assert len(scene.polyhedra[0].faces) >= 4
    assert scene.radius >= 1.0
    assert scene.render_mode in {"principled-preview", "handdrawn-preview"}
    assert any(item["kind"] == "atom" for item in scene.selection_targets)
    assert any(item["kind"] == "bond" for item in scene.selection_targets)
    assert scene.atoms[0].material.to_dict()["pipeline"] in {"principled-preview", "handdrawn-preview"}


def test_preview_model_fit_and_views():
    model = PreviewCanvasModel(PreviewSettings())
    model.set_scene(_structure(), _cfg())

    camera = model.fit_to_structure(padding=0.0)
    assert camera.center == pytest.approx(model.scene.center)
    assert camera.scale_factor >= 1.0

    model.set_view_preset("top")
    assert model.camera.view == "top"
    assert model.camera.elevation == pytest.approx(90.0)

    model.set_view_preset("front")
    assert model.camera.view == "front"
    assert model.camera.up == pytest.approx((0.0, 0.0, 1.0))

    model.set_view_preset("side")
    assert model.camera.view == "side"
    assert model.camera.right == pytest.approx((0.0, 1.0, 0.0))

    model.set_view_preset("orbit")
    assert model.camera.view == "orbit"
    assert model.camera.azimuth == pytest.approx(45.0)


def test_rotation_basis_matches_preview_view_presets():
    right, up, forward = rotation_basis(0.0, 0.0)
    assert right == pytest.approx((1.0, 0.0, 0.0))
    assert up == pytest.approx((0.0, 0.0, 1.0))
    assert forward == pytest.approx((0.0, 1.0, 0.0))

    right, up, forward = rotation_basis(90.0, 0.0)
    assert right == pytest.approx((0.0, 1.0, 0.0))
    assert up == pytest.approx((0.0, 0.0, 1.0))
    assert forward == pytest.approx((-1.0, 0.0, 0.0))

    _, up, forward = rotation_basis(0.0, 90.0)
    assert up == pytest.approx((0.0, 1.0, 0.0))
    assert forward == pytest.approx((0.0, 0.0, -1.0))


def test_preview_model_pick_and_highlight_hooks():
    model = PreviewCanvasModel(PreviewSettings(picking_radius_px=10.0))
    model.set_scene(_structure(), _cfg())
    model.set_view_preset("front")
    model.camera.scale_factor = 4.0
    model.camera.center = (0.0, 0.0, 0.0)

    picked: list[PreviewSelection | None] = []
    model.selection_changed.connect(lambda selection: picked.append(selection))

    projected = model.project_atom_positions((500, 500))
    target_index = 1
    x, y, _z = projected[target_index]
    selection = model.pick_atom_at((x, y), (500, 500))
    assert selection is not None
    assert selection.kind == "atom"
    assert selection.index == target_index
    assert picked[-1] == selection

    draw = model.atom_draw_data()
    highlighted = [item for item in draw if item["highlighted"]]
    assert highlighted and highlighted[0]["size"] > min(item["size"] for item in draw)


def test_preview_model_pick_prefers_front_atom_when_screen_positions_overlap():
    model = PreviewCanvasModel(PreviewSettings(picking_radius_px=10.0))
    model.set_scene(_depth_overlap_structure(), _cfg(draw_bonds=False, draw_cell=False))
    model.set_view_preset("front")
    model.camera.scale_factor = 4.0
    model.camera.center = (0.0, 0.0, 0.0)

    projected = model.project_atom_positions((500, 500))
    assert projected[0][:2] == pytest.approx(projected[1][:2])

    x, y, _ = projected[0]
    selection = model.pick_atom_at((x, y), (500, 500))

    assert selection is not None
    assert selection.kind == "atom"
    assert selection.index == 0


def test_preview_model_bond_selection_and_atom_priority():
    model = PreviewCanvasModel(PreviewSettings(picking_radius_px=10.0))
    model.set_scene(_structure(), _cfg())
    model.set_view_preset("front")
    model.camera.scale_factor = 4.0
    model.camera.center = (0.0, 0.0, 0.0)

    projected = model.project_atom_positions((500, 500))
    atom_x, atom_y, _ = projected[0]
    atom_hit = model.pick_selection_at((atom_x, atom_y), (500, 500))
    assert atom_hit is not None
    assert atom_hit.kind == "atom"
    assert atom_hit.index == 0

    bond_x = projected[0][0] * 0.25 + projected[1][0] * 0.75
    bond_y = projected[0][1] * 0.25 + projected[1][1] * 0.75
    bond_hit = model.pick_selection_at((bond_x, bond_y), (500, 500))
    assert bond_hit is not None
    assert bond_hit.kind == "bond"
    assert bond_hit.index == 0
    assert model.selected_object is not None
    assert model.selected_object["kind"] == "bond"
    assert model.selected_payload is not None
    assert "pipeline" in model.selected_payload


def test_preview_model_bond_does_not_highlight_when_endpoint_selected():
    model = PreviewCanvasModel(PreviewSettings())
    model.set_scene(_structure(), _cfg())
    model.select_atom(0)
    bond_data = model.bond_draw_data()
    assert bond_data[0]["highlighted"] is False


def test_preview_model_bond_highlights_when_bond_selected():
    model = PreviewCanvasModel(PreviewSettings())
    model.set_scene(_structure(), _cfg())
    model.select_bond(0)
    bond_data = model.bond_draw_data()
    assert bond_data[0]["highlighted"] is True
    assert all(segment["width"] >= 1.0 for segment in bond_data[0]["segments"])
