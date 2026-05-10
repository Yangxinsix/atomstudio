from __future__ import annotations

import pytest

from atomstudio.config import RenderJobConfig
from atomstudio.scene import SceneBuilder, build_render_scene
from atomstudio.structure import Structure
from atomstudio.structure.atom import Atom
from atomstudio.structure.cell import Cell
from atomstudio.structure.polyhedron import Polyhedron


def _cfg(**overrides) -> RenderJobConfig:
    payload = {
        "id": "scene",
        "input": {"path": "tests/data/water.xyz", "frames": "last"},
        "output": {"path": "/tmp/x.png"},
        "style": {"scene_style": "default"},
    }
    payload.update(overrides)
    return RenderJobConfig.from_dict(payload)


def test_scene_builder_builds_water_scene_with_shared_semantics():
    structure = Structure.from_dict(
        {
            "atoms": [
                {"index": 0, "atomic_number": 8, "symbol": "O", "position": [0.0, 0.0, 0.0]},
                {"index": 1, "atomic_number": 1, "symbol": "H", "position": [0.9572, 0.0, 0.0]},
                {"index": 2, "atomic_number": 1, "symbol": "H", "position": [-0.239987, 0.927297, 0.0]},
            ],
            "frame_index": 0,
        }
    )

    scene = build_render_scene(structure, _cfg())

    assert scene.atom_count == 3
    assert scene.bond_count == 2
    assert scene.visible_bond_count == 2
    assert scene.polyhedra_count == 0
    assert scene.cell_edge_count == 0
    assert scene.atoms[0].material.color == pytest.approx((1.0, 0.051, 0.051, 1.0))
    assert scene.bonds[0].material_left.color == pytest.approx(scene.atoms[0].material.color)
    assert scene.bonds[0].material_right.color == pytest.approx(scene.atoms[1].material.color)
    assert scene.bonds[0].segments and len(scene.bonds[0].segments) == 2
    assert scene.camera.position != scene.camera.target
    assert len(scene.lights) >= 1


def test_scene_builder_keeps_bonds_hidden_for_space_filling():
    structure = Structure.from_dict(
        {
            "atoms": [
                {"index": 0, "atomic_number": 8, "symbol": "O", "position": [0.0, 0.0, 0.0]},
                {"index": 1, "atomic_number": 1, "symbol": "H", "position": [0.9572, 0.0, 0.0]},
                {"index": 2, "atomic_number": 1, "symbol": "H", "position": [-0.239987, 0.927297, 0.0]},
            ],
            "frame_index": 0,
        }
    )

    scene = SceneBuilder(_cfg(structure={"representation": "space_filling"})).build(structure)

    assert scene.draw_bonds is False
    assert scene.bond_count == 0
    assert all(atom.representation == "space_filling" for atom in scene.atoms)


def test_scene_builder_emits_cell_edges_and_polyhedra():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=14, symbol="Si", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=8, symbol="O", position=(1.0, 0.0, 0.0)),
            Atom(index=2, atomic_number=8, symbol="O", position=(0.0, 1.0, 0.0)),
            Atom(index=3, atomic_number=8, symbol="O", position=(0.0, 0.0, 1.0)),
            Atom(index=4, atomic_number=8, symbol="O", position=(1.0, 1.0, 1.0)),
        ],
        polyhedra=[
            Polyhedron(
                id=0,
                center=0,
                center_symbol="Si",
                vertex_positions=[
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (1.0, 1.0, 1.0),
                ],
            )
        ],
        cell=Cell(
            vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            pbc=(True, True, True),
        ),
    )

    scene = build_render_scene(
        structure,
        _cfg(structure={"draw_cell": True, "polyhedra": {"enabled": True, "rules": []}}),
    )

    assert scene.cell_edge_count == 12
    assert scene.polyhedra_count == 1
    assert len(scene.polyhedra[0].faces) >= 4
