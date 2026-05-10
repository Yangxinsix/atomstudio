from __future__ import annotations

from atomstudio.config import RenderJobConfig
from atomstudio.preview import PreviewSettings, build_preview_scene
from atomstudio.structure import Structure
from atomstudio.structure.atom import Atom
from atomstudio.structure.cell import Cell
from atomstudio.structure.polyhedron import Polyhedron


def _cfg(**overrides) -> RenderJobConfig:
    payload = {
        "id": "preview",
        "input": {"path": "tests/data/water.xyz", "frames": "last"},
        "output": {"path": "/tmp/x.png"},
        "style": {"scene_style": "default"},
    }
    for key, value in overrides.items():
        payload[key] = value
    return RenderJobConfig.from_dict(payload)


def test_preview_builder_computes_default_water_buffers():
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

    scene = build_preview_scene(structure, _cfg(), PreviewSettings())

    assert scene.atoms.count == 3
    assert scene.bonds.segment_count == 2
    assert scene.cell.empty
    assert scene.polyhedra.empty
    assert scene.extent > 0.0
    assert len(scene.atom_records) == 3
    assert len(scene.bond_records) == 2
    assert len(scene.selection_targets) == 5
    assert scene.atom_records[0].material.pipeline == "principled"
    assert scene.selection_targets[0].kind == "atom"
    assert scene.selection_targets[-1].kind == "bond"


def test_preview_builder_respects_space_filling_bond_default():
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

    scene = build_preview_scene(
        structure,
        _cfg(structure={"representation": "space_filling"}),
        PreviewSettings(),
    )

    assert scene.atoms.count == 3
    assert scene.bonds.empty
    assert len(scene.atom_records) == 3
    assert len(scene.bond_records) == 0
    assert all(target.kind == "atom" for target in scene.selection_targets)


def test_preview_builder_emits_cell_lines_when_requested():
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=14, symbol="Si", position=(0.0, 0.0, 0.0))],
        cell=Cell(
            vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            pbc=(True, True, True),
        ),
    )

    scene = build_preview_scene(
        structure,
        _cfg(structure={"draw_cell": True}),
        PreviewSettings(),
    )

    assert scene.cell.segment_count == 12


def test_preview_builder_triangulates_polyhedra():
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
        cell=Cell(),
    )

    scene = build_preview_scene(
        structure,
        _cfg(structure={"polyhedra": {"enabled": True, "rules": []}}),
        PreviewSettings(),
    )

    assert scene.polyhedra.vertices.shape[0] == 4
    assert scene.polyhedra.face_count >= 4


def test_preview_builder_emits_split_bond_material_records():
    structure = Structure.from_dict(
        {
            "atoms": [
                {"index": 0, "atomic_number": 8, "symbol": "O", "position": [0.0, 0.0, 0.0]},
                {"index": 1, "atomic_number": 1, "symbol": "H", "position": [0.9572, 0.0, 0.0]},
            ],
            "bonds": [
                {
                    "id": 7,
                    "a": 0,
                    "b": 1,
                    "order": 1,
                    "distance": 0.9572,
                }
            ],
            "frame_index": 0,
        }
    )

    scene = build_preview_scene(structure, _cfg(), PreviewSettings())

    assert len(scene.bond_records) == 1
    bond = scene.bond_records[0]
    assert bond.id == 7
    assert bond.material_uniform.color == (0.25, 0.25, 0.25, 1.0)
    assert bond.material_left.color == tuple(scene.atom_records[0].material.color)
    assert bond.material_right.color == tuple(scene.atom_records[1].material.color)
    assert bond.metadata["split_colors"] is True
