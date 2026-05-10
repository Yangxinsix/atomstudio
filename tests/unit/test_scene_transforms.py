from __future__ import annotations

import numpy as np
import pytest

from atomstudio.config import RenderJobConfig
from atomstudio.scene.transforms import apply_boundary_expansion, apply_model_rotation, compute_bounds, resolve_model_rotation_matrix
from atomstudio.structure.atom import Atom
from atomstudio.structure.cell import Cell
from atomstudio.structure.structure import Structure


def _cfg(**overrides) -> RenderJobConfig:
    payload = {
        "id": "scene-transforms",
        "input": {"path": "tests/data/water.xyz", "frames": "last"},
        "output": {"path": "/tmp/x.png"},
    }
    payload.update(overrides)
    return RenderJobConfig.from_dict(payload)


def test_apply_boundary_expansion_reuses_structure_boundary_semantics():
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=14, symbol="Si", position=(0.5, 0.5, 0.5))],
        cell=Cell(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            pbc=(True, True, True),
        ),
    )
    cfg = _cfg(structure={"boundary": {"enabled": True, "window_frac": [[-0.5, 1.5], [0.0, 1.0], [0.0, 1.0]]}})

    expanded = apply_boundary_expansion(structure, cfg)

    offsets = sorted(tuple(atom.metadata["boundary_offset"]) for atom in expanded.atoms)
    assert offsets == [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]


def test_apply_model_rotation_rotates_atoms_and_cell():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(1.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=8, symbol="O", position=(-1.0, 0.0, 0.0)),
        ],
        cell=Cell(vectors=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], pbc=(True, True, True)),
    )

    rotated = apply_model_rotation(structure, model_rotation="-90x,-90y,0z", model_view="top")
    matrix = resolve_model_rotation_matrix(model_rotation="-90x,-90y,0z", model_view="top")

    assert matrix is not None
    expected_position = matrix @ np.asarray((1.0, 0.0, 0.0), dtype=float)
    assert np.allclose(np.asarray(rotated.atoms[0].position), expected_position)
    assert rotated.cell.vectors[0] != pytest.approx(structure.cell.vectors[0])


def test_compute_bounds_aggregates_scene_points():
    bounds = compute_bounds([(0.0, 0.0, 0.0), (2.0, -1.0, 1.0)])

    assert bounds.minimum == pytest.approx((0.0, -1.0, 0.0))
    assert bounds.maximum == pytest.approx((2.0, 0.0, 1.0))
    assert bounds.center == pytest.approx((1.0, -0.5, 0.5))
    assert bounds.radius > 0.0
