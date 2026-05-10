from __future__ import annotations

from atomstudio.config import BoundaryConfig
from atomstudio.structure.atom import Atom
from atomstudio.structure.boundary import build_boundary_expanded_structure, enumerate_offsets, normalize_window
from atomstudio.structure.cell import Cell
from atomstudio.structure.structure import Structure


def test_boundary_window_expands_atoms_for_fractional_supercell():
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.5, 0.5, 0.5))],
        cell=Cell(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            pbc=(True, True, True),
        ),
    )
    boundary = BoundaryConfig(enabled=True, window_frac=[[-0.5, 1.5], [0.0, 1.0], [0.0, 1.0]], eps=1e-6)
    expanded = build_boundary_expanded_structure(
        structure,
        window=normalize_window(boundary),
        eps=boundary.eps,
    )

    assert len(expanded.atoms) == 3
    offsets = sorted(tuple(atom.metadata["boundary_offset"]) for atom in expanded.atoms)
    assert offsets == [(-1, 0, 0), (0, 0, 0), (1, 0, 0)]
    assert all(int(atom.metadata["origin_index"]) == 0 for atom in expanded.atoms)


def test_non_periodic_axis_does_not_expand_offsets():
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.5, 0.5, 0.5))],
        cell=Cell(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            pbc=(False, True, True),
        ),
    )
    boundary = BoundaryConfig(enabled=True, window_frac=[[-0.5, 1.5], [0.0, 1.0], [0.0, 1.0]], eps=1e-6)
    expanded = build_boundary_expanded_structure(
        structure,
        window=normalize_window(boundary),
        eps=boundary.eps,
    )
    assert len(expanded.atoms) == 1
    assert tuple(expanded.atoms[0].metadata["boundary_offset"]) == (0, 0, 0)


def test_enumerate_offsets_respects_non_periodic_axes():
    window = ((-0.5, 1.5), (0.0, 1.0), (0.0, 1.0))
    offsets = enumerate_offsets(window, (False, True, True))
    assert offsets
    assert all(offset[0] == 0 for offset in offsets)
