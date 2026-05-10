from __future__ import annotations

from atomstudio.config import BondingConfig, BoundaryConfig, PolyhedraConfig, PolyhedraRuleConfig
from atomstudio.structure.atom import Atom
from atomstudio.structure.boundary import build_boundary_expanded_structure, normalize_window
from atomstudio.structure.cell import Cell
from atomstudio.structure.polyhedra_engine import PolyhedraEngine
from atomstudio.structure.structure import Structure


def test_polyhedra_engine_builds_center_neighbor_rule():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=6, symbol="C", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(0.7, 0.7, 0.7)),
            Atom(index=2, atomic_number=1, symbol="H", position=(-0.7, -0.7, 0.7)),
            Atom(index=3, atomic_number=1, symbol="H", position=(-0.7, 0.7, -0.7)),
            Atom(index=4, atomic_number=1, symbol="H", position=(0.7, -0.7, -0.7)),
        ],
        cell=Cell(),
    )
    pcfg = PolyhedraConfig(
        enabled=True,
        rules=[
            PolyhedraRuleConfig(
                center_symbols=["C"],
                neighbor_symbols=["H"],
                min_neighbors=4,
                max_neighbors=4,
            )
        ],
    )
    out = PolyhedraEngine().compute(structure, pcfg, BondingConfig(cutoff_scale=1.3))
    assert len(out) == 1
    assert out[0].center == 0
    assert out[0].center_symbol == "C"
    assert len(out[0].vertex_positions) == 4
    assert sorted(out[0].neighbor_indices) == [1, 2, 3, 4]


def test_polyhedra_engine_uses_direct_positions_without_pbc_offsets():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=6, symbol="C", position=(0.1, 0.1, 0.1)),
            Atom(index=1, atomic_number=1, symbol="H", position=(0.9, 0.1, 0.1)),
            Atom(index=2, atomic_number=1, symbol="H", position=(0.1, 0.9, 0.1)),
            Atom(index=3, atomic_number=1, symbol="H", position=(0.1, 0.1, 0.9)),
            Atom(index=4, atomic_number=1, symbol="H", position=(0.9, 0.9, 0.9)),
        ],
        cell=Cell(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            pbc=(True, True, True),
        ),
    )
    pcfg = PolyhedraConfig(
        enabled=True,
        include_periodic_images=True,
        rules=[PolyhedraRuleConfig(center_symbols=["C"], neighbor_symbols=["H"], min_neighbors=4, max_neighbors=4)],
    )
    out = PolyhedraEngine().compute(structure, pcfg, BondingConfig(cutoff_scale=1.2))
    assert len(out) == 1
    poly = out[0]
    assert all(offset == (0, 0, 0) for offset in poly.neighbor_offsets)
    assert all(
        poly.vertex_positions[i] == structure.positions[poly.neighbor_indices[i]]
        for i in range(len(poly.vertex_positions))
    )


def test_polyhedra_engine_respects_max_neighbors_and_max_distance():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=6, symbol="C", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0)),
            Atom(index=2, atomic_number=1, symbol="H", position=(-1.0, 0.0, 0.0)),
            Atom(index=3, atomic_number=1, symbol="H", position=(0.0, 1.0, 0.0)),
            Atom(index=4, atomic_number=1, symbol="H", position=(0.0, -1.0, 0.0)),
            Atom(index=5, atomic_number=1, symbol="H", position=(0.0, 0.0, 1.0)),
            Atom(index=6, atomic_number=1, symbol="H", position=(0.0, 0.0, -1.0)),
        ],
        cell=Cell(),
    )
    bounded = PolyhedraConfig(
        enabled=True,
        rules=[PolyhedraRuleConfig(center_symbols=["C"], neighbor_symbols=["H"], min_neighbors=4, max_neighbors=4)],
    )
    bounded_out = PolyhedraEngine().compute(structure, bounded, BondingConfig(cutoff_scale=1.3))
    assert len(bounded_out) == 1
    assert len(bounded_out[0].neighbor_indices) == 4

    too_short = PolyhedraConfig(
        enabled=True,
        rules=[
            PolyhedraRuleConfig(
                center_symbols=["C"],
                neighbor_symbols=["H"],
                min_neighbors=4,
                max_neighbors=6,
                max_distance=0.5,
            )
        ],
    )
    short_out = PolyhedraEngine().compute(structure, too_short, BondingConfig(cutoff_scale=1.3))
    assert short_out == []


def test_polyhedra_computes_on_boundary_expanded_structure():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=6, symbol="C", position=(0.95, 0.95, 0.95)),
            Atom(index=1, atomic_number=1, symbol="H", position=(0.05, 0.95, 0.95)),
            Atom(index=2, atomic_number=1, symbol="H", position=(0.95, 0.05, 0.95)),
            Atom(index=3, atomic_number=1, symbol="H", position=(0.95, 0.95, 0.05)),
            Atom(index=4, atomic_number=1, symbol="H", position=(0.05, 0.05, 0.05)),
        ],
        cell=Cell(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            pbc=(True, True, True),
        ),
    )
    pcfg = PolyhedraConfig(
        enabled=True,
        include_periodic_images=False,
        rules=[
            PolyhedraRuleConfig(
                center_symbols=["C"],
                neighbor_symbols=["H"],
                min_neighbors=4,
                max_neighbors=4,
                max_distance=0.3,
            )
        ],
    )
    without_expansion = PolyhedraEngine().compute(
        structure,
        pcfg,
        BondingConfig(cutoff_scale=1.0, include_periodic_images=False),
    )
    assert without_expansion == []

    boundary = BoundaryConfig(enabled=True, window_frac=[[-0.2, 1.2], [-0.2, 1.2], [-0.2, 1.2]], eps=1e-6)
    expanded = build_boundary_expanded_structure(
        structure,
        window=normalize_window(boundary),
        eps=boundary.eps,
    )
    with_expansion = PolyhedraEngine().compute(
        expanded,
        pcfg,
        BondingConfig(cutoff_scale=1.0, include_periodic_images=False),
    )
    assert len(with_expansion) > 0
