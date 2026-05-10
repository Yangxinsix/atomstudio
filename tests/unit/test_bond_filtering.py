from math import sqrt

from atomstudio.config import BoundaryConfig, BondingConfig
from atomstudio.structure.boundary import build_boundary_expanded_structure, normalize_window
from atomstudio.io.ase_loader import load_structure
from atomstudio.structure.atom import Atom
from atomstudio.structure.bonding import BondEngine, covalent_radius
from atomstudio.structure.cell import Cell
from atomstudio.structure.structure import Structure


def _max_bond_distance(bonds, structure):
    dmax = 0.0
    for bond in bonds:
        i, j = bond.a, bond.b
        p1 = structure.positions[i]
        p2 = structure.positions[j]
        d = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
        dmax = max(dmax, d)
    return dmax


def test_no_long_periodic_bonds_for_mgo_xyz():
    structure = load_structure("mgo111_adsorbates.xyz", frame="last")
    bonds = BondEngine().compute(structure, BondingConfig(include_periodic_images=False))
    assert _max_bond_distance(bonds, structure) < 4.0


def test_covalent_radius_fallback_covers_common_perovskite_elements():
    assert covalent_radius("Ti") > 1.2
    assert covalent_radius("Sr") > 1.5


def test_periodic_minimum_image_is_not_used_without_padding():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.1, 0.5, 0.5)),
            Atom(index=1, atomic_number=8, symbol="O", position=(0.9, 0.5, 0.5)),
        ],
        cell=Cell(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            pbc=(True, True, True),
        ),
    )
    bonds = BondEngine().compute(
        structure,
        BondingConfig(
            include_periodic_images=True,
            min_distance=0.0,
            pair_cutoffs={"O-O": 0.3},
        ),
    )
    assert bonds == []


def test_boundary_padding_creates_short_bondable_images():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.1, 0.5, 0.5)),
            Atom(index=1, atomic_number=8, symbol="O", position=(0.9, 0.5, 0.5)),
        ],
        cell=Cell(
            vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            pbc=(True, True, True),
        ),
    )
    expanded = build_boundary_expanded_structure(
        structure,
        window=normalize_window(BoundaryConfig(enabled=True, window_frac=[[-0.2, 1.2], [0.0, 1.0], [0.0, 1.0]])),
    )
    bonds = BondEngine().compute(
        expanded,
        BondingConfig(
            include_periodic_images=False,
            min_distance=0.0,
            pair_cutoffs={"O-O": 0.3},
        ),
    )
    assert len(bonds) > 0


def test_pair_cutoffs_filter_unwanted_bonds():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=22, symbol="Ti", position=(1.5, 0.0, 0.0)),
            Atom(index=2, atomic_number=38, symbol="Sr", position=(2.5, 0.0, 0.0)),
        ],
        cell=Cell(),
    )

    filtered_bonds = BondEngine().compute(
        structure,
        BondingConfig(
            include_periodic_images=False,
            min_distance=0.2,
            pair_cutoffs={"O-Ti": 2.4, "O-Sr": 3.2},
        ),
    )
    assert {tuple(sorted((structure.symbols[b.a], structure.symbols[b.b]))) for b in filtered_bonds} == {
        ("O", "Sr"),
        ("O", "Ti"),
    }


def test_pair_cutoff_keys_are_normalized_and_unspecified_pairs_do_not_bond():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=22, symbol="Ti", position=(2.0, 0.0, 0.0)),
            Atom(index=2, atomic_number=38, symbol="Sr", position=(2.2, 0.0, 0.0)),
        ],
        cell=Cell(),
    )

    bonds = BondEngine().compute(
        structure,
        BondingConfig(
            include_periodic_images=False,
            min_distance=0.2,
            pair_cutoffs={"Ti-O": 2.3},
        ),
    )
    assert len(bonds) == 1
    assert tuple(sorted((structure.symbols[bonds[0].a], structure.symbols[bonds[0].b]))) == ("O", "Ti")
