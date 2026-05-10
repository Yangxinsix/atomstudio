from __future__ import annotations

from atomstudio.structure.selectors import AtomSelector, BondSelector, PolyhedraSelector


def test_atom_selector_from_dict_parses_ranges_and_tags():
    selector = AtomSelector.from_dict(
        {
            "symbol": "O",
            "symbols": ["O", "N"],
            "indices": [1, "3"],
            "index_range": [0, 5],
            "z_range": [1.5, 3.5],
            "tags": ["surface", "adsorbate"],
        }
    )

    assert selector.symbol == "O"
    assert selector.symbols == ["O", "N"]
    assert selector.indices == [1, 3]
    assert selector.index_range == (0, 5)
    assert selector.z_range == (1.5, 3.5)
    assert selector.tags == ["surface", "adsorbate"]


def test_bond_selector_from_dict_parses_pairs_and_ranges():
    selector = BondSelector.from_dict(
        {
            "pair": "O-Ti",
            "pairs": ["O-Ti", "Ti-O"],
            "bond_order": 2,
            "distance_range": [1.8, 2.2],
            "index_pairs": [[3, 7], ["8", "9"]],
            "index_range": [0, 10],
        }
    )

    assert selector.pair == "O-Ti"
    assert selector.pairs == ["O-Ti", "Ti-O"]
    assert selector.bond_order == 2
    assert selector.distance_range == (1.8, 2.2)
    assert selector.index_pairs == [(3, 7), (8, 9)]
    assert selector.index_range == (0, 10)


def test_polyhedra_selector_from_dict_parses_fields():
    selector = PolyhedraSelector.from_dict(
        {
            "center_symbol": "Ti",
            "center_symbols": ["Ti", "Zr"],
            "center_indices": [0, "2"],
            "index_range": [1, 4],
            "neighbor_count_range": [5, 6],
        }
    )

    assert selector.center_symbol == "Ti"
    assert selector.center_symbols == ["Ti", "Zr"]
    assert selector.center_indices == [0, 2]
    assert selector.index_range == (1, 4)
    assert selector.neighbor_count_range == (5, 6)
