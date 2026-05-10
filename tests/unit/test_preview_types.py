from __future__ import annotations

import pytest

from atomstudio.preview.types import PreviewSelection, PreviewSelectionTarget


def test_preview_selection_supports_old_atom_index_init():
    selection = PreviewSelection(atom_index=4)

    assert selection.kind == "atom"
    assert selection.index == 4
    assert selection.atom_index == 4
    assert selection.bond_index is None


def test_preview_selection_supports_explicit_bond_kind():
    selection = PreviewSelection(kind="bond", index=7)

    assert selection.kind == "bond"
    assert selection.index == 7
    assert selection.atom_index is None
    assert selection.bond_index == 7


def test_preview_selection_target_normalizes_kind():
    target = PreviewSelectionTarget(kind="ATOM", index=3, label="Atom 3")

    assert target.kind == "atom"
    assert target.index == 3
    assert target.label == "Atom 3"
    assert target.metadata == {}


def test_preview_selection_rejects_unknown_kind():
    with pytest.raises(ValueError):
        PreviewSelection(kind="surface", index=1)
