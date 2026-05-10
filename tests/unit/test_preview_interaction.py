from __future__ import annotations

import pytest

from atomstudio.preview.interaction import AtomHit, HitTestCache, MeasurementController, SelectionController, normalize_mouse_mode


def test_selection_controller_tracks_single_multi_and_toggle() -> None:
    controller = SelectionController()

    assert controller.select_single_atom(2).index == 2
    assert controller.selected_atoms == {2}

    controller.select_atoms([1, 3], append=True)
    assert controller.selected_atoms == {1, 2, 3}
    assert controller.selected_ordered_atoms == [2, 1, 3]

    controller.toggle_atom(2)
    assert controller.selected_atoms == {1, 3}
    assert controller.selected_ordered_atoms == [1, 3]


def test_hit_test_cache_picks_nearest_visible_atom_and_rect_intersection() -> None:
    cache = HitTestCache(
        atoms=(
            # Same screen position; lower signed view depth is closer to the camera.
            AtomHit(index=1, x=50, y=50, depth=5, radius_px=10),
            AtomHit(index=2, x=50, y=50, depth=-1, radius_px=10),
            AtomHit(index=3, x=80, y=80, depth=0, radius_px=12),
        )
    )

    assert cache.pick_atom((52, 52)).index == 2
    assert set(cache.atoms_in_rect((70, 70), (72, 72))) == {3}


def test_measurement_controller_messages() -> None:
    controller = MeasurementController()

    assert normalize_mouse_mode("bad") == "rotate"
    complete, atoms = controller.add_atom("measure_distance", 0)
    assert complete is False
    complete, atoms = controller.add_atom("measure_distance", 1)
    assert complete is True
    assert atoms == [0, 1]

    with pytest.raises(KeyError):
        controller.required_count("rotate")
