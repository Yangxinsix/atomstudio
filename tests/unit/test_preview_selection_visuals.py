from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from atomstudio.preview.selection_visuals import (
    SELECTION_SHELL_COLOR,
    build_selection_shell_payload,
    empty_selection_shell_payload,
)


class _SphereMesh:
    def get_vertices(self) -> np.ndarray:
        return np.asarray(
            [
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (-1.0, 0.0, 0.0),
            ],
            dtype=float,
        )

    def get_faces(self) -> np.ndarray:
        return np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int32)


def test_empty_selection_shell_payload_has_mesh_shape() -> None:
    payload = empty_selection_shell_payload()

    assert payload["vertices"].shape == (0, 3)
    assert payload["faces"].shape == (0, 3)
    assert payload["face_colors"].shape == (0, 4)


def test_selection_shell_payload_uses_same_marker_for_multi_selection() -> None:
    scene = SimpleNamespace(
        atoms=(
            SimpleNamespace(index=0, position=(0.0, 0.0, 0.0), radius=1.0),
            SimpleNamespace(index=1, position=(2.0, 0.0, 0.0), radius=0.5),
        )
    )

    payload = build_selection_shell_payload(scene, {0, 1}, _SphereMesh(), active_atom_index=1)

    assert payload["vertices"].shape == (8, 3)
    assert payload["faces"].shape == (4, 3)
    assert np.allclose(payload["face_colors"][0], SELECTION_SHELL_COLOR)
    assert np.allclose(payload["face_colors"][-1], SELECTION_SHELL_COLOR)
