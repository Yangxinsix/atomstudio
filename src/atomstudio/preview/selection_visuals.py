from __future__ import annotations

from typing import Any

import numpy as np

from atomstudio.preview.types import PreviewRenderScene

SELECTION_SHELL_COLOR = (1.0, 0.68, 0.10, 0.34)


def empty_selection_shell_payload() -> dict[str, np.ndarray]:
    return {
        "vertices": np.zeros((0, 3), dtype=float),
        "faces": np.zeros((0, 3), dtype=np.int32),
        "face_colors": np.zeros((0, 4), dtype=float),
    }


def build_selection_shell_payload(
    scene: PreviewRenderScene | None,
    selected_atom_indices: set[int] | frozenset[int] | tuple[int, ...],
    sphere_mesh: Any,
    *,
    active_atom_index: int | None = None,
    radius_scale: float = 1.18,
) -> dict[str, np.ndarray]:
    if scene is None or sphere_mesh is None:
        return empty_selection_shell_payload()
    selected = {int(index) for index in selected_atom_indices}
    if not selected:
        return empty_selection_shell_payload()

    base_vertices = np.asarray(sphere_mesh.get_vertices(), dtype=float)
    base_faces = np.asarray(sphere_mesh.get_faces(), dtype=np.int32)
    vertices_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    face_colors: list[np.ndarray] = []
    vertex_offset = 0

    for atom in scene.atoms:
        atom_index = int(atom.index)
        if atom_index not in selected:
            continue
        radius = max(1e-4, float(atom.radius) * float(radius_scale))
        vertices = base_vertices * radius + np.asarray(atom.position, dtype=float)
        faces = base_faces + vertex_offset
        vertices_list.append(vertices)
        faces_list.append(faces)
        face_colors.append(np.tile(np.asarray(SELECTION_SHELL_COLOR, dtype=float), (faces.shape[0], 1)))
        vertex_offset += base_vertices.shape[0]

    if not vertices_list:
        return empty_selection_shell_payload()
    return {
        "vertices": np.concatenate(vertices_list, axis=0),
        "faces": np.concatenate(faces_list, axis=0),
        "face_colors": np.concatenate(face_colors, axis=0),
    }
