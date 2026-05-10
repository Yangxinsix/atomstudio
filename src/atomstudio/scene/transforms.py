from __future__ import annotations

from typing import Iterable

import numpy as np

from atomstudio.config import RenderJobConfig
from atomstudio.scene.camera_builder import _rotate_axes, camera_basis_matrix, resolve_camera_basis
from atomstudio.scene.model import SceneBounds
from atomstudio.structure.boundary import build_boundary_expanded_structure, normalize_window
from atomstudio.structure.structure import Structure


def apply_boundary_expansion(structure: Structure, cfg: RenderJobConfig) -> Structure:
    boundary_cfg = cfg.structure.boundary
    if not bool(boundary_cfg.enabled):
        return structure
    return build_boundary_expanded_structure(
        structure,
        window=normalize_window(boundary_cfg),
        eps=float(boundary_cfg.eps),
    )


def apply_model_rotation(
    structure: Structure,
    *,
    model_rotation: str | None,
    model_view: str,
) -> Structure:
    rot3 = resolve_model_rotation_matrix(model_rotation=model_rotation, model_view=model_view)
    if rot3 is None:
        return structure
    rotated = Structure.from_dict(structure.to_dict())
    if not rotated.atoms:
        return rotated
    center = rotation_center(rotated)
    for atom in rotated.atoms:
        atom.position = rotate_point(atom.position, center=center, rot3=rot3)
    rotated.cell.vectors = [list(rotate_vector(row, rot3=rot3)) for row in rotated.cell.vectors]
    for poly in rotated.polyhedra:
        poly.vertex_positions = [rotate_point(v, center=center, rot3=rot3) for v in poly.vertex_positions]
    return rotated


def resolve_model_rotation_matrix(
    *,
    model_rotation: str | None,
    model_view: str,
) -> np.ndarray | None:
    rotation = str(model_rotation).strip() if model_rotation is not None else ""
    if rotation:
        return _rotate_axes(rotation).T
    raw_view = str(model_view or "top").strip().lower()
    if raw_view == "top":
        return None
    if raw_view in {"front", "side"}:
        right, up, forward = resolve_camera_basis(view=raw_view)
        return camera_basis_matrix(right, up, forward).T
    return _rotate_axes(raw_view).T


def rotation_center(structure: Structure) -> np.ndarray:
    if not structure.atoms:
        return np.zeros(3, dtype=float)
    coords = np.array([atom.position for atom in structure.atoms], dtype=float)
    return (coords.min(axis=0) + coords.max(axis=0)) * 0.5


def rotate_point(
    point: tuple[float, float, float],
    *,
    center: np.ndarray,
    rot3: np.ndarray,
) -> tuple[float, float, float]:
    p = np.array(point, dtype=float)
    q = center + rot3 @ (p - center)
    return (float(q[0]), float(q[1]), float(q[2]))


def rotate_vector(
    vector: list[float] | tuple[float, float, float],
    *,
    rot3: np.ndarray,
) -> tuple[float, float, float]:
    v = np.array(vector, dtype=float)
    out = rot3 @ v
    return (float(out[0]), float(out[1]), float(out[2]))


def compute_bounds(points: Iterable[tuple[float, float, float]]) -> SceneBounds:
    coords = [tuple(float(v) for v in point) for point in points]
    if not coords:
        return SceneBounds()
    stacked = np.asarray(coords, dtype=float).reshape((-1, 3))
    minimum = tuple(float(v) for v in np.min(stacked, axis=0))
    maximum = tuple(float(v) for v in np.max(stacked, axis=0))
    center_arr = (np.asarray(minimum, dtype=float) + np.asarray(maximum, dtype=float)) * 0.5
    radius = float(max(np.linalg.norm(point - center_arr) for point in stacked))
    return SceneBounds(
        minimum=minimum,
        maximum=maximum,
        center=(float(center_arr[0]), float(center_arr[1]), float(center_arr[2])),
        radius=radius,
    )
