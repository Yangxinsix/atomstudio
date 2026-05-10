from __future__ import annotations

from math import cos, radians, sin, sqrt

import numpy as np

from atomstudio.preview.types import PreviewSelection, PreviewRenderScene


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def rotation_basis(
    azimuth: float,
    elevation: float,
    roll: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    az = radians(float(azimuth))
    el = radians(float(elevation))
    right = np.array([cos(az), sin(az), 0.0], dtype=float)
    forward = np.array([-sin(az) * cos(el), cos(az) * cos(el), -sin(el)], dtype=float)
    up = np.cross(right, forward)
    roll_rad = radians(float(roll))
    if abs(roll_rad) > 1e-12:
        right, up = (
            right * cos(roll_rad) + up * sin(roll_rad),
            up * cos(roll_rad) - right * sin(roll_rad),
        )
    return tuple(float(v) for v in right), tuple(float(v) for v in up), tuple(float(v) for v in forward)


def project_point(
    point: tuple[float, float, float],
    camera,
    viewport_size: tuple[int, int],
) -> tuple[float, float, float]:
    width = max(1.0, float(viewport_size[0]))
    height = max(1.0, float(viewport_size[1]))
    right, up, forward = camera.basis()
    rel = np.array(point, dtype=float) - np.array(camera.center, dtype=float)
    x = float(np.dot(rel, np.array(right, dtype=float)))
    y = float(np.dot(rel, np.array(up, dtype=float)))
    z = float(np.dot(rel, np.array(forward, dtype=float)))
    scale = max(1e-6, float(camera.scale_factor))
    sx = width * 0.5 + (x / scale) * (width * 0.5)
    sy = height * 0.5 - (y / scale) * (height * 0.5)
    return sx, sy, z


def point_distance_2d(left: tuple[float, float], right: tuple[float, float]) -> float:
    return sqrt((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2)


def segment_distance_2d(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    start_vec = np.array(start, dtype=float)
    end_vec = np.array(end, dtype=float)
    point_vec = np.array(point, dtype=float)
    segment = end_vec - start_vec
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return point_distance_2d(point, start), 0.0
    weight = float(np.dot(point_vec - start_vec, segment) / denom)
    weight = max(0.0, min(1.0, weight))
    closest = start_vec + segment * weight
    return float(np.linalg.norm(point_vec - closest)), weight


def project_atom_positions(scene: PreviewRenderScene | None, camera, viewport_size: tuple[int, int]) -> dict[int, tuple[float, float, float]]:
    if scene is None:
        return {}
    return {int(atom.index): project_point(atom.position, camera, viewport_size) for atom in scene.atoms}


def pick_atom_selection(
    scene: PreviewRenderScene | None,
    camera,
    viewport_size: tuple[int, int],
    pos: tuple[float, float],
) -> PreviewSelection | None:
    if scene is None or not scene.atoms:
        return None
    projected = project_atom_positions(scene, camera, viewport_size)
    best_index: int | None = None
    best_depth = float("inf")
    best_distance = float("inf")
    for atom in scene.atoms:
        point = projected.get(int(atom.index))
        if point is None:
            continue
        distance = point_distance_2d((point[0], point[1]), pos)
        threshold = max(6.0, float(atom.size_px) * 0.5)
        if distance > threshold:
            continue
        depth = float(point[2])
        if depth < best_depth - 1e-6 or (abs(depth - best_depth) <= 1e-6 and distance < best_distance):
            best_index = int(atom.index)
            best_depth = depth
            best_distance = distance
    if best_index is None:
        return None
    return PreviewSelection(kind="atom", index=best_index)


def pick_bond_selection(
    scene: PreviewRenderScene | None,
    camera,
    viewport_size: tuple[int, int],
    pos: tuple[float, float],
) -> PreviewSelection | None:
    if scene is None or not scene.bonds:
        return None
    best_index: int | None = None
    best_depth = float("inf")
    best_distance = float("inf")
    for bond in scene.bonds:
        for segment in bond.segments:
            start = project_point(segment.start, camera, viewport_size)
            end = project_point(segment.end, camera, viewport_size)
            distance, weight = segment_distance_2d(pos, (start[0], start[1]), (end[0], end[1]))
            threshold = max(6.0, 0.5 * float(segment.width_px))
            if distance > threshold:
                continue
            depth = (1.0 - weight) * float(start[2]) + weight * float(end[2])
            if depth < best_depth - 1e-6 or (abs(depth - best_depth) <= 1e-6 and distance < best_distance):
                best_index = int(bond.index)
                best_depth = depth
                best_distance = distance
    if best_index is None:
        return None
    return PreviewSelection(kind="bond", index=best_index)


def pick_selection(
    scene: PreviewRenderScene | None,
    camera,
    viewport_size: tuple[int, int],
    pos: tuple[float, float],
) -> PreviewSelection | None:
    atom_selection = pick_atom_selection(scene, camera, viewport_size, pos)
    if atom_selection is not None:
        return atom_selection
    return pick_bond_selection(scene, camera, viewport_size, pos)


__all__ = [
    "normalize",
    "pick_atom_selection",
    "pick_bond_selection",
    "pick_selection",
    "point_distance_2d",
    "project_atom_positions",
    "project_point",
    "rotation_basis",
    "segment_distance_2d",
]
