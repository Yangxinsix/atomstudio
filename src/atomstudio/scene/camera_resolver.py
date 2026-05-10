from __future__ import annotations

from math import cos, radians, sin, tan
from typing import Sequence

import numpy as np

from atomstudio.config import RenderJobConfig
from atomstudio.scene.camera_builder import resolve_camera_basis
from atomstudio.scene.model import SceneBounds, SceneCamera


def resolve_scene_camera(
    cfg: RenderJobConfig,
    *,
    bounds: SceneBounds,
    points: Sequence[tuple[float, float, float]] = (),
    aspect_ratio: float = 1.0,
) -> SceneCamera:
    camera_cfg = cfg.camera
    projection = str(camera_cfg.projection).upper()
    center = np.asarray(camera_cfg.center if camera_cfg.center is not None else bounds.center, dtype=float)
    point_cloud = _normalize_points(points, fallback=center)

    if camera_cfg.right is not None and camera_cfg.up is not None and camera_cfg.forward is not None:
        right = _normalized(np.asarray(camera_cfg.right, dtype=float))
        up = _normalized(np.asarray(camera_cfg.up, dtype=float))
        forward = _normalized(np.asarray(camera_cfg.forward, dtype=float))
        ortho_scale = float(camera_cfg.ortho_scale) if camera_cfg.ortho_scale is not None else (
            _ortho_scale_for_points(point_cloud, center, right, up, fit_padding=float(camera_cfg.fit_padding), aspect_ratio=aspect_ratio)
            if projection.startswith("ORTHO")
            else None
        )
        distance = float(camera_cfg.distance) if camera_cfg.distance is not None else max(float(bounds.radius) * 2.0, float(ortho_scale or 0.0), 3.0)
        position = center - forward * distance
        return SceneCamera(
            projection=projection,
            center=tuple(float(v) for v in center),
            right=tuple(float(v) for v in right),
            up=tuple(float(v) for v in up),
            forward=tuple(float(v) for v in forward),
            scale_factor=max(float(distance), float(ortho_scale or 0.0), 1.0),
            lens_mm=float(camera_cfg.lens_mm),
            clip_start=float(camera_cfg.clip_start),
            clip_end=float(camera_cfg.clip_end),
            target=tuple(float(v) for v in center),
            position=tuple(float(v) for v in position),
            fit_mode=str(camera_cfg.fit_mode),
            fit_padding=float(camera_cfg.fit_padding),
            ortho_scale=ortho_scale if projection.startswith("ORTHO") else None,
            distance=distance,
            rotation=None if camera_cfg.rotation is None else str(camera_cfg.rotation),
            rotation_euler=None if camera_cfg.rotation_euler is None else tuple(float(v) for v in camera_cfg.rotation_euler),
            view=str(camera_cfg.view),
            frame_scale=float(camera_cfg.frame_scale),
            ase_view_rotations=camera_cfg.ase_view.rotations,
            ase_view_axes_matrix=None
            if camera_cfg.ase_view.axes_matrix is None
            else tuple(tuple(float(v) for v in row) for row in camera_cfg.ase_view.axes_matrix),
            metadata={"mode": "explicit_basis", "aspect_ratio": float(aspect_ratio)},
        )

    if camera_cfg.position is not None:
        position = np.asarray(camera_cfg.position, dtype=float)
        if camera_cfg.rotation_euler is not None:
            right, up, forward = _basis_from_rotation_euler(camera_cfg.rotation_euler)
        else:
            right, up, forward = _look_at_basis(position=position, target=center)
        ortho_scale = _ortho_scale_for_points(point_cloud, center, right, up, fit_padding=float(camera_cfg.fit_padding), aspect_ratio=aspect_ratio)
        distance = float(np.linalg.norm(position - center))
        return SceneCamera(
            projection=projection,
            center=tuple(float(v) for v in center),
            right=tuple(float(v) for v in right),
            up=tuple(float(v) for v in up),
            forward=tuple(float(v) for v in forward),
            scale_factor=max(float(bounds.radius) * max(1.0, float(camera_cfg.frame_scale)), 1.0),
            lens_mm=float(camera_cfg.lens_mm),
            clip_start=float(camera_cfg.clip_start),
            clip_end=float(camera_cfg.clip_end),
            target=tuple(float(v) for v in center),
            position=tuple(float(v) for v in position),
            fit_mode=str(camera_cfg.fit_mode),
            fit_padding=float(camera_cfg.fit_padding),
            ortho_scale=ortho_scale if projection.startswith("ORTHO") else None,
            distance=distance,
            rotation=None if camera_cfg.rotation is None else str(camera_cfg.rotation),
            rotation_euler=None if camera_cfg.rotation_euler is None else tuple(float(v) for v in camera_cfg.rotation_euler),
            view=str(camera_cfg.view),
            frame_scale=float(camera_cfg.frame_scale),
            ase_view_rotations=camera_cfg.ase_view.rotations,
            ase_view_axes_matrix=None
            if camera_cfg.ase_view.axes_matrix is None
            else tuple(tuple(float(v) for v in row) for row in camera_cfg.ase_view.axes_matrix),
            metadata={"mode": "explicit_position"},
        )

    right_t, up_t, forward_t = resolve_camera_basis(
        rotation=camera_cfg.rotation,
        ase_view_axes_matrix=camera_cfg.ase_view.axes_matrix,
        ase_view_rotations=camera_cfg.ase_view.rotations,
        view=camera_cfg.view,
    )
    right = np.asarray(right_t, dtype=float)
    up = np.asarray(up_t, dtype=float)
    forward = np.asarray(forward_t, dtype=float)
    fit = _fit_camera_to_points(
        points=point_cloud,
        center=center,
        right=right,
        up=up,
        forward=forward,
        projection=projection,
        fit_mode=str(camera_cfg.fit_mode),
        fit_padding=float(camera_cfg.fit_padding),
        lens_mm=float(camera_cfg.lens_mm),
        frame_scale=float(camera_cfg.frame_scale),
        aspect_ratio=aspect_ratio,
    )
    return SceneCamera(
        projection=projection,
        center=tuple(float(v) for v in center),
        right=tuple(float(v) for v in right),
        up=tuple(float(v) for v in up),
        forward=tuple(float(v) for v in forward),
        scale_factor=float(fit["scale_factor"]),
        lens_mm=float(camera_cfg.lens_mm),
        clip_start=float(camera_cfg.clip_start),
        clip_end=float(camera_cfg.clip_end),
        target=tuple(float(v) for v in center),
        position=tuple(float(v) for v in fit["position"]),
        fit_mode=str(camera_cfg.fit_mode),
        fit_padding=float(camera_cfg.fit_padding),
        ortho_scale=fit["ortho_scale"],
        distance=fit["distance"],
        rotation=None if camera_cfg.rotation is None else str(camera_cfg.rotation),
        rotation_euler=None if camera_cfg.rotation_euler is None else tuple(float(v) for v in camera_cfg.rotation_euler),
        view=str(camera_cfg.view),
        frame_scale=float(camera_cfg.frame_scale),
        ase_view_rotations=camera_cfg.ase_view.rotations,
        ase_view_axes_matrix=None
        if camera_cfg.ase_view.axes_matrix is None
        else tuple(tuple(float(v) for v in row) for row in camera_cfg.ase_view.axes_matrix),
        metadata={"mode": "fit", "aspect_ratio": float(aspect_ratio)},
    )


def _normalize_points(
    points: Sequence[tuple[float, float, float]],
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    if points:
        return np.asarray(points, dtype=float).reshape((-1, 3))
    return np.asarray([fallback], dtype=float).reshape((-1, 3))


def _fit_camera_to_points(
    *,
    points: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    projection: str,
    fit_mode: str,
    fit_padding: float,
    lens_mm: float,
    frame_scale: float,
    aspect_ratio: float,
) -> dict[str, float | tuple[float, float, float] | None]:
    margin = max(0.01, min(0.35, float(fit_padding)))
    usable = max(0.05, 1.0 - 2.0 * margin)
    rel = points - center.reshape((1, 3))
    local_x = rel @ right
    local_y = rel @ up
    local_z = rel @ (-forward)
    max_abs_x = float(np.max(np.abs(local_x)))
    max_abs_y = float(np.max(np.abs(local_y)))
    max_z = float(np.max(local_z))
    min_z = float(np.min(local_z))

    ortho_scale = None
    if projection.startswith("ORTHO"):
        half_w = max_abs_x / usable
        half_h = max_abs_y / usable
        ortho_scale = max(2.0 * half_w, 2.0 * half_h * max(1.0, float(aspect_ratio)), 1.0)
        distance = max(max_z - min_z + 1.0, 3.0)
    else:
        tan_half = max(1e-6, tan(radians(_approximate_fov_degrees(lens_mm)) * 0.5))
        tan_x = tan_half * usable
        tan_y = tan_half * usable / max(1e-6, float(aspect_ratio))
        need_d = 0.0
        for x, y, z in zip(local_x, local_y, local_z, strict=False):
            need_d = max(need_d, float(z) + abs(float(x)) / tan_x, float(z) + abs(float(y)) / tan_y)
        distance = max(need_d + 1.2, 4.0)

    position = center - forward * distance
    if abs(float(frame_scale) - 1.0) > 1e-12:
        if ortho_scale is not None:
            ortho_scale = max(1e-6, float(ortho_scale) * float(frame_scale))
        position = center + (position - center) * float(frame_scale)
        distance = float(np.linalg.norm(position - center))
    return {
        "position": (float(position[0]), float(position[1]), float(position[2])),
        "ortho_scale": None if ortho_scale is None else float(ortho_scale),
        "distance": float(distance),
        "scale_factor": max(float(distance), float(ortho_scale or 0.0), 1.0),
    }


def _look_at_basis(*, position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = _normalized(target - position)
    if float(np.linalg.norm(forward)) <= 1e-12:
        forward = np.asarray((0.0, 0.0, -1.0), dtype=float)
    up_guess = np.asarray((0.0, 0.0, 1.0), dtype=float)
    if abs(float(np.dot(forward, up_guess))) > 0.95:
        up_guess = np.asarray((0.0, 1.0, 0.0), dtype=float)
    right = _normalized(np.cross(forward, up_guess))
    up = _normalized(np.cross(right, forward))
    return right, up, forward


def _basis_from_rotation_euler(rotation_euler: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_deg, y_deg, z_deg = (float(rotation_euler[0]), float(rotation_euler[1]), float(rotation_euler[2]))
    rot = _rotation_matrix_xyz(x_deg, y_deg, z_deg)
    right = _normalized(rot @ np.asarray((1.0, 0.0, 0.0), dtype=float))
    up = _normalized(rot @ np.asarray((0.0, 1.0, 0.0), dtype=float))
    forward = _normalized(rot @ np.asarray((0.0, 0.0, -1.0), dtype=float))
    return right, up, forward


def _rotation_matrix_xyz(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    rx = radians(x_deg)
    ry = radians(y_deg)
    rz = radians(z_deg)
    sx, cx = sin(rx), cos(rx)
    sy, cy = sin(ry), cos(ry)
    sz, cz = sin(rz), cos(rz)
    mx = np.asarray(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), dtype=float)
    my = np.asarray(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=float)
    mz = np.asarray(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)), dtype=float)
    return mz @ my @ mx


def _ortho_scale_for_points(
    points: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    *,
    fit_padding: float,
    aspect_ratio: float,
) -> float:
    margin = max(0.01, min(0.35, float(fit_padding)))
    usable = max(0.05, 1.0 - 2.0 * margin)
    rel = points - center.reshape((1, 3))
    max_abs_x = float(np.max(np.abs(rel @ right)))
    max_abs_y = float(np.max(np.abs(rel @ up)))
    half_w = max_abs_x / usable
    half_h = max_abs_y / usable
    return max(2.0 * half_w, 2.0 * half_h * max(1.0, float(aspect_ratio)), 1.0)


def _approximate_fov_degrees(lens_mm: float, sensor_width_mm: float = 36.0) -> float:
    lens = max(1e-3, float(lens_mm))
    return 2.0 * np.degrees(np.arctan(sensor_width_mm / (2.0 * lens)))


def _normalized(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return value
    return value / norm
