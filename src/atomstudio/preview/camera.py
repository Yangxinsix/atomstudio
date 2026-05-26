from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from atomstudio.preview.picking import rotation_basis


@dataclass(frozen=True, slots=True)
class CameraPlanes:
    near: float
    far: float


@dataclass(frozen=True, slots=True)
class CameraMatrices:
    view: np.ndarray
    projection: np.ndarray
    view_projection: np.ndarray
    model: np.ndarray
    near: float
    far: float


def scene_depth_planes(scene_radius: float, *, camera_distance: float | None = None) -> CameraPlanes:
    radius = max(1.0e-6, float(scene_radius))
    if camera_distance is None:
        near = max(radius * 0.001, 0.001)
        far = max(radius * 8.0, near + 1.0)
        return CameraPlanes(near=near, far=far)
    distance = max(1.0e-6, float(camera_distance))
    margin = max(radius * 2.5, 1.0)
    near = max(distance - margin, radius * 0.001, 0.001)
    far = max(distance + margin, near + 1.0)
    return CameraPlanes(near=near, far=far)


def fit_bounds(
    minimum: tuple[float, float, float] | np.ndarray,
    maximum: tuple[float, float, float] | np.ndarray,
    *,
    padding: float = 0.10,
) -> tuple[tuple[float, float, float], float, float]:
    lower = np.asarray(minimum, dtype=np.float32).reshape((3,))
    upper = np.asarray(maximum, dtype=np.float32).reshape((3,))
    extent = np.maximum(upper - lower, 0.0)
    center = (lower + upper) * 0.5
    radius = max(1.0, float(np.linalg.norm(extent) * 0.5))
    scale = max(1.0, float(extent.max()) * (1.0 + max(0.0, float(padding)) * 2.0))
    return tuple(float(value) for value in center), scale, radius


def apply_view_preset(camera: Any, view: str) -> Any:
    preset = str(view or "top").strip().lower()
    camera.view = preset
    if preset == "top":
        camera.azimuth = 0.0
        camera.elevation = 90.0
        camera.roll = 0.0
        camera.right = (1.0, 0.0, 0.0)
        camera.up = (0.0, 1.0, 0.0)
        camera.forward = (0.0, 0.0, -1.0)
    elif preset == "front":
        camera.azimuth = 180.0
        camera.elevation = 0.0
        camera.roll = 0.0
        camera.right = (-1.0, 0.0, 0.0)
        camera.up = (0.0, 0.0, 1.0)
        camera.forward = (0.0, -1.0, 0.0)
    elif preset == "side":
        camera.azimuth = 90.0
        camera.elevation = 0.0
        camera.roll = 0.0
        camera.right = (0.0, 1.0, 0.0)
        camera.up = (0.0, 0.0, 1.0)
        camera.forward = (-1.0, 0.0, 0.0)
    else:
        camera.view = "orbit"
        camera.azimuth = float(getattr(camera, "azimuth", 45.0))
        camera.elevation = float(getattr(camera, "elevation", 30.0))
        camera.roll = float(getattr(camera, "roll", 0.0))
        camera.right, camera.up, camera.forward = rotation_basis(camera.azimuth, camera.elevation, camera.roll)
    return camera


def orbit_camera(camera: Any, dx: float, dy: float, *, degrees_per_pixel: float = 0.35) -> Any:
    camera.view = "orbit"
    camera.azimuth = float(getattr(camera, "azimuth", 45.0)) + float(dx) * float(degrees_per_pixel)
    camera.elevation = max(-89.0, min(89.0, float(getattr(camera, "elevation", 30.0)) + float(dy) * float(degrees_per_pixel)))
    camera.right, camera.up, camera.forward = rotation_basis(camera.azimuth, camera.elevation, getattr(camera, "roll", 0.0))
    return camera


def reset_model_rotation(camera: Any) -> Any:
    camera.model_rotation = tuple(float(value) for value in np.eye(4, dtype=np.float32).reshape((-1,)))
    return camera


def set_model_rotation_euler_degrees(camera: Any, angles: tuple[float, float, float]) -> Any:
    x_degrees, y_degrees, z_degrees = (float(value) for value in angles)
    rotation = (
        _axis_rotation_matrix(np.asarray((0.0, 0.0, 1.0), dtype=np.float32), z_degrees)
        @ _axis_rotation_matrix(np.asarray((0.0, 1.0, 0.0), dtype=np.float32), y_degrees)
        @ _axis_rotation_matrix(np.asarray((1.0, 0.0, 0.0), dtype=np.float32), x_degrees)
    )
    camera.model_rotation = tuple(float(value) for value in rotation.astype(np.float32).reshape((-1,)))
    return camera


def model_rotation_euler_degrees(camera: Any) -> tuple[float, float, float]:
    rotation = _model_rotation_matrix(camera)[:3, :3].astype(float, copy=False)
    sy = max(-1.0, min(1.0, -float(rotation[2, 0])))
    y = float(np.arcsin(sy))
    cy = float(np.cos(y))
    if abs(cy) > 1.0e-6:
        x = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        z = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        x = 0.0
        z = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    return (
        _wrap_degrees(float(np.degrees(x))),
        _wrap_degrees(float(np.degrees(y))),
        _wrap_degrees(float(np.degrees(z))),
    )


def rotate_model(camera: Any, dx: float, dy: float, *, degrees_per_pixel: float = 0.35) -> Any:
    right = _normalize(np.asarray(getattr(camera, "right", (1.0, 0.0, 0.0)), dtype=np.float32))
    up = _normalize(np.asarray(getattr(camera, "up", (0.0, 1.0, 0.0)), dtype=np.float32))
    current = _model_rotation_matrix(camera)
    yaw = _axis_rotation_matrix(up, float(dx) * float(degrees_per_pixel))
    pitch = _axis_rotation_matrix(right, float(dy) * float(degrees_per_pixel))
    rotation = pitch @ yaw @ current
    camera.model_rotation = tuple(float(value) for value in rotation.astype(np.float32).reshape((-1,)))
    return camera


def rotate_model_trackball(
    camera: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    viewport_size: tuple[int, int],
    *,
    sensitivity: float = 1.45,
) -> Any:
    start_vec = _trackball_vector(start, viewport_size)
    end_vec = _trackball_vector(end, viewport_size)
    axis_camera = np.cross(start_vec, end_vec)
    axis_len = float(np.linalg.norm(axis_camera))
    if axis_len <= 1.0e-8:
        return camera
    dot = max(-1.0, min(1.0, float(np.dot(start_vec, end_vec))))
    angle = float(np.arctan2(axis_len, dot)) * float(sensitivity)

    right = _normalize(np.asarray(getattr(camera, "right", (1.0, 0.0, 0.0)), dtype=np.float32))
    up = _normalize(np.asarray(getattr(camera, "up", (0.0, 1.0, 0.0)), dtype=np.float32))
    forward = _normalize(np.asarray(getattr(camera, "forward", (0.0, 0.0, -1.0)), dtype=np.float32))
    axis_world = axis_camera[0] * right + axis_camera[1] * up - axis_camera[2] * forward
    axis_world = _normalize(axis_world)
    current = _model_rotation_matrix(camera)
    rotation = _axis_rotation_matrix(axis_world, float(np.degrees(angle))) @ current
    camera.model_rotation = tuple(float(value) for value in rotation.astype(np.float32).reshape((-1,)))
    return camera


def rotate_model_about_axis(camera: Any, axis: tuple[float, float, float], degrees: float) -> Any:
    current = _model_rotation_matrix(camera)
    rotation = _axis_rotation_matrix(np.asarray(axis, dtype=np.float32), float(degrees)) @ current
    camera.model_rotation = tuple(float(value) for value in rotation.astype(np.float32).reshape((-1,)))
    return camera


def pan_camera(camera: Any, dx: float, dy: float, viewport_size: tuple[int, int]) -> Any:
    width = max(1.0, float(viewport_size[0]))
    height = max(1.0, float(viewport_size[1]))
    scale = max(1.0e-6, float(getattr(camera, "scale_factor", 1.0)))
    step_x = scale / (width * 0.5)
    step_y = scale / (height * 0.5)
    right = np.asarray(getattr(camera, "right", (1.0, 0.0, 0.0)), dtype=np.float32)
    up = np.asarray(getattr(camera, "up", (0.0, 1.0, 0.0)), dtype=np.float32)
    translation = np.asarray(getattr(camera, "model_translation", (0.0, 0.0, 0.0)), dtype=np.float32)
    translation = translation + float(dx) * step_x * right + float(dy) * step_y * up
    camera.model_translation = tuple(float(value) for value in translation)
    return camera


def zoom_camera(camera: Any, factor: float) -> Any:
    factor = max(1.0e-6, float(factor))
    camera.scale_factor = max(1.0e-6, float(getattr(camera, "scale_factor", 1.0)) / factor)
    return camera


def view_matrix(camera: Any, *, scene_radius: float = 1.0) -> np.ndarray:
    center = np.asarray(getattr(camera, "center", (0.0, 0.0, 0.0)), dtype=np.float32)
    right = _normalize(np.asarray(getattr(camera, "right", (1.0, 0.0, 0.0)), dtype=np.float32))
    up = _normalize(np.asarray(getattr(camera, "up", (0.0, 1.0, 0.0)), dtype=np.float32))
    forward = _normalize(np.asarray(getattr(camera, "forward", (0.0, 0.0, -1.0)), dtype=np.float32))
    distance = camera_distance(camera, scene_radius=scene_radius)
    eye = center - forward * distance

    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(right, eye))
    matrix[1, 3] = -float(np.dot(up, eye))
    matrix[2, 3] = float(np.dot(forward, eye))
    return matrix


def model_matrix(camera: Any) -> np.ndarray:
    center = np.asarray(getattr(camera, "center", (0.0, 0.0, 0.0)), dtype=np.float32)
    translation = np.asarray(getattr(camera, "model_translation", (0.0, 0.0, 0.0)), dtype=np.float32)
    rotation = _model_rotation_matrix(camera)
    translate_to_origin = np.eye(4, dtype=np.float32)
    translate_to_origin[:3, 3] = -center
    translate_back = np.eye(4, dtype=np.float32)
    translate_back[:3, 3] = center
    translate_model = np.eye(4, dtype=np.float32)
    translate_model[:3, 3] = translation
    return (translate_model @ translate_back @ rotation @ translate_to_origin).astype(np.float32)


def projection_matrix(
    camera: Any,
    viewport_size: tuple[int, int],
    *,
    scene_radius: float = 1.0,
    projection: str = "orthographic",
) -> tuple[np.ndarray, CameraPlanes]:
    width = max(1.0, float(viewport_size[0]))
    height = max(1.0, float(viewport_size[1]))
    aspect = width / height
    distance = camera_distance(camera, scene_radius=scene_radius)
    planes = scene_depth_planes(scene_radius, camera_distance=distance)
    projection_key = str(projection or "orthographic").strip().lower()
    if projection_key.startswith("persp"):
        return perspective_projection(45.0, aspect, planes.near, planes.far), planes
    return orthographic_projection(getattr(camera, "scale_factor", 1.0), aspect, planes.near, planes.far), planes


def camera_matrices(
    camera: Any,
    viewport_size: tuple[int, int],
    *,
    scene_radius: float = 1.0,
    projection: str = "orthographic",
) -> CameraMatrices:
    view = view_matrix(camera, scene_radius=scene_radius)
    projection_mat, planes = projection_matrix(camera, viewport_size, scene_radius=scene_radius, projection=projection)
    view_projection = projection_mat @ view
    return CameraMatrices(
        view=view,
        projection=projection_mat,
        view_projection=view_projection.astype(np.float32),
        model=model_matrix(camera),
        near=planes.near,
        far=planes.far,
    )


def orthographic_projection(scale_factor: float, aspect: float, near: float, far: float) -> np.ndarray:
    half_height = max(1.0e-6, float(scale_factor)) * 0.5
    half_width = half_height * max(1.0e-6, float(aspect))
    left, right = -half_width, half_width
    bottom, top = -half_height, half_height
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = 2.0 / (right - left)
    matrix[1, 1] = 2.0 / (top - bottom)
    matrix[2, 2] = -2.0 / (float(far) - float(near))
    matrix[3, 3] = 1.0
    matrix[0, 3] = -(right + left) / (right - left)
    matrix[1, 3] = -(top + bottom) / (top - bottom)
    matrix[2, 3] = -(float(far) + float(near)) / (float(far) - float(near))
    return matrix


def perspective_projection(fov_y_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    fov = np.deg2rad(max(1.0, min(179.0, float(fov_y_degrees))))
    focal = 1.0 / np.tan(fov * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = focal / max(1.0e-6, float(aspect))
    matrix[1, 1] = focal
    matrix[2, 2] = -(float(far) + float(near)) / (float(far) - float(near))
    matrix[2, 3] = -(2.0 * float(far) * float(near)) / (float(far) - float(near))
    matrix[3, 2] = -1.0
    return matrix


def camera_distance(camera: Any, *, scene_radius: float = 1.0) -> float:
    radius = max(1.0, float(scene_radius))
    scale = max(1.0, float(getattr(camera, "scale_factor", radius)))
    return max(3.0, radius * 4.0, scale * 2.0)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return vector
    return vector / norm


def _model_rotation_matrix(camera: Any) -> np.ndarray:
    values = getattr(camera, "model_rotation", None)
    if values is None:
        return np.eye(4, dtype=np.float32)
    try:
        matrix = np.asarray(values, dtype=np.float32).reshape((4, 4))
    except (TypeError, ValueError):
        return np.eye(4, dtype=np.float32)
    return matrix


def _axis_rotation_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
    unit = _normalize(np.asarray(axis, dtype=np.float32).reshape((3,)))
    if float(np.linalg.norm(unit)) <= 1.0e-12:
        return np.eye(4, dtype=np.float32)
    angle = np.deg2rad(float(degrees))
    x, y, z = (float(value) for value in unit)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    t = 1.0 - c
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.asarray(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float32,
    )
    return matrix


def _trackball_vector(point: tuple[float, float], viewport_size: tuple[int, int]) -> np.ndarray:
    width = max(1.0, float(viewport_size[0]))
    height = max(1.0, float(viewport_size[1]))
    radius = max(1.0, min(width, height) * 0.5)
    x = (float(point[0]) - width * 0.5) / radius
    y = (height * 0.5 - float(point[1])) / radius
    distance = float(np.hypot(x, y))
    if distance <= 1.0 / np.sqrt(2.0):
        z = float(np.sqrt(max(0.0, 1.0 - distance * distance)))
    else:
        z = float(0.5 / max(distance, 1.0e-8))
    return _normalize(np.asarray((x, y, z), dtype=np.float32))


def _wrap_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


__all__ = [
    "CameraMatrices",
    "CameraPlanes",
    "apply_view_preset",
    "camera_distance",
    "camera_matrices",
    "fit_bounds",
    "model_matrix",
    "model_rotation_euler_degrees",
    "orbit_camera",
    "orthographic_projection",
    "pan_camera",
    "perspective_projection",
    "projection_matrix",
    "reset_model_rotation",
    "rotate_model",
    "rotate_model_about_axis",
    "rotate_model_trackball",
    "scene_depth_planes",
    "set_model_rotation_euler_degrees",
    "view_matrix",
    "zoom_camera",
]
