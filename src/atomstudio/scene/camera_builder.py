from __future__ import annotations

from math import cos, radians, sin, tan
from typing import Sequence

import numpy as np

from atomstudio.config import RenderJobConfig

try:
    from ase.utils import rotate as ase_rotate
except Exception:  # pragma: no cover
    ase_rotate = None

try:
    import bpy  # type: ignore
    from mathutils import Euler, Matrix, Vector  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    Euler = None
    Matrix = None
    Vector = None


_VIEW_BASIS = {
    "top": (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
    ),
    "front": (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
    ),
    "side": (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
    ),
}


def _axes_from_basis(
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    forward: tuple[float, float, float],
) -> np.ndarray:
    return np.array(
        [
            [right[0], -up[0], -forward[0]],
            [right[1], -up[1], -forward[1]],
            [right[2], -up[2], -forward[2]],
        ],
        dtype=float,
    )


def _normalized(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v
    return v / n


def _basis_from_axes(axes: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    right = _normalized(np.array((axes[0, 0], axes[1, 0], axes[2, 0]), dtype=float))
    up = _normalized(-np.array((axes[0, 1], axes[1, 1], axes[2, 1]), dtype=float))
    forward = _normalized(-np.array((axes[0, 2], axes[1, 2], axes[2, 2]), dtype=float))
    return (tuple(float(x) for x in right), tuple(float(x) for x in up), tuple(float(x) for x in forward))


def resolve_camera_basis(
    *,
    rotation: str | None = None,
    ase_view_axes_matrix: list[list[float]] | None = None,
    ase_view_rotations: str | None = None,
    view: str = "top",
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Resolve camera basis with priority: rotation > ase axes > ase rotations > semantic view."""
    if rotation:
        return _basis_from_axes(_rotate_axes(str(rotation)))

    if ase_view_axes_matrix is not None:
        axes = np.array(ase_view_axes_matrix, dtype=float)
        return _basis_from_axes(axes)

    if ase_view_rotations:
        return _basis_from_axes(_rotate_axes(str(ase_view_rotations)))

    raw_view = str(view or "top").strip().lower()
    base_view = "top"
    view_rotation: str | None = None

    if raw_view in _VIEW_BASIS:
        base_view = raw_view
    elif raw_view:
        view_rotation = raw_view

    base = _VIEW_BASIS[base_view]
    if not view_rotation:
        return base

    try:
        base_axes = _axes_from_basis(*base)
        axes = base_axes @ _rotate_axes(view_rotation)
        return _basis_from_axes(axes)
    except Exception:
        return base


def camera_basis_matrix(
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    forward: tuple[float, float, float],
) -> np.ndarray:
    return np.array(
        [
            [right[0], up[0], -forward[0]],
            [right[1], up[1], -forward[1]],
            [right[2], up[2], -forward[2]],
        ],
        dtype=float,
    )


class CameraBuilder:
    def __init__(
        self,
        *,
        projection: str = "ORTHOGRAPHIC",
        fit_mode: str = "orbit_origin",
        fit_padding: float = 0.10,
        lens_mm: float = 80.0,
        clip_start: float = 0.01,
        clip_end: float = 5000.0,
        center: tuple[float, float, float] | None = None,
        right: tuple[float, float, float] | None = None,
        up: tuple[float, float, float] | None = None,
        forward: tuple[float, float, float] | None = None,
        ortho_scale: float | None = None,
        distance: float | None = None,
        position: tuple[float, float, float] | None = None,
        rotation_euler: tuple[float, float, float] | None = None,
        rotation: str | None = None,
        view: str = "top",
        ase_view_rotations: str | None = None,
        ase_view_axes_matrix: list[list[float]] | None = None,
        frame_scale: float = 1.0,
    ) -> None:
        self.projection = str(projection).upper()
        self.fit_mode = str(fit_mode)
        self.fit_padding = float(fit_padding)
        self.lens_mm = float(lens_mm)
        self.clip_start = float(clip_start)
        self.clip_end = float(clip_end)
        self.center = None if center is None else tuple(float(v) for v in center)
        self.right = None if right is None else tuple(float(v) for v in right)
        self.up = None if up is None else tuple(float(v) for v in up)
        self.forward = None if forward is None else tuple(float(v) for v in forward)
        self.ortho_scale = None if ortho_scale is None else max(1e-6, float(ortho_scale))
        self.distance = None if distance is None else max(1e-6, float(distance))
        self.position = None if position is None else tuple(float(v) for v in position)
        self.rotation_euler = None if rotation_euler is None else tuple(float(v) for v in rotation_euler)
        self.rotation = None if rotation is None else str(rotation)
        self.view = str(view or "top").strip().lower()
        self.ase_view_rotations = None if ase_view_rotations is None else str(ase_view_rotations)
        self.ase_view_axes_matrix = None if ase_view_axes_matrix is None else [[float(v) for v in row] for row in ase_view_axes_matrix]
        self.frame_scale = max(1e-6, float(frame_scale))

    @classmethod
    def from_cfg(cls, cfg: RenderJobConfig) -> "CameraBuilder":
        cam = cfg.camera
        return cls(
            projection=cam.projection,
            fit_mode=cam.fit_mode,
            fit_padding=cam.fit_padding,
            lens_mm=cam.lens_mm,
            clip_start=cam.clip_start,
            clip_end=cam.clip_end,
            center=cam.center,
            right=cam.right,
            up=cam.up,
            forward=cam.forward,
            ortho_scale=cam.ortho_scale,
            distance=cam.distance,
            position=cam.position,
            rotation_euler=cam.rotation_euler,
            rotation=cam.rotation,
            view=cam.view,
            ase_view_rotations=cam.ase_view.rotations,
            ase_view_axes_matrix=cam.ase_view.axes_matrix,
            frame_scale=cam.frame_scale,
        )

    def build(self, points: Sequence) -> object:
        if bpy is None or Vector is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")

        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        cam.name = "RenderCamera"

        cam.data.type = "ORTHO" if self.projection.startswith("ORTHO") else "PERSP"
        cam.data.lens = self.lens_mm
        cam.data.clip_start = self.clip_start
        cam.data.clip_end = self.clip_end

        center = _center(points)
        if self.center is not None:
            center = Vector(self.center)

        if self.right is not None and self.up is not None and self.forward is not None:
            right = Vector(self.right).normalized()
            up = Vector(self.up).normalized()
            forward = Vector(self.forward).normalized()
            distance = self.distance
            if distance is None:
                distance = max(float(self.ortho_scale or 0.0), _point_depth_span(points, center, forward) + 1.0, 3.0)
            cam.location = center - forward * float(distance)
            if cam.data.type == "ORTHO":
                cam.data.ortho_scale = float(self.ortho_scale or self._fit_ortho_scale(points, center, right, up))
            _set_camera_basis(cam, right, up, forward)
            bpy.context.scene.camera = cam
            return cam

        if self.position is not None:
            cam.location = Vector(self.position)
            if self.rotation_euler is not None:
                cam.rotation_mode = "XYZ"
                cam.rotation_euler = tuple(radians(float(v)) for v in self.rotation_euler)
            else:
                cam.rotation_mode = "QUATERNION"
                cam.rotation_quaternion = (center - cam.location).to_track_quat("-Z", "Y")
            bpy.context.scene.camera = cam
            return cam

        right, up, forward = self._view_basis()
        self._fit_camera(cam, points, center, right, up, forward)
        self._apply_orbit_rotation(cam)
        self._apply_frame_scale(cam, center)

        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = (center - cam.location).to_track_quat("-Z", "Y")

        bpy.context.scene.camera = cam
        return cam

    def _fit_ortho_scale(self, points: Sequence, center: object, right: object, up: object) -> float:
        if not points:
            return 1.0
        scene = bpy.context.scene
        aspect = scene.render.resolution_x / max(1.0, scene.render.resolution_y)
        margin = max(0.01, min(0.35, self.fit_padding))
        usable = max(0.05, 1.0 - 2.0 * margin)
        rel = [p - center for p in points]
        max_abs_x = max(abs(v.dot(right)) for v in rel)
        max_abs_y = max(abs(v.dot(up)) for v in rel)
        return max(2.0 * max_abs_x / usable, 2.0 * max_abs_y * aspect / usable, 1.0)

    def _view_basis(self) -> tuple[Vector, Vector, Vector]:
        right, up, forward = resolve_camera_basis(
            rotation=self.rotation,
            ase_view_axes_matrix=self.ase_view_axes_matrix,
            ase_view_rotations=self.ase_view_rotations,
            view=self.view,
        )
        return Vector(right), Vector(up), Vector(forward)

    def _fit_camera(self, cam, points: Sequence, center: Vector, right: Vector, up: Vector, forward: Vector) -> None:
        if not points:
            points = [Vector((0.0, 0.0, 0.0))]

        margin = max(0.01, min(0.35, self.fit_padding))
        usable = max(0.05, 1.0 - 2.0 * margin)
        fit_mode = self.fit_mode.lower()

        if fit_mode == "notebook_xy" and cam.data.type == "ORTHO":
            xs = [p.x for p in points]
            ys = [p.y for p in points]
            zs = [p.z for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            z_min, z_max = min(zs), max(zs)
            x_span = x_max - x_min
            y_span = y_max - y_min

            scene = bpy.context.scene
            aspect = scene.render.resolution_x / max(1.0, scene.render.resolution_y)
            half_w = 0.5 * x_span / usable
            half_h = 0.5 * y_span / usable
            cam.data.ortho_scale = max(2.0 * half_w, 2.0 * half_h * aspect, 1.0)

            z_pad = max(3.0, (z_max - z_min) + 1.5)
            cam.location = Vector(((x_min + x_max) * 0.5, (y_min + y_max) * 0.5, z_max + z_pad))
            return

        local = []
        for p in points:
            v = p - center
            local.append(Vector((v.dot(right), v.dot(up), v.dot(-forward))))

        max_abs_x = max(abs(v.x) for v in local)
        max_abs_y = max(abs(v.y) for v in local)
        max_z = max(v.z for v in local)
        min_z = min(v.z for v in local)

        scene = bpy.context.scene
        aspect = scene.render.resolution_x / max(1.0, scene.render.resolution_y)

        if fit_mode in {"ase_gui", "auto", "orbit_origin"} and cam.data.type == "ORTHO":
            half_w = max_abs_x / usable
            half_h = max_abs_y / usable
            cam.data.ortho_scale = max(2.0 * half_w, 2.0 * half_h * aspect, 1.0)
            distance = max(max_z - min_z + 1.0, 3.0)
            cam.location = center - forward * distance
            return

        if cam.data.type == "ORTHO":
            half_w = max_abs_x / usable
            half_h = max_abs_y / usable
            cam.data.ortho_scale = max(2.0 * half_w, 2.0 * half_h * aspect, 1.0)
            distance = max(max_z - min_z + 1.0, 3.0)
        else:
            tan_x = tan(cam.data.angle_x * 0.5) * usable
            tan_y = tan(cam.data.angle_y * 0.5) * usable
            need_d = 0.0
            for v in local:
                need_d = max(
                    need_d,
                    v.z + abs(v.x) / max(1e-6, tan_x),
                    v.z + abs(v.y) / max(1e-6, tan_y),
                )
            distance = max(need_d + 1.2, 4.0)

        cam.location = center - forward * distance

    def _apply_orbit_rotation(self, cam) -> None:
        rot = self.rotation_euler
        if rot is None:
            return
        if Euler is None:
            return

        euler = Euler((radians(float(rot[0])), radians(float(rot[1])), radians(float(rot[2]))), "XYZ")
        origin = Vector((0.0, 0.0, 0.0))
        rel = cam.location - origin
        cam.location = origin + (euler.to_matrix() @ rel)

    def _apply_frame_scale(self, cam, center: Vector) -> None:
        scale = max(1e-6, float(self.frame_scale))
        if abs(scale - 1.0) <= 1e-12:
            return
        if str(getattr(cam.data, "type", "")).upper() == "ORTHO":
            cam.data.ortho_scale = max(1e-6, float(cam.data.ortho_scale) * scale)
            return
        rel = cam.location - center
        if rel.length <= 1e-9:
            return
        cam.location = center + rel * scale


def _center(points: list) -> Vector:
    if not points:
        return Vector((0.0, 0.0, 0.0))
    c = Vector((0.0, 0.0, 0.0))
    for p in points:
        c += p
    return c / len(points)


def _point_depth_span(points: Sequence, center: object, forward: object) -> float:
    if not points:
        return 0.0
    depths = [(p - center).dot(forward) for p in points]
    return float(max(depths) - min(depths))


def _set_camera_basis(cam, right, up, forward) -> None:
    if Matrix is None:
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = forward.to_track_quat("-Z", "Y")
        return
    matrix = Matrix(
        (
            (right.x, up.x, -forward.x),
            (right.y, up.y, -forward.y),
            (right.z, up.z, -forward.z),
        )
    )
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = matrix.to_quaternion()


def _rotate_axes(rotations: str) -> np.ndarray:
    if ase_rotate is not None:
        return np.array(ase_rotate(rotations), dtype=float)

    rotation = np.identity(3)
    if not rotations:
        return rotation

    for token in str(rotations).split(","):
        token = token.strip()
        if not token:
            continue
        axis = token[-1].lower()
        angle = radians(float(token[:-1]))
        s, c = sin(angle), cos(angle)
        if axis == "x":
            m = np.array([(1.0, 0.0, 0.0), (0.0, c, s), (0.0, -s, c)])
        elif axis == "y":
            m = np.array([(c, 0.0, -s), (0.0, 1.0, 0.0), (s, 0.0, c)])
        elif axis == "z":
            m = np.array([(c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0)])
        else:
            continue
        rotation = rotation @ m
    return rotation
