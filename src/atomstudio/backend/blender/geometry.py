from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    from mathutils import Vector  # type: ignore
except Exception:  # pragma: no cover
    Vector = None


def tuple3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if isinstance(value, Sequence) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return default


def set_cylinder_between(obj: Any, start: tuple[float, float, float], end: tuple[float, float, float], radius: float) -> None:
    if Vector is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")

    v1, v2 = Vector(start), Vector(end)
    vec = v2 - v1
    if vec.length < 1e-8:
        raise ValueError("Cannot update zero-length cylinder")
    obj.location = (v1 + v2) / 2.0
    obj.scale = (float(radius), float(radius), float(vec.length) * 0.5)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(vec.normalized())


def collect_bbox_points(objects: list[Any]) -> list[Any]:
    if Vector is None:
        return []
    points: list[Any] = []
    for obj in objects:
        for corner in getattr(obj, "bound_box", []):
            points.append(obj.matrix_world @ Vector(corner))
    return points
