from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    Vector = None


def tuple3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if isinstance(value, Sequence) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return default


def dashed_line_segments(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    count: int = 7,
    dash_fraction: float = 0.58,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    count = max(1, int(count))
    dash_fraction = min(1.0, max(0.05, float(dash_fraction)))
    start_v = tuple3(start)
    end_v = tuple3(end)
    delta = tuple(float(b) - float(a) for a, b in zip(start_v, end_v, strict=True))
    if sum(component * component for component in delta) < 1.0e-16:
        return ()
    out: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for index in range(count):
        t0 = float(index) / float(count)
        t1 = min(1.0, t0 + dash_fraction / float(count))
        dash_start = tuple(float(start_v[axis]) + delta[axis] * t0 for axis in range(3))
        dash_end = tuple(float(start_v[axis]) + delta[axis] * t1 for axis in range(3))
        out.append((dash_start, dash_end))
    return tuple(out)


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


def add_curve_line_between(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    radius: float,
    material: Any,
    name: str,
    *,
    collection: Any = None,
) -> Any | None:
    if bpy is None or Vector is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")

    if (Vector(end) - Vector(start)).length < 1e-8:
        return None
    curve = bpy.data.curves.new(str(name), type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = max(0.0, float(radius))
    curve.bevel_resolution = 0
    curve.use_fill_caps = False
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (float(start[0]), float(start[1]), float(start[2]), 1.0)
    spline.points[1].co = (float(end[0]), float(end[1]), float(end[2]), 1.0)
    if material is not None:
        curve.materials.append(material)
    obj = bpy.data.objects.new(str(name), curve)
    _link_object(obj, collection=collection)
    return obj


def set_curve_line_between(obj: Any, start: tuple[float, float, float], end: tuple[float, float, float], radius: float) -> None:
    if Vector is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")

    if (Vector(end) - Vector(start)).length < 1e-8:
        raise ValueError("Cannot update zero-length curve line")
    curve = getattr(obj, "data", None)
    if curve is None:
        raise ValueError("Cannot update curve line without curve data")
    curve.bevel_depth = max(0.0, float(radius))
    spline = curve.splines[0] if getattr(curve, "splines", None) else curve.splines.new("POLY")
    while len(spline.points) < 2:
        spline.points.add(1)
    spline.points[0].co = (float(start[0]), float(start[1]), float(start[2]), 1.0)
    spline.points[1].co = (float(end[0]), float(end[1]), float(end[2]), 1.0)


def collect_bbox_points(objects: list[Any]) -> list[Any]:
    if Vector is None:
        return []
    points: list[Any] = []
    for obj in objects:
        for corner in getattr(obj, "bound_box", []):
            points.append(obj.matrix_world @ Vector(corner))
    return points


def _link_object(obj: Any, *, collection: Any = None) -> None:
    if collection is not None:
        collection.objects.link(obj)
    else:
        bpy.context.collection.objects.link(obj)
