from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, tau
from typing import Any

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    Vector = None


@dataclass(frozen=True)
class BondMeshSegment:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float
    material: Any


def build_bond_mesh_batches(
    segments: list[BondMeshSegment],
    *,
    vertices: int,
    collection: Any = None,
    name_prefix: str = "Bonds",
) -> list[Any]:
    if bpy is None or Vector is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")
    if not segments:
        return []

    groups: dict[str, tuple[Any, list[BondMeshSegment]]] = {}
    for segment in segments:
        key = _material_key(segment.material)
        if key not in groups:
            groups[key] = (segment.material, [])
        groups[key][1].append(segment)

    out: list[Any] = []
    for group_index, (_key, (material, group_segments)) in enumerate(groups.items()):
        obj = _build_group_mesh(
            group_segments,
            material=material,
            vertices=max(8, int(vertices)),
            name=f"{name_prefix}_{group_index:03d}",
        )
        _link_object(obj, collection=collection)
        out.append(obj)
    return out


def _build_group_mesh(
    segments: list[BondMeshSegment],
    *,
    material: Any,
    vertices: int,
    name: str,
) -> Any:
    mesh_vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for segment in segments:
        _append_cylinder_geometry(mesh_vertices, faces, segment, vertices=vertices)

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(mesh_vertices, [], faces)
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    if material is not None:
        mesh.materials.append(material)

    obj = bpy.data.objects.new(name, mesh)
    return obj


def _append_cylinder_geometry(
    vertices_out: list[tuple[float, float, float]],
    faces_out: list[tuple[int, ...]],
    segment: BondMeshSegment,
    *,
    vertices: int,
) -> None:
    start = Vector(segment.start)
    end = Vector(segment.end)
    axis = end - start
    length = float(axis.length)
    if length < 1e-8:
        return
    direction = axis.normalized()
    basis_x, basis_y = _orthonormal_basis(direction)
    radius = max(1e-8, float(segment.radius))
    base = len(vertices_out)

    for center in (start, end):
        for idx in range(vertices):
            angle = tau * float(idx) / float(vertices)
            radial = basis_x * cos(angle) + basis_y * sin(angle)
            point = center + radial * radius
            vertices_out.append((float(point.x), float(point.y), float(point.z)))

    start_center_index = len(vertices_out)
    vertices_out.append((float(start.x), float(start.y), float(start.z)))
    end_center_index = len(vertices_out)
    vertices_out.append((float(end.x), float(end.y), float(end.z)))

    for idx in range(vertices):
        nxt = (idx + 1) % vertices
        faces_out.append((base + idx, base + nxt, base + vertices + nxt, base + vertices + idx))
        faces_out.append((start_center_index, base + nxt, base + idx))
        faces_out.append((end_center_index, base + vertices + idx, base + vertices + nxt))


def _orthonormal_basis(direction: Any) -> tuple[Any, Any]:
    ref = Vector((0.0, 0.0, 1.0))
    if abs(float(direction.dot(ref))) > 0.95:
        ref = Vector((0.0, 1.0, 0.0))
    if abs(float(direction.dot(ref))) > 0.95:
        ref = Vector((1.0, 0.0, 0.0))
    basis_x = direction.cross(ref)
    if basis_x.length < 1e-8:
        basis_x = Vector((1.0, 0.0, 0.0))
    basis_x.normalize()
    basis_y = direction.cross(basis_x)
    basis_y.normalize()
    return basis_x, basis_y


def _material_key(material: Any) -> str:
    if material is None:
        return "none"
    name = getattr(material, "name", None)
    return str(name) if name is not None else f"id:{id(material)}"


def _link_object(obj: Any, *, collection: Any = None) -> None:
    target = collection if collection is not None else bpy.context.scene.collection
    target.objects.link(obj)


__all__ = ["BondMeshSegment", "build_bond_mesh_batches"]
