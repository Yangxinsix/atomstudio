from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from typing import Any

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


@dataclass(frozen=True)
class AtomMeshInstance:
    position: tuple[float, float, float]
    radius: float
    material: Any
    segments: int
    rings: int
    index: int | None = None
    symbol: str | None = None


def build_atom_mesh_batches(
    atoms: list[AtomMeshInstance],
    *,
    collection: Any = None,
    name_prefix: str = "Atoms",
) -> list[Any]:
    if bpy is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")
    if not atoms:
        return []

    out: list[Any] = []
    for order, atom in enumerate(atoms):
        obj = _build_atom_instance(atom, name=_atom_object_name(name_prefix, atom, order))
        _link_object(obj, collection=collection)
        out.append(obj)
    return out


def _build_atom_instance(atom: AtomMeshInstance, *, name: str) -> Any:
    radius = max(1e-8, float(atom.radius))
    mesh = _cached_sphere_mesh(
        subdivisions=sphere_subdivisions(segments=int(atom.segments), rings=int(atom.rings)),
        material=atom.material,
    )
    obj = bpy.data.objects.new(str(name), mesh)
    obj.location = tuple(float(v) for v in atom.position)
    obj.scale = (radius, radius, radius)
    return obj


_SPHERE_MESH_CACHE: dict[tuple[int, str], str] = {}


def _cached_sphere_mesh(*, subdivisions: int, material: Any) -> Any:
    key = (int(subdivisions), _material_key(material))
    mesh_name = _SPHERE_MESH_CACHE.get(key)
    mesh = bpy.data.meshes.get(mesh_name) if mesh_name is not None else None
    if mesh is not None:
        return mesh

    vertices, faces = unit_icosphere_geometry(int(subdivisions))
    mesh = bpy.data.meshes.new(f"AtomIcosphere_s{int(subdivisions)}_{_mesh_suffix(material)}")
    mesh.from_pydata(list(vertices), [], list(faces))
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    if material is not None:
        mesh.materials.append(material)
    _SPHERE_MESH_CACHE[key] = str(mesh.name)
    return mesh


def sphere_subdivisions(*, segments: int, rings: int) -> int:
    detail = max(max(8, int(segments)), max(4, int(rings)) * 2)
    if detail <= 16:
        return 2
    if detail <= 64:
        return 3
    if detail <= 160:
        return 4
    return 5


@lru_cache(maxsize=8)
def unit_icosphere_geometry(subdivisions: int) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, int, int], ...]]:
    subdivisions = max(0, min(5, int(subdivisions)))
    t = (1.0 + sqrt(5.0)) * 0.5
    vertices = [
        _normalize((-1.0, t, 0.0)),
        _normalize((1.0, t, 0.0)),
        _normalize((-1.0, -t, 0.0)),
        _normalize((1.0, -t, 0.0)),
        _normalize((0.0, -1.0, t)),
        _normalize((0.0, 1.0, t)),
        _normalize((0.0, -1.0, -t)),
        _normalize((0.0, 1.0, -t)),
        _normalize((t, 0.0, -1.0)),
        _normalize((t, 0.0, 1.0)),
        _normalize((-t, 0.0, -1.0)),
        _normalize((-t, 0.0, 1.0)),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}
        next_faces: list[tuple[int, int, int]] = []
        for a, b, c in faces:
            ab = _midpoint_index(vertices, midpoints, a, b)
            bc = _midpoint_index(vertices, midpoints, b, c)
            ca = _midpoint_index(vertices, midpoints, c, a)
            next_faces.extend(
                [
                    (a, ab, ca),
                    (b, bc, ab),
                    (c, ca, bc),
                    (ab, bc, ca),
                ]
            )
        faces = next_faces
    return tuple(vertices), tuple(faces)


def _midpoint_index(
    vertices: list[tuple[float, float, float]],
    midpoints: dict[tuple[int, int], int],
    a: int,
    b: int,
) -> int:
    key = (a, b) if a < b else (b, a)
    index = midpoints.get(key)
    if index is not None:
        return index
    va = vertices[a]
    vb = vertices[b]
    vertices.append(_normalize(((va[0] + vb[0]) * 0.5, (va[1] + vb[1]) * 0.5, (va[2] + vb[2]) * 0.5)))
    index = len(vertices) - 1
    midpoints[key] = index
    return index


def _normalize(point: tuple[float, float, float]) -> tuple[float, float, float]:
    length = sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])
    if length <= 0.0:
        return (0.0, 0.0, 1.0)
    return (point[0] / length, point[1] / length, point[2] / length)


def _material_key(material: Any) -> str:
    if material is None:
        return "none"
    name = getattr(material, "name", None)
    return str(name) if name is not None else f"id:{id(material)}"


def _mesh_suffix(material: Any) -> str:
    suffix = _material_key(material)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in suffix)[:64]


def _atom_object_name(name_prefix: str, atom: AtomMeshInstance, order: int) -> str:
    label = str(atom.symbol or "X")
    index = int(atom.index) if atom.index is not None else int(order)
    return f"{name_prefix}_{index:06d}_{label}"


def _link_object(obj: Any, *, collection: Any = None) -> None:
    target = collection if collection is not None else bpy.context.scene.collection
    target.objects.link(obj)


__all__ = ["AtomMeshInstance", "build_atom_mesh_batches", "sphere_subdivisions", "unit_icosphere_geometry"]
