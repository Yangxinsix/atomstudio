from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, tau
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

    groups: dict[str, tuple[Any, list[AtomMeshInstance]]] = {}
    for atom in atoms:
        key = _material_key(atom.material)
        if key not in groups:
            groups[key] = (atom.material, [])
        groups[key][1].append(atom)

    out: list[Any] = []
    for group_index, (_key, (material, group_atoms)) in enumerate(groups.items()):
        obj = _build_group_mesh(group_atoms, material=material, name=f"{name_prefix}_{group_index:03d}")
        _link_object(obj, collection=collection)
        out.append(obj)
    return out


def _build_group_mesh(atoms: list[AtomMeshInstance], *, material: Any, name: str) -> Any:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for atom in atoms:
        _append_sphere_geometry(vertices, faces, atom)

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    if material is not None:
        mesh.materials.append(material)
    return bpy.data.objects.new(name, mesh)


def _append_sphere_geometry(
    vertices_out: list[tuple[float, float, float]],
    faces_out: list[tuple[int, ...]],
    atom: AtomMeshInstance,
) -> None:
    segments = max(8, int(atom.segments))
    rings = max(4, int(atom.rings))
    radius = max(1e-8, float(atom.radius))
    cx, cy, cz = (float(v) for v in atom.position)
    base = len(vertices_out)

    vertices_out.append((cx, cy, cz + radius))
    for ring in range(1, rings):
        phi = pi * float(ring) / float(rings)
        local_z = cos(phi)
        local_r = sin(phi)
        for segment in range(segments):
            theta = tau * float(segment) / float(segments)
            vertices_out.append(
                (
                    cx + radius * local_r * cos(theta),
                    cy + radius * local_r * sin(theta),
                    cz + radius * local_z,
                )
            )
    bottom_index = len(vertices_out)
    vertices_out.append((cx, cy, cz - radius))

    first_ring = base + 1
    last_ring = first_ring + max(0, rings - 2) * segments
    if rings <= 1:
        return

    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces_out.append((base, first_ring + segment, first_ring + nxt))

    for ring in range(rings - 2):
        row = first_ring + ring * segments
        next_row = row + segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces_out.append((row + segment, row + nxt, next_row + nxt, next_row + segment))

    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces_out.append((last_ring + segment, bottom_index, last_ring + nxt))


def _material_key(material: Any) -> str:
    if material is None:
        return "none"
    name = getattr(material, "name", None)
    return str(name) if name is not None else f"id:{id(material)}"


def _link_object(obj: Any, *, collection: Any = None) -> None:
    target = collection if collection is not None else bpy.context.scene.collection
    target.objects.link(obj)


__all__ = ["AtomMeshInstance", "build_atom_mesh_batches"]
