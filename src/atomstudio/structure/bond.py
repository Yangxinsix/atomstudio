from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from atomstudio.color_utils import coerce_color_fields
from atomstudio.scene.materials.specs import MaterialLike

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    Vector = None


_CYLINDER_MESH_CACHE: dict[tuple[int, str | None], str] = {}


@coerce_color_fields("color", "color_a", "color_b", label_prefix="bond")
@dataclass
class Bond:
    id: int
    a: int
    b: int
    bond_type: str = "covalent"
    order: int = 1
    distance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    style: str | None = None
    material: MaterialLike | None = None
    color: tuple[float, float, float, float] | None = None
    material_a: MaterialLike | None = None
    color_a: tuple[float, float, float, float] | None = None
    material_b: MaterialLike | None = None
    color_b: tuple[float, float, float, float] | None = None
    split_ratio: float = 0.5

    @classmethod
    def build(
        cls,
        bond: "Bond",
        *,
        positions: list[tuple[float, float, float]],
        radius: float,
        vertices: int = 20,
        material_a: Any = None,
        material_b: Any = None,
        split_ratio: float | None = None,
        split: bool = True,
        collection=None,
    ) -> list[Any]:
        if bpy is None or Vector is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")
        p1 = positions[int(bond.a)]
        p2 = positions[int(bond.b)]
        order = max(1, int(getattr(bond, "order", 1)))

        if order > 1:
            return add_parallel_bond_cylinders(
                p1,
                p2,
                float(radius),
                material_a,
                material_b,
                f"Bond_{bond.a}_{bond.b}",
                int(vertices),
                order=order,
                split_ratio=float(bond.split_ratio if split_ratio is None else split_ratio),
                split=bool(split),
                collection=collection,
            )

        if not bool(split):
            mat = material_a if material_a is not None else material_b
            obj = add_cylinder_between(
                p1,
                p2,
                float(radius),
                mat,
                f"Bond_{bond.a}_{bond.b}",
                int(vertices),
                collection=collection,
            )
            return [] if obj is None else [obj]
        return add_split_cylinders_between(
            p1,
            p2,
            float(radius),
            material_a,
            material_b,
            f"Bond_{bond.a}_{bond.b}",
            int(vertices),
            split_ratio=float(bond.split_ratio if split_ratio is None else split_ratio),
            collection=collection,
        )


def bond_distance(p1: tuple[float, float, float], p2: tuple[float, float, float]) -> float:
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def add_cylinder_between(p1, p2, radius: float, mat, name: str, vertices: int, collection=None):
    if bpy is None or Vector is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")

    v1, v2 = Vector(p1), Vector(p2)
    vec = v2 - v1
    length = vec.length
    if length < 1e-8:
        return None

    midpoint = (v1 + v2) / 2.0
    mesh = _cached_cylinder_mesh(vertices=max(8, int(vertices)), material=mat)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = midpoint
    obj.scale = (float(radius), float(radius), float(length) * 0.5)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(vec.normalized())
    _link_object(obj, collection=collection)

    return obj


def add_split_cylinders_between(
    p1,
    p2,
    radius: float,
    mat_left,
    mat_right,
    name_prefix: str,
    vertices: int,
    split_ratio: float = 0.5,
    collection=None,
) -> list[Any]:
    if bpy is None or Vector is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")

    ratio = min(0.95, max(0.05, float(split_ratio)))
    mid = (
        p1[0] + (p2[0] - p1[0]) * ratio,
        p1[1] + (p2[1] - p1[1]) * ratio,
        p1[2] + (p2[2] - p1[2]) * ratio,
    )
    out: list[Any] = []
    left = add_cylinder_between(
        p1,
        mid,
        radius,
        mat_left,
        f"{name_prefix}_a",
        vertices,
        collection=collection,
    )
    right = add_cylinder_between(
        mid,
        p2,
        radius,
        mat_right,
        f"{name_prefix}_b",
        vertices,
        collection=collection,
    )
    if left is not None:
        out.append(left)
    if right is not None:
        out.append(right)
    return out


def _cached_cylinder_mesh(*, vertices: int, material: Any) -> Any:
    if bpy is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")

    key = (int(vertices), _material_cache_key(material))
    mesh_name = _CYLINDER_MESH_CACHE.get(key)
    mesh = bpy.data.meshes.get(mesh_name) if mesh_name is not None else None
    if mesh is not None:
        return mesh

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=int(vertices),
        radius=1.0,
        depth=2.0,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    mesh = obj.data
    for poly in mesh.polygons:
        poly.use_smooth = True
    if material is not None:
        if len(mesh.materials) == 0:
            mesh.materials.append(material)
        else:
            mesh.materials[0] = material
    _remove_temporary_object(obj)
    _CYLINDER_MESH_CACHE[key] = str(mesh.name)
    return mesh


def _material_cache_key(material: Any) -> str | None:
    if material is None:
        return None
    name = getattr(material, "name", None)
    if name is not None:
        return str(name)
    return f"id:{id(material)}"


def _link_object(obj: Any, *, collection: Any = None) -> None:
    target = collection if collection is not None else bpy.context.scene.collection
    target.objects.link(obj)


def _remove_temporary_object(obj: Any) -> None:
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    bpy.data.objects.remove(obj, do_unlink=True)


def _bond_side_vector(p1, p2):
    if Vector is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")

    direction = Vector(p2) - Vector(p1)
    if direction.length < 1e-8:
        return None
    direction.normalize()

    ref = Vector((0.0, 0.0, 1.0))
    if abs(direction.dot(ref)) > 0.95:
        ref = Vector((0.0, 1.0, 0.0))
    if abs(direction.dot(ref)) > 0.95:
        ref = Vector((1.0, 0.0, 0.0))

    side = direction.cross(ref)
    if side.length < 1e-8:
        return None
    side.normalize()
    return side


def _parallel_offsets(order: int, spacing: float) -> list[float]:
    if order == 2:
        return [-0.5 * spacing, 0.5 * spacing]
    if order == 3:
        return [-1.0 * spacing, 0.0, 1.0 * spacing]

    center = (order - 1) * 0.5
    return [(i - center) * spacing for i in range(order)]


def add_parallel_bond_cylinders(
    p1,
    p2,
    radius: float,
    mat_left,
    mat_right,
    name_prefix: str,
    vertices: int,
    *,
    order: int,
    split_ratio: float = 0.5,
    split: bool = True,
    collection=None,
) -> list[Any]:
    if bpy is None or Vector is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")

    side = _bond_side_vector(p1, p2)
    if side is None:
        return []

    order = max(2, int(order))
    if order == 2:
        offset_step = float(radius) * 1.90
        local_radius = float(radius) * 0.54
    elif order == 3:
        offset_step = float(radius) * 2.15
        local_radius = float(radius) * 0.46
    else:
        offset_step = float(radius) * 2.00
        local_radius = float(radius) * 0.42

    v1 = Vector(p1)
    v2 = Vector(p2)
    out: list[Any] = []
    for i, offset in enumerate(_parallel_offsets(order, offset_step)):
        dv = side * float(offset)
        q1 = tuple(v1 + dv)
        q2 = tuple(v2 + dv)
        if split:
            out.extend(
                add_split_cylinders_between(
                    q1,
                    q2,
                    local_radius,
                    mat_left,
                    mat_right,
                    f"{name_prefix}_o{i}",
                    vertices,
                    split_ratio=split_ratio,
                    collection=collection,
                )
            )
        else:
            mat = mat_left if mat_left is not None else mat_right
            obj = add_cylinder_between(
                q1,
                q2,
                local_radius,
                mat,
                f"{name_prefix}_o{i}",
                vertices,
                collection=collection,
            )
            if obj is not None:
                out.append(obj)
    return out
