from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from atomstudio.color_utils import coerce_color_fields
from atomstudio.backend.blender.atom_batch import sphere_subdivisions, unit_icosphere_geometry
from atomstudio.scene.materials.specs import MaterialLike
from atomstudio.style.outline_style import OutlineRoleStyle

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


_SPHERE_MESH_CACHE: dict[tuple[int, str | None], str] = {}


_ATOMIC_NUMBER_FALLBACK = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "Mg": 12,
    "Si": 14,
}


@coerce_color_fields("color", label_prefix="atom")
@dataclass
class Atom:
    index: int
    atomic_number: int
    symbol: str
    position: tuple[float, float, float]
    radius: float | None = None
    segments: int | None = None
    rings: int | None = None
    material: MaterialLike | None = None
    color: tuple[float, float, float, float] | None = None
    style: str | None = None
    representation: str | None = None
    outline: OutlineRoleStyle = field(default_factory=OutlineRoleStyle)
    tag: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sync_color_to_material()

    @property
    def id(self) -> int:
        return self.index

    @property
    def element(self) -> str:
        return self.symbol

    @classmethod
    def from_basic(
        cls,
        *,
        index: int,
        symbol: str,
        position: tuple[float, float, float],
        atomic_number: int | None = None,
    ) -> "Atom":
        return cls(
            index=int(index),
            atomic_number=_infer_atomic_number(symbol, atomic_number),
            symbol=str(symbol),
            position=(float(position[0]), float(position[1]), float(position[2])),
        )

    @classmethod
    def build(
        cls,
        atom: "Atom",
        *,
        material: Any = None,
        collection: Any = None,
        name: str | None = None,
    ) -> Any:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")

        segments = max(8, int(atom.segments if atom.segments is not None else 32))
        rings = max(4, int(atom.rings if atom.rings is not None else 16))
        radius = float(atom.radius if atom.radius is not None else 0.5)
        mesh = _cached_icosphere_mesh(
            subdivisions=sphere_subdivisions(segments=segments, rings=rings),
            material=material,
        )
        obj = bpy.data.objects.new(name or f"Atom_{atom.index}_{atom.symbol}", mesh)
        obj.location = atom.position
        obj.scale = (radius, radius, radius)
        _link_object(obj, collection=collection)
        return obj

    def sync_color_to_material(self) -> None:
        if self.color is None or self.material is None:
            return
        self.material = replace(self.material, color=self.color)


def infer_atomic_number(symbol: str, raw_atomic_number: Any = None) -> int:
    return _infer_atomic_number(symbol, raw_atomic_number)


def _infer_atomic_number(symbol: str, raw_atomic_number: Any) -> int:
    if raw_atomic_number is not None:
        try:
            return int(raw_atomic_number)
        except Exception:
            pass
    return int(_ATOMIC_NUMBER_FALLBACK.get(str(symbol), 0))


def _cached_icosphere_mesh(*, subdivisions: int, material: Any) -> Any:
    if bpy is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")

    key = (int(subdivisions), _material_cache_key(material))
    mesh_name = _SPHERE_MESH_CACHE.get(key)
    mesh = bpy.data.meshes.get(mesh_name) if mesh_name is not None else None
    if mesh is not None:
        return mesh

    vertices, faces = unit_icosphere_geometry(int(subdivisions))
    mesh = bpy.data.meshes.new(f"AtomIcosphere_s{subdivisions}")
    mesh.from_pydata(list(vertices), [], list(faces))
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    if material is not None:
        if len(mesh.materials) == 0:
            mesh.materials.append(material)
        else:
            mesh.materials[0] = material
    _SPHERE_MESH_CACHE[key] = str(mesh.name)
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
