from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from atomstudio.color_utils import coerce_color_fields
from atomstudio.scene.materials.specs import MaterialLike, MaterialSpec
from atomstudio.structure.bond import add_cylinder_between

try:
    import bmesh  # type: ignore
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bmesh = None
    bpy = None


@coerce_color_fields("color", "edge_color", label_prefix="polyhedron")
@dataclass
class Polyhedron:
    id: int
    center: int
    center_symbol: str
    vertex_positions: list[tuple[float, float, float]] = field(default_factory=list)
    neighbor_indices: list[int] = field(default_factory=list)
    neighbor_offsets: list[tuple[int, int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    style: str | None = None
    material: MaterialLike | None = None
    color: tuple[float, float, float, float] | None = None
    show_edges: bool = False
    edge_radius: float | None = None
    edge_color: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        self.sync_color_to_material()

    def sync_color_to_material(self) -> None:
        if self.color is None:
            return
        if self.material is None:
            self.material = MaterialSpec(color=self.color)
            return
        self.material = replace(self.material, color=self.color)

    @classmethod
    def build(
        cls,
        polyhedron: "Polyhedron",
        *,
        material: Any = None,
        edge_material: Any = None,
        default_edge_radius: float = 0.015,
        collection: Any = None,
        edge_collection: Any = None,
        name: str | None = None,
    ) -> list[Any]:
        if bpy is None or bmesh is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")
        if len(polyhedron.vertex_positions) < 4:
            return []

        mesh_name = name or f"Polyhedron_{polyhedron.id}_{polyhedron.center_symbol}_{polyhedron.center}"
        mesh = bpy.data.meshes.new(mesh_name)
        bm = bmesh.new()
        verts = [bm.verts.new((float(p[0]), float(p[1]), float(p[2]))) for p in polyhedron.vertex_positions]
        bm.verts.ensure_lookup_table()

        try:
            hull = bmesh.ops.convex_hull(bm, input=verts, use_existing_faces=False)
        except Exception:
            bm.free()
            bpy.data.meshes.remove(mesh, do_unlink=True)
            return []

        delete_geom = []
        for key in ("geom_interior", "geom_unused"):
            delete_geom.extend(hull.get(key, []))
        delete_verts = [g for g in delete_geom if isinstance(g, bmesh.types.BMVert)]
        if delete_verts:
            bmesh.ops.delete(bm, geom=delete_verts, context="VERTS")

        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(mesh_name, mesh)
        for poly in obj.data.polygons:
            poly.use_smooth = True

        if material is not None:
            if len(obj.data.materials) == 0:
                obj.data.materials.append(material)
            else:
                obj.data.materials[0] = material

        if collection is not None:
            for c in list(obj.users_collection):
                c.objects.unlink(obj)
            collection.objects.link(obj)
        else:
            bpy.context.scene.collection.objects.link(obj)

        out: list[Any] = [obj]
        if not polyhedron.show_edges:
            return out

        edge_radius = max(1e-5, float(polyhedron.edge_radius if polyhedron.edge_radius is not None else default_edge_radius))
        for i, edge in enumerate(mesh.edges):
            p1 = tuple(float(v) for v in mesh.vertices[int(edge.vertices[0])].co[:])
            p2 = tuple(float(v) for v in mesh.vertices[int(edge.vertices[1])].co[:])
            edge_obj = add_cylinder_between(
                p1,
                p2,
                edge_radius,
                edge_material,
                f"{mesh_name}_edge_{i}",
                12,
                collection=edge_collection,
            )
            if edge_obj is not None:
                out.append(edge_obj)
        return out
