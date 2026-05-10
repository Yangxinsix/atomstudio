from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from atomstudio.backend.blender.atom_batch import AtomMeshInstance, build_atom_mesh_batches
from atomstudio.backend.blender.bond_batch import BondMeshSegment, build_bond_mesh_batches
from atomstudio.backend.blender.collections import ensure_collection
from atomstudio.backend.blender.geometry import collect_bbox_points, tuple3
from atomstudio.backend.blender.material_adapter import BlenderMaterialAdapter, scene_value
from atomstudio.config import RenderJobConfig
from atomstudio.scene.materials.registry import MaterialRegistry
from atomstudio.structure.bond import add_cylinder_between
from atomstudio.structure.polyhedron import Polyhedron

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


class BlenderSceneWriter:
    def __init__(
        self,
        cfg: RenderJobConfig,
        *,
        registry: MaterialRegistry,
        default_material_pipeline: str,
        default_material_style_name: str,
    ) -> None:
        self.cfg = cfg
        self.registry = registry
        self.default_material_pipeline = default_material_pipeline
        self.default_material_style_name = default_material_style_name
        self.materials = BlenderMaterialAdapter(registry)
        self.collections = {
            "Atoms": ensure_collection("Atoms"),
            "Bonds": ensure_collection("Bonds"),
            "Polyhedra": ensure_collection("Polyhedra"),
            "PolyhedraEdges": ensure_collection("PolyhedraEdges"),
            "Cell": ensure_collection("Cell"),
        }

    def clear_scene(self) -> None:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

    def write(self, scene: Any) -> tuple[list[Any], dict[str, int], list[Any]]:
        self.clear_scene()
        atom_objects, positions, atom_count = self._write_atoms(scene_value(scene, "atoms", []))
        bond_objects, bond_segment_count = self._write_bonds(scene_value(scene, "bonds", []))
        poly_objects = self._write_polyhedra(scene_value(scene, "polyhedra", []))
        cell_objects = self._write_cell_edges(scene_value(scene, "cell_edges", []))
        objects = [*atom_objects, *bond_objects, *poly_objects, *cell_objects]
        points = self._collect_bbox_points(objects)
        stats = {
            "atoms": int(atom_count),
            "atom_objects": len(atom_objects),
            "bonds": len(scene_value(scene, "bonds", [])),
            "bond_objects": len(bond_objects),
            "bond_segments": int(bond_segment_count),
            "polyhedra": len(scene_value(scene, "polyhedra", [])),
            "cell_edges": len(cell_objects),
        }
        return objects, stats, points

    def _write_atoms(self, atoms: Sequence[Any]) -> tuple[list[Any], list[tuple[float, float, float]], int]:
        instances: list[AtomMeshInstance] = []
        max_index = -1
        positions_by_index: dict[int, tuple[float, float, float]] = {}
        for raw_atom in atoms:
            index = int(scene_value(raw_atom, "index", 0))
            max_index = max(max_index, index)
            position = tuple3(scene_value(raw_atom, "position"))
            radius = float(scene_value(raw_atom, "radius", 0.5))
            segments = int(scene_value(raw_atom, "segments", self.cfg.structure.sphere_segments))
            rings = int(scene_value(raw_atom, "rings", self.cfg.structure.sphere_rings))
            material = self.materials.resolve(
                scene_value(raw_atom, "material"),
                name=f"AtomBatch_{str(scene_value(raw_atom, 'symbol', 'X'))}",
                role="atom",
                pipeline=self.default_material_pipeline,
                style_name=self.default_material_style_name,
            )
            instances.append(
                AtomMeshInstance(
                    position=position,
                    radius=radius,
                    material=material,
                    segments=segments,
                    rings=rings,
                )
            )
            positions_by_index[index] = position

        positions = [(0.0, 0.0, 0.0)] * max(0, max_index + 1)
        for index, position in positions_by_index.items():
            if index >= len(positions):
                positions.extend([(0.0, 0.0, 0.0)] * (index - len(positions) + 1))
            positions[index] = position
        objects = build_atom_mesh_batches(instances, collection=self.collections["Atoms"], name_prefix="Atoms")
        return objects, positions, len(atoms)

    def _write_bonds(self, bonds: Sequence[Any]) -> tuple[list[Any], int]:
        segments: list[BondMeshSegment] = []
        for raw_bond in bonds:
            pipeline = str(scene_value(raw_bond, "pipeline", self.default_material_pipeline))
            style_name = str(scene_value(raw_bond, "style_name", self.default_material_style_name))
            for raw_segment in scene_value(raw_bond, "segments", []) or []:
                role = _bond_segment_role(scene_value(raw_segment, "side", "uniform"))
                material = self.materials.resolve(
                    scene_value(raw_segment, "material"),
                    name=f"BondBatch_{role}",
                    role=role,
                    pipeline=pipeline,
                    style_name=style_name,
                )
                segments.append(
                    BondMeshSegment(
                        start=tuple3(scene_value(raw_segment, "start")),
                        end=tuple3(scene_value(raw_segment, "end")),
                        radius=float(scene_value(raw_segment, "radius", scene_value(raw_bond, "radius", self.cfg.structure.bond_radius))),
                        material=material,
                    )
                )
        objects = build_bond_mesh_batches(
            segments,
            vertices=int(self.cfg.structure.bond_vertices),
            collection=self.collections["Bonds"],
            name_prefix="Bonds",
        )
        return objects, len(segments)

    def _write_polyhedra(self, polyhedra: Sequence[Any]) -> list[Any]:
        out: list[Any] = []
        for raw_poly in polyhedra:
            poly = Polyhedron(
                id=int(scene_value(raw_poly, "id", 0)),
                center=int(scene_value(raw_poly, "center", 0)),
                center_symbol=str(scene_value(raw_poly, "center_symbol", "X")),
                vertex_positions=[tuple3(v) for v in scene_value(raw_poly, "vertex_positions", [])],
                neighbor_indices=[int(v) for v in scene_value(raw_poly, "neighbor_indices", [])],
                neighbor_offsets=[tuple(int(x) for x in item[:3]) for item in scene_value(raw_poly, "neighbor_offsets", [])],
                metadata=dict(scene_value(raw_poly, "metadata", {}) or {}),
                style=scene_value(raw_poly, "style"),
                show_edges=bool(scene_value(raw_poly, "show_edges", False)),
                edge_radius=float(scene_value(raw_poly, "edge_radius", self.cfg.structure.polyhedra.default_edge_radius)),
            )
            face_material = self.materials.resolve(
                scene_value(raw_poly, "material"),
                name=str(scene_value(raw_poly, "name", f"Polyhedron_{poly.id}_{poly.center_symbol}_{poly.center}")),
                role="polyhedra_face",
                pipeline=str(scene_value(raw_poly, "pipeline", self.default_material_pipeline)),
                style_name=str(scene_value(raw_poly, "style_name", self.default_material_style_name)),
            )
            edge_material = self.materials.resolve(
                scene_value(raw_poly, "edge_material"),
                name=f"{scene_value(raw_poly, 'name', f'Polyhedron_{poly.id}')}_edge",
                role="polyhedra_edge",
                pipeline=str(scene_value(raw_poly, "pipeline", self.default_material_pipeline)),
                style_name=str(scene_value(raw_poly, "style_name", self.default_material_style_name)),
            )
            out.extend(
                Polyhedron.build(
                    poly,
                    material=face_material,
                    edge_material=edge_material,
                    default_edge_radius=float(scene_value(raw_poly, "edge_radius", self.cfg.structure.polyhedra.default_edge_radius)),
                    collection=self.collections["Polyhedra"],
                    edge_collection=self.collections["PolyhedraEdges"],
                    name=str(scene_value(raw_poly, "name", f"Polyhedron_{poly.id}_{poly.center_symbol}_{poly.center}")),
                )
            )
        return out

    def _write_cell_edges(self, edges: Sequence[Any]) -> list[Any]:
        out: list[Any] = []
        for index, raw_edge in enumerate(edges):
            p1 = tuple3(scene_value(raw_edge, "start"))
            p2 = tuple3(scene_value(raw_edge, "end"))
            material = self.materials.resolve(
                scene_value(raw_edge, "material"),
                name=str(scene_value(raw_edge, "name", f"CellEdge_{index}")),
                role="cell",
                pipeline=str(scene_value(raw_edge, "pipeline", self.default_material_pipeline)),
                style_name=str(scene_value(raw_edge, "style_name", self.default_material_style_name)),
            )
            obj = add_cylinder_between(
                p1,
                p2,
                float(scene_value(raw_edge, "radius", self.cfg.structure.cell_style.radius)),
                material,
                str(scene_value(raw_edge, "name", f"CellEdge_{index}")),
                int(scene_value(raw_edge, "vertices", 12)),
                collection=self.collections["Cell"],
            )
            if obj is not None:
                out.append(obj)
        return out

    def _collect_bbox_points(self, objects: Sequence[Any]) -> list[Any]:
        return collect_bbox_points(list(objects))


def _bond_segment_role(side: Any) -> str:
    value = str(side or "uniform").strip().lower()
    if value == "left":
        return "bond_atom_i"
    if value == "right":
        return "bond_atom_j"
    return "bond_uniform"
