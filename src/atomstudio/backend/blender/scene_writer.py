from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from atomstudio.backend.blender.atom_batch import AtomMeshInstance, build_atom_mesh_batches
from atomstudio.backend.blender.bond_batch import BondMeshSegment, build_bond_mesh_batches
from atomstudio.backend.blender.collections import ensure_collection
from atomstudio.backend.blender.geometry import add_curve_line_between, collect_bbox_points, dashed_line_segments, tuple3
from atomstudio.backend.blender.material_adapter import BlenderMaterialAdapter, scene_value
from atomstudio.config import RenderJobConfig
from atomstudio.scene.materials.registry import MaterialRegistry
from atomstudio.structure.polyhedron import Polyhedron

try:
    import bpy  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    Matrix = None
    Vector = None


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
        self._apply_model_transform(objects, scene_value(scene, "camera"))
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
                    index=index,
                    symbol=str(scene_value(raw_atom, "symbol", "X")),
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
        line_objects: list[Any] = []
        for raw_bond in bonds:
            bond_id = int(scene_value(raw_bond, "id", len(line_objects)))
            is_hydrogen_bond = str(scene_value(raw_bond, "bond_type", "covalent")) == "hydrogen"
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
                start = tuple3(scene_value(raw_segment, "start"))
                end = tuple3(scene_value(raw_segment, "end"))
                radius = float(scene_value(raw_segment, "radius", scene_value(raw_bond, "radius", self.cfg.structure.bond_radius)))
                if is_hydrogen_bond:
                    for dash_index, (dash_start, dash_end) in enumerate(dashed_line_segments(start, end)):
                        obj = add_curve_line_between(
                            dash_start,
                            dash_end,
                            radius,
                            material,
                            f"HydrogenBond_{bond_id}_dash_{dash_index}",
                            collection=self.collections["Bonds"],
                        )
                        if obj is not None:
                            line_objects.append(obj)
                    continue
                segments.append(BondMeshSegment(start=start, end=end, radius=radius, material=material))
        objects = []
        if segments:
            objects.extend(
                build_bond_mesh_batches(
                    segments,
                    vertices=int(self.cfg.structure.bond_vertices),
                    collection=self.collections["Bonds"],
                    name_prefix="Bonds",
                )
            )
        objects.extend(line_objects)
        return objects, len(segments) + len(line_objects)

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
            obj = add_curve_line_between(
                p1,
                p2,
                float(scene_value(raw_edge, "radius", self.cfg.structure.cell_style.radius)),
                material,
                str(scene_value(raw_edge, "name", f"CellEdge_{index}")),
                collection=self.collections["Cell"],
            )
            if obj is not None:
                out.append(obj)
        return out

    def _collect_bbox_points(self, objects: Sequence[Any]) -> list[Any]:
        return collect_bbox_points(list(objects))

    def _apply_model_transform(self, objects: Sequence[Any], camera: Any) -> None:
        if not objects or Matrix is None or Vector is None:
            return
        values = scene_value(camera, "model_rotation")
        translation_values = scene_value(camera, "model_translation")
        matrix = Matrix.Identity(4)
        has_rotation = isinstance(values, (list, tuple)) and len(values) == 16 and not _matrix_is_identity(values)
        if has_rotation:
            try:
                matrix = Matrix(
                    (
                        tuple(float(v) for v in values[0:4]),
                        tuple(float(v) for v in values[4:8]),
                        tuple(float(v) for v in values[8:12]),
                        tuple(float(v) for v in values[12:16]),
                    )
                )
            except Exception:
                matrix = Matrix.Identity(4)
                has_rotation = False
        translation = Vector((0.0, 0.0, 0.0))
        if isinstance(translation_values, (list, tuple)) and len(translation_values) == 3:
            try:
                translation = Vector(tuple(float(v) for v in translation_values))
            except Exception:
                translation = Vector((0.0, 0.0, 0.0))
        has_translation = any(abs(float(v)) > 1.0e-12 for v in translation)
        if not has_rotation and not has_translation:
            return
        center = Vector(tuple3(scene_value(camera, "center")))
        transform = Matrix.Translation(translation) @ Matrix.Translation(center) @ matrix @ Matrix.Translation(-center)
        for obj in objects:
            try:
                obj.matrix_world = transform @ obj.matrix_world
            except Exception:
                continue


def _bond_segment_role(side: Any) -> str:
    value = str(side or "uniform").strip().lower()
    if value == "left":
        return "bond_atom_i"
    if value == "right":
        return "bond_atom_j"
    return "bond_uniform"


def _matrix_is_identity(values: Sequence[Any]) -> bool:
    identity = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    try:
        return all(abs(float(value) - identity[index]) <= 1.0e-8 for index, value in enumerate(values))
    except Exception:
        return True
