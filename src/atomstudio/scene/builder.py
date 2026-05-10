from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np

from atomstudio.config import RenderJobConfig
from atomstudio.render.space_filling import resolve_auto_space_filling_scale
from atomstudio.scene.camera_resolver import resolve_scene_camera
from atomstudio.scene.light_resolver import resolve_scene_lights
from atomstudio.scene.model import (
    RenderScene,
    SceneAtom,
    SceneBond,
    SceneBondSegment,
    SceneCellEdge,
    ScenePolyhedron,
)
from atomstudio.scene.styling import (
    resolve_atom_scene_styles,
    resolve_bond_scene_styles,
    resolve_cell_material,
    resolve_draw_bonds_for_scene,
    resolve_polyhedron_materials,
    resolve_scene_representation,
    resolve_scene_style_bundle,
)
from atomstudio.scene.transforms import apply_boundary_expansion, apply_model_rotation, compute_bounds
from atomstudio.structure.cell import cell_edges, has_cell
from atomstudio.structure.structure import Structure


@dataclass(slots=True)
class SceneBuildResult:
    scene: RenderScene
    structure: Structure


class SceneBuilder:
    def __init__(self, cfg: RenderJobConfig) -> None:
        self.cfg = cfg

    @classmethod
    def from_cfg(cls, cfg: RenderJobConfig) -> "SceneBuilder":
        return cls(cfg)

    def build(self, structure: Structure) -> RenderScene:
        return self.build_with_structure(structure).scene

    def build_with_structure(self, structure: Structure) -> SceneBuildResult:
        working = Structure.from_dict(structure.to_dict())
        effective_cfg, report = resolve_auto_space_filling_scale(working, self.cfg)
        working = apply_boundary_expansion(working, effective_cfg)
        working = apply_model_rotation(
            working,
            model_rotation=effective_cfg.structure.model_rotation,
            model_view=effective_cfg.structure.model_view,
        )
        if effective_cfg.structure.polyhedra.enabled and effective_cfg.structure.polyhedra.rules and not working.polyhedra:
            working.ensure_polyhedra(effective_cfg.structure.polyhedra, effective_cfg.structure.bonding)

        style_bundle = resolve_scene_style_bundle(effective_cfg)
        representation = resolve_scene_representation(effective_cfg, style_bundle)
        atom_styles, atom_representations, space_filling_scale = resolve_atom_scene_styles(
            working,
            effective_cfg,
            style_bundle=style_bundle,
            representation=representation,
        )
        draw_bonds = resolve_draw_bonds_for_scene(
            effective_cfg,
            representation=representation,
            atom_representations=atom_representations,
        )
        if draw_bonds and not working.bonds:
            working.ensure_bonds(effective_cfg.structure.bonding)

        bond_styles = resolve_bond_scene_styles(
            working,
            effective_cfg,
            style_bundle=style_bundle,
            atom_styles=atom_styles,
            atom_representations=atom_representations,
            base_representation=representation,
            draw_bonds=draw_bonds,
        )

        atoms = tuple(_build_scene_atom(item) for item in atom_styles)
        bonds = tuple(_build_scene_bond(item, atom_lookup={atom.index: atom for atom in atoms}) for item in bond_styles)
        cell = tuple(_build_scene_cell_edges(working, effective_cfg, style_bundle=style_bundle))
        polyhedra = tuple(_build_scene_polyhedra(working, effective_cfg, style_bundle=style_bundle))

        if effective_cfg.style.background is not None:
            background = effective_cfg.style.background
        elif effective_cfg.style.handdrawn is not None and effective_cfg.style.handdrawn.background is not None:
            background = effective_cfg.style.handdrawn.background
        else:
            background = style_bundle.background
        scene_points = list(_collect_scene_points(atoms, bonds, cell, polyhedra))
        bounds = compute_bounds(scene_points)
        camera = resolve_scene_camera(effective_cfg, bounds=bounds, points=scene_points)
        lights = tuple(resolve_scene_lights(effective_cfg, bounds=bounds, default_light_style=style_bundle.light_style_name))

        report = dict(report)
        report.setdefault("space_filling_scale", {})
        if isinstance(report["space_filling_scale"], dict):
            report["space_filling_scale"].setdefault("applied", float(space_filling_scale))
        report["counts"] = {
            "atoms": len(atoms),
            "bonds": len(bonds),
            "visible_bonds": sum(1 for bond in bonds if bond.visible),
            "cell_edges": len(cell),
            "polyhedra": len(polyhedra),
        }

        metadata = dict(working.metadata)
        metadata.setdefault("scene_style", style_bundle.scene_style_name)
        metadata["color_style"] = style_bundle.color_style_name
        metadata["material_style"] = style_bundle.material_style_name
        metadata["light_style"] = style_bundle.light_style_name

        scene = RenderScene(
            config=effective_cfg,
            structure_source=str(working.source_path),
            frame_index=int(working.frame_index),
            representation=representation,
            draw_bonds=bool(draw_bonds),
            draw_cell=bool(cell),
            atoms=atoms,
            bonds=bonds,
            polyhedra=polyhedra,
            cell_edges=cell,
            camera=camera,
            lights=lights,
            background=tuple(float(v) for v in background),
            bounds=bounds,
            metadata=metadata,
            report=report,
            space_filling_scale=float(space_filling_scale),
            style_name=style_bundle.scene_style_name,
            color_style_name=style_bundle.color_style_name,
            material_style_name=style_bundle.material_style_name,
            light_style_name=style_bundle.light_style_name,
        )
        return SceneBuildResult(scene=scene, structure=working)


def build_render_scene(structure: Structure, cfg: RenderJobConfig) -> RenderScene:
    return SceneBuilder.from_cfg(cfg).build(structure)


def _build_scene_atom(item) -> SceneAtom:
    return SceneAtom(
        index=int(item.atom_index),
        symbol=str(item.symbol),
        atomic_number=int(item.atomic_number),
        position=tuple(float(v) for v in item.position),
        radius=float(item.radius),
        representation=str(item.representation),
        material=item.material,
        selection_payload={
            "kind": "atom",
            "index": int(item.atom_index),
            "label": f"Atom {item.atom_index} {item.symbol}",
            "symbol": str(item.symbol),
            "atomic_number": int(item.atomic_number),
            "position": [float(v) for v in item.position],
            "tag": str(item.tag),
        },
        metadata=dict(item.metadata),
        style=item.style_name,
        tag=str(item.tag),
    )


def _build_scene_bond(item, *, atom_lookup: dict[int, SceneAtom]) -> SceneBond:
    atom_a = atom_lookup.get(int(item.a))
    atom_b = atom_lookup.get(int(item.b))
    p1 = atom_a.position if atom_a is not None else (0.0, 0.0, 0.0)
    p2 = atom_b.position if atom_b is not None else (0.0, 0.0, 0.0)
    segments = _build_bond_segments(
        p1,
        p2,
        radius=float(item.radius),
        order=max(1, int(item.order)),
        split_ratio=float(item.split_ratio),
        split_colors=bool(item.split_colors),
        material_uniform=item.material_uniform,
        material_left=item.material_left,
        material_right=item.material_right,
        visible=bool(item.visible),
    )
    return SceneBond(
        id=int(item.bond_id),
        a=int(item.a),
        b=int(item.b),
        order=max(1, int(item.order)),
        bond_type=str(item.bond_type),
        distance=float(item.distance),
        radius=float(item.radius),
        segments=tuple(segments),
        material_uniform=item.material_uniform,
        material_left=item.material_left,
        material_right=item.material_right,
        selection_payload={
            "kind": "bond",
            "index": int(item.bond_id),
            "label": f"Bond {item.bond_id} {item.a}-{item.b}",
            "atom_indices": [int(item.a), int(item.b)],
            "bond_type": str(item.bond_type),
            "order": max(1, int(item.order)),
            "distance": float(item.distance),
        },
        metadata=dict(item.metadata),
        split_ratio=float(item.split_ratio),
        visible=bool(item.visible),
    )


def _build_bond_segments(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    *,
    radius: float,
    order: int,
    split_ratio: float,
    split_colors: bool,
    material_uniform,
    material_left,
    material_right,
    visible: bool,
) -> list[SceneBondSegment]:
    if not visible:
        return []
    side = _bond_side_vector(p1, p2)
    if side is None:
        return []
    order = max(1, int(order))
    if order == 1:
        offsets = [0.0]
        local_radius = float(radius)
    elif order == 2:
        offsets = _parallel_offsets(order, float(radius) * 1.90)
        local_radius = float(radius) * 0.54
    elif order == 3:
        offsets = _parallel_offsets(order, float(radius) * 2.15)
        local_radius = float(radius) * 0.46
    else:
        offsets = _parallel_offsets(order, float(radius) * 2.00)
        local_radius = float(radius) * 0.42

    start = np.asarray(p1, dtype=float)
    end = np.asarray(p2, dtype=float)
    out: list[SceneBondSegment] = []
    for order_index, offset in enumerate(offsets):
        dv = side * float(offset)
        q1 = start + dv
        q2 = end + dv
        if split_colors:
            ratio = min(0.95, max(0.05, float(split_ratio)))
            mid = q1 + (q2 - q1) * ratio
            out.append(
                SceneBondSegment(
                    start=(float(q1[0]), float(q1[1]), float(q1[2])),
                    end=(float(mid[0]), float(mid[1]), float(mid[2])),
                    radius=float(local_radius),
                    material=material_left,
                    order_index=int(order_index),
                    side="left",
                )
            )
            out.append(
                SceneBondSegment(
                    start=(float(mid[0]), float(mid[1]), float(mid[2])),
                    end=(float(q2[0]), float(q2[1]), float(q2[2])),
                    radius=float(local_radius),
                    material=material_right,
                    order_index=int(order_index),
                    side="right",
                )
            )
            continue
        out.append(
            SceneBondSegment(
                start=(float(q1[0]), float(q1[1]), float(q1[2])),
                end=(float(q2[0]), float(q2[1]), float(q2[2])),
                radius=float(local_radius),
                material=material_uniform,
                order_index=int(order_index),
                side="uniform",
            )
        )
    return out


def _build_scene_cell_edges(
    structure: Structure,
    cfg: RenderJobConfig,
    *,
    style_bundle,
) -> Iterable[SceneCellEdge]:
    show = bool(cfg.structure.draw_cell or cfg.structure.cell_style.show)
    if not show or not any(structure.pbc) or not has_cell(structure.cell_vectors):
        return ()
    radius = max(0.01, float(cfg.structure.cell_style.radius))
    material = resolve_cell_material(structure, cfg, style_bundle=style_bundle)
    out: list[SceneCellEdge] = []
    for idx, (start, end) in enumerate(cell_edges(structure.cell_vectors)):
        out.append(
            SceneCellEdge(
                index=idx,
                start=tuple(float(v) for v in start),
                end=tuple(float(v) for v in end),
                radius=radius,
                material=material,
                metadata={"pbc": list(bool(v) for v in structure.pbc)},
            )
        )
    return tuple(out)


def _build_scene_polyhedra(
    structure: Structure,
    cfg: RenderJobConfig,
    *,
    style_bundle,
) -> Iterable[ScenePolyhedron]:
    if not structure.polyhedra:
        return ()
    out: list[ScenePolyhedron] = []
    default_edge_radius = max(1e-5, float(cfg.structure.polyhedra.default_edge_radius))
    for poly in structure.polyhedra:
        faces = _triangulate_polyhedron(poly.vertex_positions)
        if not faces:
            continue
        face_material, edge_material = resolve_polyhedron_materials(poly, cfg, style_bundle=style_bundle)
        out.append(
            ScenePolyhedron(
                id=int(poly.id),
                center=int(poly.center),
                center_symbol=str(poly.center_symbol),
                vertex_positions=tuple(tuple(float(v) for v in point) for point in poly.vertex_positions),
                faces=tuple(faces),
                material=face_material,
                edge_material=edge_material,
                selection_payload={
                    "kind": "polyhedron",
                    "index": int(poly.id),
                    "center": int(poly.center),
                    "center_symbol": str(poly.center_symbol),
                },
                metadata=dict(getattr(poly, "metadata", {}) or {}),
                show_edges=bool(poly.show_edges),
                edge_radius=float(poly.edge_radius if poly.edge_radius is not None else default_edge_radius),
            )
        )
    return tuple(out)


def _triangulate_polyhedron(
    vertices: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
) -> list[tuple[int, int, int]]:
    if len(vertices) < 4:
        return []
    points = np.asarray(vertices, dtype=float).reshape((-1, 3))
    try:
        from scipy.spatial import ConvexHull, QhullError

        hull = ConvexHull(points)
    except ImportError:
        return []
    except QhullError:
        return []
    return [tuple(int(v) for v in simplex) for simplex in hull.simplices]


def _collect_scene_points(
    atoms: tuple[SceneAtom, ...],
    bonds: tuple[SceneBond, ...],
    cell_edges: tuple[SceneCellEdge, ...],
    polyhedra: tuple[ScenePolyhedron, ...],
) -> Iterable[tuple[float, float, float]]:
    for atom in atoms:
        yield atom.position
    for bond in bonds:
        for segment in bond.segments:
            yield segment.start
            yield segment.end
    for edge in cell_edges:
        yield edge.start
        yield edge.end
    for poly in polyhedra:
        yield from poly.vertex_positions


def _bond_side_vector(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> np.ndarray | None:
    direction = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return None
    direction = direction / length
    ref = np.asarray((0.0, 0.0, 1.0), dtype=float)
    if abs(float(np.dot(direction, ref))) > 0.95:
        ref = np.asarray((0.0, 1.0, 0.0), dtype=float)
    if abs(float(np.dot(direction, ref))) > 0.95:
        ref = np.asarray((1.0, 0.0, 0.0), dtype=float)
    side = np.cross(direction, ref)
    side_len = float(np.linalg.norm(side))
    if side_len < 1e-8:
        return None
    return side / side_len


def _parallel_offsets(order: int, spacing: float) -> list[float]:
    if order == 2:
        return [-0.5 * spacing, 0.5 * spacing]
    if order == 3:
        return [-1.0 * spacing, 0.0, 1.0 * spacing]
    center = (order - 1) * 0.5
    return [(i - center) * spacing for i in range(order)]


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
