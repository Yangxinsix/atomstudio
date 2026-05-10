from __future__ import annotations

from typing import Iterable

import numpy as np

from atomstudio.config import RenderJobConfig
from atomstudio.preview.types import (
    AtomPreviewBuffer,
    BondPreviewBuffer,
    CellPreviewBuffer,
    PolyhedraPreviewBuffer,
    PreviewAtomRecord,
    PreviewBondRecord,
    PreviewBounds,
    PreviewMaterialPayload,
    PreviewScene,
    PreviewSelectionTarget,
    PreviewSettings,
)
from atomstudio.scene.builder import build_render_scene
from atomstudio.scene.model import RenderScene as SceneRenderScene
from atomstudio.structure.structure import Structure


def build_preview_scene(
    structure: Structure,
    cfg: RenderJobConfig,
    settings: PreviewSettings | None = None,
) -> PreviewScene:
    scene = build_render_scene(structure, cfg)
    return preview_scene_from_render_scene(scene, settings=settings)


def preview_scene_from_render_scene(
    scene: SceneRenderScene,
    *,
    settings: PreviewSettings | None = None,
) -> PreviewScene:
    resolved_settings = PreviewSettings() if settings is None else settings

    atom_records = tuple(_atom_record(atom) for atom in scene.atoms)
    atom_targets = tuple(_selection_target(atom.selection_payload, kind="atom", index=atom.index) for atom in scene.atoms)

    atom_buffer = _build_atom_buffer(scene, show=bool(resolved_settings.show_atoms))
    visible_bonds = tuple(bond for bond in scene.bonds if bond.visible)
    bond_records = tuple(_bond_record(bond) for bond in visible_bonds)
    bond_targets = tuple(_selection_target(bond.selection_payload, kind="bond", index=bond.id) for bond in visible_bonds)
    bond_buffer = _build_bond_buffer(visible_bonds, atom_lookup={atom.index: atom for atom in scene.atoms}, show=bool(resolved_settings.show_bonds))

    cell_buffer = _build_cell_buffer(scene, show=bool(resolved_settings.show_cell))
    polyhedra_buffer = _build_polyhedra_buffer(scene, show=bool(resolved_settings.show_polyhedra))

    bounds = PreviewBounds(
        minimum=tuple(float(v) for v in scene.bounds.minimum),
        maximum=tuple(float(v) for v in scene.bounds.maximum),
        center=tuple(float(v) for v in scene.bounds.center),
        radius=float(scene.bounds.radius),
    )
    metadata = dict(scene.metadata)
    metadata.setdefault("scene_style", str(scene.style_name))
    metadata.setdefault("color_style", str(scene.color_style_name))
    metadata.setdefault("material_style", str(scene.material_style_name))
    metadata.setdefault("light_style", str(scene.light_style_name))
    metadata.setdefault("source_path", str(scene.structure_source))
    metadata.setdefault("frame_index", int(scene.frame_index))

    report = dict(scene.report)
    report.setdefault("preview_source", "scene_builder")
    report["counts"] = {
        "atoms": atom_buffer.count,
        "bonds": bond_buffer.segment_count,
        "cell_edges": cell_buffer.segment_count,
        "polyhedra": polyhedra_buffer.count,
    }

    padding = max(0.0, float(resolved_settings.fit_padding))
    extent = max(float(scene.bounds.radius) * (1.0 + padding), 1.0)
    return PreviewScene(
        config=scene.config,
        style_name=str(scene.style_name),
        representation=str(scene.representation),
        draw_bonds=bool(scene.draw_bonds and resolved_settings.show_bonds and bond_buffer.segment_count),
        draw_cell=bool(scene.draw_cell and resolved_settings.show_cell and cell_buffer.segment_count),
        space_filling_scale=float(scene.space_filling_scale),
        bounds=bounds,
        atoms=atom_buffer,
        bonds=bond_buffer,
        cell=cell_buffer,
        polyhedra=polyhedra_buffer,
        atom_records=atom_records,
        bond_records=bond_records,
        selection_targets=tuple((*atom_targets, *bond_targets)),
        frame_index=int(scene.frame_index),
        source_path=str(scene.structure_source),
        bounds_min=np.asarray(bounds.minimum, dtype=np.float32),
        bounds_max=np.asarray(bounds.maximum, dtype=np.float32),
        center=np.asarray(bounds.center, dtype=np.float32),
        extent=extent,
        metadata=metadata,
        report=report,
    )


def _build_atom_buffer(scene: SceneRenderScene, *, show: bool) -> AtomPreviewBuffer:
    if not show or not scene.atoms:
        return AtomPreviewBuffer()
    positions = np.asarray([atom.position for atom in scene.atoms], dtype=np.float32).reshape((-1, 3))
    colors = np.asarray([_payload(atom.material).color for atom in scene.atoms], dtype=np.float32).reshape((-1, 4))
    radii = np.asarray([atom.radius for atom in scene.atoms], dtype=np.float32)
    atom_indices = np.asarray([atom.index for atom in scene.atoms], dtype=np.int32)
    atomic_numbers = np.asarray([atom.atomic_number for atom in scene.atoms], dtype=np.int32)
    return AtomPreviewBuffer(
        positions=positions,
        colors=colors,
        radii=radii,
        atom_indices=atom_indices,
        atomic_numbers=atomic_numbers,
        symbols=tuple(atom.symbol for atom in scene.atoms),
        representations=tuple(atom.representation for atom in scene.atoms),
    )


def _build_bond_buffer(
    bonds: tuple,
    *,
    atom_lookup: dict[int, object],
    show: bool,
) -> BondPreviewBuffer:
    if not show or not bonds:
        return BondPreviewBuffer()

    positions: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    colors: list[tuple[tuple[float, float, float, float], tuple[float, float, float, float]]] = []
    bond_ids: list[int] = []
    atom_indices: list[tuple[int, int]] = []
    radii: list[float] = []
    orders: list[int] = []
    split_ratios: list[float] = []
    split_mask: list[bool] = []
    bond_types: list[str] = []

    for bond in bonds:
        atom_a = atom_lookup.get(int(bond.a))
        atom_b = atom_lookup.get(int(bond.b))
        start = tuple(float(v) for v in getattr(atom_a, "position", (0.0, 0.0, 0.0)))
        end = tuple(float(v) for v in getattr(atom_b, "position", (0.0, 0.0, 0.0)))
        if bond.segments:
            start = tuple(float(v) for v in bond.segments[0].start)
            end = tuple(float(v) for v in bond.segments[-1].end)
        uniform = _payload(bond.material_uniform or bond.material_left or bond.material_right)
        left = _payload(bond.material_left or bond.material_uniform or bond.material_right)
        right = _payload(bond.material_right or bond.material_uniform or bond.material_left)
        positions.append((start, end))
        colors.append((left.color, right.color))
        bond_ids.append(int(bond.id))
        atom_indices.append((int(bond.a), int(bond.b)))
        radii.append(float(bond.radius))
        orders.append(max(1, int(bond.order)))
        split_ratios.append(float(bond.split_ratio))
        split_mask.append(bool(left.color != right.color))
        bond_types.append(str(bond.bond_type))

    connect = np.arange(len(positions) * 2, dtype=np.int32).reshape((-1, 2))
    return BondPreviewBuffer(
        positions=np.asarray(positions, dtype=np.float32).reshape((-1, 2, 3)),
        colors=np.asarray(colors, dtype=np.float32).reshape((-1, 2, 4)),
        connect=connect,
        bond_ids=np.asarray(bond_ids, dtype=np.int32),
        atom_indices=np.asarray(atom_indices, dtype=np.int32).reshape((-1, 2)),
        radii=np.asarray(radii, dtype=np.float32),
        orders=np.asarray(orders, dtype=np.int32),
        split_ratios=np.asarray(split_ratios, dtype=np.float32),
        split_mask=np.asarray(split_mask, dtype=np.bool_),
        bond_types=tuple(bond_types),
    )


def _build_cell_buffer(scene: SceneRenderScene, *, show: bool) -> CellPreviewBuffer:
    if not show or not scene.cell_edges:
        return CellPreviewBuffer()
    positions = np.asarray([(edge.start, edge.end) for edge in scene.cell_edges], dtype=np.float32).reshape((-1, 2, 3))
    colors = np.asarray([_payload(edge.material).color for edge in scene.cell_edges], dtype=np.float32).reshape((-1, 4))
    radii = np.asarray([edge.radius for edge in scene.cell_edges], dtype=np.float32)
    connect = np.arange(len(scene.cell_edges) * 2, dtype=np.int32).reshape((-1, 2))
    edge_indices = np.asarray([(edge.index, edge.index) for edge in scene.cell_edges], dtype=np.int32).reshape((-1, 2))
    return CellPreviewBuffer(
        positions=positions,
        colors=colors,
        connect=connect,
        edge_indices=edge_indices,
        radii=radii,
    )


def _build_polyhedra_buffer(scene: SceneRenderScene, *, show: bool) -> PolyhedraPreviewBuffer:
    if not show or not scene.polyhedra:
        return PolyhedraPreviewBuffer()

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    colors: list[tuple[float, float, float, float]] = []
    polyhedron_ids: list[int] = []
    center_indices: list[int] = []
    centers: list[tuple[float, float, float]] = []
    edge_colors: list[tuple[float, float, float, float]] = []
    show_edges: list[bool] = []
    edge_radii: list[float] = []
    vertex_atom_indices: list[int] = []
    vertex_offsets: list[int] = [0]
    face_offsets: list[int] = [0]
    atom_lookup = {atom.index: atom for atom in scene.atoms}

    for poly in scene.polyhedra:
        material = _payload(poly.material)
        edge_material = _payload(poly.edge_material or poly.material)
        vertices.extend(tuple(tuple(float(v) for v in point) for point in poly.vertex_positions))
        face_base = vertex_offsets[-1]
        faces.extend((face[0] + face_base, face[1] + face_base, face[2] + face_base) for face in poly.faces)
        colors.append(material.color)
        polyhedron_ids.append(int(poly.id))
        center_indices.append(int(poly.center))
        center_atom = atom_lookup.get(int(poly.center))
        centers.append(
            tuple(float(v) for v in getattr(center_atom, "position", _centroid(poly.vertex_positions)))
        )
        edge_colors.append(edge_material.color)
        show_edges.append(bool(poly.show_edges))
        edge_radii.append(float(poly.edge_radius or 0.02))
        vertex_atom_indices.extend([-1] * len(poly.vertex_positions))
        vertex_offsets.append(len(vertices))
        face_offsets.append(len(faces))

    return PolyhedraPreviewBuffer(
        vertices=np.asarray(vertices, dtype=np.float32).reshape((-1, 3)),
        faces=np.asarray(faces, dtype=np.int32).reshape((-1, 3)),
        colors=np.asarray(colors, dtype=np.float32).reshape((-1, 4)),
        polyhedron_ids=np.asarray(polyhedron_ids, dtype=np.int32),
        center_indices=np.asarray(center_indices, dtype=np.int32),
        centers=np.asarray(centers, dtype=np.float32).reshape((-1, 3)),
        edge_colors=np.asarray(edge_colors, dtype=np.float32).reshape((-1, 4)),
        show_edges=np.asarray(show_edges, dtype=np.bool_),
        edge_radii=np.asarray(edge_radii, dtype=np.float32),
        vertex_atom_indices=np.asarray(vertex_atom_indices, dtype=np.int32),
        vertex_offsets=np.asarray(vertex_offsets, dtype=np.int32),
        face_offsets=np.asarray(face_offsets, dtype=np.int32),
    )


def _atom_record(atom) -> PreviewAtomRecord:
    return PreviewAtomRecord(
        index=int(atom.index),
        symbol=str(atom.symbol),
        atomic_number=int(atom.atomic_number),
        position=tuple(float(v) for v in atom.position),
        radius=float(atom.radius),
        representation=str(atom.representation),
        style=atom.style,
        tag=str(atom.tag),
        material=_payload(atom.material),
        metadata=dict(atom.metadata),
    )


def _bond_record(bond) -> PreviewBondRecord:
    left = _payload(bond.material_left or bond.material_uniform or bond.material_right)
    right = _payload(bond.material_right or bond.material_uniform or bond.material_left)
    uniform = _payload(bond.material_uniform or bond.material_left or bond.material_right)
    metadata = dict(bond.metadata)
    metadata.setdefault("split_colors", bool(left.color != right.color))
    return PreviewBondRecord(
        id=int(bond.id),
        a=int(bond.a),
        b=int(bond.b),
        bond_type=str(bond.bond_type),
        order=max(1, int(bond.order)),
        distance=float(bond.distance),
        split_ratio=float(bond.split_ratio),
        material_left=left,
        material_right=right,
        material_uniform=uniform,
        metadata=metadata,
    )


def _payload(material) -> PreviewMaterialPayload:
    return PreviewMaterialPayload.from_material_like(material)


def _selection_target(
    payload: dict[str, object],
    *,
    kind: str,
    index: int,
) -> PreviewSelectionTarget:
    metadata = dict(payload)
    label = str(metadata.pop("label", f"{kind.title()} {index}"))
    metadata.setdefault("selection_kind", kind)
    metadata.setdefault("selection_index", int(index))
    return PreviewSelectionTarget(kind=kind, index=int(index), label=label, metadata=metadata)


def _centroid(points: Iterable[tuple[float, float, float]]) -> tuple[float, float, float]:
    coords = np.asarray(list(points), dtype=float).reshape((-1, 3))
    if coords.size == 0:
        return (0.0, 0.0, 0.0)
    center = coords.mean(axis=0)
    return (float(center[0]), float(center[1]), float(center[2]))


__all__ = [
    "build_preview_scene",
    "preview_scene_from_render_scene",
]
