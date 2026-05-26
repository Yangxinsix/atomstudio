from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from atomstudio.config import RenderJobConfig
from atomstudio.preview.builder import build_preview_scene as build_buffer_preview_scene
from atomstudio.preview.interaction import HitTestCache, SelectionController
from atomstudio.preview.material_adapter import mix_rgba, preview_material_payload, resolve_render_mode
from atomstudio.preview.mesh_builder import (
    atom_draw_data,
    bond_draw_data,
    build_bond_segments,
    polyhedron_edge_segments,
    triangulate_polyhedron,
)
from atomstudio.preview.picking import (
    atom_indices_in_rect,
    bond_indices_in_rect,
    pick_bond_selection,
    pick_selection,
    project_atom_positions,
    rotation_basis,
)
from atomstudio.preview.selection import cycle_atom_selection, cycle_bond_selection, selection_payload
from atomstudio.preview.types import (
    PreviewAtomRecord,
    PreviewBondRecord,
    PreviewScene,
    PreviewSelection,
    PreviewSettings as BufferPreviewSettings,
    RenderAtom,
    RenderBond,
    RenderCellEdge,
    RenderLineSegment,
    RenderPolyhedron,
    PreviewRenderScene,
)
from atomstudio.structure.structure import Structure


@dataclass(frozen=True)
class PreviewSettings:
    background: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    default_view: str = "top"
    fit_padding: float = 0.14
    atom_scale: float = 34.0
    selected_atom_scale: float = 1.04
    bond_scale: float = 18.0
    cell_scale: float = 10.0
    polyhedron_alpha: float = 0.28
    picking_radius_px: float = 20.0
    show_atoms: bool = True
    show_bonds: bool = True
    show_cell: bool = True
    show_polyhedra: bool = True
    highlight_color: tuple[float, float, float, float] = (1.0, 0.68, 0.12, 1.0)


@dataclass
class PreviewCameraState:
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_factor: float = 1.0
    view: str = "top"
    azimuth: float = 0.0
    elevation: float = 90.0
    roll: float = 0.0
    right: tuple[float, float, float] = (1.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    forward: tuple[float, float, float] = (0.0, 0.0, -1.0)
    model_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    model_rotation: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    def basis(self) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        return self.right, self.up, self.forward


class CallbackSignal:
    def __init__(self) -> None:
        self._callbacks: list[Callable[[PreviewSelection | None], None]] = []

    def connect(self, callback: Callable[[PreviewSelection | None], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, value: PreviewSelection | None) -> None:
        for callback in list(self._callbacks):
            callback(value)


def _shared_preview_settings(settings: PreviewSettings) -> BufferPreviewSettings:
    return BufferPreviewSettings(
        show_atoms=bool(settings.show_atoms),
        show_bonds=bool(settings.show_bonds),
        show_cell=bool(settings.show_cell),
        show_polyhedra=bool(settings.show_polyhedra),
        fit_padding=float(settings.fit_padding),
    )


def _buffer_bounds(scene: PreviewScene) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
    if scene.bounds is not None:
        return (
            tuple(float(v) for v in scene.bounds.minimum),
            tuple(float(v) for v in scene.bounds.maximum),
            tuple(float(v) for v in scene.bounds.center),
            max(1.0, float(scene.bounds.radius)),
        )
    return (
        tuple(float(v) for v in np.asarray(scene.bounds_min, dtype=float)),
        tuple(float(v) for v in np.asarray(scene.bounds_max, dtype=float)),
        tuple(float(v) for v in np.asarray(scene.center, dtype=float)),
        max(1.0, float(scene.extent)),
    )


def _scene_source(scene: PreviewRenderScene | PreviewScene | None) -> str:
    if scene is None:
        return ""
    source = getattr(scene, "source_path", "")
    if source:
        return str(source)
    report = getattr(scene, "report", None)
    if isinstance(report, dict):
        source = report.get("source_path")
        if source:
            return str(source)
    return ""


def _split_segment(
    segment: RenderLineSegment,
    *,
    count: int,
    duty_cycle: float,
    gradient: tuple[float, float, float, float] | None = None,
) -> tuple[RenderLineSegment, ...]:
    start = np.asarray(segment.start, dtype=float)
    end = np.asarray(segment.end, dtype=float)
    count = max(1, int(count))
    duty_cycle = max(0.05, min(1.0, float(duty_cycle)))
    pieces: list[RenderLineSegment] = []
    for idx in range(count):
        t0 = idx / count
        t1 = min(1.0, t0 + duty_cycle / count)
        color = segment.color if gradient is None else mix_rgba(segment.color, gradient, (t0 + t1) * 0.5)
        pieces.append(
            RenderLineSegment(
                start=tuple(float(v) for v in start + (end - start) * t0),
                end=tuple(float(v) for v in start + (end - start) * t1),
                color=color,
                width_px=float(segment.width_px),
            )
        )
    return tuple(pieces)


def _apply_preview_bond_style(
    segments: tuple[RenderLineSegment, ...],
    style: str,
    *,
    right_color: tuple[float, float, float, float],
) -> tuple[RenderLineSegment, ...]:
    style = str(style or "bicolor").strip().lower()
    if style not in {"dashed", "dotted", "gradient"}:
        return segments
    styled: list[RenderLineSegment] = []
    for segment in segments:
        length = float(np.linalg.norm(np.asarray(segment.end, dtype=float) - np.asarray(segment.start, dtype=float)))
        count = max(4, int(length / 0.18))
        if style == "dashed":
            styled.extend(_split_segment(segment, count=count, duty_cycle=0.58))
        elif style == "dotted":
            styled.extend(_split_segment(segment, count=max(6, count * 2), duty_cycle=0.22))
        else:
            styled.extend(_split_segment(segment, count=max(8, count * 2), duty_cycle=1.0, gradient=right_color))
    return tuple(styled)


def render_scene_from_preview_scene(scene: PreviewScene, settings: PreviewSettings) -> PreviewRenderScene:
    atoms: list[RenderAtom] = []
    atom_indices = np.asarray(scene.atoms.indices, dtype=np.int32)
    atom_positions = np.asarray(scene.atoms.positions, dtype=float)
    atom_colors = np.asarray(scene.atoms.colors, dtype=float)
    atom_radii = np.asarray(scene.atoms.radii, dtype=float)
    atom_records = tuple(getattr(scene, "atom_records", ()) or ())
    atom_record_map = {int(record.index): record for record in atom_records}
    for idx in range(scene.atoms.count):
        atom_index = int(atom_indices[idx]) if idx < atom_indices.shape[0] else idx
        radius = float(atom_radii[idx]) if idx < atom_radii.shape[0] else 0.2
        ordinal_record = atom_records[idx] if idx < len(atom_records) else None
        record = (
            ordinal_record
            if ordinal_record is not None and int(ordinal_record.index) == atom_index
            else atom_record_map.get(atom_index)
        )
        material = preview_material_payload(record.material) if record is not None else None
        color = tuple(float(v) for v in atom_colors[idx]) if idx < atom_colors.shape[0] else (
            tuple(float(v) for v in material.color) if material is not None else (0.7, 0.7, 0.7, 1.0)
        )
        atoms.append(
            RenderAtom(
                index=atom_index,
                symbol=scene.atoms.symbols[idx] if idx < len(scene.atoms.symbols) else "X",
                position=tuple(float(v) for v in atom_positions[idx]),
                radius=radius,
                representation=(
                    scene.atoms.representations[idx] if idx < len(scene.atoms.representations) else scene.representation
                ),
                color=color,
                size_px=max(4.0, radius * settings.atom_scale),
                material=material,
                record=record,
            )
        )

    bonds: list[RenderBond] = []
    bond_positions = np.asarray(scene.bonds.positions, dtype=float)
    bond_colors = np.asarray(scene.bonds.colors, dtype=float)
    bond_atom_indices = np.asarray(scene.bonds.atom_indices, dtype=np.int32)
    bond_orders = np.asarray(scene.bonds.orders, dtype=np.int32)
    bond_radii = np.asarray(scene.bonds.radii, dtype=float)
    split_ratios = np.asarray(scene.bonds.split_ratios, dtype=float)
    bond_ids = np.asarray(scene.bonds.bond_ids, dtype=np.int32)
    bond_records = tuple(getattr(scene, "bond_records", ()) or ())
    bond_record_map = {int(record.id): record for record in bond_records}
    bond_groups: dict[int, list[int]] = {}
    for idx in range(scene.bonds.segment_count):
        bond_index = int(bond_ids[idx]) if idx < bond_ids.shape[0] else idx
        bond_groups.setdefault(bond_index, []).append(idx)
    for bond_index, indexes in bond_groups.items():
        idx = indexes[0]
        start = bond_positions[idx, 0]
        end = bond_positions[idx, 1]
        left_color = tuple(float(v) for v in bond_colors[idx, 0])
        right_color = tuple(float(v) for v in bond_colors[idx, 1])
        width_px = max(1.0, float(bond_radii[idx]) * settings.bond_scale)
        order = max(1, int(bond_orders[idx]) if idx < bond_orders.shape[0] else 1)
        atom_a = int(bond_atom_indices[idx, 0]) if idx < bond_atom_indices.shape[0] else 0
        atom_b = int(bond_atom_indices[idx, 1]) if idx < bond_atom_indices.shape[0] else 0
        record = bond_record_map.get(bond_index)
        if len(indexes) > 1:
            segments = tuple(
                RenderLineSegment(
                    start=tuple(float(v) for v in bond_positions[item_idx, 0]),
                    end=tuple(float(v) for v in bond_positions[item_idx, 1]),
                    color=tuple(float(v) for v in bond_colors[item_idx, 0]),
                    width_px=max(1.0, float(bond_radii[item_idx]) * settings.bond_scale),
                )
                for item_idx in indexes
            )
        else:
            segments = build_bond_segments(
                start,
                end,
                left_color,
                right_color,
                width_px=width_px,
                order=order,
                split=bool(record is not None and record.material_left is not None and record.material_right is not None and order == 1),
                split_ratio=float(split_ratios[idx]) if idx < split_ratios.shape[0] else 0.5,
            )
        preview_bond_style = str((record.metadata if record is not None else {}).get("preview_bond_style", "bicolor"))
        segments = _apply_preview_bond_style(segments, preview_bond_style, right_color=right_color)
        bonds.append(
            RenderBond(
                index=bond_index,
                a=atom_a,
                b=atom_b,
                order=order,
                color=mix_rgba(left_color, right_color, 0.5),
                width_px=width_px,
                bond_type=record.bond_type if record is not None else (
                    scene.bonds.bond_types[idx] if idx < len(scene.bonds.bond_types) else "covalent"
                ),
                distance=record.distance if record is not None else None,
                split_ratio=float(split_ratios[idx]) if idx < split_ratios.shape[0] else 0.5,
                material_uniform=preview_material_payload(record.material_uniform) if record is not None else None,
                material_left=preview_material_payload(record.material_left) if record is not None else None,
                material_right=preview_material_payload(record.material_right) if record is not None else None,
                record=record,
                segments=segments,
            )
        )

    cell_edges: list[RenderCellEdge] = []
    cell_positions = np.asarray(scene.cell.positions, dtype=float)
    cell_colors = np.asarray(scene.cell.colors, dtype=float)
    cell_radii = np.asarray(scene.cell.radii, dtype=float)
    for idx in range(scene.cell.segment_count):
        cell_edges.append(
            RenderCellEdge(
                index=idx,
                start=tuple(float(v) for v in cell_positions[idx, 0]),
                end=tuple(float(v) for v in cell_positions[idx, 1]),
                color=tuple(float(v) for v in cell_colors[idx]) if idx < cell_colors.shape[0] else (0.4, 0.4, 0.4, 1.0),
                width_px=max(0.8, float(cell_radii[idx]) * settings.cell_scale) if idx < cell_radii.shape[0] else 1.0,
            )
        )

    polyhedra: list[RenderPolyhedron] = []
    vertex_offsets = np.asarray(scene.polyhedra.vertex_offsets, dtype=np.int32)
    face_offsets = np.asarray(scene.polyhedra.face_offsets, dtype=np.int32)
    poly_vertices = np.asarray(scene.polyhedra.vertices, dtype=float)
    poly_faces = np.asarray(scene.polyhedra.faces, dtype=np.int32)
    poly_colors = np.asarray(scene.polyhedra.colors, dtype=float)
    poly_edge_colors = np.asarray(scene.polyhedra.edge_colors, dtype=float)
    poly_edge_radii = np.asarray(scene.polyhedra.edge_radii, dtype=float)
    poly_show_edges = np.asarray(scene.polyhedra.show_edges, dtype=bool)
    poly_ids = np.asarray(scene.polyhedra.polyhedron_ids, dtype=np.int32)
    for idx in range(scene.polyhedra.count):
        v0 = int(vertex_offsets[idx]) if idx < vertex_offsets.shape[0] else 0
        v1 = int(vertex_offsets[idx + 1]) if idx + 1 < vertex_offsets.shape[0] else v0
        f0 = int(face_offsets[idx]) if idx < face_offsets.shape[0] else 0
        f1 = int(face_offsets[idx + 1]) if idx + 1 < face_offsets.shape[0] else f0
        vertices = tuple(tuple(float(v) for v in row) for row in poly_vertices[v0:v1])
        local_faces = tuple((int(face[0] - v0), int(face[1] - v0), int(face[2] - v0)) for face in poly_faces[f0:f1])
        if not local_faces:
            local_faces = triangulate_polyhedron(list(vertices))
        edge_color = tuple(float(v) for v in poly_edge_colors[idx]) if idx < poly_edge_colors.shape[0] else (0.4, 0.4, 0.4, 1.0)
        edge_width = max(0.6, float(poly_edge_radii[idx]) * settings.cell_scale) if idx < poly_edge_radii.shape[0] else 1.0
        edge_segments = tuple(
            RenderLineSegment(
                start=vertices[a_idx],
                end=vertices[b_idx],
                color=edge_color,
                width_px=edge_width,
            )
            for a_idx, b_idx in polyhedron_edge_segments(local_faces)
            if a_idx < len(vertices) and b_idx < len(vertices)
        )
        polyhedra.append(
            RenderPolyhedron(
                index=int(poly_ids[idx]) if idx < poly_ids.shape[0] else idx,
                center_index=int(scene.polyhedra.center_indices[idx]) if idx < len(scene.polyhedra.center_indices) else idx,
                vertices=vertices,
                faces=local_faces,
                face_color=tuple(float(v) for v in poly_colors[idx]) if idx < poly_colors.shape[0] else (0.7, 0.7, 0.75, settings.polyhedron_alpha),
                edge_color=edge_color,
                show_edges=bool(poly_show_edges[idx]) if idx < poly_show_edges.shape[0] else True,
                edge_segments=edge_segments,
            )
        )

    bounds_min, bounds_max, center, radius = _buffer_bounds(scene)
    return PreviewRenderScene(
        atoms=tuple(atoms),
        bonds=tuple(bonds),
        cell_edges=tuple(cell_edges),
        polyhedra=tuple(polyhedra),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        center=center,
        radius=radius,
        background=settings.background,
        style_name=str(scene.style_name),
        render_mode=resolve_render_mode(scene),
        atom_records=tuple(atom_records),
        bond_records=tuple(bond_records),
        selection_targets=tuple(
            {
                "kind": str(target.kind),
                "index": int(target.index),
                "label": str(target.label),
                "metadata": dict(target.metadata),
            }
            for target in tuple(getattr(scene, "selection_targets", ()) or ())
        ),
        report=dict(getattr(scene, "report", {}) or {}),
    )


def build_preview_scene(
    structure: Structure,
    cfg: RenderJobConfig,
    preview_settings: PreviewSettings | None = None,
) -> PreviewRenderScene:
    settings = preview_settings or PreviewSettings()
    shared_scene = build_buffer_preview_scene(structure, cfg, _shared_preview_settings(settings))
    return render_scene_from_preview_scene(shared_scene, settings)


class PreviewCanvasModel:
    def __init__(self, settings: PreviewSettings | None = None) -> None:
        self.settings = settings or PreviewSettings()
        self.scene: PreviewRenderScene | None = None
        self.shared_scene: PreviewScene | None = None
        self.cfg: RenderJobConfig | None = None
        self.camera = PreviewCameraState()
        self.selection: PreviewSelection | None = None
        self.selection_controller = SelectionController()
        self.selected_atom_indices: set[int] = set()
        self.selected_ordered_atoms: list[int] = []
        self.selected_bond_indices: set[int] = set()
        self.selected_ordered_bonds: list[int] = []
        self.selected_object: dict[str, Any] | None = None
        self.selected_payload: dict[str, Any] | None = None
        self.selection_changed = CallbackSignal()
        self.renderer_mode = "instanced"

    def set_scene(self, structure: Structure, cfg: RenderJobConfig) -> PreviewRenderScene:
        self.cfg = cfg
        shared_scene = build_buffer_preview_scene(structure, cfg, _shared_preview_settings(self.settings))
        return self.set_preview_scene(shared_scene)

    def set_preview_scene(
        self,
        preview_scene: PreviewRenderScene | PreviewScene,
        *,
        preserve_camera: bool | None = None,
    ) -> PreviewRenderScene:
        current_view = self.camera.view or self.settings.default_view
        previous_source = _scene_source(self.shared_scene or self.scene)
        next_source = _scene_source(preview_scene)
        should_preserve_camera = self.scene is not None and (
            previous_source == next_source if preserve_camera is None else bool(preserve_camera)
        )
        camera_snapshot = replace(self.camera) if should_preserve_camera else None
        self.selection = None
        self.selection_controller.clear()
        self.selected_atom_indices.clear()
        self.selected_ordered_atoms.clear()
        self.selected_bond_indices.clear()
        self.selected_ordered_bonds.clear()
        self.selected_object = None
        self.selected_payload = None
        if isinstance(preview_scene, PreviewRenderScene):
            self.scene = preview_scene
            self.shared_scene = None
        else:
            self.shared_scene = preview_scene
            self.scene = render_scene_from_preview_scene(preview_scene, self.settings)
        self.fit_to_structure()
        self.set_view_preset(current_view)
        if camera_snapshot is not None:
            self.camera = camera_snapshot
        return self.scene

    def fit_to_structure(self, padding: float | None = None) -> PreviewCameraState:
        if self.scene is None:
            self.camera.center = (0.0, 0.0, 0.0)
            self.camera.scale_factor = 1.0
            return self.camera
        resolved_padding = self.settings.fit_padding if padding is None else max(0.0, float(padding))
        extent = np.asarray(self.scene.bounds_max, dtype=float) - np.asarray(self.scene.bounds_min, dtype=float)
        span = float(max(extent.max(), 1.0))
        self.camera.center = self.scene.center
        self.camera.scale_factor = max(1.0, span * (1.0 + resolved_padding * 2.0))
        return self.camera

    def set_view_preset(self, view: str) -> PreviewCameraState:
        preset = str(view or "top").strip().lower()
        self.camera.view = preset
        if preset == "top":
            self.camera.azimuth = 0.0
            self.camera.elevation = 90.0
            self.camera.roll = 0.0
            self.camera.right = (1.0, 0.0, 0.0)
            self.camera.up = (0.0, 1.0, 0.0)
            self.camera.forward = (0.0, 0.0, -1.0)
        elif preset == "front":
            self.camera.azimuth = 180.0
            self.camera.elevation = 0.0
            self.camera.roll = 0.0
            self.camera.right = (-1.0, 0.0, 0.0)
            self.camera.up = (0.0, 0.0, 1.0)
            self.camera.forward = (0.0, -1.0, 0.0)
        elif preset == "side":
            self.camera.azimuth = 90.0
            self.camera.elevation = 0.0
            self.camera.roll = 0.0
            self.camera.right = (0.0, 1.0, 0.0)
            self.camera.up = (0.0, 0.0, 1.0)
            self.camera.forward = (-1.0, 0.0, 0.0)
        else:
            self.camera.azimuth = 45.0
            self.camera.elevation = 30.0
            self.camera.roll = 0.0
            self.camera.right, self.camera.up, self.camera.forward = rotation_basis(
                self.camera.azimuth,
                self.camera.elevation,
                self.camera.roll,
            )
        self.camera.model_rotation = tuple(float(value) for value in np.eye(4, dtype=np.float32).reshape((-1,)))
        return self.camera

    def select_preview_object(self, selection: PreviewSelection | None) -> PreviewSelection | None:
        self.selection = selection
        if selection is not None and selection.kind == "atom" and selection.index is not None:
            self.selection_controller.select_single_atom(int(selection.index))
        elif selection is not None and selection.kind == "bond" and selection.index is not None:
            self.selection_controller.select_single_bond(int(selection.index))
        else:
            self.selection_controller.clear()
        self.selected_atom_indices = set(self.selection_controller.selected_atoms)
        self.selected_ordered_atoms = list(self.selection_controller.selected_ordered_atoms)
        self.selected_bond_indices = set(self.selection_controller.selected_bonds)
        self.selected_ordered_bonds = list(self.selection_controller.selected_ordered_bonds)
        self.selected_object, self.selected_payload = selection_payload(self.scene, selection)
        self.selection_changed.emit(selection)
        return selection

    def select_atoms(self, atom_indices: list[int] | tuple[int, ...] | set[int], *, append: bool = False) -> PreviewSelection | None:
        selection = self.selection_controller.select_atoms(atom_indices, append=append)
        self.selected_atom_indices = set(self.selection_controller.selected_atoms)
        self.selected_ordered_atoms = list(self.selection_controller.selected_ordered_atoms)
        self.selected_bond_indices = set(self.selection_controller.selected_bonds)
        self.selected_ordered_bonds = list(self.selection_controller.selected_ordered_bonds)
        self.selection = selection
        self.selected_object, self.selected_payload = selection_payload(self.scene, selection)
        self.selection_changed.emit(selection)
        return selection

    def select_objects(
        self,
        atom_indices: list[int] | tuple[int, ...] | set[int],
        bond_indices: list[int] | tuple[int, ...] | set[int],
        *,
        append: bool = False,
    ) -> PreviewSelection | None:
        self.selection_controller.select_atoms(atom_indices, append=append)
        selection = self.selection_controller.select_bonds(bond_indices, append=True)
        self.selected_atom_indices = set(self.selection_controller.selected_atoms)
        self.selected_ordered_atoms = list(self.selection_controller.selected_ordered_atoms)
        self.selected_bond_indices = set(self.selection_controller.selected_bonds)
        self.selected_ordered_bonds = list(self.selection_controller.selected_ordered_bonds)
        self.selection = selection
        self.selected_object, self.selected_payload = selection_payload(self.scene, selection)
        self.selection_changed.emit(selection)
        return selection

    def toggle_atom_selection(self, atom_index: int) -> PreviewSelection | None:
        selection = self.selection_controller.toggle_atom(atom_index)
        self.selected_atom_indices = set(self.selection_controller.selected_atoms)
        self.selected_ordered_atoms = list(self.selection_controller.selected_ordered_atoms)
        self.selected_bond_indices = set(self.selection_controller.selected_bonds)
        self.selected_ordered_bonds = list(self.selection_controller.selected_ordered_bonds)
        self.selection = selection
        self.selected_object, self.selected_payload = selection_payload(self.scene, selection)
        self.selection_changed.emit(selection)
        return selection

    def toggle_bond_selection(self, bond_index: int) -> PreviewSelection | None:
        selection = self.selection_controller.toggle_bond(bond_index)
        self.selected_atom_indices = set(self.selection_controller.selected_atoms)
        self.selected_ordered_atoms = list(self.selection_controller.selected_ordered_atoms)
        self.selected_bond_indices = set(self.selection_controller.selected_bonds)
        self.selected_ordered_bonds = list(self.selection_controller.selected_ordered_bonds)
        self.selection = selection
        self.selected_object, self.selected_payload = selection_payload(self.scene, selection)
        self.selection_changed.emit(selection)
        return selection

    def update_atom_properties(self, atom_index: int, updates: dict[str, Any]) -> PreviewSelection | None:
        if self.scene is None:
            return None
        target = int(atom_index)
        atoms: list[RenderAtom] = []
        changed = False
        for atom in self.scene.atoms:
            if int(atom.index) != target:
                atoms.append(atom)
                continue
            base_position = np.asarray(updates.get("position", atom.position), dtype=float).reshape((3,))
            if atom.record is not None:
                metadata = getattr(atom.record, "metadata", {}) or {}
                cart_shift = metadata.get("boundary_atom_cart_shift") if isinstance(metadata, dict) else None
                if cart_shift is not None:
                    base_position = base_position + np.asarray(cart_shift, dtype=float).reshape((3,))
            position = tuple(float(v) for v in base_position)
            radius = float(updates.get("radius", atom.radius))
            symbol = str(updates.get("symbol", atom.symbol))
            representation = str(updates.get("representation", atom.representation))
            color = tuple(float(v) for v in updates.get("color", atom.color))
            if len(color) == 3:
                color = (*color, 1.0)
            material = self._material_with_color(atom.material, color)
            record = atom.record
            if record is not None:
                record = replace(
                    record,
                    symbol=symbol,
                    atomic_number=int(updates.get("atomic_number", record.atomic_number)),
                    position=position,
                    radius=radius,
                    representation=representation,
                    material=self._material_with_color(record.material, color) or record.material,
                )
            atoms.append(
                replace(
                    atom,
                    symbol=symbol,
                    position=position,
                    radius=radius,
                    representation=representation,
                    color=color,
                    material=material,
                    size_px=max(4.0, radius * self.settings.atom_scale),
                    record=record,
                )
            )
            changed = True
        if not changed:
            return None
        positions = {int(atom.index): atom.position for atom in atoms}
        colors = {int(atom.index): atom.color for atom in atoms}
        materials = {int(atom.index): atom.material for atom in atoms}
        bonds = tuple(self._updated_bond_geometry(bond, positions, colors, materials) for bond in self.scene.bonds)
        atom_records = tuple(atom.record for atom in atoms if atom.record is not None)
        bond_records = tuple(bond.record for bond in bonds if bond.record is not None)
        coords = np.asarray([atom.position for atom in atoms], dtype=float)
        bounds_min = tuple(float(v) for v in coords.min(axis=0)) if coords.size else self.scene.bounds_min
        bounds_max = tuple(float(v) for v in coords.max(axis=0)) if coords.size else self.scene.bounds_max
        center = tuple(float(v) for v in coords.mean(axis=0)) if coords.size else self.scene.center
        radius = max(1.0, float(np.linalg.norm(np.asarray(bounds_max) - np.asarray(bounds_min))) * 0.5)
        self.scene = replace(
            self.scene,
            atoms=tuple(atoms),
            bonds=bonds,
            atom_records=atom_records,
            bond_records=bond_records,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            center=center,
            radius=radius,
        )
        if self.shared_scene is not None:
            self.shared_scene.atom_records = atom_records
            self.shared_scene.bond_records = bond_records
        return self.select_atom(target)

    def delete_selected_objects(self) -> dict[str, int]:
        if self.scene is None:
            return {"atoms": 0, "bonds": 0}
        atom_ids = {int(index) for index in self.selected_atom_indices}
        bond_ids = {int(index) for index in self.selected_bond_indices}
        if self.selection is not None:
            if self.selection.kind == "atom" and self.selection.index is not None:
                atom_ids.add(int(self.selection.index))
            if self.selection.kind == "bond" and self.selection.index is not None:
                bond_ids.add(int(self.selection.index))
        if not atom_ids and not bond_ids:
            return {"atoms": 0, "bonds": 0}

        atoms = tuple(atom for atom in self.scene.atoms if int(atom.index) not in atom_ids)
        bonds = tuple(
            bond
            for bond in self.scene.bonds
            if int(bond.index) not in bond_ids and int(bond.a) not in atom_ids and int(bond.b) not in atom_ids
        )
        polyhedra = tuple(
            poly
            for poly in self.scene.polyhedra
            if int(poly.center_index) not in atom_ids
        )
        atom_records = tuple(record for record in self.scene.atom_records if int(record.index) not in atom_ids)
        bond_records = tuple(record for record in self.scene.bond_records if int(record.id) in {int(bond.index) for bond in bonds})
        selection_targets = tuple(
            target
            for target in self.scene.selection_targets
            if not (
                str(target.get("kind")) == "atom" and int(target.get("index", -1)) in atom_ids
            )
            and not (
                str(target.get("kind")) == "bond" and int(target.get("index", -1)) in bond_ids
            )
        )
        bounds_min, bounds_max, center, radius = self._bounds_from_atoms(atoms)
        deleted_atoms = len(self.scene.atoms) - len(atoms)
        deleted_bonds = len(self.scene.bonds) - len(bonds)
        self.scene = replace(
            self.scene,
            atoms=atoms,
            bonds=bonds,
            polyhedra=polyhedra,
            atom_records=atom_records,
            bond_records=bond_records,
            selection_targets=selection_targets,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            center=center,
            radius=radius,
        )
        self.shared_scene = None
        self.clear_selection()
        return {"atoms": int(deleted_atoms), "bonds": int(deleted_bonds)}

    @staticmethod
    def _bounds_from_atoms(
        atoms: tuple[RenderAtom, ...],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], float]:
        if not atoms:
            zero = (0.0, 0.0, 0.0)
            return zero, zero, zero, 1.0
        coords = np.asarray([atom.position for atom in atoms], dtype=float)
        radii = np.asarray([max(0.0, float(atom.radius)) for atom in atoms], dtype=float)
        bounds_min_arr = (coords - radii[:, None]).min(axis=0)
        bounds_max_arr = (coords + radii[:, None]).max(axis=0)
        center_arr = (bounds_min_arr + bounds_max_arr) * 0.5
        radius = max(1.0, float(np.linalg.norm(bounds_max_arr - bounds_min_arr) * 0.5))
        return (
            tuple(float(v) for v in bounds_min_arr),
            tuple(float(v) for v in bounds_max_arr),
            tuple(float(v) for v in center_arr),
            radius,
        )

    @staticmethod
    def _material_with_color(material, color: tuple[float, float, float, float]):
        if material is None:
            return None
        return replace(material, color=color, alpha=float(color[3]))

    def _updated_bond_geometry(
        self,
        bond: RenderBond,
        positions: dict[int, tuple[float, float, float]],
        atom_colors: dict[int, tuple[float, float, float, float]] | None = None,
        atom_materials: dict[int, Any] | None = None,
    ) -> RenderBond:
        if int(bond.a) not in positions or int(bond.b) not in positions:
            return bond
        atom_colors = atom_colors or {}
        atom_materials = atom_materials or {}
        left_color = atom_colors.get(
            int(bond.a),
            bond.material_left.color if bond.material_left is not None else bond.segments[0].color if bond.segments else bond.color,
        )
        right_color = atom_colors.get(
            int(bond.b),
            bond.material_right.color if bond.material_right is not None else bond.segments[-1].color if bond.segments else bond.color,
        )
        mixed_color = mix_rgba(left_color, right_color, 0.5)
        left_material = PreviewCanvasModel._material_with_color(
            bond.material_left or atom_materials.get(int(bond.a)),
            left_color,
        )
        right_material = PreviewCanvasModel._material_with_color(
            bond.material_right or atom_materials.get(int(bond.b)),
            right_color,
        )
        uniform_material = PreviewCanvasModel._material_with_color(bond.material_uniform, mixed_color)
        split = bool(bond.material_left is not None and bond.material_right is not None and int(bond.order) == 1)
        distance = float(np.linalg.norm(np.asarray(positions[int(bond.b)]) - np.asarray(positions[int(bond.a)])))
        record = bond.record
        if record is not None:
            record_left = PreviewCanvasModel._material_with_color(record.material_left, left_color)
            record_right = PreviewCanvasModel._material_with_color(record.material_right, right_color)
            record_uniform = PreviewCanvasModel._material_with_color(record.material_uniform, mixed_color)
            record = replace(
                record,
                distance=distance,
                material_left=record_left or record.material_left,
                material_right=record_right or record.material_right,
                material_uniform=record_uniform or record.material_uniform,
            )
        segments = build_bond_segments(
            np.asarray(positions[int(bond.a)], dtype=float),
            np.asarray(positions[int(bond.b)], dtype=float),
            left_color,
            right_color,
            width_px=float(bond.width_px),
            order=max(1, int(bond.order)),
            split=split,
            split_ratio=float(bond.split_ratio),
        )
        preview_bond_style = str((record.metadata if record is not None else {}).get("preview_bond_style", "bicolor"))
        segments = _apply_preview_bond_style(segments, preview_bond_style, right_color=right_color)
        return replace(
            bond,
            color=mixed_color,
            distance=distance,
            material_left=left_material,
            material_right=right_material,
            material_uniform=uniform_material,
            record=record,
            segments=segments,
        )

    def select_selection(self, selection: PreviewSelection | None) -> PreviewSelection | None:
        return self.select_preview_object(selection)

    def select_atom(self, atom_index: int | None) -> PreviewSelection | None:
        if atom_index is None:
            return self.select_preview_object(None)
        return self.select_preview_object(PreviewSelection(kind="atom", index=int(atom_index)))

    def select_bond(self, bond_index: int | None) -> PreviewSelection | None:
        if bond_index is None:
            return self.select_preview_object(None)
        return self.select_preview_object(PreviewSelection(kind="bond", index=int(bond_index)))

    def clear_selection(self) -> None:
        self.selection_controller.clear()
        self.selected_atom_indices.clear()
        self.selected_ordered_atoms.clear()
        self.selected_bond_indices.clear()
        self.selected_ordered_bonds.clear()
        self.select_preview_object(None)

    def scene_report(self) -> dict[str, Any]:
        if self.scene is None or self.scene.report is None:
            return {}
        return dict(self.scene.report)

    def project_atom_positions(self, viewport_size: tuple[int, int]) -> dict[int, tuple[float, float, float]]:
        return project_atom_positions(self.scene, self.camera, viewport_size)

    def hit_test_cache(self, viewport_size: tuple[int, int]) -> HitTestCache:
        return HitTestCache.from_scene(
            self.scene,
            self.camera,
            viewport_size,
            picking_radius_px=self.settings.picking_radius_px,
        )

    def pick_atom_at(self, pos: tuple[float, float], viewport_size: tuple[int, int]) -> PreviewSelection | None:
        selection = pick_selection(
            self.scene,
            self.camera,
            viewport_size,
            pos,
            picking_radius_px=self.settings.picking_radius_px,
            bond_scale=self.settings.bond_scale,
        )
        if selection is not None and selection.kind != "atom":
            selection = None
        if selection is None:
            return None
        return self.select_preview_object(selection)

    def atom_indices_in_rect(self, start: tuple[float, float], end: tuple[float, float], viewport_size: tuple[int, int]) -> tuple[int, ...]:
        return atom_indices_in_rect(
            self.scene,
            self.camera,
            viewport_size,
            start,
            end,
            picking_radius_px=self.settings.picking_radius_px,
        )

    def bond_indices_in_rect(self, start: tuple[float, float], end: tuple[float, float], viewport_size: tuple[int, int]) -> tuple[int, ...]:
        return bond_indices_in_rect(
            self.scene,
            self.camera,
            viewport_size,
            start,
            end,
            picking_radius_px=self.settings.picking_radius_px,
            bond_scale=self.settings.bond_scale,
        )

    def pick_bond_at(self, pos: tuple[float, float], viewport_size: tuple[int, int]) -> PreviewSelection | None:
        selection = pick_bond_selection(
            self.scene,
            self.camera,
            viewport_size,
            pos,
            picking_radius_px=self.settings.picking_radius_px,
            bond_scale=self.settings.bond_scale,
        )
        if selection is None:
            return None
        return self.select_preview_object(selection)

    def pick_selection_at(self, pos: tuple[float, float], viewport_size: tuple[int, int]) -> PreviewSelection | None:
        return self.pick_at(pos, viewport_size)

    def peek_selection_at(self, pos: tuple[float, float], viewport_size: tuple[int, int]) -> PreviewSelection | None:
        return pick_selection(
            self.scene,
            self.camera,
            viewport_size,
            pos,
            picking_radius_px=self.settings.picking_radius_px,
            bond_scale=self.settings.bond_scale,
        )

    def pick_at(self, pos: tuple[float, float], viewport_size: tuple[int, int]) -> PreviewSelection | None:
        selection = self.peek_selection_at(pos, viewport_size)
        if selection is None:
            return None
        return self.select_preview_object(selection)

    def select_next_atom(self) -> PreviewSelection | None:
        selection = cycle_atom_selection(self.scene, self.selection, +1)
        return self.select_preview_object(selection) if selection is not None else None

    def select_previous_atom(self) -> PreviewSelection | None:
        selection = cycle_atom_selection(self.scene, self.selection, -1)
        return self.select_preview_object(selection) if selection is not None else None

    def select_next_bond(self) -> PreviewSelection | None:
        selection = cycle_bond_selection(self.scene, self.selection, +1)
        return self.select_preview_object(selection) if selection is not None else None

    def select_previous_bond(self) -> PreviewSelection | None:
        selection = cycle_bond_selection(self.scene, self.selection, -1)
        return self.select_preview_object(selection) if selection is not None else None

    def atom_draw_data(self) -> tuple[dict[str, Any], ...]:
        return atom_draw_data(self.scene, self.selection, self.settings, self.selected_atom_indices)

    def bond_draw_data(self) -> tuple[dict[str, Any], ...]:
        return bond_draw_data(self.scene, self.selection, self.settings, self.selected_bond_indices)


__all__ = [
    "CallbackSignal",
    "PreviewCameraState",
    "PreviewCanvasModel",
    "PreviewRenderScene",
    "PreviewSettings",
    "build_preview_scene",
    "render_scene_from_preview_scene",
]
