from __future__ import annotations

from typing import Any

import numpy as np

from atomstudio.preview.material_adapter import mix_rgba
from atomstudio.preview.picking import normalize
from atomstudio.preview.types import PreviewSelection, RenderLineSegment, PreviewRenderScene
from atomstudio.visual_defaults import HYDROGEN_BOND_LINE_COLOR, HYDROGEN_BOND_LINE_WIDTH


def bond_offset_vector(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return np.zeros(3, dtype=float)
    axis = direction / length
    normal = np.cross(axis, np.array([0.0, 0.0, 1.0], dtype=float))
    if float(np.linalg.norm(normal)) <= 1e-12:
        normal = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=float))
    return normalize(normal)


def build_bond_segments(
    start: np.ndarray,
    end: np.ndarray,
    left_color: tuple[float, float, float, float],
    right_color: tuple[float, float, float, float],
    *,
    width_px: float,
    order: int,
    split: bool,
    split_ratio: float,
) -> tuple[RenderLineSegment, ...]:
    base_color = mix_rgba(left_color, right_color, 0.5)
    offset = bond_offset_vector(start, end) * max(0.0, float(width_px) * 0.35)
    offsets = np.linspace(-(order - 1) * 0.5, (order - 1) * 0.5, order) if order > 1 else (0.0,)
    segments: list[RenderLineSegment] = []
    ratio = max(0.05, min(0.95, float(split_ratio)))
    for delta in offsets:
        shift = offset * float(delta)
        shifted_start = start + shift
        shifted_end = end + shift
        if split and order == 1:
            mid = shifted_start + (shifted_end - shifted_start) * ratio
            segments.append(
                RenderLineSegment(
                    start=tuple(float(v) for v in shifted_start),
                    end=tuple(float(v) for v in mid),
                    color=left_color,
                    width_px=width_px,
                )
            )
            segments.append(
                RenderLineSegment(
                    start=tuple(float(v) for v in mid),
                    end=tuple(float(v) for v in shifted_end),
                    color=right_color,
                    width_px=width_px,
                )
            )
            continue
        segments.append(
            RenderLineSegment(
                start=tuple(float(v) for v in shifted_start),
                end=tuple(float(v) for v in shifted_end),
                color=base_color,
                width_px=width_px,
            )
        )
    return tuple(segments)


def triangulate_polyhedron(vertices: list[tuple[float, float, float]]) -> tuple[tuple[int, int, int], ...]:
    if len(vertices) < 4:
        return ()
    if len(vertices) == 4:
        return ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    pts = np.asarray(vertices, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_u = vh[0]
        axis_v = vh[1]
    except Exception:
        axis_u = np.array([1.0, 0.0, 0.0], dtype=float)
        axis_v = np.array([0.0, 1.0, 0.0], dtype=float)
    angles = np.arctan2(centered @ axis_v, centered @ axis_u)
    order = np.argsort(angles)
    if len(order) < 3:
        return ()
    faces: list[tuple[int, int, int]] = []
    for idx in range(1, len(order) - 1):
        faces.append((int(order[0]), int(order[idx]), int(order[idx + 1])))
    return tuple(faces)


def polyhedron_edge_segments(faces: tuple[tuple[int, int, int], ...]) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges.add((min(int(a), int(b)), max(int(a), int(b))))
    return tuple(sorted(edges))


def atom_draw_data(
    scene: PreviewRenderScene | None,
    selection: PreviewSelection | None,
    settings,
    selected_atom_indices: set[int] | frozenset[int] | tuple[int, ...] = (),
) -> tuple[dict[str, Any], ...]:
    if scene is None:
        return ()
    out: list[dict[str, Any]] = []
    selected_atoms = {int(index) for index in selected_atom_indices}
    for atom in scene.atoms:
        highlighted = int(atom.index) in selected_atoms or (
            selection is not None and selection.kind == "atom" and selection.index == atom.index
        )
        color = atom.color
        size = atom.size_px
        radius = atom.radius
        if highlighted:
            color = mix_rgba(color, settings.highlight_color, 0.18)
            size *= settings.selected_atom_scale
            radius *= settings.selected_atom_scale
        out.append(
            {
                "index": atom.index,
                "position": atom.position,
                "radius": radius,
                "size": size,
                "face_color": color,
                "edge_color": settings.highlight_color if highlighted else (0.18, 0.18, 0.18, 1.0),
                "material": atom.material,
                "highlighted": highlighted,
            }
        )
    return tuple(out)


def bond_draw_data(
    scene: PreviewRenderScene | None,
    selection: PreviewSelection | None,
    settings,
    selected_bond_indices: set[int] | frozenset[int] | tuple[int, ...] = (),
) -> tuple[dict[str, Any], ...]:
    if scene is None:
        return ()
    selected_bond = selection.index if selection is not None and selection.kind == "bond" else None
    selected_bonds = {int(index) for index in selected_bond_indices}
    out: list[dict[str, Any]] = []
    for bond in scene.bonds:
        highlighted = bool(selected_bond == bond.index or int(bond.index) in selected_bonds)
        color = mix_rgba(bond.color, settings.highlight_color, 0.25) if highlighted else bond.color
        out.append(
            {
                "index": bond.index,
                "bond_type": bond.bond_type,
                "segments": tuple(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "color": mix_rgba(segment.color, settings.highlight_color, 0.15) if highlighted else segment.color,
                        "width": float(segment.width_px) * (1.15 if highlighted else 1.0),
                        "material": bond.material_left if idx == 0 and bond.material_left is not None else (
                            bond.material_right if idx == len(bond.segments) - 1 and bond.material_right is not None else bond.material_uniform
                        ),
                    }
                    for idx, segment in enumerate(bond.segments)
                ),
                "color": color,
                "width": bond.width_px * (1.15 if highlighted else 1.0),
                "highlighted": highlighted,
            }
        )
    return tuple(out)


def segment_basis(start: np.ndarray, end: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 1e-12:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
        length = 1e-6
    axis = axis / length
    side = np.cross(axis, np.array([0.0, 0.0, 1.0], dtype=float))
    if float(np.linalg.norm(side)) <= 1e-12:
        side = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=float))
    side = normalize(side) * radius
    up = normalize(np.cross(axis, side)) * radius
    return side, up, axis, length


def build_atom_mesh_payload(draw_data: tuple[dict[str, Any], ...], sphere_mesh) -> dict[str, np.ndarray]:
    if not draw_data:
        return {
            "vertices": np.zeros((0, 3), dtype=float),
            "faces": np.zeros((0, 3), dtype=np.int32),
            "face_colors": np.zeros((0, 4), dtype=float),
        }
    base_vertices = np.asarray(sphere_mesh.get_vertices(), dtype=float)
    base_faces = np.asarray(sphere_mesh.get_faces(), dtype=np.int32)
    vertices_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    face_colors: list[np.ndarray] = []
    vertex_offset = 0
    for item in draw_data:
        radius = max(1e-4, float(item["radius"]))
        transformed = base_vertices * radius + np.asarray(item["position"], dtype=float)
        vertices_list.append(transformed)
        faces = base_faces + vertex_offset
        faces_list.append(faces)
        face_colors.append(np.tile(np.asarray(item["face_color"], dtype=float), (faces.shape[0], 1)))
        vertex_offset += base_vertices.shape[0]
    return {
        "vertices": np.concatenate(vertices_list, axis=0),
        "faces": np.concatenate(faces_list, axis=0),
        "face_colors": np.concatenate(face_colors, axis=0),
    }


def build_atom_instance_payload(draw_data: tuple[dict[str, Any], ...]) -> dict[str, np.ndarray]:
    if not draw_data:
        return {
            "instance_positions": np.zeros((0, 3), dtype=np.float32),
            "instance_transforms": np.zeros((0, 3, 3), dtype=np.float32),
            "instance_colors": np.zeros((0, 4), dtype=np.float32),
        }
    positions: list[tuple[float, float, float]] = []
    transforms: list[np.ndarray] = []
    colors: list[tuple[float, float, float, float]] = []
    for item in draw_data:
        radius = max(1e-4, float(item["radius"]))
        positions.append(tuple(float(v) for v in item["position"]))
        transforms.append(np.eye(3, dtype=np.float32) * radius)
        colors.append(tuple(float(v) for v in item["face_color"]))
    return {
        "instance_positions": np.asarray(positions, dtype=np.float32).reshape((-1, 3)),
        "instance_transforms": np.asarray(transforms, dtype=np.float32).reshape((-1, 3, 3)),
        "instance_colors": np.asarray(colors, dtype=np.float32).reshape((-1, 4)),
    }


def build_bond_mesh_payload(
    draw_data: tuple[dict[str, Any], ...],
    cylinder_mesh,
    *,
    bond_scale: float,
) -> dict[str, np.ndarray]:
    if not draw_data:
        return {
            "vertices": np.zeros((0, 3), dtype=float),
            "faces": np.zeros((0, 3), dtype=np.int32),
            "face_colors": np.zeros((0, 4), dtype=float),
        }
    base_vertices = np.asarray(cylinder_mesh.get_vertices(), dtype=float)
    base_faces = np.asarray(cylinder_mesh.get_faces(), dtype=np.int32)
    vertices_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    face_colors: list[np.ndarray] = []
    vertex_offset = 0
    for item in draw_data:
        if str(item.get("bond_type", "covalent")) == "hydrogen":
            continue
        for segment in item["segments"]:
            start = np.asarray(segment["start"], dtype=float)
            end = np.asarray(segment["end"], dtype=float)
            radius = max(1e-4, float(segment["width"]) / max(float(bond_scale), 1.0))
            side, up, axis, length = segment_basis(start, end, radius)
            transformed = (
                start[None, :]
                + np.outer(base_vertices[:, 0], side)
                + np.outer(base_vertices[:, 1], up)
                + np.outer(base_vertices[:, 2], axis * length)
            )
            vertices_list.append(transformed)
            faces = base_faces + vertex_offset
            faces_list.append(faces)
            face_colors.append(np.tile(np.asarray(segment["color"], dtype=float), (faces.shape[0], 1)))
            vertex_offset += base_vertices.shape[0]
    if not vertices_list:
        return {
            "vertices": np.zeros((0, 3), dtype=float),
            "faces": np.zeros((0, 3), dtype=np.int32),
            "face_colors": np.zeros((0, 4), dtype=float),
        }
    return {
        "vertices": np.concatenate(vertices_list, axis=0),
        "faces": np.concatenate(faces_list, axis=0),
        "face_colors": np.concatenate(face_colors, axis=0),
    }


def build_hydrogen_bond_line_payload(draw_data: tuple[dict[str, Any], ...]) -> dict[str, np.ndarray | float | str]:
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    for item in draw_data:
        if str(item.get("bond_type", "covalent")) != "hydrogen":
            continue
        for segment in item["segments"]:
            positions.append(tuple(float(v) for v in segment["start"]))
            positions.append(tuple(float(v) for v in segment["end"]))
            colors.append(HYDROGEN_BOND_LINE_COLOR)
            colors.append(HYDROGEN_BOND_LINE_COLOR)
    if not positions:
        return {"pos": np.zeros((0, 3), dtype=float)}
    return {
        "pos": np.asarray(positions, dtype=float),
        "color": np.asarray(colors, dtype=float),
        "width": HYDROGEN_BOND_LINE_WIDTH,
        "connect": "segments",
    }


def build_bond_instance_payload(
    draw_data: tuple[dict[str, Any], ...],
    *,
    bond_scale: float,
) -> dict[str, np.ndarray]:
    if not draw_data:
        return {
            "instance_positions": np.zeros((0, 3), dtype=np.float32),
            "instance_transforms": np.zeros((0, 3, 3), dtype=np.float32),
            "instance_colors": np.zeros((0, 4), dtype=np.float32),
        }
    positions: list[tuple[float, float, float]] = []
    transforms: list[np.ndarray] = []
    colors: list[tuple[float, float, float, float]] = []
    for item in draw_data:
        for segment in item["segments"]:
            start = np.asarray(segment["start"], dtype=float)
            end = np.asarray(segment["end"], dtype=float)
            radius = max(1e-4, float(segment["width"]) / max(float(bond_scale), 1.0))
            side, up, axis, length = segment_basis(start, end, radius)
            positions.append(tuple(float(v) for v in start))
            transforms.append(np.asarray((side, up, axis * length), dtype=np.float32).T)
            colors.append(tuple(float(v) for v in segment["color"]))
    return {
        "instance_positions": np.asarray(positions, dtype=np.float32).reshape((-1, 3)),
        "instance_transforms": np.asarray(transforms, dtype=np.float32).reshape((-1, 3, 3)),
        "instance_colors": np.asarray(colors, dtype=np.float32).reshape((-1, 4)),
    }


def build_cell_visual_payload(scene: PreviewRenderScene | None) -> dict[str, np.ndarray | float | str]:
    if scene is None or not scene.cell_edges:
        return {"pos": np.zeros((0, 3), dtype=float)}
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    widths: list[float] = []
    for item in scene.cell_edges:
        positions.append(item.start)
        positions.append(item.end)
        colors.append(item.color)
        colors.append(item.color)
        widths.append(item.width_px)
        widths.append(item.width_px)
    return {
        "pos": np.asarray(positions, dtype=float),
        "color": np.asarray(colors, dtype=float),
        "width": max(widths) if widths else 1.0,
        "connect": "segments",
    }


def build_poly_visual_payload(scene: PreviewRenderScene | None) -> dict[str, Any]:
    if scene is None or not scene.polyhedra:
        return {
            "mesh": {
                "vertices": np.zeros((0, 3), dtype=float),
                "faces": np.zeros((0, 3), dtype=np.int32),
                "face_colors": np.zeros((0, 4), dtype=float),
            },
            "edges": {"pos": np.zeros((0, 3), dtype=float)},
            "visible": False,
        }
    all_vertices: list[tuple[float, float, float]] = []
    all_faces: list[tuple[int, int, int]] = []
    face_colors: list[tuple[float, float, float, float]] = []
    edge_positions: list[tuple[float, float, float]] = []
    edge_colors: list[tuple[float, float, float, float]] = []
    vertex_offset = 0
    for item in scene.polyhedra:
        all_vertices.extend(item.vertices)
        for face in item.faces:
            all_faces.append((face[0] + vertex_offset, face[1] + vertex_offset, face[2] + vertex_offset))
            face_colors.append(item.face_color)
        if item.show_edges:
            for segment in item.edge_segments:
                edge_positions.append(segment.start)
                edge_positions.append(segment.end)
                edge_colors.append(segment.color)
                edge_colors.append(segment.color)
        vertex_offset += len(item.vertices)
    return {
        "mesh": {
            "vertices": np.asarray(all_vertices, dtype=float),
            "faces": np.asarray(all_faces, dtype=np.int32),
            "face_colors": np.asarray(face_colors, dtype=float),
        },
        "edges": (
            {
                "pos": np.asarray(edge_positions, dtype=float),
                "color": np.asarray(edge_colors, dtype=float),
                "width": 1.2,
                "connect": "segments",
            }
            if edge_positions
            else {"pos": np.zeros((0, 3), dtype=float)}
        ),
        "visible": True,
    }


__all__ = [
    "atom_draw_data",
    "bond_draw_data",
    "bond_offset_vector",
    "build_atom_instance_payload",
    "build_atom_mesh_payload",
    "build_bond_instance_payload",
    "build_bond_mesh_payload",
    "build_hydrogen_bond_line_payload",
    "build_bond_segments",
    "build_cell_visual_payload",
    "build_poly_visual_payload",
    "polyhedron_edge_segments",
    "segment_basis",
    "triangulate_polyhedron",
]
