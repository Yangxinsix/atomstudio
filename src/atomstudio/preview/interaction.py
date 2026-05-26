from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from atomstudio.preview.picking import point_distance_2d, project_point
from atomstudio.preview.types import PreviewRenderScene, PreviewSelection


MOUSE_MODES = {"rotate", "select", "pan", "measure_distance", "measure_angle", "measure_dihedral"}


def normalize_mouse_mode(mode: str | None) -> str:
    value = str(mode or "rotate").strip().lower()
    return value if value in MOUSE_MODES else "rotate"


@dataclass(frozen=True, slots=True)
class AtomHit:
    index: int
    x: float
    y: float
    depth: float
    radius_px: float

    def distance_to(self, pos: tuple[float, float]) -> float:
        return point_distance_2d((self.x, self.y), pos)

    def intersects_rect(self, rect: tuple[float, float, float, float]) -> bool:
        x0, y0, x1, y1 = rect
        closest_x = min(max(self.x, x0), x1)
        closest_y = min(max(self.y, y0), y1)
        return point_distance_2d((self.x, self.y), (closest_x, closest_y)) <= self.radius_px


@dataclass(frozen=True, slots=True)
class HitTestCache:
    atoms: tuple[AtomHit, ...] = ()

    @classmethod
    def from_scene(
        cls,
        scene: PreviewRenderScene | None,
        camera,
        viewport_size: tuple[int, int],
        *,
        picking_radius_px: float = 20.0,
    ) -> "HitTestCache":
        if scene is None:
            return cls()
        hits = []
        radius_floor = max(1.0, float(picking_radius_px))
        scene_radius = max(1.0, float(getattr(scene, "radius", 1.0) or 1.0))
        for atom in scene.atoms:
            x, y, depth = project_point(atom.position, camera, viewport_size, scene_radius=scene_radius)
            hits.append(
                AtomHit(
                    index=int(atom.index),
                    x=float(x),
                    y=float(y),
                    depth=float(depth),
                    radius_px=max(radius_floor, float(atom.size_px) * 0.5),
                )
            )
        return cls(tuple(hits))

    def pick_atom(self, pos: tuple[float, float]) -> PreviewSelection | None:
        candidates = []
        for atom in self.atoms:
            distance = atom.distance_to(pos)
            if distance <= atom.radius_px:
                candidates.append((atom.depth, distance, atom.index))
        if not candidates:
            return None
        _depth, _distance, index = min(candidates)
        return PreviewSelection(kind="atom", index=int(index))

    def atoms_in_rect(self, start: tuple[float, float], end: tuple[float, float]) -> tuple[int, ...]:
        x0, x1 = sorted((float(start[0]), float(end[0])))
        y0, y1 = sorted((float(start[1]), float(end[1])))
        rect = (x0, y0, x1, y1)
        return tuple(sorted(atom.index for atom in self.atoms if atom.intersects_rect(rect)))


class SelectionController:
    def __init__(self) -> None:
        self.selected_atoms: set[int] = set()
        self.selected_ordered_atoms: list[int] = []
        self.selected_bonds: set[int] = set()
        self.selected_ordered_bonds: list[int] = []

    def clear(self) -> None:
        self.selected_atoms.clear()
        self.selected_ordered_atoms.clear()
        self.selected_bonds.clear()
        self.selected_ordered_bonds.clear()

    def select_single_atom(self, atom_index: int) -> PreviewSelection:
        index = int(atom_index)
        self.clear()
        self.selected_atoms = {index}
        self.selected_ordered_atoms = [index]
        return PreviewSelection(kind="atom", index=index)

    def select_atoms(self, atom_indices: Iterable[int], *, append: bool = False) -> PreviewSelection | None:
        indices = [int(index) for index in atom_indices]
        if not append:
            self.clear()
        for index in indices:
            if index not in self.selected_atoms:
                self.selected_ordered_atoms.append(index)
            self.selected_atoms.add(index)
        if not self.selected_ordered_atoms:
            return self._active_selection()
        return PreviewSelection(kind="atom", index=self.selected_ordered_atoms[0])

    def toggle_atom(self, atom_index: int) -> PreviewSelection | None:
        index = int(atom_index)
        if index in self.selected_atoms:
            self.selected_atoms.remove(index)
            self.selected_ordered_atoms = [item for item in self.selected_ordered_atoms if item != index]
        else:
            self.selected_atoms.add(index)
            self.selected_ordered_atoms.append(index)
        return self._active_selection(prefer_kind="atom")

    def select_single_bond(self, bond_index: int) -> PreviewSelection:
        index = int(bond_index)
        self.clear()
        self.selected_bonds = {index}
        self.selected_ordered_bonds = [index]
        return PreviewSelection(kind="bond", index=index)

    def select_bonds(self, bond_indices: Iterable[int], *, append: bool = False) -> PreviewSelection | None:
        indices = [int(index) for index in bond_indices]
        if not append:
            self.clear()
        for index in indices:
            if index not in self.selected_bonds:
                self.selected_ordered_bonds.append(index)
            self.selected_bonds.add(index)
        return self._active_selection(prefer_kind="bond")

    def toggle_bond(self, bond_index: int) -> PreviewSelection | None:
        index = int(bond_index)
        if index in self.selected_bonds:
            self.selected_bonds.remove(index)
            self.selected_ordered_bonds = [item for item in self.selected_ordered_bonds if item != index]
        else:
            self.selected_bonds.add(index)
            self.selected_ordered_bonds.append(index)
        return self._active_selection(prefer_kind="bond")

    def _active_selection(self, *, prefer_kind: str = "atom") -> PreviewSelection | None:
        if prefer_kind == "bond" and self.selected_ordered_bonds:
            return PreviewSelection(kind="bond", index=self.selected_ordered_bonds[-1])
        if self.selected_ordered_atoms:
            return PreviewSelection(kind="atom", index=self.selected_ordered_atoms[-1])
        if self.selected_ordered_bonds:
            return PreviewSelection(kind="bond", index=self.selected_ordered_bonds[-1])
        return None


class MeasurementController:
    _REQUIRED = {"measure_distance": 2, "measure_angle": 3, "measure_dihedral": 4}

    def __init__(self) -> None:
        self.atom_indices: list[int] = []

    def clear(self) -> None:
        self.atom_indices.clear()

    def required_count(self, mode: str) -> int:
        return self._REQUIRED[normalize_mouse_mode(mode)]

    def add_atom(self, mode: str, atom_index: int) -> tuple[bool, list[int]]:
        if int(atom_index) not in self.atom_indices:
            self.atom_indices.append(int(atom_index))
        required = self.required_count(mode)
        complete = len(self.atom_indices) >= required
        return complete, list(self.atom_indices[:required])

    def message(self, scene: PreviewRenderScene | None, atom_indices: list[int]) -> str:
        positions: dict[int, np.ndarray] = {}
        for atom in scene.atoms if scene else ():
            positions.setdefault(int(atom.index), np.asarray(atom.position, dtype=float))
        points = [positions[index] for index in atom_indices if index in positions]
        if len(points) == 2:
            distance = float(np.linalg.norm(points[1] - points[0]))
            return f"Distance {atom_indices[0]}-{atom_indices[1]}: {distance:.4f} A"
        if len(points) == 3:
            angle = self._angle_degrees(points[0] - points[1], points[2] - points[1])
            return f"Angle {atom_indices[0]}-{atom_indices[1]}-{atom_indices[2]}: {angle:.2f} deg"
        if len(points) == 4:
            dihedral = self._dihedral_degrees(points[0], points[1], points[2], points[3])
            return f"Dihedral {atom_indices[0]}-{atom_indices[1]}-{atom_indices[2]}-{atom_indices[3]}: {dihedral:.2f} deg"
        return "Measurement unavailable"

    @staticmethod
    def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            return 0.0
        cos_value = float(np.dot(left, right) / (left_norm * right_norm))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_value))))

    @staticmethod
    def _dihedral_degrees(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        b0 = -(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        norm = float(np.linalg.norm(b1))
        if norm <= 1e-12:
            return 0.0
        b1 = b1 / norm
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = float(np.dot(v, w))
        y = float(np.dot(np.cross(b1, v), w))
        return math.degrees(math.atan2(y, x))


__all__ = [
    "AtomHit",
    "HitTestCache",
    "MeasurementController",
    "SelectionController",
    "normalize_mouse_mode",
]
