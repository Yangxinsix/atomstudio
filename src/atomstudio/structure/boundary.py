from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

from atomstudio.structure.structure import Structure

if TYPE_CHECKING:
    from atomstudio.config import BoundaryConfig


def normalize_window(boundary_cfg: "BoundaryConfig") -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    window_raw = getattr(boundary_cfg, "window_frac", [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    if not isinstance(window_raw, (list, tuple)) or len(window_raw) != 3:
        raise ValueError("structure.boundary.window_frac must be a 3x2 array in fractional coordinates.")
    out: list[tuple[float, float]] = []
    for axis in range(3):
        row = window_raw[axis]
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("structure.boundary.window_frac must be a 3x2 array in fractional coordinates.")
        lo = float(row[0])
        hi = float(row[1])
        if lo > hi:
            raise ValueError("structure.boundary.window_frac axis range must satisfy min <= max.")
        out.append((lo, hi))
    return (out[0], out[1], out[2])


def fractional_positions(structure: Structure) -> np.ndarray:
    if not structure.atoms:
        return np.zeros((0, 3), dtype=float)
    cell = np.asarray(structure.cell_vectors, dtype=float).reshape(3, 3)
    if abs(float(np.linalg.det(cell))) < 1e-12:
        raise ValueError("Structure cell is singular; cannot compute fractional coordinates.")
    positions = np.asarray(structure.positions, dtype=float).reshape((-1, 3))
    inv_cell = np.linalg.inv(cell)
    return positions @ inv_cell


def enumerate_offsets(
    window: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    pbc: tuple[bool, bool, bool],
) -> list[tuple[int, int, int]]:
    axis_ranges: list[range] = []
    for axis in range(3):
        if not bool(pbc[axis]):
            axis_ranges.append(range(0, 1))
            continue
        lo, hi = window[axis]
        start = int(np.floor(float(lo)))
        stop = int(np.floor(float(hi)))
        axis_ranges.append(range(start, stop + 1))
    return [tuple(int(v) for v in offset) for offset in product(*axis_ranges)]


def build_boundary_expanded_structure(
    structure: Structure,
    window: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    eps: float = 1e-6,
) -> Structure:
    expanded = Structure.from_dict(structure.to_dict())
    expanded.bonds = []
    expanded.polyhedra = []
    if not expanded.atoms:
        return expanded

    tol = max(0.0, float(eps))
    try:
        frac = fractional_positions(structure)
    except Exception:
        return expanded

    cell = np.asarray(structure.cell_vectors, dtype=float).reshape(3, 3)
    template_atoms = expanded.atoms
    template_positions = np.asarray([atom.position for atom in template_atoms], dtype=float).reshape((-1, 3))
    expanded_atoms = []

    for offset in enumerate_offsets(window, structure.pbc):
        offset_vec = np.asarray(offset, dtype=float)
        shifted_frac = frac + offset_vec
        keep = np.ones(len(template_atoms), dtype=bool)
        for axis in range(3):
            lo, hi = window[axis]
            values = shifted_frac[:, axis]
            keep &= values >= float(lo) - tol
            keep &= values <= float(hi) + tol
        if not np.any(keep):
            continue
        cart_shift = offset_vec @ cell
        for atom_idx in np.where(keep)[0]:
            atom = template_atoms[int(atom_idx)]
            shifted_pos = template_positions[int(atom_idx)] + cart_shift
            metadata = dict(atom.metadata)
            metadata["origin_index"] = int(atom.index)
            metadata["boundary_offset"] = [int(offset[0]), int(offset[1]), int(offset[2])]
            expanded_atoms.append(
                replace(
                    atom,
                    index=len(expanded_atoms),
                    position=(float(shifted_pos[0]), float(shifted_pos[1]), float(shifted_pos[2])),
                    metadata=metadata,
                )
            )

    expanded.atoms = expanded_atoms
    return expanded


def wrap_structure_into_cell(structure: Structure) -> Structure:
    wrapped = Structure.from_dict(structure.to_dict())
    if not wrapped.atoms or not any(bool(v) for v in wrapped.pbc):
        return wrapped
    positions = _ase_wrapped_positions(wrapped)
    if positions is None:
        positions = _fractional_wrapped_positions(wrapped)
    if positions is None:
        return wrapped
    for atom, position in zip(wrapped.atoms, positions, strict=True):
        atom.position = (float(position[0]), float(position[1]), float(position[2]))
    return wrapped


def _ase_wrapped_positions(structure: Structure) -> np.ndarray | None:
    try:
        from ase import Atoms

        atoms = Atoms(
            symbols=structure.symbols,
            positions=structure.positions,
            cell=structure.cell_vectors,
            pbc=structure.pbc,
        )
        atoms.wrap()
        return np.asarray(atoms.get_positions(), dtype=float).reshape((-1, 3))
    except Exception:
        return None


def _fractional_wrapped_positions(structure: Structure) -> np.ndarray | None:
    try:
        frac = fractional_positions(structure)
    except Exception:
        return None

    for axis, periodic in enumerate(structure.pbc):
        if periodic:
            frac[:, axis] = frac[:, axis] % 1.0

    cell = np.asarray(structure.cell_vectors, dtype=float).reshape(3, 3)
    return frac @ cell
