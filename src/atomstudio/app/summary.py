from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def build_structure_summary(
    *,
    structure: Any | None,
    preview_scene: Any | None,
    source_path: str = "",
    graphics: dict[str, Any] | None = None,
) -> str:
    graphics = dict(graphics or {})
    lines: list[str] = []
    lines.extend(_graphics_lines(graphics))
    lines.append("")
    lines.append(str(source_path or getattr(structure, "source_path", "") or "<memory>"))
    lines.append("=" * 84)
    lines.append(f"Title               {_title(structure)}")
    lines.append("")
    lines.append("Lattice type        P")
    lines.append("Space group name    P 1")
    lines.append("Space group number  1")
    lines.append("Setting number      1")
    lines.append("")
    lines.append("Lattice parameters")
    lines.append("")
    lines.append("   a        b        c       alpha    beta     gamma")
    a, b, c, alpha, beta, gamma, volume = _cell_parameters(structure)
    lines.append(f"{a:8.5f} {b:8.5f} {c:8.5f} {alpha:8.4f} {beta:8.4f} {gamma:8.4f}")
    lines.append("")
    lines.append(f"Unit-cell volume = {volume:.6f} Å^3")
    lines.append("")
    lines.append("Structure parameters")
    lines.append("")
    lines.append("                        x          y          z          Occ.     U      Site      Sym.")
    lines.extend(_atom_table_lines(structure))
    lines.append("=" * 84)
    atom_count = _count(getattr(preview_scene, "atoms", None), "count", len(getattr(structure, "atoms", ()) or ()))
    bond_count = _count(getattr(preview_scene, "bonds", None), "count", len(getattr(structure, "bonds", ()) or ()))
    poly_count = _count(getattr(preview_scene, "polyhedra", None), "count", len(getattr(structure, "polyhedra", ()) or ()))
    lines.append(f"Number of polygons and unique vertices on isosurface = 0 (0)")
    lines.append(f"{atom_count} atoms, {bond_count} bonds, {poly_count} polyhedra")
    return "\n".join(lines)


def _graphics_lines(graphics: dict[str, Any]) -> list[str]:
    version = graphics.get("opengl_version") or "unavailable"
    renderer = graphics.get("renderer") or graphics.get("video_configuration") or "unavailable"
    viewport = graphics.get("max_viewport_dims") or "unavailable"
    depth = graphics.get("depth_bits")
    if depth is None:
        depth = "unavailable"
    lines = [
        f"OpenGL version: {version}",
        f"Video configuration: {renderer}",
        f"Maximum supported width and height of the viewport: {viewport}",
        f"OpenGL depth buffer bit: {depth}",
    ]
    diagnostics = [
        ("Qt platform", graphics.get("qt_platform")),
        ("VisPy backend", graphics.get("vispy_backend")),
        ("GL backend", graphics.get("gl_backend")),
        ("Instancing requested", graphics.get("instancing_requested")),
        ("Instancing supported", graphics.get("instancing_supported")),
        ("Instancing reason", graphics.get("instancing_reason")),
        ("Preview renderer", graphics.get("preview_renderer")),
        ("DISPLAY", graphics.get("display")),
        ("WAYLAND_DISPLAY", graphics.get("wayland_display")),
    ]
    lines.extend(f"{label}: {value}" for label, value in diagnostics if value)
    instances = graphics.get("preview_instances")
    if isinstance(instances, dict):
        lines.append(
            "Preview instances: "
            f"mode={instances.get('mode', 'unavailable')}, "
            f"atoms={instances.get('atoms', 0)}, "
            f"bond_segments={instances.get('bond_segments', 0)}"
        )
    return lines


def _title(structure: Any | None) -> str:
    metadata = getattr(structure, "metadata", {}) if structure is not None else {}
    if isinstance(metadata, dict):
        for key in ("title", "name", "formula"):
            value = metadata.get(key)
            if value:
                return str(value)
    atoms = list(getattr(structure, "atoms", ()) or ())
    if not atoms:
        return "Untitled"
    counts = Counter(str(getattr(atom, "symbol", "X")) for atom in atoms)
    return " ".join(f"{symbol}{'' if count == 1 else count}" for symbol, count in sorted(counts.items()))


def _cell_parameters(structure: Any | None) -> tuple[float, float, float, float, float, float, float]:
    vectors = _cell_vectors(structure)
    if vectors is None:
        return 0.0, 0.0, 0.0, 90.0, 90.0, 90.0, 0.0
    a_vec, b_vec, c_vec = vectors
    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))
    c = float(np.linalg.norm(c_vec))
    alpha = _angle_degrees(b_vec, c_vec)
    beta = _angle_degrees(a_vec, c_vec)
    gamma = _angle_degrees(a_vec, b_vec)
    volume = abs(float(np.linalg.det(vectors)))
    return a, b, c, alpha, beta, gamma, volume


def _cell_vectors(structure: Any | None) -> np.ndarray | None:
    raw = getattr(structure, "cell_vectors", None) if structure is not None else None
    if raw is None:
        return None
    try:
        vectors = np.asarray(raw, dtype=float).reshape((3, 3))
    except Exception:
        return None
    if not np.any(np.abs(vectors) > 1e-8):
        return None
    return vectors


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 90.0
    value = float(np.dot(left, right) / denom)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, value)))))


def _atom_table_lines(structure: Any | None) -> list[str]:
    atoms = list(getattr(structure, "atoms", ()) or ())
    if not atoms:
        return ["   <no atoms>"]
    frac = _fractional_positions(structure, atoms)
    out: list[str] = []
    for row, atom in enumerate(atoms, start=1):
        symbol = str(getattr(atom, "symbol", "X"))
        tag = str(getattr(atom, "tag", "") or f"{symbol}{row - 1}")
        x, y, z = frac[row - 1]
        out.append(
            f"{row:4d} {symbol:<4} {tag:<10} {x:10.5f} {y:10.5f} {z:10.5f}    1.000   -0.000    1a         1"
        )
    return out


def _fractional_positions(structure: Any | None, atoms: list[Any]) -> np.ndarray:
    positions = np.asarray([getattr(atom, "position", (0.0, 0.0, 0.0)) for atom in atoms], dtype=float).reshape((-1, 3))
    vectors = _cell_vectors(structure)
    if vectors is None or abs(float(np.linalg.det(vectors))) <= 1e-12:
        return positions
    return positions @ np.linalg.inv(vectors)


def _count(value: Any, attr: str, fallback: int) -> int:
    if value is None:
        return int(fallback)
    try:
        return int(getattr(value, attr))
    except Exception:
        return int(fallback)
