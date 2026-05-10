from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from atomstudio.config import RenderJobConfig
from atomstudio.scene.style_helpers import render_atom_radius, resolve_atom_style_state, resolve_representation
from atomstudio.structure.structure import Structure
from atomstudio.style.registry import get_radius_style

AUTO_SPACE_FILLING_SCALE = "auto"
AUTO_SPACE_FILLING_TARGET_RATIO = 0.96
_MIN_AUTO_SPACE_FILLING_SCALE = 0.01


def resolve_auto_space_filling_scale(
    structure: Structure,
    cfg: RenderJobConfig,
) -> tuple[RenderJobConfig, dict[str, Any]]:
    requested = cfg.structure.space_filling_scale
    if str(requested).strip().lower() != AUTO_SPACE_FILLING_SCALE:
        return cfg, {}

    applied, report = _compute_auto_space_filling_scale(structure, cfg)
    if int(report.get("space_filling_atom_count", 0)) < 2:
        return cfg, {}
    next_cfg = replace(
        cfg,
        structure=replace(cfg.structure, space_filling_scale=float(applied)),
    )
    return next_cfg, {"space_filling_scale": report}


def _compute_auto_space_filling_scale(
    structure: Structure,
    cfg: RenderJobConfig,
) -> tuple[float, dict[str, Any]]:
    default_representation = resolve_representation(cfg.structure.representation, cfg.style.scene_style)
    radius_style = get_radius_style(str(cfg.style.radius_style or "").strip().lower() or "atomic")

    atom_indices: list[int] = []
    positions: list[tuple[float, float, float]] = []
    fixed_radii: list[float] = []
    scalable_radii: list[float] = []

    for atom in structure.atoms:
        state = resolve_atom_style_state(
            atom=atom,
            style_cfg=cfg.style,
            default_representation=default_representation,
        )
        if state.representation != "space_filling":
            continue
        atom_indices.append(int(atom.index))
        positions.append(tuple(float(v) for v in atom.position))
        if state.radius is not None:
            fixed_radii.append(float(state.radius))
            scalable_radii.append(0.0)
            continue
        fixed_radii.append(0.0)
        scalable_radii.append(
            render_atom_radius(
                atom.symbol,
                radius_style=radius_style,
                atom_scale=cfg.structure.atom_scale,
                element_scale=cfg.structure.element_scale,
                representation="space_filling",
                space_filling_scale=1.0,
                radii_scale=cfg.structure.radii_scale,
            )
        )

    base_report = {
        "mode": "auto",
        "requested": "auto",
        "applied": 1.0,
        "adjusted": False,
        "target_contact_ratio": AUTO_SPACE_FILLING_TARGET_RATIO,
        "space_filling_atom_count": len(atom_indices),
    }

    if len(atom_indices) < 2:
        base_report["reason"] = "fewer than two space-filling atoms"
        return 1.0, base_report

    fixed = np.asarray(fixed_radii, dtype=float)
    scalable = np.asarray(scalable_radii, dtype=float)
    pos = np.asarray(positions, dtype=float).reshape((-1, 3))

    max_radius = float(np.max(fixed + scalable))
    if max_radius <= 1e-12:
        base_report["reason"] = "space-filling atoms use zero effective radius"
        return 1.0, base_report

    best_scale = 1.0
    best_pair: tuple[int, int] | None = None
    best_distance: float | None = None
    best_sum_unit: float | None = None
    limited_by_fixed_radii = False

    search_cutoff = max(1e-6, (2.0 * max_radius) / AUTO_SPACE_FILLING_TARGET_RATIO)
    cell_size = search_cutoff
    cells: dict[tuple[int, int, int], list[int]] = {}
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    for i in range(len(atom_indices)):
        cell = tuple(int(v) for v in np.floor(pos[i] / cell_size))
        for dx, dy, dz in neighbor_offsets:
            other = cells.get((cell[0] + dx, cell[1] + dy, cell[2] + dz))
            if other is None:
                continue
            for j in other:
                delta = pos[i] - pos[j]
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-12 or distance > search_cutoff:
                    continue
                numerator = AUTO_SPACE_FILLING_TARGET_RATIO * distance - fixed[i] - fixed[j]
                denom = scalable[i] + scalable[j]
                pair_scale = 1.0
                if denom > 1e-12:
                    pair_scale = numerator / denom
                elif numerator < 0.0:
                    pair_scale = 0.0
                    limited_by_fixed_radii = True
                else:
                    continue
                if numerator < 0.0:
                    limited_by_fixed_radii = True
                if pair_scale < best_scale:
                    best_scale = pair_scale
                    best_pair = (atom_indices[j], atom_indices[i])
                    best_distance = distance
                    best_sum_unit = float(fixed[j] + fixed[i] + scalable[j] + scalable[i])
        cells.setdefault(cell, []).append(i)

    applied = max(_MIN_AUTO_SPACE_FILLING_SCALE, min(1.0, best_scale))
    base_report["applied"] = float(applied)
    base_report["adjusted"] = bool(applied < 1.0 - 1e-12)
    base_report["limited_by_fixed_radii"] = bool(limited_by_fixed_radii)
    if best_pair is None:
        base_report["reason"] = "no overlapping or near-touching space-filling pairs at unit scale"
        return float(applied), base_report

    base_report["reason"] = "reduced scale to preserve separation between nearest space-filling neighbors"
    base_report["limiting_pair"] = [int(best_pair[0]), int(best_pair[1])]
    base_report["limiting_distance"] = float(best_distance) if best_distance is not None else None
    base_report["limiting_sum_radius_at_unit_scale"] = float(best_sum_unit) if best_sum_unit is not None else None
    return float(applied), base_report
