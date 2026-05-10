#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.morse import MorsePotential
from ase.constraints import FixedPlane, Hookean
from ase.io import write
from ase.optimize import FIRE


def _triangular_layer(nx: int, ny: int, a: float) -> Atoms:
    positions = []
    for j in range(int(ny)):
        y = j * (np.sqrt(3.0) * 0.5 * a)
        shift = 0.5 * a if (j % 2) else 0.0
        for i in range(int(nx)):
            x = i * a + shift
            positions.append((x, y, 0.0))
    atoms = Atoms("Cu" * len(positions), positions=positions, pbc=False)
    atoms.set_cell([nx * a + 6.0, ny * np.sqrt(3.0) * 0.5 * a + 6.0, 20.0])
    atoms.center()
    return atoms


def _nearest_index(points_xy: np.ndarray, target_xy: np.ndarray) -> int:
    d2 = np.sum((points_xy - target_xy[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def _interior_indices(points_xy: np.ndarray, a: float, margin: float = 1.4) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    m = float(margin) * float(a)
    keep = (x > x.min() + m) & (x < x.max() - m) & (y > y.min() + m) & (y < y.max() - m)
    return np.where(keep)[0]


def _neighbor_indices(points_xy: np.ndarray, center_idx: int, cutoff: float) -> np.ndarray:
    d = np.linalg.norm(points_xy - points_xy[int(center_idx)][None, :], axis=1)
    mask = (d > 1e-10) & (d <= float(cutoff))
    return np.where(mask)[0]


def _apply_radial_neighbor_shift(
    points: np.ndarray,
    *,
    center_idx: int,
    neighbor_ids: np.ndarray,
    shift: float,
) -> None:
    c = points[int(center_idx), :2].copy()
    for j in neighbor_ids:
        v = points[int(j), :2] - c
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            continue
        points[int(j), :2] = points[int(j), :2] + (float(shift) * v / n)


def _find_hollow_centers(points_xy: np.ndarray, a: float) -> np.ndarray:
    n = len(points_xy)
    cutoff = 1.22 * float(a)
    adj = [set() for _ in range(n)]
    for i in range(n):
        pi = points_xy[i]
        for j in range(i + 1, n):
            if float(np.linalg.norm(points_xy[j] - pi)) <= cutoff:
                adj[i].add(j)
                adj[j].add(i)

    centers: list[np.ndarray] = []
    for i in range(n):
        for j in [x for x in adj[i] if x > i]:
            common = [k for k in adj[i].intersection(adj[j]) if k > j]
            for k in common:
                dij = float(np.linalg.norm(points_xy[i] - points_xy[j]))
                dik = float(np.linalg.norm(points_xy[i] - points_xy[k]))
                djk = float(np.linalg.norm(points_xy[j] - points_xy[k]))
                if max(abs(dij - a), abs(dik - a), abs(djk - a)) > 0.28 * float(a):
                    continue
                centers.append((points_xy[i] + points_xy[j] + points_xy[k]) / 3.0)
    if not centers:
        return np.zeros((0, 2), dtype=float)
    return np.array(centers, dtype=float)


def _pick_hollow_near_target(
    hollows: np.ndarray,
    *,
    target_xy: np.ndarray,
    vacancy_xy: np.ndarray | None = None,
    a: float = 1.0,
    right_lower_of_vacancy: bool = False,
    min_vac_dist: float = 0.0,
    max_vac_dist: float = 1.0e9,
) -> np.ndarray:
    if len(hollows) == 0:
        return target_xy.copy()
    keep = np.ones(len(hollows), dtype=bool)
    if vacancy_xy is not None:
        dv = np.linalg.norm(hollows - vacancy_xy[None, :], axis=1)
        keep &= (dv >= float(min_vac_dist)) & (dv <= float(max_vac_dist))
        if right_lower_of_vacancy:
            keep &= (hollows[:, 0] > float(vacancy_xy[0]) + 0.10 * float(a))
            keep &= (hollows[:, 1] < float(vacancy_xy[1]) - 0.10 * float(a))
    cands = hollows[keep]
    if len(cands) == 0:
        cands = hollows
    d2 = np.sum((cands - target_xy[None, :]) ** 2, axis=1)
    return cands[int(np.argmin(d2))]


def build_defective_layer(nx: int, ny: int, a: float) -> tuple[Atoms, dict]:
    atoms = _triangular_layer(nx=nx, ny=ny, a=a)
    p = atoms.get_positions()
    xy = p[:, :2]
    center = np.array([xy[:, 0].mean(), xy[:, 1].mean()], dtype=float)
    span = np.array([np.ptp(xy[:, 0]), np.ptp(xy[:, 1])], dtype=float)
    interior = _interior_indices(xy, a=a)

    # 1) Large substitution defect (much larger than host, marked by Au).
    large_target = center + np.array([-0.30 * span[0], 0.24 * span[1]], dtype=float)
    large_sub_idx = int(interior[_nearest_index(xy[interior], large_target)])
    atoms[large_sub_idx].symbol = "Au"

    # 2) Small substitution defect (much smaller than host, marked by Ni).
    small_target = center + np.array([0.20 * span[0], 0.20 * span[1]], dtype=float)
    small_sub_idx = int(interior[_nearest_index(xy[interior], small_target)])
    if small_sub_idx == large_sub_idx:
        d2 = np.sum((xy[interior] - small_target[None, :]) ** 2, axis=1)
        order = np.argsort(d2)
        for k in order:
            cand = int(interior[int(k)])
            if cand != large_sub_idx:
                small_sub_idx = cand
                break
    atoms[small_sub_idx].symbol = "Ni"

    # 3) Emulate larger/smaller atomic size by perturbing local first-shell neighbors,
    # then let Morse relaxation settle.
    p = atoms.get_positions()
    xy = p[:, :2]
    large_nbr = _neighbor_indices(xy, center_idx=large_sub_idx, cutoff=1.35 * a)
    small_nbr = _neighbor_indices(xy, center_idx=small_sub_idx, cutoff=1.35 * a)
    _apply_radial_neighbor_shift(p, center_idx=large_sub_idx, neighbor_ids=large_nbr, shift=0.40 * a)
    _apply_radial_neighbor_shift(p, center_idx=small_sub_idx, neighbor_ids=small_nbr, shift=-0.30 * a)
    atoms.set_positions(p)

    # 4) Keep the original vacancy + interstitial defect (extra vacancy not from Frenkel).
    p = atoms.get_positions()
    xy = p[:, :2]
    interior = _interior_indices(xy, a=a)
    reserved = {large_sub_idx, small_sub_idx}

    # Put the "extra vacancy" at lower-left so it is clearly separated from Frenkel vacancy.
    vac0_target = center + np.array([-0.42 * span[0], -0.34 * span[1]], dtype=float)
    vac1_target = center + np.array([0.22 * span[0], -0.10 * span[1]], dtype=float)

    def _pick_vacancy(target: np.ndarray, forbidden: set[int]) -> int:
        d2 = np.sum((xy[interior] - target[None, :]) ** 2, axis=1)
        for k in np.argsort(d2):
            cand = int(interior[int(k)])
            if cand not in forbidden:
                return cand
        raise RuntimeError("failed to pick vacancy index")

    vac0_idx_old = _pick_vacancy(vac0_target, reserved)
    reserved.add(vac0_idx_old)
    vac1_idx_old = _pick_vacancy(vac1_target, reserved)
    reserved.add(vac1_idx_old)

    vacancy0_site = p[vac0_idx_old].copy()
    vacancy1_site = p[vac1_idx_old].copy()

    removed_sorted = sorted([vac0_idx_old, vac1_idx_old])
    for ridx in sorted(removed_sorted, reverse=True):
        del atoms[int(ridx)]

    def _remap_after_deletions(idx: int) -> int:
        shift = sum(1 for d in removed_sorted if d < idx)
        return int(idx - shift)

    large_sub_idx = _remap_after_deletions(large_sub_idx)
    small_sub_idx = _remap_after_deletions(small_sub_idx)

    # Add two interstitials with user-requested layout.
    # Snap to nearby hollow centers so they stay "interstitial-like" after relaxation.
    mid = 0.5 * (vacancy0_site + vacancy1_site)
    p_now = atoms.get_positions()
    hollows = _find_hollow_centers(p_now[:, :2], a=a)

    int0_target_xy = (mid + np.array([0.10 * a, -1.10 * a, 0.0], dtype=float))[:2]
    int0_xy = _pick_hollow_near_target(
        hollows,
        target_xy=int0_target_xy,
        vacancy_xy=vacancy0_site[:2],
        a=a,
        min_vac_dist=1.2 * float(a),
        max_vac_dist=8.0 * float(a),
    )

    int1_target_xy = (vacancy1_site.copy() + np.array([1.35 * a, -0.95 * a, 0.0], dtype=float))[:2]
    int1_xy = _pick_hollow_near_target(
        hollows,
        target_xy=int1_target_xy,
        vacancy_xy=vacancy1_site[:2],
        a=a,
        right_lower_of_vacancy=True,
        min_vac_dist=1.5 * float(a),
        max_vac_dist=2.6 * float(a),
    )
    int0_pos = np.array([float(int0_xy[0]), float(int0_xy[1]), float(vacancy0_site[2])], dtype=float)
    int1_pos = np.array([float(int1_xy[0]), float(int1_xy[1]), float(vacancy1_site[2])], dtype=float)
    # Use distinct species so interstitials are visually identifiable in rendering.
    atoms += Atoms(symbols=["Ag", "Pd"], positions=[tuple(int0_pos), tuple(int1_pos)])
    int0_idx = len(atoms) - 2
    int1_idx = len(atoms) - 1

    defect_code = np.zeros(len(atoms), dtype=int)  # 0 host
    defect_code[large_sub_idx] = 1  # large substitution
    defect_code[small_sub_idx] = 2  # small substitution
    defect_code[int0_idx] = 3  # extra interstitial
    defect_code[int1_idx] = 4  # interstitial (Frenkel)
    atoms.new_array("defect_code", defect_code)

    # Soft anchors to keep defects visible after relaxation (not fixed atoms):
    # - keep both interstitials near their target interstitial sites;
    # - softly keep first-shell vacancy neighbors around original shell positions.
    pos_now = atoms.get_positions()
    soft_anchors: list[dict] = []
    soft_anchors.append({"index": int(int0_idx), "point": [float(int0_pos[0]), float(int0_pos[1]), float(int0_pos[2])], "k": 24.0, "rt": 0.20 * float(a)})
    # Frenkel interstitial: strong always-on tether to keep off-lattice identity.
    soft_anchors.append({"index": int(int1_idx), "point": [float(int1_pos[0]), float(int1_pos[1]), float(int1_pos[2])], "k": 130.0, "rt": 0.0})

    vacancy_sites = [vacancy0_site, vacancy1_site]
    for site in vacancy_sites:
        d = np.linalg.norm(pos_now[:, :2] - site[None, :2], axis=1)
        order = np.argsort(d)
        picked = []
        for idx in order:
            ii = int(idx)
            if ii in {int0_idx, int1_idx}:
                continue
            if d[ii] < 0.45 * a or d[ii] > 1.50 * a:
                continue
            picked.append(ii)
            if len(picked) == 6:
                break
        for ii in picked:
            anchor = pos_now[ii].copy()
            soft_anchors.append({"index": int(ii), "point": [float(anchor[0]), float(anchor[1]), float(anchor[2])], "k": 12.0, "rt": 0.06 * float(a)})

    # Keep substitution contrast visible: large-shell expanded, small-shell contracted.
    d_large = np.linalg.norm(pos_now[:, :2] - pos_now[large_sub_idx][None, :2], axis=1)
    d_small = np.linalg.norm(pos_now[:, :2] - pos_now[small_sub_idx][None, :2], axis=1)
    large_shell = [int(i) for i in np.where((d_large > 1e-8) & (d_large < 1.50 * a))[0][:6]]
    small_shell = [int(i) for i in np.where((d_small > 1e-8) & (d_small < 1.50 * a))[0][:6]]
    for ii in large_shell:
        anchor = pos_now[ii].copy()
        soft_anchors.append({"index": int(ii), "point": [float(anchor[0]), float(anchor[1]), float(anchor[2])], "k": 9.0, "rt": 0.05 * float(a)})
    for ii in small_shell:
        anchor = pos_now[ii].copy()
        soft_anchors.append({"index": int(ii), "point": [float(anchor[0]), float(anchor[1]), float(anchor[2])], "k": 9.0, "rt": 0.05 * float(a)})

    meta = {
        "lattice": {"type": "triangular_monolayer", "element": "Cu", "a": float(a), "nx": int(nx), "ny": int(ny)},
        "defects": {
            "vacancy_site_extra": [float(vacancy0_site[0]), float(vacancy0_site[1]), float(vacancy0_site[2])],
            "vacancy_site_frenkel": [float(vacancy1_site[0]), float(vacancy1_site[1]), float(vacancy1_site[2])],
            "vacancy_pair_distance_initial_A": float(np.linalg.norm(vacancy0_site[:2] - vacancy1_site[:2])),
            "extra_vacancy_to_interstitial_distance_initial_A": float(np.linalg.norm(int0_pos[:2] - vacancy0_site[:2])),
            "frenkel_vacancy_to_interstitial_distance_initial_A": float(np.linalg.norm(int1_pos[:2] - vacancy1_site[:2])),
            "large_substitution_index": int(large_sub_idx),
            "large_substitution_species": "Au_on_Cu",
            "small_substitution_index": int(small_sub_idx),
            "small_substitution_species": "Ni_on_Cu",
            "interstitial_index_extra": int(int0_idx),
            "interstitial_index_frenkel": int(int1_idx),
            "interstitial_species_extra": "Ag_i (extra defect)",
            "interstitial_species_frenkel": "Pd_i (Frenkel pair)",
            "interstitial_target_extra": [float(int0_pos[0]), float(int0_pos[1]), float(int0_pos[2])],
            "interstitial_target_frenkel": [float(int1_pos[0]), float(int1_pos[1]), float(int1_pos[2])],
        },
        "soft_anchor_count": int(len(soft_anchors)),
        "soft_anchors": soft_anchors,
    }
    return atoms, meta


def relax_with_potential(
    atoms: Atoms,
    *,
    meta: dict,
    calculator: str,
    a: float,
    fmax: float,
    steps: int,
    log_path: Path,
    traj_path: Path,
    morse_epsilon: float,
    morse_rho0: float,
) -> dict:
    constraints = [FixedPlane(indices=list(range(len(atoms))), direction=[0, 0, 1])]
    for item in meta.get("soft_anchors", []):
        constraints.append(
            Hookean(
                int(item["index"]),
                [float(item["point"][0]), float(item["point"][1]), float(item["point"][2])],
                k=float(item["k"]),
                rt=float(item["rt"]),
            )
        )
    atoms.set_constraint(constraints)
    calc_name = str(calculator).strip().lower()
    if calc_name == "emt":
        atoms.calc = EMT()
    elif calc_name == "morse":
        atoms.calc = MorsePotential(
            epsilon=float(morse_epsilon),
            rho0=float(morse_rho0),
            r0=float(a),
            rcut1=1.45 * float(a),
            rcut2=1.75 * float(a),
        )
    else:
        raise ValueError("calculator must be one of: emt, morse")
    e0 = float(atoms.get_potential_energy())

    dyn = FIRE(atoms, trajectory=str(traj_path), logfile=str(log_path))
    dyn.run(fmax=float(fmax), steps=int(steps))

    e1 = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    max_force = float(np.max(np.linalg.norm(forces, axis=1)))
    return {
        "calculator": calc_name,
        "energy_initial_eV": e0,
        "energy_final_eV": e1,
        "energy_drop_eV": e0 - e1,
        "max_force_eV_per_A": max_force,
        "converged": bool(max_force <= float(fmax) + 1e-12),
        "n_fixed_boundary_atoms": 0,
        "n_soft_anchors": int(len(meta.get("soft_anchors", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2D metallic defective structure and relax it with ASE EMT/Morse")
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--a", type=float, default=2.56, help="Nearest-neighbor spacing in Angstrom")
    parser.add_argument("--fmax", type=float, default=0.03, help="FIRE convergence force threshold (eV/Angstrom)")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--calculator", choices=["morse", "emt"], default="morse")
    parser.add_argument("--morse-epsilon", type=float, default=0.35, help="Morse epsilon (eV), only used when --calculator morse")
    parser.add_argument("--morse-rho0", type=float, default=5.0, help="Morse rho0, only used when --calculator morse")
    parser.add_argument("--out-dir", type=str, default="outputs/emt_2d_metal_defects")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms, meta = build_defective_layer(nx=int(args.nx), ny=int(args.ny), a=float(args.a))
    atoms_initial = atoms.copy()

    initial_xyz = out_dir / "metal_2d_defects_initial.xyz"
    initial_extxyz = out_dir / "metal_2d_defects_initial.extxyz"
    final_xyz = out_dir / "metal_2d_defects_relaxed.xyz"
    final_extxyz = out_dir / "metal_2d_defects_relaxed.extxyz"
    traj_path = out_dir / "metal_2d_defects_fire.traj"
    log_path = out_dir / "metal_2d_defects_fire.log"
    report_path = out_dir / "metal_2d_defects_report.json"

    write(str(initial_xyz), atoms_initial)
    write(str(initial_extxyz), atoms_initial)
    relax_info = relax_with_potential(
        atoms,
        meta=meta,
        calculator=str(args.calculator),
        a=float(args.a),
        fmax=float(args.fmax),
        steps=int(args.steps),
        log_path=log_path,
        traj_path=traj_path,
        morse_epsilon=float(args.morse_epsilon),
        morse_rho0=float(args.morse_rho0),
    )
    write(str(final_xyz), atoms)
    write(str(final_extxyz), atoms)

    disp = np.linalg.norm(atoms.get_positions() - atoms_initial.get_positions(), axis=1)
    pos = atoms.get_positions()
    defects = meta.get("defects", {})
    i_extra = int(defects.get("interstitial_index_extra", -1))
    i_frenkel = int(defects.get("interstitial_index_frenkel", -1))
    t_extra = np.array(defects.get("interstitial_target_extra", [0.0, 0.0, 0.0]), dtype=float)
    t_frenkel = np.array(defects.get("interstitial_target_frenkel", [0.0, 0.0, 0.0]), dtype=float)
    v_extra = np.array(defects.get("vacancy_site_extra", [0.0, 0.0, 0.0]), dtype=float)
    v_frenkel = np.array(defects.get("vacancy_site_frenkel", [0.0, 0.0, 0.0]), dtype=float)
    d_extra_to_target = float(np.linalg.norm(pos[i_extra] - t_extra)) if 0 <= i_extra < len(pos) else None
    d_frenkel_to_target = float(np.linalg.norm(pos[i_frenkel] - t_frenkel)) if 0 <= i_frenkel < len(pos) else None
    d_extra_to_vac = float(np.linalg.norm(pos[i_extra][:2] - v_extra[:2])) if 0 <= i_extra < len(pos) else None
    d_frenkel_to_vac = float(np.linalg.norm(pos[i_frenkel][:2] - v_frenkel[:2])) if 0 <= i_frenkel < len(pos) else None

    report = {
        "meta": meta,
        "optimization": relax_info,
        "displacement_A": {
            "max": float(np.max(disp)),
            "mean": float(np.mean(disp)),
            "p95": float(np.quantile(disp, 0.95)),
        },
        "defect_visibility_check": {
            "extra_interstitial_to_target_A": d_extra_to_target,
            "frenkel_interstitial_to_target_A": d_frenkel_to_target,
            "extra_interstitial_to_vacancy_A": d_extra_to_vac,
            "frenkel_interstitial_to_vacancy_A": d_frenkel_to_vac,
        },
        "files": {
            "initial_xyz": str(initial_xyz),
            "initial_extxyz": str(initial_extxyz),
            "final_xyz": str(final_xyz),
            "final_extxyz": str(final_extxyz),
            "traj": str(traj_path),
            "log": str(log_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Initial XYZ: {initial_xyz}")
    print(f"Final XYZ:   {final_xyz}")
    print(f"Report:      {report_path}")
    print(json.dumps(report["optimization"], indent=2, ensure_ascii=False))
    print(json.dumps(report["displacement_A"], indent=2, ensure_ascii=False))
    print(json.dumps(report["defect_visibility_check"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
