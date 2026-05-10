from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np
from ase import Atoms


DefectType = Literal["substitution", "interstitial", "vacancy"]


@dataclass
class DefectRelaxResult:
    atoms: Atoms
    defect_type: DefectType
    defect_index: int | None
    vacancy_site: tuple[float, float] | None
    fixed_indices: list[int]
    max_displacement: float
    mean_displacement: float
    iterations: int
    converged: bool
    energy: float


def _triangular_positions(nx: int, ny: int, a: float) -> np.ndarray:
    out: list[tuple[float, float]] = []
    for j in range(int(ny)):
        y = j * (sqrt(3.0) * 0.5 * a)
        shift = 0.5 * a if (j % 2) else 0.0
        for i in range(int(nx)):
            x = i * a + shift
            out.append((x, y))
    return np.array(out, dtype=float)


def _nearest_index(points: np.ndarray, target: tuple[float, float]) -> int:
    t = np.array(target, dtype=float)
    d2 = np.sum((points - t[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def _choose_default_target(points: np.ndarray) -> tuple[float, float]:
    return (float(np.mean(points[:, 0])), float(np.mean(points[:, 1])))


def _build_edges(points: np.ndarray, cutoff: float) -> list[tuple[int, int]]:
    n = len(points)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[j] - pi))
            if d <= cutoff:
                edges.append((i, j))
    return edges


def _boundary_mask(points: np.ndarray, a: float, margin: float = 0.80) -> np.ndarray:
    minx = float(np.min(points[:, 0]))
    maxx = float(np.max(points[:, 0]))
    miny = float(np.min(points[:, 1]))
    maxy = float(np.max(points[:, 1]))
    m = float(margin) * float(a)
    fixed = (
        (points[:, 0] <= minx + m)
        | (points[:, 0] >= maxx - m)
        | (points[:, 1] <= miny + m)
        | (points[:, 1] >= maxy - m)
    )
    return fixed


def _energy_and_forces(
    x: np.ndarray,
    *,
    edges: list[tuple[int, int]],
    edge_rest: np.ndarray,
    radii: np.ndarray,
    k_spring: float,
    k_repulsion: float,
    repulsion_scale: float,
    a: float,
    vacancy_neighbors: list[int],
    vacancy_site: np.ndarray | None,
    vacancy_pull: float,
    vacancy_target_scale: float,
) -> tuple[float, np.ndarray]:
    n = len(x)
    f = np.zeros((n, 2), dtype=float)
    e = 0.0
    eps = 1e-12

    for idx, (i, j) in enumerate(edges):
        v = x[j] - x[i]
        d = float(np.linalg.norm(v))
        if d < eps:
            continue
        u = v / d
        dr = d - float(edge_rest[idx])
        e += 0.5 * k_spring * dr * dr
        force = k_spring * dr * u
        f[i] += force
        f[j] -= force

    for i in range(n):
        for j in range(i + 1, n):
            v = x[j] - x[i]
            d = float(np.linalg.norm(v))
            if d < eps:
                continue
            contact = float(repulsion_scale) * float(a) * float(radii[i] + radii[j]) * 0.5
            if d >= contact:
                continue
            u = v / d
            dr = d - contact
            e += 0.5 * k_repulsion * dr * dr
            force = k_repulsion * dr * u
            f[i] += force
            f[j] -= force

    if vacancy_site is not None and vacancy_neighbors and vacancy_pull > 0.0:
        tgt = float(vacancy_target_scale) * float(a)
        for i in vacancy_neighbors:
            v = x[i] - vacancy_site
            d = float(np.linalg.norm(v))
            if d < eps:
                continue
            u = v / d
            dr = d - tgt
            e += 0.5 * vacancy_pull * dr * dr
            f[i] += -vacancy_pull * dr * u

    return e, f


def solve_defect_relaxation(
    *,
    nx: int = 16,
    ny: int = 9,
    lattice_constant: float = 1.0,
    defect_type: DefectType = "interstitial",
    defect_target_xy: tuple[float, float] | None = None,
    substitution_size_scale: float = 1.25,
    interstitial_size_scale: float = 1.00,
    interstitial_offset_xy: tuple[float, float] | None = None,
    rattle: float = 0.0,
    seed: int = 7,
    k_spring: float = 10.0,
    k_repulsion: float = 80.0,
    repulsion_scale: float = 0.95,
    vacancy_pull: float = 3.0,
    vacancy_target_scale: float = 0.80,
    max_iter: int = 1600,
    tol_force: float = 1e-4,
) -> DefectRelaxResult:
    a = float(lattice_constant)
    base = _triangular_positions(nx=nx, ny=ny, a=a)
    host_r = np.ones(len(base), dtype=float)
    host_symbols = ["N"] * len(base)

    tgt = _choose_default_target(base) if defect_target_xy is None else tuple(map(float, defect_target_xy))
    defect_index: int | None = None
    vacancy_site: np.ndarray | None = None
    vacancy_neighbors: list[int] = []

    points = base.copy()
    radii = host_r.copy()
    symbols = list(host_symbols)

    if defect_type == "substitution":
        defect_index = _nearest_index(points, tgt)
        radii[defect_index] *= float(substitution_size_scale)
        symbols[defect_index] = "Si"
    elif defect_type == "interstitial":
        c = np.array(tgt, dtype=float)
        if interstitial_offset_xy is not None:
            c = c + np.array(interstitial_offset_xy, dtype=float)
        points = np.vstack([points, c[None, :]])
        radii = np.concatenate([radii, np.array([float(interstitial_size_scale)], dtype=float)])
        symbols.append("O")
        defect_index = len(points) - 1
    elif defect_type == "vacancy":
        remove_idx = _nearest_index(points, tgt)
        vacancy_site = points[remove_idx].copy()
        old_edges = _build_edges(points, cutoff=1.25 * a)
        old_neighbor_ids = sorted({j if i == remove_idx else i for i, j in old_edges if (i == remove_idx or j == remove_idx)})
        keep = np.ones(len(points), dtype=bool)
        keep[remove_idx] = False
        points = points[keep]
        radii = radii[keep]
        symbols = [s for i, s in enumerate(symbols) if i != remove_idx]
        remap = {}
        ni = 0
        for oi in range(len(keep)):
            if keep[oi]:
                remap[oi] = ni
                ni += 1
        vacancy_neighbors = [remap[i] for i in old_neighbor_ids if i in remap]
    else:
        raise ValueError("defect_type must be one of: substitution, interstitial, vacancy")

    if float(rattle) > 0.0:
        rng = np.random.default_rng(int(seed))
        points = points + rng.normal(0.0, float(rattle), size=points.shape)

    edges = _build_edges(points, cutoff=1.25 * a)
    edge_rest = []
    for i, j in edges:
        ratio = 0.5 * float(radii[i] + radii[j])
        edge_rest.append(a * ratio)
    edge_rest_arr = np.array(edge_rest, dtype=float)

    fixed_mask = _boundary_mask(points, a=a, margin=0.80)
    free_mask = ~fixed_mask
    if int(np.sum(free_mask)) == 0:
        free_mask[:] = True

    x = points.copy()
    e, f = _energy_and_forces(
        x,
        edges=edges,
        edge_rest=edge_rest_arr,
        radii=radii,
        k_spring=float(k_spring),
        k_repulsion=float(k_repulsion),
        repulsion_scale=float(repulsion_scale),
        a=a,
        vacancy_neighbors=vacancy_neighbors,
        vacancy_site=vacancy_site,
        vacancy_pull=float(vacancy_pull),
        vacancy_target_scale=float(vacancy_target_scale),
    )

    step = 0.02 * a
    converged = False
    iters = 0
    for it in range(int(max_iter)):
        iters = it + 1
        ff = f.copy()
        ff[~free_mask] = 0.0
        max_force = float(np.max(np.linalg.norm(ff, axis=1)))
        if max_force < float(tol_force):
            converged = True
            break

        trial = x + step * ff
        e_trial, f_trial = _energy_and_forces(
            trial,
            edges=edges,
            edge_rest=edge_rest_arr,
            radii=radii,
            k_spring=float(k_spring),
            k_repulsion=float(k_repulsion),
            repulsion_scale=float(repulsion_scale),
            a=a,
            vacancy_neighbors=vacancy_neighbors,
            vacancy_site=vacancy_site,
            vacancy_pull=float(vacancy_pull),
            vacancy_target_scale=float(vacancy_target_scale),
        )
        if e_trial <= e:
            x = trial
            e = e_trial
            f = f_trial
            step = min(step * 1.03, 0.08 * a)
        else:
            step = max(step * 0.5, 1e-5 * a)

    disp = np.linalg.norm(x - points, axis=1)
    pos3 = np.column_stack([x[:, 0], x[:, 1], np.zeros(len(x), dtype=float)])
    atoms = Atoms(symbols=symbols, positions=pos3, pbc=False)
    atoms.set_cell([float(np.max(x[:, 0]) - np.min(x[:, 0]) + 4.0 * a), float(np.max(x[:, 1]) - np.min(x[:, 1]) + 4.0 * a), 20.0])
    before_center = atoms.get_positions().copy()
    atoms.center()
    after_center = atoms.get_positions()
    shift_xy = np.array([float(after_center[0, 0] - before_center[0, 0]), float(after_center[0, 1] - before_center[0, 1])], dtype=float)
    vacancy_site_out = None if vacancy_site is None else (float(vacancy_site[0] + shift_xy[0]), float(vacancy_site[1] + shift_xy[1]))

    return DefectRelaxResult(
        atoms=atoms,
        defect_type=defect_type,
        defect_index=defect_index,
        vacancy_site=vacancy_site_out,
        fixed_indices=[int(i) for i in np.where(fixed_mask)[0]],
        max_displacement=float(np.max(disp)) if len(disp) else 0.0,
        mean_displacement=float(np.mean(disp)) if len(disp) else 0.0,
        iterations=int(iters),
        converged=bool(converged),
        energy=float(e),
    )
