#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from ase import Atoms

from atomstudio.structure import AtomSelector, Structure


def _build_single_layer_hex_lattice(nx: int, ny: int, spacing: float) -> Atoms:
    positions: list[tuple[float, float, float]] = []
    for j in range(ny):
        y = j * (math.sqrt(3.0) * 0.5 * spacing)
        shift = 0.5 * spacing if (j % 2) else 0.0
        for i in range(nx):
            x = i * spacing + shift
            positions.append((x, y, 0.0))
    atoms = Atoms("N" * len(positions), positions=positions, pbc=False)
    atoms.center(vacuum=4.0)
    return atoms


def _argmin_xy_distance(positions: np.ndarray, target_xy: np.ndarray) -> int:
    delta = positions[:, :2] - target_xy[None, :]
    dist2 = np.sum(delta * delta, axis=1)
    return int(np.argmin(dist2))


def _pick_top_middle_index(positions: np.ndarray) -> int:
    y_threshold = float(np.quantile(positions[:, 1], 0.82))
    candidate_ids = np.where(positions[:, 1] >= y_threshold)[0]
    center_x = float(np.mean(positions[:, 0]))
    if len(candidate_ids) == 0:
        return _argmin_xy_distance(positions, np.array([center_x, float(np.max(positions[:, 1]))], dtype=float))
    xs = positions[candidate_ids, 0]
    return int(candidate_ids[int(np.argmin(np.abs(xs - center_x)))])


def build_frenkel_layer(
    *,
    nx: int = 16,
    ny: int = 9,
    spacing: float = 1.38,
    rattle_stdev: float = 0.02,
    seed: int = 7,
) -> tuple[Structure, np.ndarray]:
    atoms = _build_single_layer_hex_lattice(nx=nx, ny=ny, spacing=spacing)
    original_positions = atoms.get_positions()
    center_xy = np.mean(original_positions[:, :2], axis=0)

    vacancy_index = _argmin_xy_distance(original_positions, center_xy)
    vacancy_site = original_positions[vacancy_index].copy()
    del atoms[vacancy_index]

    interstitial_site = vacancy_site + np.array([0.62 * spacing, -0.45 * spacing, 0.0], dtype=float)
    atoms += Atoms("N", positions=[tuple(interstitial_site.tolist())])
    interstitial_index = len(atoms) - 1

    if rattle_stdev > 0.0:
        atoms.rattle(stdev=float(rattle_stdev), seed=int(seed))
    jittered_positions = atoms.get_positions()
    jittered_positions[:, 2] = 0.0
    atoms.set_positions(jittered_positions)

    structure = Structure.from_ase(atoms)

    # Base lattice color.
    for atom in structure.atoms:
        atom.color = (0.22, 0.56, 0.83, 1.0)

    # Interstitial atom (red).
    structure.assign_atom_style(
        AtomSelector(indices=[interstitial_index]),
        color=(0.83, 0.22, 0.24, 1.0),
        radius=0.58,
    )

    # A highlighted atom near top-middle (purple), close to the reference figure.
    positions_now = np.array(structure.positions, dtype=float)
    accent_index = _pick_top_middle_index(positions_now)
    if accent_index != interstitial_index:
        structure.assign_atom_style(
            AtomSelector(indices=[accent_index]),
            color=(0.72, 0.58, 0.77, 1.0),
            radius=0.66,
        )

    # A pale cyan atom near upper-left as a minor visual accent.
    upper_left_target = np.array([np.min(positions_now[:, 0]) + 2.0 * spacing, np.max(positions_now[:, 1]) - spacing], dtype=float)
    cyan_index = _argmin_xy_distance(positions_now, upper_left_target)
    if cyan_index not in {interstitial_index, accent_index}:
        structure.assign_atom_style(
            AtomSelector(indices=[cyan_index]),
            color=(0.63, 0.84, 0.93, 1.0),
            radius=0.62,
        )

    return structure, vacancy_site


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single-layer cartoon Frenkel-pair-like schematic")
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=9)
    parser.add_argument("--spacing", type=float, default=1.38, help="In-plane atom spacing (Angstrom)")
    parser.add_argument("--rattle", type=float, default=0.02, help="Very small random displacement stdev (Angstrom)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=1700)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).expanduser().resolve() if args.out else (root / "outputs" / "frenkel_pair_cartoon.png").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    structure, vacancy_site = build_frenkel_layer(
        nx=int(args.nx),
        ny=int(args.ny),
        spacing=float(args.spacing),
        rattle_stdev=float(args.rattle),
        seed=int(args.seed),
    )

    rendered = structure.get_image(
        str(out),
        style="handdrawn",
        representation="space_filling",
        engine="eevee",
        draw_cell=False,
        draw_bonds=False,
        view="top",
        camera_view="top",
        frame_scale=1.02,
        return_type="path",
        overrides={
            "render": {
                "transparent_bg": False,
                "resolution": [int(args.width), int(args.height)],
                "samples": int(args.samples),
            },
            "style": {
                "handdrawn": {
                    "background": [0.93, 0.93, 0.93, 1.0],
                    "jmol_desaturate": 0.12,
                    "jmol_lighten": 0.05,
                    "shadow_strength": 0.36,
                    "highlight_strength": 0.18,
                    "outline_surface": 2.6,
                    "outline_molecule": 2.8,
                }
            },
            "lighting": {"light_style": "handdrawn_soft", "intensity": 1.05},
            "camera": {"fit_padding": 0.04},
            "structure": {"space_filling_scale": 1.0},
        },
    )

    print(f"Rendered: {rendered}")
    print(f"rattle_stdev_A: {float(args.rattle):.4f}")
    print(f"vacancy_site_A: [{vacancy_site[0]:.3f}, {vacancy_site[1]:.3f}, {vacancy_site[2]:.3f}]")


if __name__ == "__main__":
    main()
