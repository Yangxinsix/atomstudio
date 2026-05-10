#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


def _nearest_index(points: np.ndarray, target_xy: tuple[float, float]) -> int:
    target = np.array(target_xy, dtype=float)
    delta = points[:, :2] - target[None, :]
    dist2 = np.sum(delta * delta, axis=1)
    return int(np.argmin(dist2))


def build_reference_layout_1to1() -> tuple[list[dict], dict]:
    """
    Build a 1:1 layout in image-pixel coordinates from the reference Frenkel-pair figure.
    Coordinate frame:
      - x: pixel to the right
      - y: pixel downward
      - z: 0 for all atoms (single layer)
    """
    # Hand-tuned lattice to match the displayed figure footprint.
    # The underlying layer is close to a triangular packing.
    y0 = 53.0
    dy = 31.0
    x_even = 46.5
    x_odd = 62.0
    dx = 31.8
    n_rows = 9
    n_cols = 15

    atoms: list[dict] = []
    for row in range(n_rows):
        y = y0 + row * dy
        x0 = x_even if (row % 2 == 0) else x_odd
        for col in range(n_cols):
            x = x0 + col * dx
            atoms.append(
                {
                    "id": len(atoms),
                    "row": row,
                    "col": col,
                    "role": "host",
                    "symbol": "N",
                    "color": "blue",
                    "position": [float(x), float(y), 0.0],
                }
            )

    # Vacancy (remove one host site near the dashed circle in the reference).
    points = np.array([a["position"] for a in atoms], dtype=float)
    vacancy_target = (126.0, 206.0)
    vacancy_idx = _nearest_index(points, vacancy_target)
    vacancy_site = atoms[vacancy_idx]["position"][:]
    removed = atoms.pop(vacancy_idx)

    # Reindex after deletion.
    for i, atom in enumerate(atoms):
        atom["id"] = i

    points = np.array([a["position"] for a in atoms], dtype=float)

    # Color accents from the reference figure.
    # 1) light-cyan near upper-left
    idx_top_cyan = _nearest_index(points, (112.0, 95.0))
    atoms[idx_top_cyan]["role"] = "accent_top_cyan"
    atoms[idx_top_cyan]["symbol"] = "C"
    atoms[idx_top_cyan]["color"] = "light_cyan"

    # 2) purple near upper-middle
    idx_purple = _nearest_index(points, (286.0, 130.0))
    if idx_purple == idx_top_cyan:
        dist = np.sum((points[:, :2] - np.array([286.0, 130.0])) ** 2, axis=1)
        order = np.argsort(dist)
        idx_purple = int(order[1])
    atoms[idx_purple]["role"] = "accent_purple"
    atoms[idx_purple]["symbol"] = "Si"
    atoms[idx_purple]["color"] = "purple"

    # 3) light-cyan near lower-right
    idx_bottom_cyan = _nearest_index(points, (468.0, 284.0))
    if idx_bottom_cyan in {idx_top_cyan, idx_purple}:
        dist = np.sum((points[:, :2] - np.array([468.0, 284.0])) ** 2, axis=1)
        for cand in np.argsort(dist):
            if int(cand) not in {idx_top_cyan, idx_purple}:
                idx_bottom_cyan = int(cand)
                break
    atoms[idx_bottom_cyan]["role"] = "accent_bottom_cyan"
    atoms[idx_bottom_cyan]["symbol"] = "C"
    atoms[idx_bottom_cyan]["color"] = "light_cyan"

    # Red interstitial atom.
    red_interstitial = {
        "id": len(atoms),
        "row": None,
        "col": None,
        "role": "interstitial",
        "symbol": "O",
        "color": "red",
        "position": [495.0, 222.0, 0.0],
    }
    atoms.append(red_interstitial)

    meta = {
        "image_size_px": [569, 373],
        "coordinate_system": "pixel_1to1_from_reference_image",
        "notes": [
            "All atoms are in a single z=0 layer.",
            "One host site removed as vacancy; one red interstitial added.",
            "Three host atoms are recolored to match cyan/purple accents in the figure.",
        ],
        "vacancy_site": vacancy_site,
        "vacancy_removed_atom": removed,
        "counts": {
            "total_atoms": len(atoms),
            "host_blue": sum(1 for a in atoms if a["role"] == "host"),
            "accent_top_cyan": sum(1 for a in atoms if a["role"] == "accent_top_cyan"),
            "accent_purple": sum(1 for a in atoms if a["role"] == "accent_purple"),
            "accent_bottom_cyan": sum(1 for a in atoms if a["role"] == "accent_bottom_cyan"),
            "interstitial_red": sum(1 for a in atoms if a["role"] == "interstitial"),
        },
    }
    return atoms, meta


def atoms_to_ase(atoms_data: list[dict]) -> Atoms:
    symbols = [str(a["symbol"]) for a in atoms_data]
    positions = [tuple(float(v) for v in a["position"]) for a in atoms_data]
    ase_atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    ase_atoms.set_cell([569.0, 373.0, 20.0])
    ase_atoms.set_tags(list(range(len(atoms_data))))
    return ase_atoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Create 1:1 coordinate structure from the reference Frenkel-pair image")
    parser.add_argument("--out-dir", type=str, default="outputs/frenkel_pair_1to1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms_data, meta = build_reference_layout_1to1()
    ase_atoms = atoms_to_ase(atoms_data)

    json_path = out_dir / "frenkel_pair_1to1_coords.json"
    xyz_path = out_dir / "frenkel_pair_1to1.xyz"
    extxyz_path = out_dir / "frenkel_pair_1to1.extxyz"

    payload = {
        "metadata": meta,
        "atoms": atoms_data,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write(str(xyz_path), ase_atoms)
    write(str(extxyz_path), ase_atoms)

    print(f"JSON: {json_path}")
    print(f"XYZ: {xyz_path}")
    print(f"EXTXYZ: {extxyz_path}")
    print(f"Total atoms: {len(atoms_data)}")
    print(f"Vacancy site: {meta['vacancy_site']}")


if __name__ == "__main__":
    main()
