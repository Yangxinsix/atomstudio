#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import write

from atomstudio.structure import AtomSelector, Structure
from defect_relaxation import solve_defect_relaxation


def _nearest_indices_xy(points: np.ndarray, target_xy: tuple[float, float], k: int) -> list[int]:
    t = np.array(target_xy, dtype=float)
    d2 = np.sum((points[:, :2] - t[None, :]) ** 2, axis=1)
    return [int(i) for i in np.argsort(d2)[: int(k)]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2D defect relaxation solver and generate a test structure")
    parser.add_argument("--defect-type", choices=["substitution", "interstitial", "vacancy"], default="interstitial")
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=9)
    parser.add_argument("--a", type=float, default=1.38)
    parser.add_argument("--out-dir", type=str, default="outputs/defect_solver_demo")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    result = solve_defect_relaxation(
        nx=int(args.nx),
        ny=int(args.ny),
        lattice_constant=float(args.a),
        defect_type=str(args.defect_type),
        substitution_size_scale=1.24,
        interstitial_size_scale=1.05,
        k_spring=10.0,
        k_repulsion=95.0,
        repulsion_scale=0.96,
        vacancy_pull=3.2,
        vacancy_target_scale=0.80,
        max_iter=1800,
        tol_force=1e-4,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = out_dir / f"{args.defect_type}_relaxed.xyz"
    extxyz_path = out_dir / f"{args.defect_type}_relaxed.extxyz"
    write(str(xyz_path), result.atoms)
    write(str(extxyz_path), result.atoms)

    report = {
        "defect_type": result.defect_type,
        "defect_index": result.defect_index,
        "vacancy_site": result.vacancy_site,
        "iterations": result.iterations,
        "converged": result.converged,
        "energy": result.energy,
        "max_displacement": result.max_displacement,
        "mean_displacement": result.mean_displacement,
        "num_atoms": len(result.atoms),
        "num_fixed": len(result.fixed_indices),
        "xyz": str(xyz_path),
        "extxyz": str(extxyz_path),
    }
    json_path = out_dir / f"{args.defect_type}_relaxed_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    png_path = None
    if bool(args.render):
        structure = Structure.from_ase(result.atoms)
        for atom in structure.atoms:
            atom.color = (0.22, 0.56, 0.83, 1.0)

        if result.defect_index is not None:
            if args.defect_type == "substitution":
                structure.assign_atom_style(AtomSelector(indices=[int(result.defect_index)]), color=(0.72, 0.58, 0.77, 1.0), radius=0.66)
            else:
                structure.assign_atom_style(AtomSelector(indices=[int(result.defect_index)]), color=(0.83, 0.22, 0.24, 1.0), radius=0.62)

        if result.vacancy_site is not None:
            pts = np.array(structure.positions, dtype=float)
            near = _nearest_indices_xy(pts, target_xy=result.vacancy_site, k=6)
            structure.assign_atom_style(AtomSelector(indices=near), color=(0.63, 0.84, 0.93, 1.0), radius=0.60)

        png_path = out_dir / f"{args.defect_type}_relaxed.png"
        try:
            structure.get_image(
                str(png_path),
                style="handdrawn",
                representation="space_filling",
                engine="eevee",
                draw_cell=False,
                draw_bonds=False,
                view="top",
                camera_view="top",
                frame_scale=1.03,
                return_type="path",
                overrides={
                    "render": {"transparent_bg": False, "resolution": [1700, 1100], "samples": 72},
                    "style": {
                        "handdrawn": {
                            "background": [0.93, 0.93, 0.93, 1.0],
                            "outline_surface": 2.5,
                            "outline_molecule": 2.8,
                            "jmol_desaturate": 0.10,
                            "jmol_lighten": 0.04,
                            "shadow_strength": 0.36,
                            "highlight_strength": 0.18,
                        }
                    },
                    "lighting": {"light_style": "handdrawn_soft", "intensity": 1.03},
                },
            )
        except Exception as exc:
            png_path = None
            report["render_error"] = str(exc)
            json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"XYZ: {xyz_path}")
    print(f"EXTXYZ: {extxyz_path}")
    print(f"REPORT: {json_path}")
    if png_path is not None:
        print(f"PNG: {png_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
