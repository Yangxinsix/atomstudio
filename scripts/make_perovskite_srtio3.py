#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import write
from ase.spacegroup import crystal


def build_srtio3(*, a: float, supercell: tuple[int, int, int]):
    atoms = crystal(
        symbols=["Sr", "Ti", "O"],
        basis=[
            (0.0, 0.0, 0.0),  # A site
            (0.5, 0.5, 0.5),  # B site
            (0.5, 0.5, 0.0),  # O site (others from symmetry)
        ],
        spacegroup=221,  # Pm-3m
        cellpar=[a, a, a, 90.0, 90.0, 90.0],
    )
    atoms = atoms.repeat(supercell)
    atoms.wrap()
    return atoms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cubic perovskite SrTiO3 with ASE.")
    parser.add_argument("--a", type=float, default=3.905, help="Lattice constant in Angstrom.")
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        metavar=("NX", "NY", "NZ"),
        default=(3, 3, 3),
        help="Supercell replication along x/y/z.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/perovskite/srtio3_3x3x3.cif",
        help="Output CIF path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    supercell = tuple(int(v) for v in args.supercell)
    if any(v <= 0 for v in supercell):
        raise ValueError("All supercell dimensions must be positive.")

    atoms = build_srtio3(a=float(args.a), supercell=supercell)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_path), atoms)

    formula = atoms.get_chemical_formula(mode="hill")
    print(f"[ok] wrote: {out_path}")
    print(f"[ok] formula: {formula}")
    print(f"[ok] atoms: {len(atoms)}")
    print(f"[ok] cell: {atoms.cell.tolist()}")


if __name__ == "__main__":
    main()
