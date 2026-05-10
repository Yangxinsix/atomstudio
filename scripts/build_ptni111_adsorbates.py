from __future__ import annotations

import argparse
from pathlib import Path

from ase import Atoms
from ase.build import fcc111
from ase.io import read, write


_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
}


def _layer_groups_by_z(z_values: list[float], tol: float = 1e-3) -> list[list[int]]:
    ordered = sorted(range(len(z_values)), key=lambda i: z_values[i])
    groups: list[list[int]] = []
    for idx in ordered:
        z = float(z_values[idx])
        if not groups:
            groups.append([idx])
            continue
        z_ref = float(z_values[groups[-1][0]])
        if abs(z - z_ref) <= tol:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def _row_id_by_y(y_values: list[float], tol: float = 1e-3) -> list[int]:
    """Map each y value to a compact row id with tolerance merge."""
    out = [-1] * len(y_values)
    refs: list[float] = []
    for i, y in enumerate(y_values):
        yf = float(y)
        assigned = False
        for row_id, y_ref in enumerate(refs):
            if abs(yf - y_ref) <= tol:
                out[i] = row_id
                assigned = True
                break
        if not assigned:
            refs.append(yf)
            out[i] = len(refs) - 1
    return out


def _adsorbate_components(
    symbols: list[str],
    positions,
    indices: list[int],
    cutoff_scale: float = 1.1,
) -> list[list[int]]:
    neigh: dict[int, set[int]] = {i: set() for i in indices}
    for a in indices:
        sa = str(symbols[a])
        ra = _COVALENT_RADII.get(sa)
        if ra is None:
            continue
        for b in indices:
            if b <= a:
                continue
            sb = str(symbols[b])
            rb = _COVALENT_RADII.get(sb)
            if rb is None:
                continue
            d = float(((positions[a] - positions[b]) ** 2).sum()) ** 0.5
            if d <= (ra + rb) * float(cutoff_scale):
                neigh[a].add(b)
                neigh[b].add(a)

    seen: set[int] = set()
    comps: list[list[int]] = []
    for i in indices:
        if i in seen:
            continue
        stack = [i]
        comp: list[int] = []
        seen.add(i)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in sorted(neigh[cur]):
                if nb in seen:
                    continue
                seen.add(nb)
                stack.append(nb)
        comps.append(sorted(comp))
    return comps


def _anchor_index(component: list[int], symbols: list[str]) -> int:
    for i in component:
        if str(symbols[i]) != "H":
            return i
    return component[0]


def build_ptni111_adsorbates(
    *,
    source_adsorbates: Path,
    out_xyz: Path,
    out_cif: Path,
    size: tuple[int, int, int] = (6, 6, 6),
    lattice_a: float = 3.78,
    vacuum: float = 12.0,
    adsorbate_lift: float = 1.2,
    adsorbate_shift_y: float = 0.0,
) -> Atoms:
    src = read(str(source_adsorbates))
    src_symbols = src.get_chemical_symbols()
    src_pos = src.get_positions()
    src_cell = src.cell.lengths()

    # Current mgo111_adsorbates data keeps substrate in first 144 atoms.
    substrate_count = 144
    ads_idx = list(range(substrate_count, len(src)))
    top_src_sub = max(float(src_pos[i, 2]) for i in range(substrate_count))

    slab = fcc111("Pt", size=size, a=lattice_a, vacuum=vacuum, orthogonal=True)
    slab_pos = slab.get_positions()
    slab_z = [float(p[2]) for p in slab_pos]
    layers = _layer_groups_by_z(slab_z, tol=1e-3)

    # Row-alternating alloy pattern: one row Pt, next row Ni, with layer phase shift.
    for layer_id, layer in enumerate(layers):
        y_values = [float(slab_pos[i, 1]) for i in layer]
        row_ids = _row_id_by_y(y_values, tol=1e-3)
        for local_i, idx in enumerate(layer):
            row_id = int(row_ids[local_i])
            slab[idx].symbol = "Pt" if ((row_id + layer_id) % 2 == 0) else "Ni"

    top_new_sub = max(float(p[2]) for p in slab.get_positions())
    new_lx, new_ly = slab.cell.lengths()[0], slab.cell.lengths()[1]
    old_lx, old_ly = src_cell[0], src_cell[1]

    ads_symbols: list[str] = [str(src_symbols[i]) for i in ads_idx]
    ads_positions: dict[int, tuple[float, float, float]] = {}
    components = _adsorbate_components(src_symbols, src_pos, ads_idx, cutoff_scale=1.1)
    for comp in components:
        anchor = _anchor_index(comp, src_symbols)
        ax, ay, az = (float(v) for v in src_pos[anchor])
        fx = ax / float(old_lx)
        fy = ay / float(old_ly)
        target_x = fx * float(new_lx)
        target_y = (fy * float(new_ly) + float(adsorbate_shift_y)) % float(new_ly)
        target_z = float(top_new_sub) + (az - float(top_src_sub)) + float(adsorbate_lift)
        dx = target_x - ax
        dy = target_y - ay
        dz = target_z - az
        for i in comp:
            sx, sy, sz = (float(v) for v in src_pos[i])
            ads_positions[i] = (sx + dx, sy + dy, sz + dz)

    ordered_ads_positions = [ads_positions[i] for i in ads_idx]

    ads = Atoms(symbols=ads_symbols, positions=ordered_ads_positions)
    out = slab + ads
    out.set_cell(slab.cell)
    out.set_pbc((True, True, False))

    out_xyz.parent.mkdir(parents=True, exist_ok=True)
    out_cif.parent.mkdir(parents=True, exist_ok=True)
    write(str(out_xyz), out)
    write(str(out_cif), out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PtNi(111) slab with copied adsorbates from mgo111_adsorbates.")
    parser.add_argument("--size-x", type=int, default=6, help="Slab repetition count along x.")
    parser.add_argument("--size-y", type=int, default=6, help="Slab repetition count along y.")
    parser.add_argument("--size-z", type=int, default=6, help="Number of atomic layers.")
    parser.add_argument("--adsorbate-lift", type=float, default=1.2, help="Lift adsorbates in Angstrom along +z.")
    parser.add_argument("--adsorbate-shift-y", type=float, default=-1.0, help="Shift adsorbates in Angstrom along y.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = root / "mgo111_adsorbates.xyz"
    out_xyz = root / "ptni111_adsorbates.xyz"
    out_cif = root / "ptni111_adsorbates.cif"
    atoms = build_ptni111_adsorbates(
        source_adsorbates=source,
        out_xyz=out_xyz,
        out_cif=out_cif,
        size=(int(args.size_x), int(args.size_y), int(args.size_z)),
        adsorbate_lift=float(args.adsorbate_lift),
        adsorbate_shift_y=float(args.adsorbate_shift_y),
    )
    counts: dict[str, int] = {}
    for s in atoms.get_chemical_symbols():
        counts[s] = counts.get(s, 0) + 1
    print("wrote:", out_xyz)
    print("wrote:", out_cif)
    print("atoms:", len(atoms), counts)
    print("size:", (int(args.size_x), int(args.size_y), int(args.size_z)))
    print("adsorbate_lift:", float(args.adsorbate_lift))
    print("adsorbate_shift_y:", float(args.adsorbate_shift_y))


if __name__ == "__main__":
    main()
