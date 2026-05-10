import argparse
from pathlib import Path

import numpy as np
from ase.build import molecule

from atomstudio.structure import BondingConfig, Structure


def find_cross_bonds(structure: Structure) -> list:
    cross = []
    for bond in structure.bonds:
        if structure.atoms[bond.a].tag != structure.atoms[bond.b].tag:
            cross.append(bond)
    return cross


def group_center(structure: Structure, tag: str) -> np.ndarray:
    pts = [structure.positions[i] for i, atom in enumerate(structure.atoms) if str(atom.tag) == str(tag)]
    arr = np.array(pts, dtype=float)
    return arr.mean(axis=0)


def auto_resolve_cross_bonds(structure: Structure, max_iter: int = 40, step: float = 0.50) -> int:
    moved_steps = 0
    for iter_idx in range(max_iter):
        structure.compute_bonds(BondingConfig(cutoff_scale=1.1, min_distance=0.2))
        cross = find_cross_bonds(structure)
        if not cross:
            return moved_steps

        shifts: dict[str, np.ndarray] = {}
        for bond in cross:
            tag_a = str(structure.atoms[bond.a].tag)
            tag_b = str(structure.atoms[bond.b].tag)
            pa = np.array(structure.positions[int(bond.a)], dtype=float)
            pb = np.array(structure.positions[int(bond.b)], dtype=float)
            direction = pb - pa
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                ca = group_center(structure, tag_a)
                cb = group_center(structure, tag_b)
                direction = cb - ca
                norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                direction = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                direction = direction / norm

            # 当前误连键距离越短，推开幅度越大；后续迭代轻微增大步长，避免卡住。
            dist = float(bond.distance)
            min_sep = 1.35
            extra = max(0.0, min_sep - dist) * 1.4
            magnitude = step + extra + 0.03 * float(iter_idx)
            vec = direction * magnitude

            # 尽量保持中心分子(tag=0)稳定，优先移动外围分子。
            if tag_a == "0" and tag_b != "0":
                move_tag = tag_b
                move_vec = vec
            elif tag_b == "0" and tag_a != "0":
                move_tag = tag_a
                move_vec = -vec
            else:
                move_tag = tag_b if int(tag_b) >= int(tag_a) else tag_a
                move_vec = vec if move_tag == tag_b else -vec
            shifts[move_tag] = shifts.get(move_tag, np.zeros(3, dtype=float)) + move_vec

        for i, atom in enumerate(structure.atoms):
            tag = str(atom.tag)
            if tag not in shifts:
                continue
            shift = shifts[tag]
            shift_norm = float(np.linalg.norm(shift))
            if shift_norm > 1.6:
                shift = shift / shift_norm * 1.6
            p = np.array(structure.positions[i], dtype=float) + shift
            atom.position = (float(p[0]), float(p[1]), float(p[2]))
        moved_steps += 1

    return moved_steps


def apply_manual_bond_orders(structure: Structure, group_name_by_tag: dict[str, str]) -> None:
    group_atoms: dict[str, list[int]] = {}
    for i, atom in enumerate(structure.atoms):
        group_atoms.setdefault(str(atom.tag), []).append(i)

    for tag, atom_ids in group_atoms.items():
        name = group_name_by_tag.get(str(tag))
        if not name:
            continue

        group_set = set(atom_ids)
        group_bonds = [b for b in structure.bonds if int(b.a) in group_set and int(b.b) in group_set]
        bond_map = {tuple(sorted((int(b.a), int(b.b)))): b for b in group_bonds}
        symbols = structure.symbols

        def set_order(i: int, j: int, order: int) -> None:
            key = tuple(sorted((int(i), int(j))))
            if key in bond_map:
                bond_map[key].order = max(int(bond_map[key].order), int(order))

        if name == "C2H4":
            for b in group_bonds:
                if symbols[int(b.a)] == "C" and symbols[int(b.b)] == "C":
                    b.order = max(int(b.order), 2)

        if name in {"CO2", "SO2"}:
            center = "C" if name == "CO2" else "S"
            for b in group_bonds:
                sa = symbols[int(b.a)]
                sb = symbols[int(b.b)]
                if {sa, sb} == {center, "O"}:
                    b.order = max(int(b.order), 2)

        if name == "CH3CONH2":
            neighbors: dict[int, list[int]] = {i: [] for i in atom_ids}
            for b in group_bonds:
                a = int(b.a)
                c = int(b.b)
                neighbors[a].append(c)
                neighbors[c].append(a)
            for i in atom_ids:
                if symbols[i] != "C":
                    continue
                neigh_syms = {symbols[n] for n in neighbors[i]}
                if "O" in neigh_syms and "N" in neigh_syms:
                    for n in neighbors[i]:
                        if symbols[n] == "O":
                            set_order(i, n, 2)

        if name == "C6H6":
            carbon_ids = [i for i in atom_ids if symbols[i] == "C"]
            if len(carbon_ids) == 6:
                pos = np.array([structure.positions[i] for i in carbon_ids], dtype=float)
                center = pos.mean(axis=0)
                angles = np.arctan2(pos[:, 1] - center[1], pos[:, 0] - center[0])
                ordered = [carbon_ids[k] for k in np.argsort(angles)]
                for i in range(6):
                    if i % 2 != 0:
                        continue
                    set_order(ordered[i], ordered[(i + 1) % 6], 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render handdrawn molecule collage")
    parser.add_argument(
        "--preset",
        default="compact_mixed",
        choices=["compact_mixed", "md17_like_mixed"],
        help="Molecule layout preset",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_path = root / "outputs" / "common_molecule_collage_handdrawn.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 可切换分子方案：
    # - compact_mixed: 紧凑构图（默认）
    # - md17_like_mixed: 含 MD17 常见分子（benzene/ethanol 对应 C6H6/CH3CH2OH）
    # 说明：MD22 里的超大分子一般不在 ase.build.molecule 内置库里，通常需从 xyz 文件读入。
    presets = {
        "compact_mixed": [
            # 每个条目格式:
            # ("分子名", (x, y, z), (rx, ry, rz))
            # 位置单位是 Å；旋转单位是度（分别绕 x/y/z）。
            # 想调位置就改 (x, y, z)；想调朝向就改 (rx, ry, rz)。
            ("CH3CONH2", (0.0, 0.0, 0.0), (18, 6, 16)),
            ("NH3", (-4.2, 0.6, 0.0), (-8, 12, -6)),
            ("CH4", (4.2, -1.1, 0.0), (8, -10, 10)),
            ("H2O", (0.1, -4.3, 0.0), (12, 60, 8)),
            ("C6H6", (-4.0, -3.6, 0.0), (4, 2, 12)),
            ("CH3CH2OH", (3.7, -4.9, 0.0), (5, 6, -8)),
            # C2H4 改成近正面，优先显示 C=C
            ("C2H4", (-0.7, 3.2, 0.0), (0, 60, 30)),
            # 右侧整体下移，避免顶部出框
            ("SO2", (4.8, 1.0, 0.0), (0, 6, 60)),
        ],
        "md17_like_mixed": [
            ("CH3CONH2", (0.0, 0.0, 0.0), (16, 6, 14)),
            ("NH3", (-4.3, 0.9, 0.0), (-8, 10, -4)),
            ("CH4", (4.2, -1.0, 0.0), (8, -10, 10)),
            ("H2O", (0.0, -4.4, 0.0), (10, 0, 8)),
            ("C6H6", (-4.1, -3.7, 0.0), (3, 2, 8)),
            ("CH3CH2OH", (3.8, -4.8, 0.0), (5, 6, -8)),
            ("CO2", (-0.6, 3.6, 0.0), (0, 2, 6)),
            ("C2H4", (4.0, 1.1, 0.0), (0, 0, 0)),
        ],
    }
    specs = presets[args.preset]
    group_name_by_tag = {str(i): name for i, (name, _, _) in enumerate(specs)}

    combo = None
    for group_id, (name, shift, rotation) in enumerate(specs):
        atoms = molecule(name)
        atoms.rotate(rotation[0], "x", center="COM")
        atoms.rotate(rotation[1], "y", center="COM")
        atoms.rotate(rotation[2], "z", center="COM")
        atoms.translate(shift)
        atoms.set_tags([group_id] * len(atoms))
        combo = atoms if combo is None else combo + atoms

    combo.set_cell([38, 42, 28])
    combo.center()

    structure = Structure.from_ase(combo)
    moved_steps = auto_resolve_cross_bonds(structure, max_iter=40, step=0.50)
    structure.compute_bonds(BondingConfig(cutoff_scale=1.1, min_distance=0.2))

    cross = find_cross_bonds(structure)
    if cross:
        details = [
            (
                int(b.a),
                int(b.b),
                f"{group_name_by_tag.get(str(structure.atoms[b.a].tag), 'UNKNOWN')}[{str(structure.atoms[b.a].tag)}]",
                f"{group_name_by_tag.get(str(structure.atoms[b.b].tag), 'UNKNOWN')}[{str(structure.atoms[b.b].tag)}]",
                round(float(b.distance), 3),
            )
            for b in cross[:5]
        ]
        raise RuntimeError(f"检测到跨分子误连键(自动避碰后仍存在): {details}")

    apply_manual_bond_orders(structure, group_name_by_tag)

    rendered = structure.get_image(
        str(out_path),
        style="handdrawn",
        representation="ball_stick",
        engine="eevee",
        quality="low",
        draw_cell=False,
        draw_bonds=True,
        # 全局视角减倾斜，减少“看不到双键”的情况
        view="-10x,5y,10z",
        # >1 更容易把边缘分子纳入画面
        frame_scale=1.08,
        return_type="path",
        overrides={
            "render": {
                "transparent_bg": False,
                "resolution": [1500, 1100],
                "samples": 64,
            },
            "style": {
                "handdrawn": {
                    "background": [0.90, 0.89, 0.84, 1.0],
                    "outline_molecule": 2.8,
                    "outline_bond": 1.7,
                    "jmol_desaturate": 0.12,
                    "jmol_lighten": 0.05,
                    "shadow_strength": 0.40,
                    "highlight_strength": 0.20,
                }
            },
            # fit_padding 越大越不容易裁边
            "camera": {"fit_padding": 0.09},
            "structure": {
                "draw_cell": False,
                "bonding": {"cutoff_scale": 1.1},
                "bond_radius": 0.10,
                "atom_scale": 1.05,
            },
            "lighting": {"light_style": "handdrawn_soft", "intensity": 1.08},
        },
    )

    print("跨分子误连键检查通过")
    print(f"自动避碰步数: {moved_steps}")
    print(f"双键数量: {sum(1 for b in structure.bonds if int(b.order) == 2)}")
    print(f"三键数量: {sum(1 for b in structure.bonds if int(b.order) == 3)}")
    print(f"分子方案: {args.preset}")
    print(f"渲染完成: {rendered}")


if __name__ == "__main__":
    main()
