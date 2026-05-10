#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from PIL import Image
from scipy import ndimage as ndi


@dataclass
class Component:
    pixels: list[tuple[int, int]]
    area: int
    cx: float
    cy: float
    width: int
    height: int
    fill_ratio: float
    mean_rgb: tuple[float, float, float]


def _connected_components(mask: np.ndarray) -> list[Component]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[Component] = []

    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or visited[y0, x0]:
                continue

            stack = [(y0, x0)]
            visited[y0, x0] = True
            pts: list[tuple[int, int]] = []
            min_x = x0
            max_x = x0
            min_y = y0
            max_y = y0

            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

                for ny in (y - 1, y, y + 1):
                    for nx in (x - 1, x, x + 1):
                        if ny == y and nx == x:
                            continue
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            area = len(pts)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            fill_ratio = float(area) / float(width * height)
            ys = np.array([p[0] for p in pts], dtype=float)
            xs = np.array([p[1] for p in pts], dtype=float)
            comp = Component(
                pixels=pts,
                area=area,
                cx=float(xs.mean()),
                cy=float(ys.mean()),
                width=width,
                height=height,
                fill_ratio=fill_ratio,
                mean_rgb=(0.0, 0.0, 0.0),
            )
            components.append(comp)

    return components


def _classify_color(mean_rgb: tuple[float, float, float]) -> tuple[str, str]:
    r, g, b = mean_rgb
    if r > 140 and r > g + 40 and r > b + 40:
        return "interstitial_red", "O"
    if r > 150 and b > 140 and g > 100:
        return "accent_purple", "Si"
    if b > r + 20 and g > r + 10 and r >= 120:
        return "accent_cyan", "C"
    if b > r + 15 and g > r:
        return "host", "N"
    return "host", "N"


def extract_atoms_from_image(image_path: Path, threshold: float) -> tuple[list[dict], dict]:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    h, w, _ = arr.shape

    corners = np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]], axis=0)
    bg = np.median(corners, axis=0)

    dist = np.sqrt(np.sum((arr - bg[None, None, :]) ** 2, axis=2))
    mask = dist >= float(threshold)

    # For close-packed circles that touch, connected-components cannot separate atoms well.
    # Use distance-transform local maxima as circle-center candidates.
    distmap = ndi.distance_transform_edt(mask)
    maxf = ndi.maximum_filter(distmap, size=17, mode="nearest")
    peaks = (distmap == maxf) & (distmap >= 6.0)
    ys, xs = np.where(peaks)

    # Keep stronger peaks first; suppress near-duplicates.
    order = np.argsort(distmap[ys, xs])[::-1]
    picked: list[tuple[int, int]] = []
    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        if len(picked) > 0:
            d2 = [(py - y) * (py - y) + (px - x) * (px - x) for py, px in picked]
            if min(d2) < 80:  # roughly < 9 px
                continue
        # reject very dark/text pixels
        rgb = arr[y, x, :]
        if float(rgb.mean()) < 45:
            continue
        picked.append((y, x))

    atoms: list[dict] = []
    for y, x in picked:
        # local mean color in a small disk around center
        rr = int(max(3, min(8, distmap[y, x] * 0.55)))
        y0 = max(0, y - rr)
        y1 = min(h, y + rr + 1)
        x0 = max(0, x - rr)
        x1 = min(w, x + rr + 1)
        patch = arr[y0:y1, x0:x1, :]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (yy - y) * (yy - y) + (xx - x) * (xx - x) <= rr * rr
        if np.count_nonzero(disk) == 0:
            continue
        mean_rgb = patch[disk].mean(axis=0)
        role, symbol = _classify_color((float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])))
        atoms.append(
            {
                "id": len(atoms),
                "role": role,
                "symbol": symbol,
                "position": [float(x), float(y), 0.0],
                "mean_rgb": [float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])],
                "radius_px": float(distmap[y, x]),
            }
        )

    atoms.sort(key=lambda a: (a["position"][1], a["position"][0]))
    for i, a in enumerate(atoms):
        a["id"] = i

    meta = {
        "image": str(image_path),
        "image_size_px": [w, h],
        "background_rgb_estimate": [float(bg[0]), float(bg[1]), float(bg[2])],
        "threshold": float(threshold),
        "total_atoms": len(atoms),
        "counts_by_role": {
            role: int(sum(1 for a in atoms if a["role"] == role))
            for role in sorted({a["role"] for a in atoms})
        },
    }
    return atoms, meta


def atoms_to_ase(atoms_data: list[dict], image_w: int, image_h: int) -> Atoms:
    symbols = [str(a["symbol"]) for a in atoms_data]
    positions = [tuple(float(v) for v in a["position"]) for a in atoms_data]
    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    atoms.set_cell([float(image_w), float(image_h), 20.0])
    atoms.set_tags(list(range(len(atoms_data))))
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract all atom-center coordinates from a Frenkel-pair schematic image")
    parser.add_argument("image", type=str, help="Input image path")
    parser.add_argument("--out-dir", type=str, default="outputs/frenkel_pair_extracted")
    parser.add_argument("--threshold", type=float, default=18.0, help="Foreground-vs-background RGB distance threshold")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms_data, meta = extract_atoms_from_image(image_path, threshold=float(args.threshold))
    if not atoms_data:
        raise RuntimeError("No atom-like blobs were detected. Try a smaller --threshold (e.g. 12).")

    w, h = meta["image_size_px"]
    ase_atoms = atoms_to_ase(atoms_data, image_w=int(w), image_h=int(h))

    stem = image_path.stem
    json_path = out_dir / f"{stem}_coords.json"
    xyz_path = out_dir / f"{stem}.xyz"
    extxyz_path = out_dir / f"{stem}.extxyz"

    payload = {"metadata": meta, "atoms": atoms_data}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write(str(xyz_path), ase_atoms)
    write(str(extxyz_path), ase_atoms)

    print(f"JSON: {json_path}")
    print(f"XYZ: {xyz_path}")
    print(f"EXTXYZ: {extxyz_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
