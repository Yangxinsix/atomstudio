#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def avg_hash(path: Path) -> str:
    img = Image.open(path).convert("L").resize((8, 8))
    px = list(img.getdata())
    mean = sum(px) / len(px)
    bits = "".join("1" if p >= mean else "0" for p in px)
    return f"{int(bits, 2):016x}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Update visual regression baseline hashes")
    parser.add_argument("--confirm", action="store_true", help="Required flag to write baselines")
    args = parser.parse_args()

    if not args.confirm:
        print("Refusing to update baselines without --confirm")
        return 2

    root = Path(__file__).resolve().parents[1]
    baseline_path = root / "tests" / "golden" / "baselines.json"
    image_dir = root / "tests" / "golden" / "images"

    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    for key in data:
        image = image_dir / f"{key}.png"
        if image.exists():
            data[key]["hash"] = avg_hash(image)

    baseline_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
