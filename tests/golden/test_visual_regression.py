from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "tests" / "golden" / "baselines.json"
GOLDEN_IMAGES = ROOT / "tests" / "golden" / "images"


def _avg_hash(path: Path) -> str:
    from PIL import Image

    img = Image.open(path).convert("L").resize((8, 8))
    pixels = list(img.getdata())
    mean = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= mean else "0" for p in pixels)
    return f"{int(bits, 2):016x}"


@pytest.mark.golden
@pytest.mark.skipif(not os.environ.get("ATOMSTUDIO_RUN_GOLDEN"), reason="set ATOMSTUDIO_RUN_GOLDEN=1 to run")
def test_visual_hashes_match_baseline():
    pytest.importorskip("PIL")
    data = json.loads(BASELINES.read_text(encoding="utf-8"))

    missing = []
    for key, info in data.items():
        target = GOLDEN_IMAGES / f"{key}.png"
        if not target.exists():
            missing.append(str(target))
            continue
        if info["hash"]:
            assert _avg_hash(target) == info["hash"], f"hash mismatch for {key}"

    assert not missing, f"Missing golden image files: {missing}"

