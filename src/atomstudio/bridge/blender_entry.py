from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atomstudio.backend.blender.renderer import render_animation_payload, render_payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="atomstudio blender entry")
    parser.add_argument("--payload", required=True, help="JSON payload path")
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    payload_path = Path(args.payload).resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if payload.get("schema") == "atomstudio.animation.v1":
        result = render_animation_payload(payload)
    else:
        result = render_payload(payload)

    result_path = Path(payload["result_path"]).resolve()
    result_path.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
