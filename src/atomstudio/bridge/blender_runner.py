from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from atomstudio.backend.blender.renderer import build_render_scene_payload
from atomstudio.config import RenderJobConfig
from atomstudio.structure.structure import Structure


def _find_blender(blender_path: str | None = None) -> str:
    if blender_path:
        return blender_path
    env_path = os.environ.get("ATOMSTUDIO_BLENDER")
    if env_path:
        return env_path
    which = shutil.which("blender")
    if which:
        return which
    raise FileNotFoundError("Blender executable not found. Set ATOMSTUDIO_BLENDER or pass blender_path.")


def run_blender_render(
    structure: Structure,
    cfg: RenderJobConfig,
    blender_path: str | None = None,
    timeout_seconds: int = 1800,
    *,
    render_scene_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blender_bin = _find_blender(blender_path)
    entry_script = Path(__file__).with_name("blender_entry.py")
    src_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory(prefix="atomstudio_") as tmp:
        tmp_dir = Path(tmp)
        payload_path = tmp_dir / "payload.json"
        result_path = tmp_dir / "result.json"

        payload = {
            **(render_scene_payload or build_render_scene_payload(structure, cfg)),
            "result_path": str(result_path),
        }
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        cmd = [
            blender_bin,
            "--background",
            "--factory-startup",
            "--python",
            str(entry_script),
            "--",
            "--payload",
            str(payload_path),
        ]

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing}" if existing else str(src_root)

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_seconds)
        if proc.returncode != 0:
            raise RuntimeError(
                "Blender render failed\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        if not result_path.exists():
            raise RuntimeError(
                "Blender finished without result file.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        return json.loads(result_path.read_text(encoding="utf-8"))


def run_blender_animation(
    frames: list[Structure] | tuple[Structure, ...],
    cfg: RenderJobConfig,
    *,
    output_dir: str,
    filename_template: str = "frame_{frame:04d}.png",
    blender_path: str | None = None,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    blender_bin = _find_blender(blender_path)
    entry_script = Path(__file__).with_name("blender_entry.py")
    src_root = Path(__file__).resolve().parents[2]
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_payloads: list[dict[str, Any]] = []
    for sequence_index, structure in enumerate(frames):
        frame_number = int(getattr(structure, "frame_index", sequence_index))
        output_path = out_dir / filename_template.format(
            job_id=cfg.id,
            frame=frame_number,
            index=sequence_index,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_cfg = cfg.with_output_path(str(output_path))
        payload = build_render_scene_payload(structure, frame_cfg)
        frame_payloads.append(
            {
                "config": payload["config"],
                "render_scene": payload["render_scene"],
                "frame_index": frame_number,
                "output_path": str(output_path),
            }
        )

    with tempfile.TemporaryDirectory(prefix="atomstudio_anim_") as tmp:
        tmp_dir = Path(tmp)
        payload_path = tmp_dir / "payload.json"
        result_path = tmp_dir / "result.json"
        payload = {
            "schema": "atomstudio.animation.v1",
            "source": "scene_builder",
            "config": cfg.to_dict(),
            "output_dir": str(out_dir),
            "filename_template": str(filename_template),
            "frames": frame_payloads,
            "result_path": str(result_path),
        }
        payload_path.write_text(json.dumps(payload), encoding="utf-8")

        cmd = [
            blender_bin,
            "--background",
            "--factory-startup",
            "--python",
            str(entry_script),
            "--",
            "--payload",
            str(payload_path),
        ]

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing}" if existing else str(src_root)

        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_seconds)
        if proc.returncode != 0:
            raise RuntimeError(
                "Blender animation render failed\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        if not result_path.exists():
            raise RuntimeError(
                "Blender finished without animation result file.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        return json.loads(result_path.read_text(encoding="utf-8"))
