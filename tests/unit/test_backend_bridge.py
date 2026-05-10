from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import atomstudio.bridge.blender_entry as entry_mod
import atomstudio.bridge.blender_runner as runner_mod
from atomstudio.config import RenderJobConfig
from atomstudio.structure.atom import Atom
from atomstudio.structure.cell import Cell
from atomstudio.structure.structure import Structure


def _structure() -> Structure:
    return Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
        source_path="tests/data/water.xyz",
        frame_index=0,
    )


def _cfg(tmp_path: Path) -> RenderJobConfig:
    return RenderJobConfig.from_dict(
        {
            "id": "bridge",
            "input": {"path": "tests/data/water.xyz", "frames": "0"},
            "output": {"path": str(tmp_path / "bridge.png")},
        }
    )


def test_run_blender_render_writes_render_scene_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cfg = _cfg(tmp_path)
    structure = _structure()
    captured: dict[str, object] = {}

    monkeypatch.setattr(runner_mod, "_find_blender", lambda blender_path=None: "/usr/bin/blender")
    monkeypatch.setattr(
        runner_mod,
        "build_render_scene_payload",
        lambda structure_arg, cfg_arg: {
            "schema": "atomstudio.render_scene.v1",
            "source": "unit_test",
            "render_scene": {"atoms": [], "metadata": {"frame_index": structure_arg.frame_index}},
            "config": cfg_arg.to_dict(),
        },
    )

    def fake_run(cmd, capture_output, text, env, timeout):
        payload_path = Path(cmd[-1])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        captured["payload"] = payload
        result_path = Path(payload["result_path"])
        result_path.write_text(
            json.dumps(
                {
                    "success": True,
                    "output_path": payload["config"]["output"]["path"],
                    "frame_index": payload["render_scene"]["metadata"]["frame_index"],
                    "message": "ok",
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner_mod.subprocess, "run", fake_run)

    result = runner_mod.run_blender_render(structure, cfg)

    assert result["success"] is True
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["source"] == "unit_test"
    assert "render_scene" in payload
    assert payload["config"]["output"]["path"] == str(tmp_path / "bridge.png")


def test_blender_entry_main_delegates_to_backend_renderer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    payload_path = tmp_path / "payload.json"
    result_path = tmp_path / "result.json"
    payload = {
        "schema": "atomstudio.render_scene.v1",
        "source": "unit_test",
        "render_scene": {"metadata": {"frame_index": 7}},
        "config": _cfg(tmp_path).to_dict(),
        "result_path": str(result_path),
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        entry_mod,
        "render_payload",
        lambda loaded: {
            "success": True,
            "output_path": loaded["config"]["output"]["path"],
            "frame_index": loaded["render_scene"]["metadata"]["frame_index"],
            "message": "ok",
            "stats": {"atoms": 1},
        },
    )
    monkeypatch.setattr(sys, "argv", ["blender", "--", "--payload", str(payload_path)])

    entry_mod.main()

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["success"] is True
    assert result["frame_index"] == 7
