from __future__ import annotations

import json
from pathlib import Path

import pytest

import atomstudio.render.pipeline as pipeline_mod
from atomstudio.config import BatchConfig, RenderJobConfig
from atomstudio.render.results import RenderResult
from atomstudio.structure.atom import Atom
from atomstudio.structure.cell import Cell
from atomstudio.structure.structure import Structure


def _dense_carbon_pair() -> Structure:
    return Structure(
        atoms=[
            Atom(index=0, atomic_number=6, symbol="C", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=6, symbol="C", position=(1.2, 0.0, 0.0)),
        ],
        bonds=[],
        cell=Cell(),
    )


def test_render_job_config_accepts_auto_space_filling_scale():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "auto_scale",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/out.png"},
            "structure": {"representation": "space_filling"},
        }
    )
    assert cfg.structure.space_filling_scale == "auto"


def test_render_job_config_accepts_internal_space_filling_scale_override():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "override_auto_scale",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/out.png"},
            "structure": {"representation": "space_filling", "space_filling_scale": 0.85},
        }
    )
    assert cfg.structure.space_filling_scale == pytest.approx(0.85)


def test_render_structure_auto_space_filling_scale_is_applied_and_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    structure = _dense_carbon_pair()
    cfg = RenderJobConfig.from_dict(
        {
            "id": "auto_scale",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": str(tmp_path / "auto_scale.png")},
            "style": {"scene_style": "handdrawn"},
            "structure": {"representation": "space_filling"},
        }
    )

    captured: dict[str, object] = {}

    def _fake_run_blender_render(structure_arg, cfg_arg, blender_path=None, timeout_seconds=1800):
        captured["applied_scale"] = cfg_arg.structure.space_filling_scale
        Path(cfg_arg.output.path).write_bytes(b"fake image")
        return {
            "success": True,
            "output_path": cfg_arg.output.path,
            "frame_index": structure_arg.frame_index,
            "message": "ok",
        }

    monkeypatch.setattr(pipeline_mod, "run_blender_render", _fake_run_blender_render)

    result = pipeline_mod.render_structure(structure, cfg)

    expected = 0.96 * 1.2 / (0.77 + 0.77)
    assert float(captured["applied_scale"]) == pytest.approx(expected)
    assert cfg.structure.space_filling_scale == "auto"
    assert result.success is True
    assert result.adjustments["space_filling_scale"]["mode"] == "auto"
    assert result.adjustments["space_filling_scale"]["adjusted"] is True
    assert result.adjustments["space_filling_scale"]["applied"] == pytest.approx(expected)
    assert result.adjustments["space_filling_scale"]["limiting_pair"] == [0, 1]


def test_render_structure_reports_failure_when_output_file_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    structure = _dense_carbon_pair()
    cfg = RenderJobConfig.from_dict(
        {
            "id": "missing_output",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": str(tmp_path / "missing.png")},
        }
    )

    def _fake_run_blender_render(structure_arg, cfg_arg, blender_path=None, timeout_seconds=1800):
        return {
            "success": True,
            "output_path": cfg_arg.output.path,
            "frame_index": structure_arg.frame_index,
            "message": "ok",
        }

    monkeypatch.setattr(pipeline_mod, "run_blender_render", _fake_run_blender_render)

    result = pipeline_mod.render_structure(structure, cfg)

    assert result.success is False
    assert "output file was not created" in result.message


def test_render_batch_job_report_includes_frame_adjustments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    structure = _dense_carbon_pair()
    job = RenderJobConfig.from_dict(
        {
            "id": "auto_scale_job",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"dir": str(tmp_path / "outputs")},
            "style": {"scene_style": "handdrawn"},
            "structure": {"representation": "space_filling"},
        }
    )
    batch = BatchConfig(version=2, jobs=[job])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline_mod, "load_batch_config", lambda path: batch)
    monkeypatch.setattr(pipeline_mod, "load_trajectory", lambda path, frames: [structure])
    monkeypatch.setattr(
        pipeline_mod,
        "render_structure",
        lambda structure_arg, cfg_arg: RenderResult(
            success=True,
            output_path=str(cfg_arg.output.path),
            frame_index=structure_arg.frame_index,
            message="ok",
            adjustments={"space_filling_scale": {"mode": "auto", "applied": 0.91, "adjusted": True}},
        ),
    )

    result = pipeline_mod.render_batch("dummy.yaml")

    assert result.success is True
    assert len(result.reports) == 1
    assert result.reports[0].frame_reports[0].adjustments["space_filling_scale"]["applied"] == pytest.approx(0.91)

    report_path = tmp_path / "outputs" / "job_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["frame_reports"][0]["adjustments"]["space_filling_scale"]["mode"] == "auto"
    assert payload["frame_reports"][0]["adjustments"]["space_filling_scale"]["applied"] == pytest.approx(0.91)
