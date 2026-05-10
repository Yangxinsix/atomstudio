from __future__ import annotations

import sys
import types

import pytest

from atomstudio.app import workers
from atomstudio.app.state import LoadedFrameBundle
from atomstudio.config import RenderJobConfig
from atomstudio.render.results import RenderResult
from atomstudio.structure.structure import Structure


def _render_cfg() -> RenderJobConfig:
    return RenderJobConfig.from_dict(
        {
            "id": "app",
            "input": {"path": "example.xyz", "frames": "last"},
            "output": {"path": "/tmp/app.png"},
            "style": {"scene_style": "default"},
        }
    )


def test_load_structure_bundle_uses_trajectory_loader(monkeypatch, tmp_path):
    seen = {}

    def fake_load_trajectory(path: str, frame_selector: str = "all"):
        seen["path"] = path
        seen["selector"] = frame_selector
        return [
            Structure(frame_index=4, source_path=path),
            Structure(frame_index=9, source_path=path),
        ]

    monkeypatch.setattr(workers, "load_trajectory", fake_load_trajectory)
    bundle = workers.load_structure_bundle(
        workers.LoadStructureRequest(input_path=str(tmp_path / "sample.xyz"), frame_selector="0:2")
    )

    assert seen["selector"] == "0:2"
    assert bundle.frame_count == 2
    assert isinstance(bundle, LoadedFrameBundle)
    assert bundle.current().frame_index == 4


def test_preview_worker_uses_lazy_builder(monkeypatch):
    preview_package = types.ModuleType("atomstudio.preview")
    preview_module = types.ModuleType("atomstudio.preview.builder")
    seen = {}

    def fake_build_preview_scene(structure, render_config, preview_settings):
        seen["structure"] = structure
        seen["render_config"] = render_config
        seen["preview_settings"] = preview_settings
        return {"scene": True, "frame": structure.frame_index}

    preview_module.build_preview_scene = fake_build_preview_scene  # type: ignore[attr-defined]
    preview_package.builder = preview_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "atomstudio.preview", preview_package)
    monkeypatch.setitem(sys.modules, "atomstudio.preview.builder", preview_module)

    structure = Structure(frame_index=3, source_path="/tmp/sample.xyz")
    request = workers.PreviewRequest(structure=structure, render_config=_render_cfg(), preview_settings={"mode": "simple"})
    result = workers.PreviewWorker(request).run()

    assert result == {"scene": True, "frame": 3}
    assert seen["structure"] is structure
    assert seen["render_config"].id == "app"
    assert seen["preview_settings"] == {"mode": "simple"}


def test_render_worker_passes_through_to_pipeline(monkeypatch):
    seen = {}

    def fake_render_structure(structure, render_config, *, blender_path=None, timeout_seconds=1800):
        seen["structure"] = structure
        seen["render_config"] = render_config
        seen["blender_path"] = blender_path
        seen["timeout_seconds"] = timeout_seconds
        return RenderResult(success=True, output_path="/tmp/out.png", frame_index=structure.frame_index, message="ok")

    monkeypatch.setattr(workers, "render_structure", fake_render_structure)
    structure = Structure(frame_index=12, source_path="/tmp/sample.xyz")
    request = workers.RenderRequest(structure=structure, render_config=_render_cfg(), blender_path="/opt/blender", timeout_seconds=7)

    result = workers.RenderWorker(request).run()

    assert result.success is True
    assert seen["blender_path"] == "/opt/blender"
    assert seen["timeout_seconds"] == 7
    assert seen["render_config"].id == "app"


def test_animation_render_worker_passes_through_to_pipeline(monkeypatch):
    seen = {}

    def fake_render_animation(structures, render_config, *, output_dir, filename_template, blender_path=None, timeout_seconds=7200):
        seen["structures"] = structures
        seen["render_config"] = render_config
        seen["output_dir"] = output_dir
        seen["filename_template"] = filename_template
        seen["blender_path"] = blender_path
        seen["timeout_seconds"] = timeout_seconds
        return {"success": True, "output_dir": output_dir}

    monkeypatch.setattr(workers, "render_animation", fake_render_animation)
    structures = (
        Structure(frame_index=1, source_path="/tmp/sample.xyz"),
        Structure(frame_index=2, source_path="/tmp/sample.xyz"),
    )
    request = workers.AnimationRenderRequest(
        structures=structures,
        render_config=_render_cfg(),
        output_dir="/tmp/frames",
        filename_template="frame_{frame:04d}.png",
        blender_path="/opt/blender",
        timeout_seconds=9,
    )

    result = workers.AnimationRenderWorker(request).run()

    assert result["success"] is True
    assert seen["structures"] == structures
    assert seen["output_dir"] == "/tmp/frames"
    assert seen["filename_template"] == "frame_{frame:04d}.png"
    assert seen["blender_path"] == "/opt/blender"
    assert seen["timeout_seconds"] == 9


def test_start_background_task_falls_back_to_sync_execution(monkeypatch):
    monkeypatch.setattr(workers, "QtCore", None)
    seen = {}

    def task():
        seen["task"] = True
        return 42

    def on_result(value):
        seen["result"] = value

    workers.start_background_task(task, on_result=on_result, label="sync")
    assert seen == {"task": True, "result": 42}
