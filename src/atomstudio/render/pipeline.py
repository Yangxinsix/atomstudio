from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from atomstudio.bridge.blender_runner import run_blender_animation, run_blender_render
from atomstudio.io.ase_loader import load_structure, load_trajectory
from atomstudio.paths import normalize_host_path
from atomstudio.render.config_resolver import load_batch_config
from atomstudio.render.results import AnimationResult, BatchResult, JobReport, RenderResult
from atomstudio.render.space_filling import resolve_auto_space_filling_scale
from atomstudio.config import RenderJobConfig
from atomstudio.structure.structure import Structure


def _missing_output_message(output_path: str) -> str:
    return f"Render completed but output file was not created: {normalize_host_path(output_path) or '<empty output path>'}"


def _output_file_exists(output_path: str) -> bool:
    if not output_path:
        return False
    return Path(normalize_host_path(output_path)).expanduser().is_file()


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def render_structure(
    structure: Structure,
    cfg: RenderJobConfig,
    *,
    blender_path: str | None = None,
    timeout_seconds: int = 1800,
) -> RenderResult:
    start = time.perf_counter()
    effective_cfg, adjustments = resolve_auto_space_filling_scale(structure, cfg)
    try:
        result = run_blender_render(
            structure,
            effective_cfg,
            blender_path=blender_path,
            timeout_seconds=int(timeout_seconds),
        )
        elapsed = time.perf_counter() - start
        output_path = normalize_host_path(str(result.get("output_path", effective_cfg.output.path or "")))
        success = bool(result.get("success", True))
        message = str(result.get("message", "ok"))
        if success and not _output_file_exists(output_path):
            success = False
            message = _missing_output_message(output_path)
        return RenderResult(
            success=success,
            output_path=output_path,
            frame_index=int(result.get("frame_index", structure.frame_index)),
            message=message,
            elapsed_seconds=elapsed,
            adjustments=adjustments,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return RenderResult(
            success=False,
            output_path=normalize_host_path(str(effective_cfg.output.path or "")),
            frame_index=structure.frame_index,
            message=str(exc),
            elapsed_seconds=elapsed,
            adjustments=adjustments,
        )


def render_animation(
    frames: list[Structure] | tuple[Structure, ...],
    cfg: RenderJobConfig,
    *,
    output_dir: str,
    filename_template: str = "frame_{frame:04d}.png",
    blender_path: str | None = None,
    timeout_seconds: int = 7200,
) -> AnimationResult:
    start = time.perf_counter()
    frame_adjustments: dict[int, dict[str, Any]] = {}
    for structure in frames:
        _effective_cfg, adjustments = resolve_auto_space_filling_scale(structure, cfg)
        frame_adjustments[int(structure.frame_index)] = adjustments
    try:
        result = run_blender_animation(
            list(frames),
            cfg,
            output_dir=output_dir,
            filename_template=filename_template,
            blender_path=blender_path,
            timeout_seconds=int(timeout_seconds),
        )
        elapsed = time.perf_counter() - start
        outputs = [normalize_host_path(str(path)) for path in result.get("outputs", [])]
        failed_frames = [int(frame) for frame in result.get("failed_frames", [])]
        frame_reports: list[RenderResult] = []
        for raw in result.get("frame_results", []) or []:
            output_path = normalize_host_path(str(raw.get("output_path", "")))
            success = bool(raw.get("success", False))
            frame_index = int(raw.get("frame_index", 0))
            message = str(raw.get("message", "ok"))
            if success and not _output_file_exists(output_path):
                success = False
                message = _missing_output_message(output_path)
                failed_frames.append(frame_index)
            frame_reports.append(
                RenderResult(
                    success=success,
                    output_path=output_path,
                    frame_index=frame_index,
                    message=message,
                    elapsed_seconds=0.0,
                    adjustments=frame_adjustments.get(frame_index, {}),
                )
            )
        failed_frames = sorted(set(failed_frames))
        return AnimationResult(
            success=not failed_frames and bool(result.get("success", True)),
            output_dir=normalize_host_path(str(result.get("output_dir", output_dir))),
            outputs=[path for path in outputs if _output_file_exists(path)],
            failed_frames=failed_frames,
            frame_reports=frame_reports,
            elapsed_seconds=elapsed,
            message=str(result.get("message", "ok")) if not failed_frames else "some frames failed",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return AnimationResult(
            success=False,
            output_dir=normalize_host_path(str(output_dir)),
            failed_frames=[int(frame.frame_index) for frame in frames],
            elapsed_seconds=elapsed,
            message=str(exc),
        )


def render_batch(config_path: str) -> BatchResult:
    cfg = load_batch_config(config_path)

    log_path = Path.cwd() / "render.log"
    reports: list[JobReport] = []

    for job in cfg.jobs:
        job_start = time.perf_counter()
        _append_log(log_path, f"[job:{job.id}] start input={job.input.path} frames={job.input.frames}")

        frames = load_trajectory(job.input.path, job.input.frames)
        outputs: list[str] = []
        failed_frames: list[int] = []
        frame_reports: list[RenderResult] = []

        for structure in frames:
            output_path = job.output.resolve_path(job.id, structure.frame_index)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            render_cfg = job.with_output_path(str(output_path))

            result = render_structure(structure, render_cfg)
            if not result.success:
                _append_log(log_path, f"[job:{job.id}] frame={structure.frame_index} failed first try: {result.message}")
                result = render_structure(structure, render_cfg)

            if result.success:
                outputs.append(result.output_path)
                _append_log(log_path, f"[job:{job.id}] frame={structure.frame_index} ok {result.output_path}")
            else:
                failed_frames.append(structure.frame_index)
                _append_log(log_path, f"[job:{job.id}] frame={structure.frame_index} failed: {result.message}")
            frame_reports.append(result)

        elapsed = time.perf_counter() - job_start
        job_success = len(failed_frames) == 0
        report = JobReport(
            job_id=job.id,
            success=job_success,
            outputs=outputs,
            failed_frames=failed_frames,
            elapsed_seconds=elapsed,
            message="ok" if job_success else "some frames failed",
            frame_reports=frame_reports,
        )

        report_dir = (
            Path(job.output.dir).expanduser().resolve()
            if job.output.dir
            else Path(outputs[0]).resolve().parent if outputs else Path.cwd()
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "job_report.json"
        report_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        reports.append(report)
        _append_log(log_path, f"[job:{job.id}] done success={job_success} elapsed={elapsed:.2f}s")

    return BatchResult(success=all(r.success for r in reports), reports=reports)


def render_single_from_config(config_path: str) -> RenderResult:
    cfg = load_batch_config(config_path)
    if len(cfg.jobs) != 1:
        raise ValueError("render --config expects exactly one job. Use batch for multiple jobs.")

    job = cfg.jobs[0]
    structure = load_structure(job.input.path, frame=job.input.frames)
    output = job.output.resolve_path(job.id, structure.frame_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    return render_structure(structure, job.with_output_path(str(output)))
