from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from atomstudio.config import RenderJobConfig
from atomstudio.paths import normalize_host_path
from atomstudio.structure.structure import Structure

from .state import LoadedFrameBundle

try:  # pragma: no cover - Qt is optional in tests
    from PySide6 import QtCore  # type: ignore
except Exception:  # pragma: no cover
    QtCore = None


@dataclass(frozen=True)
class LoadStructureRequest:
    input_path: str
    frame_selector: str = "all"
    selected_index: int | None = None


@dataclass(frozen=True)
class PreviewRequest:
    structure: Structure
    render_config: RenderJobConfig
    preview_settings: Any | None = None


@dataclass(frozen=True)
class RenderRequest:
    structure: Structure
    render_config: RenderJobConfig
    blender_path: str | None = None
    timeout_seconds: int = 1800


@dataclass(frozen=True)
class AnimationRenderRequest:
    structures: tuple[Structure, ...]
    render_config: RenderJobConfig
    output_dir: str
    filename_template: str = "frame_{frame:04d}.png"
    blender_path: str | None = None
    timeout_seconds: int = 7200


def load_trajectory(path: str, frame_selector: str = "all") -> list[Structure]:
    from atomstudio.io.ase_loader import load_trajectory as _load_trajectory

    return _load_trajectory(path, frame_selector=frame_selector)


def selected_frame_indices(path: str, frame_selector: str = "all") -> list[int]:
    from atomstudio.io.ase_loader import count_trajectory_frames, parse_frame_selector

    return parse_frame_selector(frame_selector, count_trajectory_frames(path))


def render_structure(
    structure: Structure,
    render_config: RenderJobConfig,
    *,
    blender_path: str | None = None,
    timeout_seconds: int = 1800,
):
    from atomstudio.render.pipeline import render_structure as _render_structure

    return _render_structure(
        structure,
        render_config,
        blender_path=blender_path,
        timeout_seconds=timeout_seconds,
    )


def render_animation(
    structures: tuple[Structure, ...],
    render_config: RenderJobConfig,
    *,
    output_dir: str,
    filename_template: str = "frame_{frame:04d}.png",
    blender_path: str | None = None,
    timeout_seconds: int = 7200,
):
    from atomstudio.render.pipeline import render_animation as _render_animation

    return _render_animation(
        list(structures),
        render_config,
        output_dir=output_dir,
        filename_template=filename_template,
        blender_path=blender_path,
        timeout_seconds=timeout_seconds,
    )


def _default_output_path(structure: Structure) -> str:
    source = Path(normalize_host_path(structure.source_path)).expanduser() if structure.source_path else None
    if source and source.name:
        stem = source.stem or "render"
        return str((Path.cwd() / f"{stem}_atomstudio.png").resolve())
    return str((Path.cwd() / "atomstudio_render.png").resolve())


def build_default_render_config(structure: Structure, output_path: str | None = None) -> RenderJobConfig:
    path = str(output_path or _default_output_path(structure))
    return RenderJobConfig.from_dict(
        {
            "id": "atomstudio_app",
            "input": {"path": structure.source_path or "<memory>", "frames": "last"},
            "output": {"path": path},
            "style": {"scene_style": "default"},
            "render": {"engine": "cycles", "device": "auto", "transparent_bg": True},
        }
    ).with_output_path(path)


def load_structure_bundle(request: LoadStructureRequest) -> LoadedFrameBundle:
    path = str(Path(normalize_host_path(request.input_path)).expanduser().resolve())
    selector = str(request.frame_selector or "all")
    frame_indices = selected_frame_indices(path, selector)
    if not frame_indices:
        raise ValueError(f"No frames were loaded from {path}")
    selected_index = 0 if request.selected_index is None else int(request.selected_index)
    bundle = LoadedFrameBundle.lazy(
        source_path=path,
        frame_selector=selector,
        frame_indices=frame_indices,
        selected_index=selected_index,
    )
    bundle.current()
    return bundle


def build_preview_scene_for_structure(
    structure: Structure,
    render_config: RenderJobConfig,
    preview_settings: Any | None = None,
) -> Any:
    try:
        from atomstudio.preview.builder import build_preview_scene  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only when preview package is missing
        raise RuntimeError("atomstudio.preview.builder.build_preview_scene is unavailable") from exc
    return build_preview_scene(structure, render_config, preview_settings)


def render_with_blender(request: RenderRequest):
    return render_structure(
        request.structure,
        request.render_config,
        blender_path=request.blender_path,
        timeout_seconds=int(request.timeout_seconds),
    )


def render_animation_with_blender(request: AnimationRenderRequest):
    return render_animation(
        tuple(request.structures),
        request.render_config,
        output_dir=request.output_dir,
        filename_template=request.filename_template,
        blender_path=request.blender_path,
        timeout_seconds=int(request.timeout_seconds),
    )


class LoadStructureWorker:
    def __init__(self, request: LoadStructureRequest) -> None:
        self.request = request

    def run(self) -> LoadedFrameBundle:
        return load_structure_bundle(self.request)


class PreviewWorker:
    def __init__(self, request: PreviewRequest) -> None:
        self.request = request

    def run(self) -> Any:
        return build_preview_scene_for_structure(
            self.request.structure,
            self.request.render_config,
            self.request.preview_settings,
        )


class RenderWorker:
    def __init__(self, request: RenderRequest) -> None:
        self.request = request

    def run(self):
        return render_with_blender(self.request)


class AnimationRenderWorker:
    def __init__(self, request: AnimationRenderRequest) -> None:
        self.request = request

    def run(self):
        return render_animation_with_blender(self.request)


if QtCore is not None:  # pragma: no cover - exercised only with Qt installed

    class _TaskRunner(QtCore.QObject):
        finished = QtCore.Signal(object)
        failed = QtCore.Signal(object)

        def __init__(self, task: Callable[[], Any], *, label: str = "task") -> None:
            super().__init__()
            self._task = task
            self._label = label

        @QtCore.Slot()
        def run(self) -> None:
            try:
                result = self._task()
            except Exception as exc:  # pragma: no cover - routed through UI
                self.failed.emit(exc)
                return
            self.finished.emit(result)


    class _TaskCallbacks(QtCore.QObject):
        finished = QtCore.Signal(object)
        failed = QtCore.Signal(object)

        @QtCore.Slot(object)
        def deliver_finished(self, result: object) -> None:
            self.finished.emit(result)

        @QtCore.Slot(object)
        def deliver_failed(self, error: object) -> None:
            self.failed.emit(error)


def start_background_task(
    task: Callable[[], Any],
    *,
    on_result: Callable[[Any], None] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
    parent: Any | None = None,
    label: str = "task",
):
    if QtCore is None:  # pragma: no cover - simple synchronous fallback for tests
        try:
            result = task()
        except BaseException as exc:  # noqa: BLE001 - propagate to callback
            if on_error is not None:
                on_error(exc)
            else:
                raise
        else:
            if on_result is not None:
                on_result(result)
        return None

    thread = QtCore.QThread(parent)
    runner = _TaskRunner(task, label=label)
    callbacks = _TaskCallbacks(parent)
    runner.moveToThread(thread)

    thread.started.connect(runner.run)
    if on_result is not None:
        callbacks.finished.connect(on_result)
    if on_error is not None:
        callbacks.failed.connect(on_error)
    runner.finished.connect(callbacks.deliver_finished)
    runner.failed.connect(callbacks.deliver_failed)
    runner.finished.connect(thread.quit)
    runner.failed.connect(thread.quit)
    thread.finished.connect(runner.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread._runner = runner  # type: ignore[attr-defined]
    thread._callbacks = callbacks  # type: ignore[attr-defined]
    thread.start()
    return thread
