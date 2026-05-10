from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from atomstudio.config import RenderJobConfig
from atomstudio.preview.types import PreviewSelection
from atomstudio.structure.structure import Structure


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AppLogEntry:
    level: str
    message: str
    source: str = "app"
    timestamp: str = field(default_factory=_utc_now_iso)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppLogEntry":
        return cls(
            level=str(data.get("level", "info")),
            message=str(data.get("message", "")),
            source=str(data.get("source", "app")),
            timestamp=str(data.get("timestamp", _utc_now_iso())),
        )


@dataclass
class LoadedFrameBundle:
    source_path: str
    frame_selector: str = "all"
    frames: list[Structure] = field(default_factory=list)
    selected_index: int = 0

    def __post_init__(self) -> None:
        self.selected_index = self._clamp_index(self.selected_index)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def _clamp_index(self, index: int) -> int:
        if not self.frames:
            return 0
        return max(0, min(int(index), len(self.frames) - 1))

    def set_selected_index(self, index: int) -> None:
        self.selected_index = self._clamp_index(index)

    def current(self) -> Structure | None:
        if not self.frames:
            return None
        return self.frames[self.selected_index]

    def frame_indices(self) -> list[int]:
        return [int(frame.frame_index) for frame in self.frames]

    def to_dict(self) -> dict[str, Any]:
        current = self.current()
        return {
            "source_path": self.source_path,
            "frame_selector": self.frame_selector,
            "selected_index": int(self.selected_index),
            "frame_count": self.frame_count,
            "frame_indices": self.frame_indices(),
            "current_frame_index": None if current is None else int(current.frame_index),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoadedFrameBundle":
        frame_count = int(data.get("frame_count", 0))
        frame_indices = [int(v) for v in data.get("frame_indices", [])]
        frames = [
            Structure(frame_index=index, source_path=str(data.get("source_path", "")))
            for index in frame_indices
        ]
        if not frames and frame_count > 0:
            frames = [
                Structure(frame_index=index, source_path=str(data.get("source_path", "")))
                for index in range(frame_count)
            ]
        return cls(
            source_path=str(data.get("source_path", "")),
            frame_selector=str(data.get("frame_selector", "all")),
            frames=frames,
            selected_index=int(data.get("selected_index", 0)),
        )


@dataclass
class DockVisibilityState:
    inspector_visible: bool = True
    log_visible: bool = True
    axis_overlay_visible: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DockVisibilityState":
        src = {} if data is None else dict(data)
        return cls(
            inspector_visible=bool(src.get("inspector_visible", True)),
            log_visible=bool(src.get("log_visible", True)),
            axis_overlay_visible=bool(src.get("axis_overlay_visible", True)),
        )


def _selection_to_dict(selection: PreviewSelection | None) -> dict[str, Any] | None:
    if selection is None:
        return None
    return {"kind": str(selection.kind), "index": int(selection.index)}


def _selection_from_dict(data: dict[str, Any] | None) -> PreviewSelection | None:
    if not isinstance(data, dict):
        return None
    kind = str(data.get("kind", "")).strip().lower()
    if not kind:
        return None
    return PreviewSelection(kind=kind, index=int(data.get("index", 0)))


@dataclass
class AppState:
    bundle: LoadedFrameBundle | None = None
    render_config: RenderJobConfig | None = None
    preview_scene: Any | None = None
    dirty: bool = False
    status: str = "Ready"
    logs: list[AppLogEntry] = field(default_factory=list)
    last_render_output: str | None = None
    last_error: str | None = None
    selected_object: PreviewSelection | None = None
    selected_payload: dict[str, Any] | None = None
    dock_visibility: DockVisibilityState = field(default_factory=DockVisibilityState)

    def current_structure(self) -> Structure | None:
        if self.bundle is None:
            return None
        return self.bundle.current()

    def current_frame_index(self) -> int | None:
        if self.bundle is None:
            return None
        current = self.bundle.current()
        return None if current is None else int(current.frame_index)

    def set_loaded_frames(
        self,
        bundle: LoadedFrameBundle,
        *,
        render_config: RenderJobConfig | None = None,
        status: str | None = None,
    ) -> None:
        self.bundle = bundle
        self.render_config = render_config
        self.preview_scene = None
        self.selected_object = None
        self.selected_payload = None
        self.dirty = False
        self.last_error = None
        self.status = status or f"Loaded {bundle.frame_count} frame(s)"
        self.append_log("info", f"Loaded {bundle.frame_count} frame(s) from {bundle.source_path}")

    def set_render_config(self, cfg: RenderJobConfig | None, *, mark_dirty: bool = False) -> None:
        self.render_config = cfg
        self.dirty = bool(mark_dirty)

    def set_preview_scene(self, scene: Any | None) -> None:
        self.preview_scene = scene
        self.selected_object = None
        self.selected_payload = None

    def select_frame(self, index: int) -> Structure | None:
        if self.bundle is None:
            return None
        self.bundle.set_selected_index(index)
        current = self.bundle.current()
        if current is not None:
            self.status = f"Frame {current.frame_index} selected"
        return current

    def mark_dirty(self, reason: str | None = None) -> None:
        self.dirty = True
        self.status = reason or "Unsaved changes"
        if reason:
            self.append_log("warning", reason)

    def mark_clean(self) -> None:
        self.dirty = False

    def set_status(self, status: str) -> None:
        self.status = str(status)

    def append_log(self, level: str, message: str, *, source: str = "app") -> AppLogEntry:
        entry = AppLogEntry(level=str(level), message=str(message), source=str(source))
        self.logs.append(entry)
        return entry

    def clear_logs(self) -> None:
        self.logs.clear()

    def set_selection(
        self,
        selection: PreviewSelection | None,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.selected_object = selection
        self.selected_payload = payload

    def clear_selection(self) -> None:
        self.set_selection(None, payload=None)

    def set_dock_visibility(
        self,
        *,
        inspector_visible: bool | None = None,
        log_visible: bool | None = None,
        axis_overlay_visible: bool | None = None,
    ) -> None:
        if inspector_visible is not None:
            self.dock_visibility.inspector_visible = bool(inspector_visible)
        if log_visible is not None:
            self.dock_visibility.log_visible = bool(log_visible)
        if axis_overlay_visible is not None:
            self.dock_visibility.axis_overlay_visible = bool(axis_overlay_visible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": None if self.bundle is None else self.bundle.to_dict(),
            "render_config": None if self.render_config is None else self.render_config.to_dict(),
            "dirty": bool(self.dirty),
            "status": self.status,
            "logs": [asdict(entry) for entry in self.logs],
            "last_render_output": self.last_render_output,
            "last_error": self.last_error,
            "selected_object": _selection_to_dict(self.selected_object),
            "selected_payload": self.selected_payload,
            "dock_visibility": asdict(self.dock_visibility),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppState":
        bundle_data = data.get("bundle")
        render_config_data = data.get("render_config")
        return cls(
            bundle=LoadedFrameBundle.from_dict(bundle_data) if isinstance(bundle_data, dict) else None,
            render_config=RenderJobConfig.from_dict(render_config_data) if isinstance(render_config_data, dict) else None,
            dirty=bool(data.get("dirty", False)),
            status=str(data.get("status", "Ready")),
            logs=[AppLogEntry.from_dict(entry) for entry in data.get("logs", []) if isinstance(entry, dict)],
            last_render_output=(
                str(data["last_render_output"]) if data.get("last_render_output") is not None else None
            ),
            last_error=str(data["last_error"]) if data.get("last_error") is not None else None,
            selected_object=_selection_from_dict(data.get("selected_object")),
            selected_payload=(
                dict(data["selected_payload"]) if isinstance(data.get("selected_payload"), dict) else None
            ),
            dock_visibility=DockVisibilityState.from_dict(data.get("dock_visibility")),
        )

    def title_suffix(self) -> str:
        parts: list[str] = []
        if self.bundle is not None:
            parts.append(self.bundle.source_path)
            current = self.bundle.current()
            if current is not None:
                parts.append(f"frame {current.frame_index}")
        if self.dirty:
            parts.append("*")
        return " | ".join(parts) if parts else "AtomStudio"
