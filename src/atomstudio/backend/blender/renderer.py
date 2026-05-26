from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from atomstudio.backend.blender.animation_renderer import BlenderAnimationRenderer
from atomstudio.backend.blender.material_adapter import scene_value
from atomstudio.backend.blender.scene_setup import apply_camera_lights_effects, apply_render_environment, write_scene_metadata
from atomstudio.backend.blender.scene_writer import BlenderSceneWriter
from atomstudio.config import RenderJobConfig
from atomstudio.scene.builder import bake_preview_model_transform, build_render_scene
from atomstudio.scene.materials.registry import MaterialRegistry
from atomstudio.structure.structure import Structure
from atomstudio.style.registry import get_scene_style
from atomstudio.style.resolver import resolve_style_bundle

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {str(k): _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        out = value.to_dict()
        if isinstance(out, dict):
            return {str(k): _jsonable(v) for k, v in out.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return value


def build_render_scene_payload(structure: Structure, cfg: RenderJobConfig) -> dict[str, Any]:
    scene = bake_preview_model_transform(build_render_scene(structure, cfg))
    return {
        "schema": "atomstudio.render_scene.v1",
        "source": "scene_builder",
        "render_scene": _jsonable(scene),
        "config": cfg.to_dict(),
    }


class BlenderRenderer:
    def __init__(self, cfg: RenderJobConfig) -> None:
        self.cfg = cfg
        self.scene_style = get_scene_style(cfg.style.scene_style)
        self.style_bundle = resolve_style_bundle(cfg.style, self.scene_style)
        self.registry = MaterialRegistry()

    def render_scene(self, scene: Any) -> dict[str, Any]:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")

        background = scene_value(scene, "background", self.style_bundle.background)
        apply_render_environment(self.cfg, background=background)

        writer = BlenderSceneWriter(
            self.cfg,
            registry=self.registry,
            default_material_pipeline=str(self.style_bundle.material_style.pipeline),
            default_material_style_name=str(self.style_bundle.material_style_name),
        )
        _objects, stats, points = writer.write(scene)

        ground_spec = apply_camera_lights_effects(
            self.cfg,
            scene,
            points,
            registry=self.registry,
            style_bundle=self.style_bundle,
        )
        write_scene_metadata(scene, style_bundle=self.style_bundle, ground_spec=ground_spec)

        output_path = self._render_output()
        metadata = dict(scene_value(scene, "metadata", {}) or {})
        return {
            "success": True,
            "output_path": output_path,
            "frame_index": int(metadata.get("frame_index", 0)),
            "stats": stats,
            "message": "ok",
        }

    def _render_output(self) -> str:
        output_path = self.cfg.output.path
        if not output_path:
            raise ValueError("output.path must be set before rendering")
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        return str(path)


def render_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = RenderJobConfig.from_dict(payload["config"])
    scene = payload["render_scene"]
    return BlenderRenderer(cfg).render_scene(scene)


def render_animation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    frames = list(payload.get("frames", []) or [])
    cfg = RenderJobConfig.from_dict(frames[0]["config"] if frames else payload["config"])
    return BlenderAnimationRenderer(cfg).render_frames(frames, output_dir=str(payload.get("output_dir", "")))


__all__ = [
    "BlenderRenderer",
    "BlenderAnimationRenderer",
    "build_render_scene_payload",
    "render_animation_payload",
    "render_payload",
    "scene_value",
]
