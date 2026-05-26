from __future__ import annotations

from typing import Any

from atomstudio.backend.blender.camera_writer import BlenderCameraWriter
from atomstudio.backend.blender.light_writer import BlenderLightWriter
from atomstudio.backend.blender.material_adapter import scene_value
from atomstudio.config import RenderJobConfig
from atomstudio.scene.color_management import ColorManagementBuilder
from atomstudio.scene.effects_builder import RenderEffectsBuilder
from atomstudio.scene.ground_builder import GroundBuilder
from atomstudio.scene.materials.registry import MaterialRegistry
from atomstudio.scene.outline_builder import OutlineBuilder
from atomstudio.scene.render_setup import RenderSetup
from atomstudio.scene.style_helpers import resolve_handdrawn_config
from atomstudio.scene.sunbeam_builder import SunbeamBuilder

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


def apply_render_environment(cfg: RenderJobConfig, *, background: Any) -> None:
    RenderSetup.from_cfg(cfg).apply()
    ColorManagementBuilder.from_cfg(cfg).apply()
    from atomstudio.scene.world_builder import WorldBuilder

    bg = tuple(float(v) for v in background)
    WorldBuilder.from_cfg(cfg, background=bg).apply()
    RenderEffectsBuilder.from_cfg(cfg, background=bg).apply()


def apply_camera_lights_effects(
    cfg: RenderJobConfig,
    scene: Any,
    points: list[Any],
    *,
    registry: MaterialRegistry,
    style_bundle: Any,
) -> dict[str, Any]:
    BlenderCameraWriter(cfg).write(scene, points)
    BlenderLightWriter(cfg, default_light_style=style_bundle.light_style_name).write(scene, points)
    _, ground_spec = GroundBuilder.from_cfg(cfg, registry=registry).build(points)
    SunbeamBuilder.from_cfg(cfg).apply(points)

    handdrawn = resolve_handdrawn_config(
        style_bundle.material_style.pipeline,
        cfg,
        material_style=style_bundle.material_style,
    )
    OutlineBuilder.from_cfg(
        cfg,
        style_name=style_bundle.scene_style_name,
        material_pipeline=style_bundle.material_style.pipeline,
        outline=style_bundle.outline,
        handdrawn=handdrawn,
    ).apply()
    return dict(ground_spec)


def write_scene_metadata(scene: Any, *, style_bundle: Any, ground_spec: dict[str, Any] | None = None) -> None:
    if bpy is None:
        raise RuntimeError("bpy is not available. Run this function inside Blender.")

    metadata = dict(scene_value(scene, "metadata", {}) or {})
    bpy.context.scene["atomstudio_source"] = str(metadata.get("source_path", ""))
    bpy.context.scene["atomstudio_frame"] = int(scene_value(scene, "frame_index", metadata.get("frame_index", 0)))
    bpy.context.scene["atomstudio_style"] = style_bundle.scene_style_name
    bpy.context.scene["atomstudio_color_style"] = style_bundle.color_style_name
    bpy.context.scene["atomstudio_material_style"] = style_bundle.material_style_name
    bpy.context.scene["atomstudio_light_style"] = style_bundle.light_style_name

    if ground_spec is not None:
        bpy.context.scene["atomstudio_ground_requested_mode"] = str(ground_spec["requested_mode"])
        bpy.context.scene["atomstudio_ground_effective_mode"] = str(ground_spec["effective_mode"])
        bpy.context.scene["atomstudio_ground_enabled"] = bool(ground_spec["enabled"])
