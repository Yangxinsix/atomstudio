from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atomstudio.config import StyleConfig
from atomstudio.style.outline_style import OutlineStyle
from atomstudio.style.color_style import ColorStyle
from atomstudio.style.material_style import MaterialStyle
from atomstudio.style.registry import (
    get_color_style,
    get_light_style_name,
    get_material_style,
    get_scene_style,
)
from atomstudio.style.scene_style import SceneStyle


@dataclass
class ResolvedStyleBundle:
    scene_style_name: str
    color_style_name: str
    material_style_name: str
    light_style_name: str
    color_style: ColorStyle
    material_style: MaterialStyle
    background: tuple[float, float, float, float]
    outline: OutlineStyle
    structure_tokens: dict[str, Any]
    camera_tokens: dict[str, Any]
    render_tokens: dict[str, Any]


def resolve_style_bundle(
    style_cfg: StyleConfig,
    scene_style: SceneStyle | None = None,
) -> ResolvedStyleBundle:
    base = scene_style if scene_style is not None else get_scene_style(style_cfg.scene_style)

    color_style_name = str(style_cfg.color_style or base.color_style.name).strip().lower()
    material_style_name = str(style_cfg.material_style or base.material_style.name).strip().lower()
    light_style_name = get_light_style_name(str(style_cfg.light_style or base.light_style).strip().lower())

    color_style = get_color_style(color_style_name)
    material_style = get_material_style(material_style_name)
    background = style_cfg.background if style_cfg.background is not None else base.background
    base_outline = OutlineStyle.from_any(base.outline)
    if style_cfg.outline is None:
        outline = base_outline
    else:
        # Keep scene-style per-object outline profile while honoring user overrides.
        outline = OutlineStyle.from_any(style_cfg.outline, fallback=base_outline)

    return ResolvedStyleBundle(
        scene_style_name=base.name,
        color_style_name=color_style_name,
        material_style_name=material_style_name,
        light_style_name=light_style_name,
        color_style=color_style,
        material_style=material_style,
        background=background,
        outline=outline,
        structure_tokens=dict(base.structure_tokens),
        camera_tokens=dict(base.camera_tokens),
        render_tokens=dict(base.render_tokens),
    )
