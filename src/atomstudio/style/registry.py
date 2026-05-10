from __future__ import annotations

from copy import deepcopy

from atomstudio.style.color_style import COLOR_STYLE_LIBRARY, ColorStyle
from atomstudio.style.light_style import LIGHT_STYLE_LIBRARY
from atomstudio.style.material_style import MATERIAL_STYLE_LIBRARY, MaterialStyle
from atomstudio.style.radius_style import RADIUS_STYLE_LIBRARY, RadiusStyle
from atomstudio.style.scene_style import (
    DARKLAB_SCENE_STYLE,
    DEFAULT_SCENE_STYLE,
    HANDDRAWN_SCENE_STYLE,
    HANDDRAWN_V2_SCENE_STYLE,
    MONOCHROME_SCENE_STYLE,
    SceneStyle,
)

_STYLE_LIBRARY: dict[str, SceneStyle] = {
    "default": DEFAULT_SCENE_STYLE,
    "darklab": DARKLAB_SCENE_STYLE,
    "monochrome": MONOCHROME_SCENE_STYLE,
    "handdrawn": HANDDRAWN_SCENE_STYLE,
    "handdrawn_v2": HANDDRAWN_V2_SCENE_STYLE,
}

_HIDDEN_STYLE_LIBRARY: dict[str, SceneStyle] = {}

_PUBLIC_MATERIAL_STYLE_NAMES = (
    "clean",
    "glass",
    "ceramic",
    "metallic",
    "emissive",
    "marble",
    "handdrawn",
)

LIGHT_STYLE_ALIAS: dict[str, str] = {
    "default": "batoms_soft",
    "darklab": "three_point",
    "monochrome": "three_point",
    "handdrawn": "handdrawn_soft",
    "handdrawn_v2": "handdrawn_soft_spot",
}


def style_choices() -> list[str]:
    return list(_STYLE_LIBRARY.keys())


def scene_style_choices() -> list[str]:
    return style_choices()


def color_style_choices() -> list[str]:
    return list(COLOR_STYLE_LIBRARY.keys())


def material_style_choices() -> list[str]:
    return list(_PUBLIC_MATERIAL_STYLE_NAMES)


def radius_style_choices() -> list[str]:
    return list(RADIUS_STYLE_LIBRARY.keys())


def light_style_choices() -> list[str]:
    return sorted(set(LIGHT_STYLE_ALIAS.keys()) | set(LIGHT_STYLE_LIBRARY.keys()))


def normalize_style_name(name: str) -> str:
    style = str(name or "default").strip().lower()
    if style not in _STYLE_LIBRARY:
        raise ValueError(f"Unknown style '{style}'. Valid styles: {', '.join(style_choices())}")
    return style


def get_scene_style(name: str) -> SceneStyle:
    style = str(name or "default").strip().lower()
    if style in _STYLE_LIBRARY:
        return deepcopy(_STYLE_LIBRARY[style])
    if style in _HIDDEN_STYLE_LIBRARY:
        return deepcopy(_HIDDEN_STYLE_LIBRARY[style])
    raise ValueError(f"Unknown style '{style}'. Valid styles: {', '.join(style_choices())}")


def get_color_style(name: str) -> ColorStyle:
    style = str(name or "default").strip().lower()
    if style not in COLOR_STYLE_LIBRARY:
        raise ValueError(f"Unknown color_style '{style}'. Valid styles: {', '.join(color_style_choices())}")
    return deepcopy(COLOR_STYLE_LIBRARY[style])


def get_material_style(name: str) -> MaterialStyle:
    style = str(name or "default").strip().lower()
    if style not in MATERIAL_STYLE_LIBRARY:
        raise ValueError(
            f"Unknown material_style '{style}'. Valid styles: {', '.join(material_style_choices())}"
        )
    return deepcopy(MATERIAL_STYLE_LIBRARY[style])


def get_radius_style(name: str) -> RadiusStyle:
    style = str(name or "atomic").strip().lower()
    if style not in RADIUS_STYLE_LIBRARY:
        raise ValueError(f"Unknown radius_style '{style}'. Valid styles: {', '.join(radius_style_choices())}")
    return deepcopy(RADIUS_STYLE_LIBRARY[style])


def get_light_style_name(name: str) -> str:
    style = str(name or "default").strip().lower()
    if style in LIGHT_STYLE_ALIAS:
        return str(LIGHT_STYLE_ALIAS[style])
    if style in LIGHT_STYLE_LIBRARY:
        return style
    raise ValueError(f"Unknown light_style '{style}'. Valid styles: {', '.join(light_style_choices())}")


def is_handdrawn_style(name: str) -> bool:
    style = str(name or "").strip().lower()
    if style in MATERIAL_STYLE_LIBRARY:
        return get_material_style(style).pipeline == "handdrawn"
    if style in _STYLE_LIBRARY:
        return get_scene_style(style).material_style.pipeline == "handdrawn"
    return style == "handdrawn"
