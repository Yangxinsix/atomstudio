from __future__ import annotations

from copy import deepcopy

from atomstudio.style.color_style import COLOR_STYLE_LIBRARY, ColorStyle
from atomstudio.style.light_style import LIGHT_STYLE_LIBRARY
from atomstudio.style.material_style import MATERIAL_STYLE_LIBRARY, MaterialStyle
from atomstudio.style.radius_style import RADIUS_STYLE_LIBRARY, RadiusStyle
from atomstudio.style.scene_style import (
    CERAMIC_STUDIO_SCENE_STYLE,
    CLEAN_GLOSSY_SCENE_STYLE,
    DARKLAB_SCENE_STYLE,
    DEFAULT_SCENE_STYLE,
    GLASS_LAB_SCENE_STYLE,
    HANDDRAWN_SCENE_STYLE,
    HANDDRAWN_V2_SCENE_STYLE,
    HOLOGRAPHIC_SCENE_STYLE,
    JADE_SCENE_STYLE,
    MONOCHROME_SCENE_STYLE,
    PEARL_SCENE_STYLE,
    PORCELAIN_SCENE_STYLE,
    SceneStyle,
    SOLID_GLASS_SCENE_STYLE,
    STUDIO_CRYSTAL_PRODUCT_SCENE_STYLE,
    STUDIO_COOL_RIM_SCENE_STYLE,
    STUDIO_DARKLAB_SCENE_STYLE,
    STUDIO_HIGHKEY_CLEAN_SCENE_STYLE,
    STUDIO_HIGHKEY_SCENE_STYLE,
    STUDIO_MACRO_SCENE_STYLE,
    STUDIO_PEARL_SCENE_STYLE,
    STUDIO_SOFTMETAL_SCENE_STYLE,
    STUDIO_WARM_SOFT_SCENE_STYLE,
    WARM_CLAY_SCENE_STYLE,
)

_STYLE_LIBRARY: dict[str, SceneStyle] = {
    "default": DEFAULT_SCENE_STYLE,
    "darklab": DARKLAB_SCENE_STYLE,
    "monochrome": MONOCHROME_SCENE_STYLE,
    "ceramic_studio": CERAMIC_STUDIO_SCENE_STYLE,
    "glass_lab": GLASS_LAB_SCENE_STYLE,
    "clean_glossy": CLEAN_GLOSSY_SCENE_STYLE,
    "porcelain": PORCELAIN_SCENE_STYLE,
    "solid_glass": SOLID_GLASS_SCENE_STYLE,
    "jade": JADE_SCENE_STYLE,
    "pearl": PEARL_SCENE_STYLE,
    "holographic": HOLOGRAPHIC_SCENE_STYLE,
    "warm_clay": WARM_CLAY_SCENE_STYLE,
    "studio_highkey": STUDIO_HIGHKEY_SCENE_STYLE,
    "studio_highkey_clean": STUDIO_HIGHKEY_CLEAN_SCENE_STYLE,
    "studio_darklab": STUDIO_DARKLAB_SCENE_STYLE,
    "studio_warm_soft": STUDIO_WARM_SOFT_SCENE_STYLE,
    "studio_cool_rim": STUDIO_COOL_RIM_SCENE_STYLE,
    "studio_macro": STUDIO_MACRO_SCENE_STYLE,
    "studio_crystal_product": STUDIO_CRYSTAL_PRODUCT_SCENE_STYLE,
    "studio_pearl": STUDIO_PEARL_SCENE_STYLE,
    "studio_softmetal": STUDIO_SOFTMETAL_SCENE_STYLE,
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
    "clean_glossy",
    "porcelain",
    "solid_glass",
    "jade",
    "pearl",
    "holographic",
    "warm_clay",
    "studio_satin",
    "studio_pearl",
    "studio_softmetal",
    "studio_white_ceramic",
    "studio_macro_gloss",
    "studio_product_gloss",
    "handdrawn",
)

LIGHT_STYLE_ALIAS: dict[str, str] = {
    "default": "batoms_soft",
    "darklab": "darklab_rim",
    "monochrome": "monochrome_softbox",
    "ceramic_studio": "ceramic_softbox",
    "glass_lab": "glass_lab_rim",
    "clean_glossy": "style_sphere_showcase",
    "porcelain": "style_sphere_showcase",
    "solid_glass": "style_sphere_showcase",
    "jade": "style_sphere_showcase",
    "pearl": "style_sphere_showcase",
    "holographic": "style_sphere_showcase",
    "warm_clay": "style_sphere_showcase",
    "studio_highkey": "studio_highkey_softbox",
    "studio_highkey_clean": "studio_highkey_softbox",
    "studio_darklab": "style_sphere_showcase",
    "studio_warm_soft": "studio_warm_softbox",
    "studio_cool_rim": "studio_cool_rim_softbox",
    "studio_macro": "studio_macro_softbox",
    "studio_crystal_product": "studio_crystal_tabletop",
    "studio_pearl": "studio_highkey_softbox",
    "studio_softmetal": "studio_softmetal_strip",
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
