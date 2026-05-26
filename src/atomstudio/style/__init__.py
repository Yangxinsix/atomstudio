from __future__ import annotations

_COLOR_EXPORTS = {
    "COLOR_STYLE_LIBRARY",
    "CPK_COLOR_STYLE",
    "ColorStyle",
    "JMOL_COLOR_STYLE",
    "JMOL_SOFT_COLOR_STYLE",
    "MONO_GRAY_COLOR_STYLE",
    "STUDIO_HIGHKEY_COLOR_STYLE",
    "STUDIO_PRODUCT_COLOR_STYLE",
    "VESTA_COLOR_STYLE",
}
_MATERIAL_EXPORTS = {
    "CERAMIC_MATERIAL_STYLE",
    "CLEAN_MATERIAL_STYLE",
    "CLEAN_GLOSSY_MATERIAL_STYLE",
    "EMISSIVE_MATERIAL_STYLE",
    "GLASS_MATERIAL_STYLE",
    "HOLOGRAPHIC_MATERIAL_STYLE",
    "JADE_MATERIAL_STYLE",
    "MATERIAL_STYLE_LIBRARY",
    "METALLIC_MATERIAL_STYLE",
    "PEARL_MATERIAL_STYLE",
    "PORCELAIN_MATERIAL_STYLE",
    "SOLID_GLASS_MATERIAL_STYLE",
    "STUDIO_PRODUCT_GLOSS_MATERIAL_STYLE",
    "WARM_CLAY_MATERIAL_STYLE",
    "MaterialStyle",
    "build_handdrawn_material_style",
    "tune_rgb",
    "tune_rgba",
}
_RADIUS_EXPORTS = {
    "ATOMIC_RADIUS_STYLE",
    "COVALENT_RADIUS_STYLE",
    "IONIC_RADIUS_STYLE",
    "RADIUS_STYLE_LIBRARY",
    "RadiusStyle",
    "VDW_RADIUS_STYLE",
}
_REGISTRY_EXPORTS = {
    "color_style_choices",
    "get_color_style",
    "get_light_style_name",
    "get_material_style",
    "get_radius_style",
    "get_scene_style",
    "is_handdrawn_style",
    "light_style_choices",
    "material_style_choices",
    "normalize_style_name",
    "radius_style_choices",
    "scene_style_choices",
    "style_choices",
}


def __getattr__(name: str):
    if name == "MaterialSpec":
        from atomstudio.scene.materials.specs import MaterialSpec

        return MaterialSpec
    if name in _COLOR_EXPORTS:
        from atomstudio.style import color_style

        return getattr(color_style, name)
    if name == "LIGHT_STYLE_LIBRARY":
        from atomstudio.style.light_style import LIGHT_STYLE_LIBRARY

        return LIGHT_STYLE_LIBRARY
    if name in _MATERIAL_EXPORTS:
        from atomstudio.style import material_style

        return getattr(material_style, name)
    if name in {"OutlineRoleStyle", "OutlineStyle"}:
        from atomstudio.style import outline_style

        return getattr(outline_style, name)
    if name in _RADIUS_EXPORTS:
        from atomstudio.style import radius_style

        return getattr(radius_style, name)
    if name in _REGISTRY_EXPORTS:
        from atomstudio.style import registry

        return getattr(registry, name)
    if name in {"ResolvedStyleBundle", "resolve_style_bundle"}:
        from atomstudio.style import resolver

        return getattr(resolver, name)
    if name == "SceneStyle":
        from atomstudio.style.scene_style import SceneStyle

        return SceneStyle
    raise AttributeError(name)


__all__ = [
    "ColorStyle",
    "JMOL_COLOR_STYLE",
    "JMOL_SOFT_COLOR_STYLE",
    "MONO_GRAY_COLOR_STYLE",
    "STUDIO_HIGHKEY_COLOR_STYLE",
    "STUDIO_PRODUCT_COLOR_STYLE",
    "CPK_COLOR_STYLE",
    "VESTA_COLOR_STYLE",
    "COLOR_STYLE_LIBRARY",
    "LIGHT_STYLE_LIBRARY",
    "MaterialSpec",
    "MaterialStyle",
    "CLEAN_MATERIAL_STYLE",
    "CLEAN_GLOSSY_MATERIAL_STYLE",
    "GLASS_MATERIAL_STYLE",
    "CERAMIC_MATERIAL_STYLE",
    "METALLIC_MATERIAL_STYLE",
    "EMISSIVE_MATERIAL_STYLE",
    "PORCELAIN_MATERIAL_STYLE",
    "SOLID_GLASS_MATERIAL_STYLE",
    "STUDIO_PRODUCT_GLOSS_MATERIAL_STYLE",
    "JADE_MATERIAL_STYLE",
    "PEARL_MATERIAL_STYLE",
    "HOLOGRAPHIC_MATERIAL_STYLE",
    "WARM_CLAY_MATERIAL_STYLE",
    "MATERIAL_STYLE_LIBRARY",
    "build_handdrawn_material_style",
    "RadiusStyle",
    "ATOMIC_RADIUS_STYLE",
    "IONIC_RADIUS_STYLE",
    "VDW_RADIUS_STYLE",
    "COVALENT_RADIUS_STYLE",
    "RADIUS_STYLE_LIBRARY",
    "OutlineRoleStyle",
    "OutlineStyle",
    "SceneStyle",
    "tune_rgb",
    "tune_rgba",
    "ResolvedStyleBundle",
    "resolve_style_bundle",
    "get_scene_style",
    "get_color_style",
    "get_material_style",
    "get_radius_style",
    "get_light_style_name",
    "is_handdrawn_style",
    "normalize_style_name",
    "scene_style_choices",
    "color_style_choices",
    "material_style_choices",
    "radius_style_choices",
    "light_style_choices",
    "style_choices",
]
