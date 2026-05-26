from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atomstudio.style.color_style import (
    ColorStyle,
    JMOL_COLOR_STYLE,
    JMOL_SOFT_COLOR_STYLE,
    MONO_GRAY_COLOR_STYLE,
    STUDIO_HIGHKEY_COLOR_STYLE,
    STUDIO_MONO_COLOR_STYLE,
    STUDIO_PRODUCT_COLOR_STYLE,
    STUDIO_SOFT_COLOR_STYLE,
)
from atomstudio.style.material_style import (
    CERAMIC_MATERIAL_STYLE,
    CLEAN_MATERIAL_STYLE,
    CLEAN_GLOSSY_MATERIAL_STYLE,
    GLASS_MATERIAL_STYLE,
    HANDDRAWN_MATERIAL_STYLE,
    HANDDRAWN_V2_MATERIAL_STYLE,
    HOLOGRAPHIC_MATERIAL_STYLE,
    JADE_MATERIAL_STYLE,
    MaterialStyle,
    PEARL_MATERIAL_STYLE,
    PORCELAIN_MATERIAL_STYLE,
    SOLID_GLASS_MATERIAL_STYLE,
    STUDIO_MACRO_GLOSS_MATERIAL_STYLE,
    STUDIO_PEARL_MATERIAL_STYLE,
    STUDIO_PRODUCT_GLOSS_MATERIAL_STYLE,
    STUDIO_SATIN_MATERIAL_STYLE,
    STUDIO_SOFTMETAL_MATERIAL_STYLE,
    STUDIO_WHITE_CERAMIC_MATERIAL_STYLE,
    WARM_CLAY_MATERIAL_STYLE,
)
from atomstudio.style.outline_style import DEFAULT_OUTLINE_STYLE, HANDDRAWN_OUTLINE_STYLE, OutlineStyle


@dataclass
class SceneStyle:
    name: str
    color_style: ColorStyle
    material_style: MaterialStyle
    light_style: str
    background: tuple[float, float, float, float]
    outline: OutlineStyle = field(default_factory=OutlineStyle)
    camera_tokens: dict[str, Any] = field(default_factory=dict)
    lighting_tokens: dict[str, Any] = field(default_factory=dict)
    render_tokens: dict[str, Any] = field(default_factory=dict)
    structure_tokens: dict[str, Any] = field(default_factory=dict)





_STYLE_SPHERE_RENDER_TOKENS = {
    "color_management": {
        "view_transform": "Filmic",
        "look": "Medium High Contrast",
        "exposure": 0.0,
        "gamma": 1.0,
    }
}

_STUDIO_RENDER_TOKENS = {
    "engine": "cycles",
    "samples": 192,
    "transparent_bg": False,
    "color_management": {
        "view_transform": "Standard",
        "look": "None",
        "exposure": 0.0,
        "gamma": 1.0,
    },
}

_STUDIO_CAMERA_TOKENS = {
    "projection": "PERSP",
    "fit_mode": "ase_gui",
    "fit_padding": 0.16,
    "lens_mm": 95.0,
    "frame_scale": 1.08,
    "dof_enabled": False,
    "aperture_fstop": 11.0,
}

_STUDIO_STRUCTURE_TOKENS = {
    "representation": "ball_stick",
    "draw_cell": False,
    "cell_style": {"show": False},
    "bonding": {"hbond": {"enabled": False}},
    "bond_radius": 0.045,
    "sphere_segments": 48,
    "sphere_rings": 24,
    "bond_vertices": 32,
}


def _studio_lighting_tokens(
    *,
    color: tuple[float, float, float, float] = (0.72, 0.745, 0.755, 1.0),
    roughness: float = 0.56,
    specular: float = 0.12,
    coat: float = 0.08,
    coat_roughness: float = 0.28,
    size_scale: float = 3.6,
    z_offset_scale: float = 0.035,
) -> dict[str, Any]:
    return {
        "ground": {
            "enabled": True,
            "mode": "visible",
            "size_scale": size_scale,
            "z_offset_scale": z_offset_scale,
            "color": color,
            "roughness": roughness,
            "specular": specular,
            "metallic": 0.0,
            "coat": coat,
            "coat_roughness": coat_roughness,
        }
    }


def _studio_highkey_lighting_tokens() -> dict[str, Any]:
    return _studio_sweep_lighting_tokens(
        color=(0.70, 0.725, 0.735, 1.0),
        roughness=0.64,
        specular=0.08,
        coat=0.04,
        coat_roughness=0.36,
    )


def _studio_sweep_lighting_tokens(
    *,
    color: tuple[float, float, float, float],
    roughness: float,
    specular: float,
    coat: float,
    coat_roughness: float,
    width_scale: float = 9.0,
    width_segments: int = 32,
    floor_depth_scale: float = 7.0,
    wall_height_scale: float = 8.0,
    radius_scale: float = 1.6,
    floor_offset_scale: float = 0.020,
    wall_offset_scale: float = 0.35,
    segments: int = 40,
    gradient_enabled: bool = False,
    bottom_color: tuple[float, float, float, float] | None = None,
    top_color: tuple[float, float, float, float] | None = None,
    spot_color: tuple[float, float, float, float] | None = None,
    spot_strength: float = 0.0,
    spot_x: float = 0.50,
    spot_y: float = 0.72,
    spot_radius: float = 0.32,
    vignette_strength: float = 0.0,
) -> dict[str, Any]:
    return {
        "ground": {"enabled": False},
        "sweep": {
            "enabled": True,
            "width_scale": width_scale,
            "width_segments": width_segments,
            "floor_depth_scale": floor_depth_scale,
            "wall_height_scale": wall_height_scale,
            "radius_scale": radius_scale,
            "floor_offset_scale": floor_offset_scale,
            "wall_offset_scale": wall_offset_scale,
            "segments": segments,
            "color": color,
            "roughness": roughness,
            "specular": specular,
            "metallic": 0.0,
            "coat": coat,
            "coat_roughness": coat_roughness,
            "gradient_enabled": gradient_enabled,
            "bottom_color": bottom_color or color,
            "top_color": top_color or color,
            "spot_color": spot_color or (1.0, 1.0, 1.0, 1.0),
            "spot_strength": spot_strength,
            "spot_x": spot_x,
            "spot_y": spot_y,
            "spot_radius": spot_radius,
            "vignette_strength": vignette_strength,
        },
    }


def _studio_effects_tokens(
    *,
    ao_factor: float,
    ao_distance: float,
    bloom_intensity: float,
    atmosphere_density: float = 0.0,
    atmosphere_color: tuple[float, float, float, float] = (0.82, 0.88, 0.94, 1.0),
    vignette_intensity: float = 0.0,
) -> dict[str, Any]:
    return {
        "effects": {
            "ambient_occlusion": {"enabled": True, "factor": ao_factor, "distance": ao_distance},
            "bloom": {"enabled": bloom_intensity > 0.0, "threshold": 0.90, "intensity": bloom_intensity, "size": 5},
            "atmosphere": {"enabled": atmosphere_density > 0.0, "density": atmosphere_density, "color": atmosphere_color},
            "vignette": {"enabled": vignette_intensity > 0.0, "intensity": vignette_intensity, "softness": 0.62},
        }
    }


def _studio_scene(
    name: str,
    *,
    color_style: ColorStyle,
    material_style: MaterialStyle,
    light_style: str,
    background: tuple[float, float, float, float],
    render_tokens: dict[str, Any] | None = None,
    camera_tokens: dict[str, Any] | None = None,
    lighting_tokens: dict[str, Any] | None = None,
    structure_tokens: dict[str, Any] | None = None,
) -> SceneStyle:
    return SceneStyle(
        name=name,
        color_style=color_style,
        material_style=material_style,
        light_style=light_style,
        background=background,
        outline=DEFAULT_OUTLINE_STYLE.copy(),
        camera_tokens={**_STUDIO_CAMERA_TOKENS, **(camera_tokens or {})},
        lighting_tokens=lighting_tokens or _studio_lighting_tokens(),
        render_tokens={**_STUDIO_RENDER_TOKENS, **(render_tokens or {})},
        structure_tokens={**_STUDIO_STRUCTURE_TOKENS, **(structure_tokens or {})},
    )


def _style_sphere_scene(name: str, material_style: MaterialStyle) -> SceneStyle:
    return SceneStyle(
        name=name,
        color_style=JMOL_SOFT_COLOR_STYLE,
        material_style=material_style,
        light_style="style_sphere_showcase",
        background=(0.045, 0.052, 0.062, 1.0),
        outline=DEFAULT_OUTLINE_STYLE.copy(),
        camera_tokens={"projection": "ORTHOGRAPHIC", "fit_mode": "ase_gui", "fit_padding": 0.12, "lens_mm": 80.0},
        render_tokens=dict(_STYLE_SPHERE_RENDER_TOKENS),
        structure_tokens={"representation": "ball_stick"},
    )


DEFAULT_SCENE_STYLE = SceneStyle(
    name="default",
    color_style=JMOL_SOFT_COLOR_STYLE,
    material_style=CLEAN_MATERIAL_STYLE,
    light_style="homogeneous",
    background=(1.0, 1.0, 1.0, 1.0),
    outline=DEFAULT_OUTLINE_STYLE.copy(),
    camera_tokens={"projection": "ORTHOGRAPHIC", "fit_mode": "ase_gui", "fit_padding": 0.12, "lens_mm": 80.0},
    render_tokens={"color_management": {"view_transform": "Standard", "exposure": -0.15}},
    structure_tokens={"representation": "ball_stick"},
)

DARKLAB_SCENE_STYLE = SceneStyle(
    name="darklab",
    color_style=JMOL_SOFT_COLOR_STYLE,
    material_style=CLEAN_MATERIAL_STYLE,
    light_style="darklab_rim",
    background=(0.07, 0.08, 0.10, 1.0),
    outline=DEFAULT_OUTLINE_STYLE.copy(),
    camera_tokens={"projection": "ORTHOGRAPHIC", "fit_mode": "ase_gui", "fit_padding": 0.12, "lens_mm": 80.0},
    render_tokens={"color_management": {"view_transform": "Standard", "exposure": -0.15}},
    structure_tokens={"representation": "ball_stick"},
)


MONOCHROME_SCENE_STYLE = SceneStyle(
    name="monochrome",
    color_style=MONO_GRAY_COLOR_STYLE,
    material_style=CLEAN_MATERIAL_STYLE,
    light_style="monochrome_softbox",
    background=(1.0, 1.0, 1.0, 1.0),
    outline=DEFAULT_OUTLINE_STYLE.copy(),
    camera_tokens={"projection": "ORTHOGRAPHIC", "fit_mode": "ase_gui", "fit_padding": 0.10, "lens_mm": 80.0},
    render_tokens={"color_management": {"view_transform": "Standard", "exposure": -0.10}},
    structure_tokens={"representation": "ball_stick"},
)


CERAMIC_STUDIO_SCENE_STYLE = SceneStyle(
    name="ceramic_studio",
    color_style=JMOL_SOFT_COLOR_STYLE,
    material_style=CERAMIC_MATERIAL_STYLE,
    light_style="ceramic_softbox",
    background=(0.93, 0.94, 0.92, 1.0),
    outline=DEFAULT_OUTLINE_STYLE.copy(),
    camera_tokens={"projection": "ORTHOGRAPHIC", "fit_mode": "ase_gui", "fit_padding": 0.11, "lens_mm": 80.0},
    render_tokens={"color_management": {"view_transform": "Standard", "exposure": -0.04}},
    structure_tokens={"representation": "ball_stick"},
)


GLASS_LAB_SCENE_STYLE = SceneStyle(
    name="glass_lab",
    color_style=JMOL_SOFT_COLOR_STYLE,
    material_style=GLASS_MATERIAL_STYLE,
    light_style="glass_lab_rim",
    background=(0.025, 0.035, 0.055, 1.0),
    outline=DEFAULT_OUTLINE_STYLE.copy(),
    camera_tokens={"projection": "ORTHOGRAPHIC", "fit_mode": "ase_gui", "fit_padding": 0.12, "lens_mm": 80.0},
    render_tokens={"color_management": {"view_transform": "Standard", "exposure": -0.12}},
    structure_tokens={"representation": "ball_stick"},
)
CLEAN_GLOSSY_SCENE_STYLE = _style_sphere_scene("clean_glossy", CLEAN_GLOSSY_MATERIAL_STYLE)
PORCELAIN_SCENE_STYLE = _style_sphere_scene("porcelain", PORCELAIN_MATERIAL_STYLE)
SOLID_GLASS_SCENE_STYLE = _style_sphere_scene("solid_glass", SOLID_GLASS_MATERIAL_STYLE)
JADE_SCENE_STYLE = _style_sphere_scene("jade", JADE_MATERIAL_STYLE)
PEARL_SCENE_STYLE = _style_sphere_scene("pearl", PEARL_MATERIAL_STYLE)
HOLOGRAPHIC_SCENE_STYLE = _style_sphere_scene("holographic", HOLOGRAPHIC_MATERIAL_STYLE)
WARM_CLAY_SCENE_STYLE = _style_sphere_scene("warm_clay", WARM_CLAY_MATERIAL_STYLE)


STUDIO_HIGHKEY_SCENE_STYLE = _studio_scene(
    "studio_highkey",
    color_style=STUDIO_HIGHKEY_COLOR_STYLE,
    material_style=STUDIO_SATIN_MATERIAL_STYLE,
    light_style="studio_highkey_softbox",
    background=(0.79, 0.815, 0.835, 1.0),
    lighting_tokens=_studio_highkey_lighting_tokens(),
    structure_tokens={"bond_radius": 0.0675},
)

STUDIO_HIGHKEY_CLEAN_SCENE_STYLE = _studio_scene(
    "studio_highkey_clean",
    color_style=STUDIO_HIGHKEY_COLOR_STYLE,
    material_style=STUDIO_SATIN_MATERIAL_STYLE,
    light_style="studio_highkey_softbox",
    background=(0.79, 0.815, 0.835, 1.0),
    lighting_tokens=_studio_highkey_lighting_tokens(),
    structure_tokens={"bond_radius": 0.0675},
)

STUDIO_DARKLAB_SCENE_STYLE = _studio_scene(
    "studio_darklab",
    color_style=JMOL_SOFT_COLOR_STYLE,
    material_style=SOLID_GLASS_MATERIAL_STYLE,
    light_style="style_sphere_showcase",
    background=(0.045, 0.052, 0.062, 1.0),
    camera_tokens={"projection": "ORTHOGRAPHIC", "fit_padding": 0.12, "lens_mm": 80.0, "frame_scale": 1.0},
    lighting_tokens=_studio_sweep_lighting_tokens(
        color=(0.055, 0.063, 0.075, 1.0),
        roughness=0.78,
        specular=0.08,
        coat=0.0,
        coat_roughness=0.15,
        width_scale=9.0,
        floor_depth_scale=10.0,
        wall_height_scale=7.0,
        radius_scale=1.35,
        floor_offset_scale=0.0,
        wall_offset_scale=0.40,
        segments=40,
    ),
    render_tokens={
        "samples": 192,
        "color_management": {
            "view_transform": "Filmic",
            "look": "Medium High Contrast",
            "exposure": 0.0,
            "gamma": 1.0,
        },
        **_studio_effects_tokens(
            ao_factor=0.72,
            ao_distance=2.8,
            bloom_intensity=0.0,
            vignette_intensity=0.08,
        ),
    },
    structure_tokens={"bond_radius": 0.0675},
)

STUDIO_WARM_SOFT_SCENE_STYLE = _studio_scene(
    "studio_warm_soft",
    color_style=STUDIO_PRODUCT_COLOR_STYLE,
    material_style=STUDIO_PRODUCT_GLOSS_MATERIAL_STYLE,
    light_style="studio_warm_softbox",
    background=(0.70, 0.625, 0.555, 1.0),
    camera_tokens={"lens_mm": 105.0, "frame_scale": 0.92, "aperture_fstop": 12.0},
    lighting_tokens=_studio_sweep_lighting_tokens(
        color=(0.68, 0.600, 0.525, 1.0),
        roughness=0.58,
        specular=0.14,
        coat=0.07,
        coat_roughness=0.28,
        floor_offset_scale=0.0,
        floor_depth_scale=24.0,
        wall_height_scale=8.0,
        radius_scale=1.55,
        gradient_enabled=True,
        bottom_color=(0.62, 0.535, 0.455, 1.0),
        top_color=(0.82, 0.745, 0.665, 1.0),
        spot_color=(1.0, 0.88, 0.64, 1.0),
        spot_strength=0.46,
        spot_x=0.66,
        spot_y=0.73,
        spot_radius=0.34,
        vignette_strength=0.22,
    ),
    render_tokens={
        "samples": 192,
        "color_management": {"view_transform": "Standard", "look": "Medium High Contrast", "exposure": -0.50, "gamma": 1.0},
        **_studio_effects_tokens(
            ao_factor=0.66,
            ao_distance=2.4,
            bloom_intensity=0.0,
            atmosphere_density=0.0,
            atmosphere_color=(1.0, 0.90, 0.78, 1.0),
        ),
    },
    structure_tokens={"bond_radius": 0.0675},
)

STUDIO_COOL_RIM_SCENE_STYLE = _studio_scene(
    "studio_cool_rim",
    color_style=STUDIO_PRODUCT_COLOR_STYLE,
    material_style=STUDIO_PRODUCT_GLOSS_MATERIAL_STYLE,
    light_style="studio_cool_rim_softbox",
    background=(0.42, 0.515, 0.620, 1.0),
    camera_tokens={"lens_mm": 110.0, "frame_scale": 0.92, "aperture_fstop": 11.0},
    lighting_tokens=_studio_sweep_lighting_tokens(
        color=(0.40, 0.500, 0.600, 1.0),
        roughness=0.56,
        specular=0.16,
        coat=0.08,
        coat_roughness=0.28,
        floor_offset_scale=0.0,
        floor_depth_scale=24.0,
        wall_height_scale=8.5,
        radius_scale=1.50,
        gradient_enabled=True,
        bottom_color=(0.30, 0.405, 0.520, 1.0),
        top_color=(0.58, 0.695, 0.805, 1.0),
        spot_color=(0.82, 0.93, 1.0, 1.0),
        spot_strength=0.38,
        spot_x=0.58,
        spot_y=0.76,
        spot_radius=0.40,
        vignette_strength=0.28,
    ),
    render_tokens={
        "samples": 192,
        "color_management": {"view_transform": "Standard", "look": "High Contrast", "exposure": -0.30, "gamma": 1.0},
        **_studio_effects_tokens(
            ao_factor=0.84,
            ao_distance=2.8,
            bloom_intensity=0.0,
            atmosphere_density=0.0,
            atmosphere_color=(0.70, 0.82, 1.0, 1.0),
            vignette_intensity=0.10,
        ),
    },
    structure_tokens={"bond_radius": 0.0675},
)

STUDIO_MACRO_SCENE_STYLE = _studio_scene(
    "studio_macro",
    color_style=STUDIO_SOFT_COLOR_STYLE,
    material_style=STUDIO_MACRO_GLOSS_MATERIAL_STYLE,
    light_style="studio_macro_softbox",
    background=(0.78, 0.805, 0.825, 1.0),
    camera_tokens={"fit_padding": 0.11, "lens_mm": 115.0, "frame_scale": 0.96, "aperture_fstop": 9.0},
    lighting_tokens=_studio_lighting_tokens(color=(0.70, 0.725, 0.735, 1.0), roughness=0.50, specular=0.16, coat=0.10, size_scale=3.2),
)

STUDIO_CRYSTAL_PRODUCT_SCENE_STYLE = _studio_scene(
    "studio_crystal_product",
    color_style=STUDIO_MONO_COLOR_STYLE,
    material_style=STUDIO_PEARL_MATERIAL_STYLE,
    light_style="studio_crystal_tabletop",
    background=(0.77, 0.795, 0.815, 1.0),
    camera_tokens={"fit_padding": 0.12, "lens_mm": 110.0, "frame_scale": 1.02, "aperture_fstop": 10.0},
    lighting_tokens=_studio_lighting_tokens(color=(0.69, 0.715, 0.725, 1.0), roughness=0.48, specular=0.18, coat=0.12, size_scale=3.8),
)

STUDIO_PEARL_SCENE_STYLE = _studio_scene(
    "studio_pearl",
    color_style=STUDIO_MONO_COLOR_STYLE,
    material_style=STUDIO_WHITE_CERAMIC_MATERIAL_STYLE,
    light_style="studio_highkey_softbox",
    background=(0.80, 0.805, 0.805, 1.0),
    camera_tokens={"lens_mm": 100.0, "frame_scale": 1.05, "aperture_fstop": 11.0},
    lighting_tokens=_studio_lighting_tokens(color=(0.73, 0.73, 0.715, 1.0), roughness=0.58, specular=0.11, coat=0.06),
)

STUDIO_SOFTMETAL_SCENE_STYLE = _studio_scene(
    "studio_softmetal",
    color_style=STUDIO_SOFT_COLOR_STYLE,
    material_style=STUDIO_SOFTMETAL_MATERIAL_STYLE,
    light_style="studio_softmetal_strip",
    background=(0.76, 0.790, 0.820, 1.0),
    camera_tokens={"lens_mm": 105.0, "frame_scale": 1.05, "aperture_fstop": 10.0},
    lighting_tokens=_studio_lighting_tokens(color=(0.68, 0.705, 0.720, 1.0), roughness=0.50, specular=0.18, coat=0.10, size_scale=3.5),
)


HANDDRAWN_SCENE_STYLE = SceneStyle(
    name="handdrawn",
    color_style=JMOL_COLOR_STYLE,
    material_style=HANDDRAWN_MATERIAL_STYLE,
    light_style="handdrawn_soft",
    background=(1.0, 1.0, 1.0, 1.0),
    outline=HANDDRAWN_OUTLINE_STYLE.copy(),
    camera_tokens={"projection": "ORTHOGRAPHIC", "fit_mode": "ase_gui", "fit_padding": 0.11, "lens_mm": 80.0},
    render_tokens={"color_management": {"view_transform": "Standard", "exposure": 0.0}},
    structure_tokens={
        "representation": "space_filling",
        "draw_surface_bonds": False,
    },
)


HANDDRAWN_V2_SCENE_STYLE = SceneStyle(
    name="handdrawn_v2",
    color_style=JMOL_COLOR_STYLE,
    material_style=HANDDRAWN_V2_MATERIAL_STYLE,
    light_style="handdrawn_soft_spot",
    background=(1.0, 1.0, 1.0, 1.0),
    outline=HANDDRAWN_OUTLINE_STYLE.copy(),
    camera_tokens={"projection": "ORTHOGRAPHIC", "fit_mode": "ase_gui", "fit_padding": 0.10, "lens_mm": 80.0},
    render_tokens={"color_management": {"view_transform": "Standard", "exposure": 0.02}},
    structure_tokens={
        "representation": "space_filling",
        "draw_surface_bonds": False,
    },
)
