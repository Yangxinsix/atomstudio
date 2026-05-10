from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atomstudio.style.color_style import ColorStyle, JMOL_COLOR_STYLE, JMOL_SOFT_COLOR_STYLE
from atomstudio.style.material_style import (
    CLEAN_MATERIAL_STYLE,
    HANDDRAWN_MATERIAL_STYLE,
    HANDDRAWN_V2_MATERIAL_STYLE,
    MaterialStyle,
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
    render_tokens: dict[str, Any] = field(default_factory=dict)
    structure_tokens: dict[str, Any] = field(default_factory=dict)





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
    light_style="three_point",
    background=(0.07, 0.08, 0.10, 1.0),
    outline=DEFAULT_OUTLINE_STYLE.copy(),
    camera_tokens={"projection": "ORTHOGRAPHIC", "fit_mode": "ase_gui", "fit_padding": 0.12, "lens_mm": 80.0},
    render_tokens={"color_management": {"view_transform": "Standard", "exposure": -0.15}},
    structure_tokens={"representation": "ball_stick"},
)


MONOCHROME_SCENE_STYLE = SceneStyle(
    name="monochrome",
    color_style=JMOL_COLOR_STYLE,
    material_style=CLEAN_MATERIAL_STYLE,
    light_style="three_point",
    background=(1.0, 1.0, 1.0, 1.0),
    outline=DEFAULT_OUTLINE_STYLE.copy(),
    camera_tokens={"projection": "ORTHOGRAPHIC", "fit_mode": "ase_gui", "fit_padding": 0.10, "lens_mm": 80.0},
    render_tokens={"color_management": {"view_transform": "Standard", "exposure": -0.10}},
    structure_tokens={"representation": "ball_stick"},
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
