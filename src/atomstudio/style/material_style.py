from __future__ import annotations

from dataclasses import dataclass, field, replace

from atomstudio.color_utils import clamp01
from atomstudio.scene.materials.specs import (
    HandDrawnMaterialSpec,
    MaterialLike,
    MaterialSpec,
    as_handdrawn_spec,
    as_material_spec,
)


@dataclass
class MaterialStyle:
    name: str
    pipeline: str = "principled"
    atom_default: MaterialLike = field(default_factory=MaterialSpec)
    bond_default: MaterialLike = field(default_factory=MaterialSpec)
    polyhedra_default: MaterialLike = field(default_factory=MaterialSpec)
    cell_default: MaterialLike = field(default_factory=MaterialSpec)
    handdrawn_spec: HandDrawnMaterialSpec | None = None

    def __post_init__(self) -> None:
        pipeline = str(self.pipeline).strip().lower()
        if pipeline == "handdrawn":
            base = HandDrawnMaterialSpec() if self.handdrawn_spec is None else replace(self.handdrawn_spec)
            self.handdrawn_spec = base
            self.atom_default = as_handdrawn_spec(self.atom_default, fallback=base)
            self.bond_default = as_handdrawn_spec(self.bond_default, fallback=base)
            self.polyhedra_default = as_handdrawn_spec(self.polyhedra_default, fallback=base)
            self.cell_default = as_handdrawn_spec(self.cell_default, fallback=base)
            return

        self.atom_default = as_material_spec(self.atom_default)
        self.bond_default = as_material_spec(self.bond_default)
        self.polyhedra_default = as_material_spec(self.polyhedra_default)
        self.cell_default = as_material_spec(self.cell_default)
        self.handdrawn_spec = replace(self.handdrawn_spec) if self.handdrawn_spec is not None else None

    def atom_for(self, _: str) -> MaterialLike:
        return self.atom_default


_UNRESOLVED_COLOR = (0.6, 0.6, 0.6, 1.0)


def _style_with_uniform_elements(
    name: str,
    atom_default: MaterialLike,
    bond_default: MaterialLike,
    polyhedra_default: MaterialLike,
    cell_default: MaterialLike,
    *,
    pipeline: str = "principled",
    handdrawn_spec: HandDrawnMaterialSpec | None = None,
) -> MaterialStyle:
    return MaterialStyle(
        name=name,
        pipeline=pipeline,
        atom_default=atom_default,
        bond_default=bond_default,
        polyhedra_default=polyhedra_default,
        cell_default=cell_default,
        handdrawn_spec=handdrawn_spec,
    )


def _mix(a: float, b: float, t: float) -> float:
    x = clamp01(t)
    return (1.0 - x) * float(a) + x * float(b)


def tune_rgb(
    rgb: tuple[float, float, float],
    desaturate: float = 0.0,
    lighten: float = 0.0,
) -> tuple[float, float, float]:
    r, g, b = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    gray = (r + g + b) / 3.0
    d = clamp01(desaturate)
    r = _mix(r, gray, d)
    g = _mix(g, gray, d)
    b = _mix(b, gray, d)
    l = clamp01(lighten)
    r = _mix(r, 1.0, l)
    g = _mix(g, 1.0, l)
    b = _mix(b, 1.0, l)
    return (clamp01(r), clamp01(g), clamp01(b))


def tune_rgba(
    rgba: tuple[float, float, float, float],
    desaturate: float = 0.0,
    lighten: float = 0.0,
) -> tuple[float, float, float, float]:
    rgb = tune_rgb((rgba[0], rgba[1], rgba[2]), desaturate=desaturate, lighten=lighten)
    return (rgb[0], rgb[1], rgb[2], float(rgba[3]))


def scale_rgba(
    rgba: tuple[float, float, float, float],
    factor: float,
    *,
    alpha: float | None = None,
) -> tuple[float, float, float, float]:
    f = max(0.0, float(factor))
    a = float(rgba[3]) if alpha is None else clamp01(alpha)
    return (clamp01(rgba[0] * f), clamp01(rgba[1] * f), clamp01(rgba[2] * f), a)


def _candidate_material_style(
    name: str,
    atom: MaterialSpec,
    *,
    polyhedra_alpha: float = 0.32,
    cell_factor: float = 0.55,
) -> MaterialStyle:
    polyhedra = replace(
        atom,
        color=(0.56, 0.62, 0.70, polyhedra_alpha),
        alpha=polyhedra_alpha,
    )
    cell = replace(atom, color=scale_rgba(_UNRESOLVED_COLOR, cell_factor), alpha=1.0)
    return _style_with_uniform_elements(
        name,
        replace(atom),
        replace(atom),
        polyhedra,
        cell,
    )


CLEAN_MATERIAL_STYLE = _style_with_uniform_elements(
    "clean",
    MaterialSpec(color=_UNRESOLVED_COLOR, roughness=0.32, specular=0.35, metallic=0.05, ior=1.45),
    MaterialSpec(color=(0.25, 0.25, 0.25, 1.0), roughness=0.38, specular=0.28, metallic=0.02, ior=1.45),
    MaterialSpec(color=(0.38, 0.42, 0.48, 0.35), roughness=0.30, specular=0.24, metallic=0.0, ior=1.45, alpha=0.35),
    MaterialSpec(color=(0.22, 0.22, 0.22, 1.0), roughness=0.42, specular=0.24, metallic=0.02, ior=1.45),
)

GLASS_MATERIAL_STYLE = _style_with_uniform_elements(
    "glass",
    MaterialSpec(
        color=_UNRESOLVED_COLOR,
        roughness=0.045,
        specular=0.88,
        metallic=0.0,
        ior=1.52,
        transmission=0.0,
        coat=0.72,
        coat_roughness=0.035,
        alpha=1.0,
    ),
    MaterialSpec(
        color=(0.62, 0.70, 0.82, 1.0),
        roughness=0.075,
        specular=0.56,
        metallic=0.0,
        ior=1.52,
        transmission=0.0,
        coat=0.28,
        coat_roughness=0.060,
        alpha=1.0,
    ),
    MaterialSpec(
        color=(0.72, 0.80, 0.92, 0.35),
        roughness=0.055,
        specular=0.72,
        metallic=0.0,
        ior=1.52,
        transmission=0.20,
        alpha=0.35,
    ),
    MaterialSpec(
        color=(0.42, 0.48, 0.58, 1.0),
        roughness=0.095,
        specular=0.42,
        metallic=0.0,
        ior=1.52,
        transmission=0.0,
        coat=0.18,
        coat_roughness=0.080,
        alpha=1.0,
    ),
)

CERAMIC_MATERIAL_STYLE = _style_with_uniform_elements(
    "ceramic",
    MaterialSpec(color=_UNRESOLVED_COLOR, roughness=0.86, specular=0.045, metallic=0.0, ior=1.38),
    MaterialSpec(color=(0.24, 0.24, 0.24, 1.0), roughness=0.88, specular=0.035, metallic=0.0, ior=1.38),
    MaterialSpec(color=(0.44, 0.48, 0.54, 0.32), roughness=0.84, specular=0.030, metallic=0.0, ior=1.38, alpha=0.32),
    MaterialSpec(color=(0.24, 0.24, 0.24, 1.0), roughness=0.90, specular=0.030, metallic=0.0, ior=1.38),
)

METALLIC_MATERIAL_STYLE = _style_with_uniform_elements(
    "metallic",
    MaterialSpec(color=_UNRESOLVED_COLOR, roughness=0.36, specular=0.42, metallic=0.68),
    MaterialSpec(color=(0.46, 0.46, 0.48, 1.0), roughness=0.44, specular=0.30, metallic=0.40),
    MaterialSpec(color=(0.60, 0.64, 0.70, 0.35), roughness=0.36, specular=0.30, metallic=0.25, alpha=0.35),
    MaterialSpec(color=(0.42, 0.42, 0.44, 1.0), roughness=0.46, specular=0.28, metallic=0.45),
)

EMISSIVE_MATERIAL_STYLE = _style_with_uniform_elements(
    "emissive",
    MaterialSpec(color=_UNRESOLVED_COLOR, roughness=0.14, specular=0.72, metallic=0.08, coat=0.12, coat_roughness=0.18),
    MaterialSpec(color=(0.36, 0.36, 0.40, 1.0), roughness=0.24, specular=0.52, metallic=0.05, coat=0.08, coat_roughness=0.20),
    MaterialSpec(color=(0.55, 0.62, 0.72, 0.35), roughness=0.20, specular=0.46, metallic=0.02, coat=0.06, coat_roughness=0.20, alpha=0.35),
    MaterialSpec(color=(0.34, 0.34, 0.36, 1.0), roughness=0.30, specular=0.45, metallic=0.04, coat=0.06, coat_roughness=0.24),
)


CLEAN_GLOSSY_MATERIAL_STYLE = _candidate_material_style(
    "clean_glossy",
    MaterialSpec(
        color=_UNRESOLVED_COLOR,
        roughness=0.16,
        specular=0.72,
        metallic=0.0,
        ior=1.45,
        coat=0.28,
        coat_roughness=0.08,
    ),
)

PORCELAIN_MATERIAL_STYLE = _candidate_material_style(
    "porcelain",
    MaterialSpec(
        color=_UNRESOLVED_COLOR,
        roughness=0.58,
        specular=0.20,
        metallic=0.0,
        ior=1.45,
        coat=0.24,
        coat_roughness=0.22,
    ),
)

SOLID_GLASS_MATERIAL_STYLE = _candidate_material_style(
    "solid_glass",
    MaterialSpec(
        color=_UNRESOLVED_COLOR,
        roughness=0.045,
        specular=0.88,
        metallic=0.0,
        ior=1.52,
        coat=0.72,
        coat_roughness=0.035,
    ),
)

JADE_MATERIAL_STYLE = _candidate_material_style(
    "jade",
    MaterialSpec(
        color=_UNRESOLVED_COLOR,
        roughness=0.20,
        specular=0.55,
        metallic=0.0,
        ior=1.45,
        subsurface=0.28,
        coat=0.38,
        coat_roughness=0.075,
    ),
)

PEARL_MATERIAL_STYLE = _candidate_material_style(
    "pearl",
    MaterialSpec(
        color=_UNRESOLVED_COLOR,
        roughness=0.18,
        specular=0.80,
        metallic=0.0,
        ior=1.45,
        sheen=0.45,
        subsurface=0.12,
        coat=0.80,
        coat_roughness=0.055,
    ),
)

HOLOGRAPHIC_MATERIAL_STYLE = _candidate_material_style(
    "holographic",
    MaterialSpec(
        color=_UNRESOLVED_COLOR,
        roughness=0.12,
        specular=0.90,
        metallic=0.0,
        ior=1.45,
        coat=0.90,
        coat_roughness=0.025,
        emission_strength=0.035,
    ),
)

WARM_CLAY_MATERIAL_STYLE = _candidate_material_style(
    "warm_clay",
    MaterialSpec(
        color=_UNRESOLVED_COLOR,
        roughness=0.95,
        specular=0.025,
        metallic=0.0,
        ior=1.45,
    ),
)

STUDIO_SATIN_MATERIAL_STYLE = _style_with_uniform_elements(
    "studio_satin",
    MaterialSpec(color=_UNRESOLVED_COLOR, roughness=0.22, specular=0.66, metallic=0.0, ior=1.46, coat=0.58, coat_roughness=0.075),
    MaterialSpec(color=(0.34, 0.38, 0.40, 1.0), roughness=0.30, specular=0.44, metallic=0.10, ior=1.46, coat=0.22, coat_roughness=0.12),
    MaterialSpec(color=(0.48, 0.56, 0.62, 0.28), roughness=0.32, specular=0.36, metallic=0.0, alpha=0.28),
    MaterialSpec(color=(0.42, 0.44, 0.44, 1.0), roughness=0.46, specular=0.24, metallic=0.0),
)

STUDIO_PEARL_MATERIAL_STYLE = _style_with_uniform_elements(
    "studio_pearl",
    MaterialSpec(color=_UNRESOLVED_COLOR, roughness=0.20, specular=0.76, metallic=0.0, ior=1.45, sheen=0.18, subsurface=0.025, coat=0.70, coat_roughness=0.065),
    MaterialSpec(color=(0.40, 0.42, 0.42, 1.0), roughness=0.32, specular=0.42, metallic=0.04, ior=1.45, coat=0.16, coat_roughness=0.12),
    MaterialSpec(color=(0.58, 0.62, 0.64, 0.26), roughness=0.28, specular=0.38, alpha=0.26),
    MaterialSpec(color=(0.48, 0.48, 0.46, 1.0), roughness=0.44, specular=0.22, metallic=0.0),
)

STUDIO_SOFTMETAL_MATERIAL_STYLE = _style_with_uniform_elements(
    "studio_softmetal",
    MaterialSpec(color=_UNRESOLVED_COLOR, roughness=0.18, specular=0.62, metallic=0.46, ior=1.45, coat=0.42, coat_roughness=0.085),
    MaterialSpec(color=(0.30, 0.34, 0.36, 1.0), roughness=0.30, specular=0.38, metallic=0.30, coat=0.16, coat_roughness=0.16),
    MaterialSpec(color=(0.48, 0.56, 0.64, 0.26), roughness=0.30, specular=0.34, metallic=0.08, alpha=0.26),
    MaterialSpec(color=(0.38, 0.42, 0.44, 1.0), roughness=0.36, specular=0.32, metallic=0.22),
)

STUDIO_WHITE_CERAMIC_MATERIAL_STYLE = _style_with_uniform_elements(
    "studio_white_ceramic",
    MaterialSpec(color=_UNRESOLVED_COLOR, roughness=0.42, specular=0.28, metallic=0.0, ior=1.42, coat=0.24, coat_roughness=0.16),
    MaterialSpec(color=(0.36, 0.38, 0.38, 1.0), roughness=0.50, specular=0.18, metallic=0.0),
    MaterialSpec(color=(0.62, 0.62, 0.60, 0.24), roughness=0.48, specular=0.16, alpha=0.24),
    MaterialSpec(color=(0.54, 0.54, 0.52, 1.0), roughness=0.56, specular=0.12, metallic=0.0),
)

STUDIO_MACRO_GLOSS_MATERIAL_STYLE = _style_with_uniform_elements(
    "studio_macro_gloss",
    MaterialSpec(color=_UNRESOLVED_COLOR, roughness=0.16, specular=0.78, metallic=0.0, ior=1.48, coat=0.76, coat_roughness=0.055),
    MaterialSpec(color=(0.30, 0.34, 0.36, 1.0), roughness=0.24, specular=0.50, metallic=0.08, coat=0.24, coat_roughness=0.08),
    MaterialSpec(color=(0.52, 0.60, 0.66, 0.26), roughness=0.22, specular=0.44, alpha=0.26),
    MaterialSpec(color=(0.44, 0.46, 0.46, 1.0), roughness=0.34, specular=0.30, metallic=0.0),
)

STUDIO_PRODUCT_GLOSS_MATERIAL_STYLE = _style_with_uniform_elements(
    "studio_product_gloss",
    MaterialSpec(
        color=_UNRESOLVED_COLOR,
        roughness=0.24,
        specular=0.52,
        metallic=0.0,
        ior=1.48,
        coat=0.26,
        coat_roughness=0.12,
    ),
    MaterialSpec(
        color=(0.38, 0.42, 0.43, 1.0),
        roughness=0.26,
        specular=0.46,
        metallic=0.05,
        ior=1.46,
        coat=0.18,
        coat_roughness=0.10,
    ),
    MaterialSpec(color=(0.54, 0.62, 0.70, 0.26), roughness=0.20, specular=0.46, alpha=0.26),
    MaterialSpec(color=(0.44, 0.46, 0.46, 1.0), roughness=0.32, specular=0.34, metallic=0.0),
)


HANDDRAWN_PROFILE_DEFAULT = HandDrawnMaterialSpec(
    color=(0.63, 0.68, 0.75, 1.0),
    alpha=1.0,
    roughness=0.90,
    specular=0.015,
    jmol_desaturate=0.10,
    jmol_lighten=0.04,
    light_direction=(0.68, 0.36, 0.62),
    shadow_area=0.34,
    shadow_strength=0.42,
    shadow_softness=0.12,
    highlight_strength=0.16,
    highlight_direction=(0.78, 0.62, 0.0),
    highlight_arc_length=0.22,
    highlight_band_inner=0.56,
    highlight_band_outer=0.90,
    outline_surface=2.0,
    outline_molecule=2.4,
    outline_bond=1.6,
    outline_secondary_thickness=0.8,
    outline_secondary_color=(0.76, 0.82, 0.92, 1.0),
)

HANDDRAWN_PROFILE_V2 = HandDrawnMaterialSpec(
    color=(0.63, 0.68, 0.75, 1.0),
    alpha=1.0,
    roughness=0.86,
    specular=0.03,
    jmol_desaturate=0.10,
    jmol_lighten=0.04,
    light_direction=(0.74, 0.42, 0.52),
    shadow_area=0.52,
    shadow_strength=0.38,
    shadow_softness=0.11,
    highlight_strength=0.55,
    highlight_direction=(0.62, 0.52, 0.52),
    highlight_arc_length=0.08,
    highlight_band_inner=0.42,
    highlight_band_outer=0.62,
    outline_surface=2.0,
    outline_molecule=2.5,
    outline_bond=1.6,
    outline_secondary_thickness=0.8,
    outline_secondary_color=(0.78, 0.84, 0.94, 1.0),
)


def _handdrawn_role_spec(
    profile: HandDrawnMaterialSpec,
    *,
    color: tuple[float, float, float, float],
    alpha: float,
    roughness: float,
    specular: float,
) -> HandDrawnMaterialSpec:
    return HandDrawnMaterialSpec(
        color=color,
        alpha=alpha,
        roughness=roughness,
        specular=specular,
        jmol_desaturate=profile.jmol_desaturate,
        jmol_lighten=profile.jmol_lighten,
        light_direction=profile.light_direction,
        shadow_area=profile.shadow_area,
        shadow_strength=profile.shadow_strength,
        shadow_softness=profile.shadow_softness,
        highlight_strength=profile.highlight_strength,
        highlight_direction=profile.highlight_direction,
        highlight_arc_length=profile.highlight_arc_length,
        highlight_band_inner=profile.highlight_band_inner,
        highlight_band_outer=profile.highlight_band_outer,
        outline_surface=profile.outline_surface,
        outline_molecule=profile.outline_molecule,
        outline_bond=profile.outline_bond,
        outline_secondary_thickness=profile.outline_secondary_thickness,
        outline_secondary_color=profile.outline_secondary_color,
    )


def _default_handdrawn_role_defaults(profile: HandDrawnMaterialSpec) -> dict[str, HandDrawnMaterialSpec]:
    atom_color = (
        float(profile.color[0]),
        float(profile.color[1]),
        float(profile.color[2]),
        float(profile.alpha),
    )
    atom_alpha = float(profile.alpha)
    atom = _handdrawn_role_spec(
        profile,
        color=atom_color,
        alpha=atom_alpha,
        roughness=clamp01(profile.roughness - 0.02),
        specular=clamp01(max(0.03, profile.specular * 2.0)),
    )
    bond = _handdrawn_role_spec(
        profile,
        color=scale_rgba(atom_color, 0.68),
        alpha=atom_alpha,
        roughness=clamp01(profile.roughness + 0.02),
        specular=clamp01(profile.specular * 0.55),
    )
    polyhedra_alpha = min(0.25, atom_alpha)
    polyhedra = _handdrawn_role_spec(
        profile,
        color=scale_rgba(tune_rgba(atom_color, lighten=0.06), 1.0, alpha=polyhedra_alpha),
        alpha=polyhedra_alpha,
        roughness=clamp01(profile.roughness),
        specular=clamp01(max(0.01, profile.specular * 0.66)),
    )
    cell = _handdrawn_role_spec(
        profile,
        color=scale_rgba(atom_color, 0.72),
        alpha=atom_alpha,
        roughness=clamp01(profile.roughness + 0.04),
        specular=clamp01(profile.specular * 0.55),
    )
    return {
        "atom": atom,
        "bond": bond,
        "polyhedra": polyhedra,
        "cell": cell,
    }


HANDDRAWN_ROLE_DEFAULTS = _default_handdrawn_role_defaults(HANDDRAWN_PROFILE_DEFAULT)
HANDDRAWN_V2_ROLE_DEFAULTS = _default_handdrawn_role_defaults(HANDDRAWN_PROFILE_V2)


def make_handdrawn_style(
    *,
    name: str = "handdrawn",
    profile: HandDrawnMaterialSpec | None = None,
    role_defaults: dict[str, HandDrawnMaterialSpec] | None = None,
) -> MaterialStyle:
    active_profile = replace(HANDDRAWN_PROFILE_DEFAULT if profile is None else profile)
    defaults = (
        _default_handdrawn_role_defaults(active_profile)
        if role_defaults is None
        else {str(k): as_handdrawn_spec(v, fallback=active_profile) for k, v in role_defaults.items()}
    )
    return _style_with_uniform_elements(
        str(name),
        defaults["atom"],
        defaults["bond"],
        defaults["polyhedra"],
        defaults["cell"],
        pipeline="handdrawn",
        handdrawn_spec=active_profile,
    )


def build_handdrawn_material_style(spec: HandDrawnMaterialSpec | None = None) -> MaterialStyle:
    return make_handdrawn_style(name="handdrawn", profile=spec)


HANDDRAWN_MATERIAL_STYLE = make_handdrawn_style(
    name="handdrawn",
    profile=HANDDRAWN_PROFILE_DEFAULT,
    role_defaults=HANDDRAWN_ROLE_DEFAULTS,
)

HANDDRAWN_V2_MATERIAL_STYLE = make_handdrawn_style(
    name="handdrawn_v2",
    profile=HANDDRAWN_PROFILE_V2,
    role_defaults=HANDDRAWN_V2_ROLE_DEFAULTS,
)


MATERIAL_STYLE_LIBRARY: dict[str, MaterialStyle] = {
    "clean": CLEAN_MATERIAL_STYLE,
    "clean_glossy": CLEAN_GLOSSY_MATERIAL_STYLE,
    "glass": GLASS_MATERIAL_STYLE,
    "ceramic": CERAMIC_MATERIAL_STYLE,
    "metallic": METALLIC_MATERIAL_STYLE,
    "emissive": EMISSIVE_MATERIAL_STYLE,
    "porcelain": PORCELAIN_MATERIAL_STYLE,
    "solid_glass": SOLID_GLASS_MATERIAL_STYLE,
    "jade": JADE_MATERIAL_STYLE,
    "pearl": PEARL_MATERIAL_STYLE,
    "holographic": HOLOGRAPHIC_MATERIAL_STYLE,
    "warm_clay": WARM_CLAY_MATERIAL_STYLE,
    "studio_satin": STUDIO_SATIN_MATERIAL_STYLE,
    "studio_pearl": STUDIO_PEARL_MATERIAL_STYLE,
    "studio_softmetal": STUDIO_SOFTMETAL_MATERIAL_STYLE,
    "studio_white_ceramic": STUDIO_WHITE_CERAMIC_MATERIAL_STYLE,
    "studio_macro_gloss": STUDIO_MACRO_GLOSS_MATERIAL_STYLE,
    "studio_product_gloss": STUDIO_PRODUCT_GLOSS_MATERIAL_STYLE,
    "handdrawn": HANDDRAWN_MATERIAL_STYLE,
    "handdrawn_v2": HANDDRAWN_V2_MATERIAL_STYLE,
}
