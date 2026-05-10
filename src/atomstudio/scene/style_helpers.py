from __future__ import annotations

from dataclasses import dataclass

from atomstudio.config import (
    AtomStylePresetConfig,
    AtomStyleRuleConfig,
    HanddrawnStyleConfig,
    MaterialPolicy,
    RenderJobConfig,
    StyleConfig,
)
from atomstudio.scene.materials.specs import (
    HandDrawnMaterialSpec,
    MaterialLike,
    MaterialSpec,
    as_handdrawn_spec,
    as_material_spec,
    handdrawn_spec_from_any,
)
from atomstudio.structure.structure import Structure
from atomstudio.style.material_style import MaterialStyle
from atomstudio.style.radius_style import RadiusStyle

DEFAULT_RADII_SCALE = 0.40


@dataclass
class ResolvedAtomStyleState:
    style: str | None
    representation: str
    radius: float | None
    material: MaterialLike | None
    color: tuple[float, float, float, float] | None


def use_atom_matched_split_bonds(material_pipeline: str, policy: MaterialPolicy) -> bool:
    if policy.bond_defaults:
        return False
    if policy.bond_rules:
        return False
    if policy.bond_overrides_by_index:
        return False
    if policy.bond_overrides_by_pair:
        return False
    return True


def material_specs_equivalent(a: MaterialLike, b: MaterialLike, tol: float = 1e-6) -> bool:
    left = as_material_spec(a)
    right = as_material_spec(b)
    if not _material_spec_base_equal(left, right, tol):
        return False
    if isinstance(a, HandDrawnMaterialSpec) or isinstance(b, HandDrawnMaterialSpec):
        return _handdrawn_spec_equal(as_handdrawn_spec(a), as_handdrawn_spec(b), tol)
    return True


def resolve_handdrawn_config(
    material_pipeline: str,
    cfg: RenderJobConfig,
    *,
    material_style: MaterialStyle | None = None,
) -> HanddrawnStyleConfig | None:
    if str(material_pipeline).strip().lower() != "handdrawn":
        return None
    if cfg.style.handdrawn is not None:
        return cfg.style.handdrawn
    if material_style is not None and material_style.handdrawn_spec is not None:
        return HanddrawnStyleConfig.from_material_spec(material_style.handdrawn_spec)
    return HanddrawnStyleConfig.from_dict({})


def resolve_handdrawn_profile(
    material_pipeline: str,
    material_style: MaterialStyle,
    handdrawn_cfg: HanddrawnStyleConfig | None,
) -> HandDrawnMaterialSpec | None:
    if str(material_pipeline).strip().lower() != "handdrawn":
        return None
    base = material_style.handdrawn_spec or HandDrawnMaterialSpec()
    return handdrawn_spec_from_any(handdrawn_cfg, fallback=base)


def render_atom_radius(
    symbol: str,
    *,
    radius_style: RadiusStyle,
    atom_scale: float,
    element_scale: dict[str, float],
    representation: str,
    space_filling_scale: float,
    radii_scale: float = DEFAULT_RADII_SCALE,
) -> float:
    rep = str(representation).strip().lower()
    base = float(radius_style.radius_for(symbol))
    scale = float(atom_scale) * float(element_scale.get(symbol, 1.0))
    if rep == "space_filling":
        scale *= float(space_filling_scale)
    elif rep == "ball_stick":
        scale *= float(radii_scale)
    return float(max(0.01, base * scale))


def resolve_representation(representation: str | None, style_name: str) -> str:
    rep = str(representation or "auto").strip().lower()
    if rep not in {"auto", "space_filling", "ball_stick"}:
        rep = "auto"
    if rep == "auto":
        return "space_filling" if str(style_name).lower() == "handdrawn" else "ball_stick"
    return rep


def normalize_atom_representation(representation: str | None, *, default: str | None = None) -> str:
    if representation is None:
        if default is None:
            raise ValueError("Atom representation must be one of: ball_stick, space_filling.")
        return str(default).strip().lower()
    rep = str(representation).strip().lower()
    if rep not in {"ball_stick", "space_filling"}:
        raise ValueError("Atom representation must be one of: ball_stick, space_filling.")
    return rep


def resolve_atom_style_state(
    *,
    atom,
    style_cfg: StyleConfig,
    default_representation: str,
) -> ResolvedAtomStyleState:
    state = ResolvedAtomStyleState(
        style=None,
        representation=normalize_atom_representation(default_representation, default=default_representation),
        radius=None,
        material=None,
        color=None,
    )
    presets = style_cfg.atom_styles
    for rule in style_cfg.atom_style_rules:
        if not rule.selector.matches(atom.index, atom.symbol, atom.position, atom.tag):
            continue
        if rule.style is not None:
            _apply_atom_style_preset(
                state=state,
                preset=presets.get(rule.style),
                style_name=rule.style,
            )
        _apply_atom_style_rule_overrides(state=state, rule=rule)

    if atom.representation is not None:
        state.representation = normalize_atom_representation(atom.representation)
    if atom.radius is not None:
        state.radius = float(atom.radius)
    if atom.material is not None:
        state.material = atom.material
    if atom.color is not None:
        state.color = atom.color
    return state


def resolve_atom_representations(
    structure: Structure,
    *,
    style_cfg: StyleConfig,
    default_representation: str,
) -> dict[int, str]:
    out: dict[int, str] = {}
    for idx, atom in enumerate(structure.atoms):
        state = resolve_atom_style_state(
            atom=atom,
            style_cfg=style_cfg,
            default_representation=default_representation,
        )
        out[int(atom.index)] = state.representation
        if int(atom.index) != idx and idx not in out:
            out[idx] = state.representation
    return out


def resolve_draw_bonds(draw_bonds: bool | None, representation: str) -> bool:
    if draw_bonds is None:
        return str(representation).lower() == "ball_stick"
    return bool(draw_bonds)


def resolve_draw_bonds_with_atom_representations(
    *,
    draw_bonds: bool | None,
    representation: str,
    atom_representations: dict[int, str],
) -> bool:
    if draw_bonds is not None:
        return bool(draw_bonds)
    base = normalize_atom_representation(representation, default=representation)
    mixed = any(rep != base for rep in atom_representations.values())
    if mixed and any(rep == "ball_stick" for rep in atom_representations.values()):
        return True
    return resolve_draw_bonds(None, base)


def build_surface_layer_map(
    structure: Structure,
    surface_symbols: set[str],
    tolerance: float,
    enabled: bool,
) -> dict[int, int]:
    if not enabled or not surface_symbols:
        return {}

    z_items = [
        (idx, float(atom.position[2]))
        for idx, atom in enumerate(structure.atoms)
        if atom.symbol in surface_symbols
    ]
    if not z_items:
        return {}

    z_items.sort(key=lambda item: item[1])
    out: dict[int, int] = {}
    current_layer = 0
    prev_z = z_items[0][1]
    out[z_items[0][0]] = current_layer

    tol = max(1e-6, float(tolerance))
    for idx, z in z_items[1:]:
        if abs(z - prev_z) > tol:
            current_layer += 1
        out[idx] = current_layer
        prev_z = z
    return out


def _apply_atom_style_preset(
    *,
    state: ResolvedAtomStyleState,
    preset: AtomStylePresetConfig | None,
    style_name: str | None = None,
) -> None:
    if preset is None:
        return
    if style_name is not None:
        state.style = str(style_name)
    if preset.representation is not None:
        state.representation = normalize_atom_representation(preset.representation)
    if preset.radius is not None:
        state.radius = float(preset.radius)
    if preset.material is not None:
        state.material = preset.material
    if preset.color is not None:
        state.color = preset.color


def _apply_atom_style_rule_overrides(*, state: ResolvedAtomStyleState, rule: AtomStyleRuleConfig) -> None:
    if rule.style is not None:
        state.style = str(rule.style)
    if rule.representation is not None:
        state.representation = normalize_atom_representation(rule.representation)
    if rule.radius is not None:
        state.radius = float(rule.radius)
    if rule.material is not None:
        state.material = rule.material
    if rule.color is not None:
        state.color = rule.color


def _material_spec_base_equal(left: MaterialSpec, right: MaterialSpec, tol: float) -> bool:
    for field_name in ("roughness", "specular", "metallic", "coat", "coat_roughness", "specular_tint", "alpha"):
        if not _close(float(getattr(left, field_name)), float(getattr(right, field_name)), tol):
            return False
    if not _all_close(left.color, right.color, tol):
        return False
    if left.ior is None or right.ior is None:
        return left.ior is None and right.ior is None
    return _close(float(left.ior), float(right.ior), tol)


def _handdrawn_spec_equal(left: HandDrawnMaterialSpec, right: HandDrawnMaterialSpec, tol: float) -> bool:
    for field_name in (
        "jmol_desaturate",
        "jmol_lighten",
        "shadow_area",
        "shadow_strength",
        "shadow_softness",
        "highlight_strength",
        "highlight_arc_length",
        "highlight_band_inner",
        "highlight_band_outer",
        "outline_surface",
        "outline_molecule",
        "outline_bond",
        "outline_secondary_thickness",
    ):
        if not _close(float(getattr(left, field_name)), float(getattr(right, field_name)), tol):
            return False
    if not _all_close(left.light_direction, right.light_direction, tol):
        return False
    if not _all_close(left.highlight_direction, right.highlight_direction, tol):
        return False
    return _all_close(left.outline_secondary_color, right.outline_secondary_color, tol)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def _all_close(left: tuple[float, ...], right: tuple[float, ...], tol: float) -> bool:
    if len(left) != len(right):
        return False
    return all(_close(float(va), float(vb), tol) for va, vb in zip(left, right))
