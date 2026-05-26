from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, TypeAlias

from atomstudio.color_utils import coerce_color_fields, rgba_from_any


_PROVIDED_FIELDS_ATTR = "_provided_fields"


def _set_provided_fields(spec: object, keys: Iterable[str]) -> None:
    setattr(spec, _PROVIDED_FIELDS_ATTR, {str(k) for k in keys})


def _get_provided_fields(spec: object) -> set[str] | None:
    raw = getattr(spec, _PROVIDED_FIELDS_ATTR, None)
    if isinstance(raw, set):
        return {str(v) for v in raw}
    return None


@coerce_color_fields("color", label_prefix="material")
@dataclass
class BaseMaterialSpec:
    color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    alpha: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": list(self.color),
            "alpha": float(self.alpha),
        }


@coerce_color_fields("color", "emission_color", label_prefix="material")
@dataclass
class MaterialSpec(BaseMaterialSpec):
    roughness: float = 0.35
    specular: float = 0.30
    metallic: float = 0.0
    ior: float | None = None
    transmission: float = 0.0
    coat: float = 0.0
    coat_roughness: float = 0.08
    specular_tint: float = 0.0
    sheen: float = 0.0
    subsurface: float = 0.0
    emission_color: tuple[float, float, float, float] | None = None
    emission_strength: float = 0.0

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None = None,
        fallback_color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0),
    ) -> "MaterialSpec":
        src = {} if data is None else dict(data)
        color = _to_rgba(src.get("color"), fallback=fallback_color)
        alpha = float(src.get("alpha", color[3]))
        color = (color[0], color[1], color[2], alpha)
        emission_color = None
        if src.get("emission_color") is not None:
            emission_color = _to_rgba(src.get("emission_color"), fallback=fallback_color)
        out = cls(
            color=color,
            roughness=float(src.get("roughness", 0.35)),
            specular=float(src.get("specular", 0.30)),
            metallic=float(src.get("metallic", 0.0)),
            ior=float(src["ior"]) if src.get("ior") is not None else None,
            transmission=float(src.get("transmission", 0.0)),
            coat=float(src.get("coat", 0.0)),
            coat_roughness=float(src.get("coat_roughness", 0.08)),
            specular_tint=float(src.get("specular_tint", 0.0)),
            sheen=float(src.get("sheen", 0.0)),
            subsurface=float(src.get("subsurface", 0.0)),
            emission_color=emission_color,
            emission_strength=float(src.get("emission_strength", 0.0)),
            alpha=alpha,
        )
        _set_provided_fields(out, src.keys())
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": list(self.color),
            "roughness": self.roughness,
            "specular": self.specular,
            "metallic": self.metallic,
            "ior": self.ior,
            "transmission": self.transmission,
            "coat": self.coat,
            "coat_roughness": self.coat_roughness,
            "specular_tint": self.specular_tint,
            "sheen": self.sheen,
            "subsurface": self.subsurface,
            "emission_color": None if self.emission_color is None else list(self.emission_color),
            "emission_strength": self.emission_strength,
            "alpha": self.alpha,
        }


@coerce_color_fields("outline_secondary_color", label_prefix="handdrawn_material")
@dataclass
class HandDrawnMaterialSpec(BaseMaterialSpec):
    color: tuple[float, float, float, float] = (0.63, 0.68, 0.75, 1.0)
    alpha: float = 1.0
    roughness: float = 0.90
    specular: float = 0.015
    jmol_desaturate: float = 0.10
    jmol_lighten: float = 0.04
    light_direction: tuple[float, float, float] = (0.68, 0.36, 0.62)
    shadow_area: float = 0.34
    shadow_strength: float = 0.42
    shadow_softness: float = 0.12
    highlight_strength: float = 0.16
    highlight_direction: tuple[float, float, float] = (0.78, 0.62, 0.0)
    # 0..1; larger means a longer visible arc segment.
    highlight_arc_length: float = 0.22
    highlight_band_inner: float = 0.56
    highlight_band_outer: float = 0.90
    outline_surface: float = 2.0
    outline_molecule: float = 2.4
    outline_bond: float = 1.6
    outline_secondary_thickness: float = 0.8
    outline_secondary_color: tuple[float, float, float, float] = (0.76, 0.82, 0.92, 1.0)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None = None,
        *,
        fallback: HandDrawnMaterialSpec | None = None,
    ) -> "HandDrawnMaterialSpec":
        src = {} if data is None else dict(data)
        base = HandDrawnMaterialSpec() if fallback is None else replace(fallback)
        direction = src.get("light_direction", base.light_direction)
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            direction = base.light_direction
        highlight_direction = src.get("highlight_direction", base.highlight_direction)
        if not isinstance(highlight_direction, (list, tuple)) or len(highlight_direction) != 3:
            highlight_direction = base.highlight_direction
        arc_length_raw = src.get("highlight_arc_length")
        if arc_length_raw is None and src.get("highlight_direction_threshold") is not None:
            # Backward compatibility: old field was inverted semantics.
            arc_length_raw = 1.0 - float(src.get("highlight_direction_threshold"))
        if arc_length_raw is None:
            arc_length_raw = base.highlight_arc_length
        arc_length = max(0.0, min(1.0, float(arc_length_raw)))
        out = cls(
            color=_to_rgba(src.get("color"), base.color),
            alpha=float(src.get("alpha", base.alpha)),
            roughness=float(src.get("roughness", base.roughness)),
            specular=float(src.get("specular", base.specular)),
            jmol_desaturate=float(src.get("jmol_desaturate", base.jmol_desaturate)),
            jmol_lighten=float(src.get("jmol_lighten", base.jmol_lighten)),
            light_direction=(float(direction[0]), float(direction[1]), float(direction[2])),
            shadow_area=max(0.0, min(1.0, float(src.get("shadow_area", base.shadow_area)))),
            shadow_strength=max(0.0, min(1.0, float(src.get("shadow_strength", base.shadow_strength)))),
            shadow_softness=max(0.0, min(1.0, float(src.get("shadow_softness", base.shadow_softness)))),
            highlight_strength=max(0.0, min(0.9, float(src.get("highlight_strength", base.highlight_strength)))),
            highlight_direction=(
                float(highlight_direction[0]),
                float(highlight_direction[1]),
                float(highlight_direction[2]),
            ),
            highlight_arc_length=arc_length,
            highlight_band_inner=max(0.0, min(1.0, float(src.get("highlight_band_inner", base.highlight_band_inner)))),
            highlight_band_outer=max(0.0, min(1.0, float(src.get("highlight_band_outer", base.highlight_band_outer)))),
            outline_surface=max(0.0, float(src.get("outline_surface", base.outline_surface))),
            outline_molecule=max(0.0, float(src.get("outline_molecule", base.outline_molecule))),
            outline_bond=max(0.0, float(src.get("outline_bond", base.outline_bond))),
            outline_secondary_thickness=max(
                0.0,
                float(src.get("outline_secondary_thickness", base.outline_secondary_thickness)),
            ),
            outline_secondary_color=_to_rgba(src.get("outline_secondary_color"), base.outline_secondary_color),
        )
        provided_keys = set(src.keys())
        if "highlight_direction_threshold" in provided_keys:
            provided_keys.add("highlight_arc_length")
        _set_provided_fields(out, provided_keys)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": list(self.color),
            "alpha": self.alpha,
            "roughness": self.roughness,
            "specular": self.specular,
            "jmol_desaturate": self.jmol_desaturate,
            "jmol_lighten": self.jmol_lighten,
            "light_direction": list(self.light_direction),
            "shadow_area": self.shadow_area,
            "shadow_strength": self.shadow_strength,
            "shadow_softness": self.shadow_softness,
            "highlight_strength": self.highlight_strength,
            "highlight_direction": list(self.highlight_direction),
            "highlight_arc_length": self.highlight_arc_length,
            "highlight_band_inner": self.highlight_band_inner,
            "highlight_band_outer": self.highlight_band_outer,
            "outline_surface": self.outline_surface,
            "outline_molecule": self.outline_molecule,
            "outline_bond": self.outline_bond,
            "outline_secondary_thickness": self.outline_secondary_thickness,
            "outline_secondary_color": list(self.outline_secondary_color),
        }

    @property
    def highlight_direction_threshold(self) -> float:
        # Backward-compat alias: old value was inverse of arc length semantics.
        return max(0.0, min(1.0, 1.0 - float(self.highlight_arc_length)))

    @highlight_direction_threshold.setter
    def highlight_direction_threshold(self, value: float) -> None:
        self.highlight_arc_length = max(0.0, min(1.0, 1.0 - float(value)))


MaterialLike: TypeAlias = MaterialSpec | HandDrawnMaterialSpec

HANDDRAWN_FIELDS: set[str] = {
    "jmol_desaturate",
    "jmol_lighten",
    "light_direction",
    "shadow_area",
    "shadow_strength",
    "shadow_softness",
    "highlight_strength",
    "highlight_direction",
    "highlight_arc_length",
    "highlight_direction_threshold",
    "highlight_band_inner",
    "highlight_band_outer",
    "outline_surface",
    "outline_molecule",
    "outline_bond",
    "outline_secondary_thickness",
    "outline_secondary_color",
}


_DEFAULT_HANDDRAWN = HandDrawnMaterialSpec()


def is_handdrawn_material_payload(data: dict[str, Any] | None) -> bool:
    if not isinstance(data, dict):
        return False
    if str(data.get("pipeline", "")).strip().lower() == "handdrawn":
        return True
    return any(k in HANDDRAWN_FIELDS for k in data.keys())


def material_from_dict(
    data: dict[str, Any] | None,
    *,
    fallback_color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0),
    handdrawn_fallback: HandDrawnMaterialSpec | None = None,
) -> MaterialLike:
    src = {} if data is None else dict(data)
    if is_handdrawn_material_payload(src):
        return HandDrawnMaterialSpec.from_dict(src, fallback=handdrawn_fallback)
    return MaterialSpec.from_dict(src, fallback_color=fallback_color)


def as_material_spec(
    material: MaterialLike,
    *,
    fallback: MaterialSpec | None = None,
) -> MaterialSpec:
    base = MaterialSpec() if fallback is None else replace(fallback)
    if isinstance(material, MaterialSpec):
        provided = _get_provided_fields(material)
        if fallback is None or provided is None or len(provided) == 0:
            return replace(material)
        kwargs: dict[str, Any] = {}
        for name in (
            "color",
            "alpha",
            "roughness",
            "specular",
            "metallic",
            "ior",
            "transmission",
            "coat",
            "coat_roughness",
            "specular_tint",
            "sheen",
            "subsurface",
            "emission_color",
            "emission_strength",
        ):
            if name in provided:
                kwargs[name] = getattr(material, name)
        return replace(base, **kwargs)
    alpha = float(material.alpha)
    return replace(
        base,
        color=(float(material.color[0]), float(material.color[1]), float(material.color[2]), alpha),
        alpha=alpha,
        roughness=float(material.roughness),
        specular=float(material.specular),
        metallic=0.0,
        ior=None,
        transmission=0.0,
        coat=0.0,
        coat_roughness=0.08,
        specular_tint=0.0,
        sheen=0.0,
        subsurface=0.0,
        emission_color=None,
        emission_strength=0.0,
    )


def as_handdrawn_spec(
    material: MaterialLike,
    *,
    fallback: HandDrawnMaterialSpec | None = None,
) -> HandDrawnMaterialSpec:
    base = HandDrawnMaterialSpec() if fallback is None else replace(fallback)
    if isinstance(material, HandDrawnMaterialSpec):
        provided = _get_provided_fields(material)
        if provided is None or len(provided) == 0:
            return replace(material)
        kwargs: dict[str, Any] = {}
        for name in (
            "color",
            "alpha",
            "roughness",
            "specular",
            "jmol_desaturate",
            "jmol_lighten",
            "light_direction",
            "shadow_area",
            "shadow_strength",
            "shadow_softness",
            "highlight_strength",
            "highlight_direction",
            "highlight_arc_length",
            "highlight_band_inner",
            "highlight_band_outer",
            "outline_surface",
            "outline_molecule",
            "outline_bond",
            "outline_secondary_thickness",
            "outline_secondary_color",
        ):
            if name in provided:
                kwargs[name] = getattr(material, name)
        return replace(base, **kwargs)
    if isinstance(material, MaterialSpec):
        provided = _get_provided_fields(material)
        if provided is None or len(provided) == 0:
            return replace(
                base,
                color=tuple(float(v) for v in material.color),
                alpha=float(material.alpha),
                roughness=float(material.roughness),
                specular=float(material.specular),
            )
        kwargs: dict[str, Any] = {}
        if "color" in provided:
            kwargs["color"] = tuple(float(v) for v in material.color)
        if "alpha" in provided:
            kwargs["alpha"] = float(material.alpha)
        if "roughness" in provided:
            kwargs["roughness"] = float(material.roughness)
        if "specular" in provided:
            kwargs["specular"] = float(material.specular)
        return replace(base, **kwargs)
    return replace(
        base,
        color=tuple(float(v) for v in material.color),
        alpha=float(material.alpha),
        roughness=float(material.roughness),
        specular=float(material.specular),
    )


def handdrawn_spec_from_any(
    handdrawn: Any | None,
    *,
    fallback: HandDrawnMaterialSpec | None = None,
) -> HandDrawnMaterialSpec:
    if handdrawn is None:
        return replace(_DEFAULT_HANDDRAWN if fallback is None else fallback)

    base = _DEFAULT_HANDDRAWN if fallback is None else fallback
    direction = getattr(handdrawn, "light_direction", base.light_direction)
    if not isinstance(direction, (list, tuple)) or len(direction) != 3:
        direction = base.light_direction
    highlight_direction = getattr(handdrawn, "highlight_direction", base.highlight_direction)
    if not isinstance(highlight_direction, (list, tuple)) or len(highlight_direction) != 3:
        highlight_direction = base.highlight_direction
    arc_length_raw = getattr(handdrawn, "highlight_arc_length", None)
    if arc_length_raw is None and getattr(handdrawn, "highlight_direction_threshold", None) is not None:
        arc_length_raw = 1.0 - float(getattr(handdrawn, "highlight_direction_threshold"))
    if arc_length_raw is None:
        arc_length_raw = base.highlight_arc_length
    return HandDrawnMaterialSpec(
        color=base.color,
        alpha=float(getattr(handdrawn, "alpha", base.alpha)),
        roughness=float(getattr(handdrawn, "roughness", base.roughness)),
        specular=float(getattr(handdrawn, "specular", base.specular)),
        jmol_desaturate=float(getattr(handdrawn, "jmol_desaturate", base.jmol_desaturate)),
        jmol_lighten=float(getattr(handdrawn, "jmol_lighten", base.jmol_lighten)),
        light_direction=(float(direction[0]), float(direction[1]), float(direction[2])),
        shadow_area=max(0.0, min(1.0, float(getattr(handdrawn, "shadow_area", base.shadow_area)))),
        shadow_strength=max(0.0, min(1.0, float(getattr(handdrawn, "shadow_strength", base.shadow_strength)))),
        shadow_softness=max(0.0, min(1.0, float(getattr(handdrawn, "shadow_softness", base.shadow_softness)))),
        highlight_strength=max(0.0, min(0.9, float(getattr(handdrawn, "highlight_strength", base.highlight_strength)))),
        highlight_direction=(float(highlight_direction[0]), float(highlight_direction[1]), float(highlight_direction[2])),
        highlight_arc_length=max(0.0, min(1.0, float(arc_length_raw))),
        highlight_band_inner=max(0.0, min(1.0, float(getattr(handdrawn, "highlight_band_inner", base.highlight_band_inner)))),
        highlight_band_outer=max(0.0, min(1.0, float(getattr(handdrawn, "highlight_band_outer", base.highlight_band_outer)))),
        outline_surface=max(0.0, float(getattr(handdrawn, "outline_surface", base.outline_surface))),
        outline_molecule=max(0.0, float(getattr(handdrawn, "outline_molecule", base.outline_molecule))),
        outline_bond=max(0.0, float(getattr(handdrawn, "outline_bond", base.outline_bond))),
        outline_secondary_thickness=max(
            0.0,
            float(getattr(handdrawn, "outline_secondary_thickness", base.outline_secondary_thickness)),
        ),
        outline_secondary_color=_to_rgba(
            getattr(handdrawn, "outline_secondary_color", base.outline_secondary_color),
            base.outline_secondary_color,
        ),
    )

def _to_rgba(value: Any, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return rgba_from_any(value, fallback=fallback)
