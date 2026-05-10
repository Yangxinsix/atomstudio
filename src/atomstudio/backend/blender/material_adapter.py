from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from atomstudio.scene.materials.registry import MaterialRegistry
from atomstudio.scene.materials.request import MaterialRequest
from atomstudio.scene.materials.specs import HandDrawnMaterialSpec, MaterialLike, MaterialSpec


_HANDDRAWN_HINT_FIELDS = {
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
}


def scene_value(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        return node.get(key, default)
    return getattr(node, key, default)


def scene_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if hasattr(value, "to_dict"):
        out = value.to_dict()
        if isinstance(out, Mapping):
            return {str(k): v for k, v in out.items()}
    if hasattr(value, "__dict__"):
        return {str(k): v for k, v in vars(value).items() if not str(k).startswith("_")}
    raise TypeError(f"Expected a mapping-like scene object, got {type(value)!r}.")


def coerce_material_spec(payload: Any, *, pipeline: str | None = None) -> MaterialLike:
    if isinstance(payload, (MaterialSpec, HandDrawnMaterialSpec)):
        return payload
    if payload is None:
        raise ValueError("material payload is required")

    src = scene_dict(payload)
    raw_pipeline = str(pipeline or src.get("pipeline") or "").strip().lower()
    spec_data = src.get("spec", src)
    if isinstance(spec_data, (MaterialSpec, HandDrawnMaterialSpec)):
        return spec_data
    if not isinstance(spec_data, Mapping):
        raise TypeError("material payload spec must be a mapping")

    spec_dict = {str(k): v for k, v in spec_data.items()}
    if raw_pipeline == "handdrawn" or any(key in spec_dict for key in _HANDDRAWN_HINT_FIELDS):
        return HandDrawnMaterialSpec.from_dict(spec_dict)
    return MaterialSpec.from_dict(spec_dict)


class BlenderMaterialAdapter:
    def __init__(self, registry: MaterialRegistry) -> None:
        self.registry = registry

    def resolve(
        self,
        payload: Any,
        *,
        name: str,
        role: str,
        pipeline: str | None = None,
        style_name: str | None = None,
    ) -> Any:
        if payload is None:
            return None

        raw_pipeline = str(pipeline or scene_value(payload, "pipeline", "") or "").strip().lower()
        spec = coerce_material_spec(payload, pipeline=raw_pipeline or None)
        pipeline_name = raw_pipeline or ("handdrawn" if isinstance(spec, HandDrawnMaterialSpec) else "principled")
        request_style = str(style_name or scene_value(payload, "style_name", "") or pipeline_name)

        if pipeline_name == "handdrawn":
            request = MaterialRequest.handdrawn(
                name=name,
                material=spec,
                role=role,
                style_name=request_style,
            )
        else:
            request = MaterialRequest.principled(
                name=name,
                material=spec,
                role=role,
                style_name=request_style,
            )
        return self.registry.get(request)
