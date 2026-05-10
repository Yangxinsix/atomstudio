from __future__ import annotations

from atomstudio.preview.types import PreviewMaterialPayload, PreviewScene


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def rgba(
    value: tuple[float, float, float, float] | None,
    fallback: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if value is None:
        return fallback
    return tuple(clamp01(float(v)) for v in value)  # type: ignore[return-value]


def mix_rgba(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    weight: float,
) -> tuple[float, float, float, float]:
    resolved = clamp01(weight)
    return tuple(clamp01((1.0 - resolved) * float(a) + resolved * float(b)) for a, b in zip(left, right))  # type: ignore[return-value]


def scale_rgba(
    color: tuple[float, float, float, float],
    factor: float,
    alpha: float | None = None,
) -> tuple[float, float, float, float]:
    return (
        clamp01(float(color[0]) * factor),
        clamp01(float(color[1]) * factor),
        clamp01(float(color[2]) * factor),
        float(color[3]) if alpha is None else clamp01(alpha),
    )


def preview_material_payload(material: PreviewMaterialPayload | None) -> PreviewMaterialPayload | None:
    if material is None:
        return None
    payload = material.to_dict()
    payload["pipeline"] = f"{material.pipeline}-preview"
    return PreviewMaterialPayload(**payload)


def resolve_render_mode(scene: PreviewScene) -> str:
    if scene.atom_records:
        pipeline = str(scene.atom_records[0].material.pipeline or "principled").strip().lower()
        return f"{pipeline}-preview"
    return "principled-preview"


__all__ = [
    "clamp01",
    "mix_rgba",
    "preview_material_payload",
    "resolve_render_mode",
    "rgba",
    "scale_rgba",
]
