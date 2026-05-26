from __future__ import annotations

import numpy as np

from atomstudio.config import RenderJobConfig
from atomstudio.scene.lights.builder import resolve_lighting_specs
from atomstudio.scene.model import SceneBounds, SceneLight


def resolve_scene_lights(
    cfg: RenderJobConfig,
    *,
    bounds: SceneBounds,
    default_light_style: str | None = None,
) -> list[SceneLight]:
    specs = resolve_lighting_specs(
        cfg,
        center=tuple(float(v) for v in bounds.center),
        extent=max(float(bounds.radius) * 2.0, 1.0),
        default_light_style=default_light_style,
    )
    center = np.asarray(bounds.center, dtype=float)
    source = "custom" if cfg.lighting.lights else "preset"
    resolved: list[SceneLight] = []
    for idx, spec in enumerate(specs):
        location = tuple(float(v) for v in spec["location"])
        direction = _direction_towards_center(location, center=center)
        resolved.append(
            SceneLight(
                type=str(spec["type"]),
                location=location,
                energy=float(spec["energy"]),
                size=float(spec["size"]),
                size_y=None if spec.get("size_y") is None else float(spec["size_y"]),
                shape=None if spec.get("shape") is None else str(spec["shape"]),
                color=None if spec.get("color") is None else tuple(float(v) for v in spec["color"]),
                direction=direction,
                lock_to_camera=bool(spec.get("lock_to_camera", False)),
                metadata={
                    "index": idx,
                    "style_name": str(cfg.lighting.light_style or default_light_style or ""),
                    "source": source,
                },
            )
        )
    return resolved


def _direction_towards_center(
    location: tuple[float, float, float],
    *,
    center: np.ndarray,
) -> tuple[float, float, float] | None:
    delta = center - np.asarray(location, dtype=float)
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-12:
        return None
    out = delta / norm
    return (float(out[0]), float(out[1]), float(out[2]))
