from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreviewLightingSettings:
    # Direction in screen/camera space. The z component points from the camera
    # into the scene because VisPy's ShadingFilter stores photon travel
    # direction, not light position.
    light_dir_screen: tuple[float, float, float] = (0.0, 0.0, 1.0)
    ambient_light: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.52)
    diffuse_light: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.46)
    specular_light: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.08)
    ambient_coefficient: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    diffuse_coefficient: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    specular_coefficient: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    shininess: float = 48.0


DEFAULT_PREVIEW_LIGHTING = PreviewLightingSettings()


def _normalized(direction: np.ndarray) -> tuple[float, float, float]:
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return (0.0, 0.0, -1.0)
    out = direction / norm
    return (float(out[0]), float(out[1]), float(out[2]))


def preview_light_dir(settings: PreviewLightingSettings = DEFAULT_PREVIEW_LIGHTING) -> tuple[float, float, float]:
    direction = np.asarray(settings.light_dir_screen, dtype=float)
    return _normalized(direction)


def screen_space_light_dir(camera: Any, settings: PreviewLightingSettings = DEFAULT_PREVIEW_LIGHTING) -> tuple[float, float, float]:
    right = np.asarray(getattr(camera, "right", (1.0, 0.0, 0.0)), dtype=float)
    up = np.asarray(getattr(camera, "up", (0.0, 1.0, 0.0)), dtype=float)
    forward = np.asarray(getattr(camera, "forward", (0.0, 0.0, -1.0)), dtype=float)
    sx, sy, sz = settings.light_dir_screen
    return _normalized(float(sx) * right + float(sy) * up + float(sz) * forward)


def configure_shading_filter(
    visual: Any,
    camera: Any,
    settings: PreviewLightingSettings = DEFAULT_PREVIEW_LIGHTING,
) -> bool:
    shading_filter = getattr(visual, "shading_filter", None)
    if shading_filter is None:
        return False
    try:
        shading_filter.light_dir = screen_space_light_dir(camera, settings)
        shading_filter.ambient_light = settings.ambient_light
        shading_filter.diffuse_light = settings.diffuse_light
        shading_filter.specular_light = settings.specular_light
        shading_filter.ambient_coefficient = settings.ambient_coefficient
        shading_filter.diffuse_coefficient = settings.diffuse_coefficient
        shading_filter.specular_coefficient = settings.specular_coefficient
        shading_filter.shininess = float(settings.shininess)
    except (AttributeError, ValueError, TypeError):
        return False
    return True


__all__ = [
    "DEFAULT_PREVIEW_LIGHTING",
    "PreviewLightingSettings",
    "configure_shading_filter",
    "preview_light_dir",
    "screen_space_light_dir",
]
