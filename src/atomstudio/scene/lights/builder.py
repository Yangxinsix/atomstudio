from __future__ import annotations

from copy import deepcopy
from math import radians
from typing import Sequence

from atomstudio.config import RenderJobConfig
from atomstudio.scene.lights.specs import LightSpec
from atomstudio.style.light_style import LIGHT_STYLE_LIBRARY

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    Vector = None


_PRESET_LIGHT_REFERENCE_EXTENT = 4.0
_PRESET_LIGHT_MIN_ENERGY_SCALE = 0.0625
_PRESET_LIGHT_MAX_ENERGY_SCALE = 2000.0
_REFERENCE_CLAMPED_LIGHT_STYLES = {"style_sphere_showcase"}


class LightingBuilder:
    def __init__(
        self,
        *,
        light_style: str = "three_point",
        intensity: float = 1.0,
        lights: list[LightSpec] | None = None,
        default_light_style: str | None = None,
    ) -> None:
        self.light_style = str(light_style)
        self.intensity = float(intensity)
        self.lights = [] if lights is None else [deepcopy(item) for item in lights]
        self.default_light_style = None if default_light_style is None else str(default_light_style)

    @classmethod
    def from_cfg(
        cls,
        cfg: RenderJobConfig,
        *,
        default_light_style: str | None = None,
    ) -> "LightingBuilder":
        lighting = cfg.lighting
        return cls(
            light_style=(lighting.light_style or ""),
            intensity=lighting.intensity,
            lights=lighting.lights,
            default_light_style=default_light_style,
        )

    def build(self, points: Sequence) -> list:
        if bpy is None or Vector is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")

        center, extent = _center_extent(points)
        specs = self.resolve_specs(
            center=(float(center.x), float(center.y), float(center.z)),
            extent=float(extent),
        )

        style_name = self.effective_light_style_name()
        objs = []
        for i, spec in enumerate(specs):
            bpy.ops.object.light_add(type=spec["type"], location=spec["location"])
            obj = bpy.context.active_object
            obj.name = f"Light_{style_name}_{i}"
            obj.data.energy = float(spec["energy"])
            if spec["type"] == "AREA":
                obj.data.size = float(spec["size"])
                if spec.get("shape") is not None and hasattr(obj.data, "shape"):
                    obj.data.shape = str(spec["shape"]).upper()
                if spec.get("size_y") is not None and hasattr(obj.data, "size_y"):
                    obj.data.size_y = float(spec["size_y"])
            if spec["type"] == "SUN":
                _apply_sun_angle(obj.data, spec.get("size", 1.0))
            color = spec.get("color")
            if color is not None:
                obj.data.color = color[:3]
            if bool(spec.get("lock_to_camera", False)):
                self._lock_to_camera(obj)
            objs.append(obj)
        return objs

    def _lock_to_camera(self, light_obj) -> None:
        cam = bpy.context.scene.camera
        if cam is None:
            return
        constraint = light_obj.constraints.new(type="COPY_ROTATION")
        constraint.target = cam
        constraint.target_space = "WORLD"
        constraint.owner_space = "WORLD"

    def effective_light_style_name(self) -> str:
        return str(self.light_style or self.default_light_style or "three_point")

    def resolve_specs(
        self,
        *,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        extent: float = 1.0,
    ) -> list[dict]:
        if self.lights:
            return [light.to_runtime_spec(center=center, extent=extent, intensity=self.intensity) for light in self.lights]

        style_name = self.effective_light_style_name()
        preset = LIGHT_STYLE_LIBRARY.get(style_name, LIGHT_STYLE_LIBRARY["three_point"])
        out: list[dict] = []
        for light in preset:
            runtime = deepcopy(light)
            if style_name != "preview_softbox":
                runtime.lock_to_camera = False
            runtime_extent = _preset_light_runtime_extent(style_name, extent)
            spec = runtime.to_runtime_spec(center=center, extent=runtime_extent, intensity=self.intensity)
            if runtime.placement == "scaled_offset":
                spec["energy"] = float(spec["energy"]) * _preset_light_energy_scale(runtime_extent)
            out.append(spec)
        return out


def _center_extent(points: Sequence) -> tuple[Vector, float]:
    if not points:
        return Vector((0.0, 0.0, 0.0)), 1.0
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    center = Vector(((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, (min(zs) + max(zs)) * 0.5))
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)
    return center, extent


def _preset_light_energy_scale(extent: float) -> float:
    scale = max(1.0e-6, float(extent)) / _PRESET_LIGHT_REFERENCE_EXTENT
    return max(_PRESET_LIGHT_MIN_ENERGY_SCALE, min(_PRESET_LIGHT_MAX_ENERGY_SCALE, scale * scale))


def _preset_light_runtime_extent(style_name: str, extent: float) -> float:
    if str(style_name) in _REFERENCE_CLAMPED_LIGHT_STYLES:
        return max(_PRESET_LIGHT_REFERENCE_EXTENT, float(extent))
    return float(extent)


def resolve_lighting_specs(
    cfg: RenderJobConfig,
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extent: float = 1.0,
    default_light_style: str | None = None,
) -> list[dict]:
    return LightingBuilder.from_cfg(cfg, default_light_style=default_light_style).resolve_specs(
        center=center,
        extent=extent,
    )


def _apply_sun_angle(light_data, size_value: float) -> None:
    if not hasattr(light_data, "angle"):
        return
    # Interpret style "size" as degrees for SUN to keep presets readable.
    deg = max(0.1, min(10.0, float(size_value)))
    light_data.angle = radians(deg)
