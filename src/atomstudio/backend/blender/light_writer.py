from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import radians
from typing import Any

from atomstudio.config import RenderJobConfig
from atomstudio.scene.lights.builder import LightingBuilder

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


def _value(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        return node.get(key, default)
    return getattr(node, key, default)


class BlenderLightWriter:
    def __init__(self, cfg: RenderJobConfig, *, default_light_style: str | None = None) -> None:
        self.cfg = cfg
        self.default_light_style = default_light_style

    def write(self, scene: Any, points: Sequence[Any]) -> list[Any]:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")

        lights = _value(scene, "lights")
        if not lights:
            return LightingBuilder.from_cfg(self.cfg, default_light_style=self.default_light_style).build(points)

        out: list[Any] = []
        for index, light in enumerate(lights):
            runtime = dict(light.items()) if isinstance(light, Mapping) else {str(k): v for k, v in vars(light).items()}
            bpy.ops.object.light_add(type=str(runtime.get("type", "AREA")).upper(), location=runtime.get("location", (0.0, 0.0, 0.0)))
            obj = bpy.context.active_object
            obj.name = str(runtime.get("name", f"Light_{index}"))
            obj.data.energy = float(runtime.get("energy", 100.0))
            if str(runtime.get("type", "AREA")).upper() == "AREA":
                obj.data.size = float(runtime.get("size", 1.0))
                shape = runtime.get("shape")
                if shape is not None and hasattr(obj.data, "shape"):
                    obj.data.shape = str(shape).upper()
                if runtime.get("size_y") is not None and hasattr(obj.data, "size_y"):
                    obj.data.size_y = float(runtime.get("size_y"))
            if str(runtime.get("type", "")).upper() == "SUN" and hasattr(obj.data, "angle"):
                obj.data.angle = radians(max(0.1, min(10.0, float(runtime.get("size", 1.0)))))
            color = runtime.get("color")
            if isinstance(color, Sequence) and len(color) >= 3:
                obj.data.color = (float(color[0]), float(color[1]), float(color[2]))
            if bool(runtime.get("lock_to_camera", False)):
                camera = bpy.context.scene.camera
                if camera is not None:
                    constraint = obj.constraints.new(type="COPY_ROTATION")
                    constraint.target = camera
                    constraint.target_space = "WORLD"
                    constraint.owner_space = "WORLD"
            out.append(obj)
        return out
