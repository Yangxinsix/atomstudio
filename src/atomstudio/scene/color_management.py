from __future__ import annotations

from copy import deepcopy

from atomstudio.config import RenderJobConfig

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


class ColorManagementBuilder:
    def __init__(self, *, color_management: dict | None = None) -> None:
        self.color_management = {} if color_management is None else deepcopy(color_management)

    @classmethod
    def from_cfg(cls, cfg: RenderJobConfig) -> "ColorManagementBuilder":
        return cls(color_management=cfg.render.color_management)

    def apply(self) -> None:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")
        scene = bpy.context.scene
        cm = self.color_management
        scene.view_settings.view_transform = str(cm.get("view_transform", "Standard"))
        scene.view_settings.look = str(cm.get("look", "None"))
        scene.view_settings.exposure = float(cm.get("exposure", 0.0))
        scene.view_settings.gamma = float(cm.get("gamma", 1.0))
