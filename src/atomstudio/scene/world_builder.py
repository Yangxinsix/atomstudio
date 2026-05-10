from __future__ import annotations

from atomstudio.config import RenderJobConfig

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


class WorldBuilder:
    def __init__(
        self,
        *,
        background: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        transparent_bg: bool = True,
    ) -> None:
        self.background = background
        self.transparent_bg = bool(transparent_bg)

    @classmethod
    def from_cfg(
        cls,
        cfg: RenderJobConfig,
        *,
        background: tuple[float, float, float, float],
    ) -> "WorldBuilder":
        return cls(background=background, transparent_bg=cfg.render.transparent_bg)

    def apply(self) -> None:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")

        scene = bpy.context.scene
        world = bpy.data.worlds.get("World")
        if world is None:
            world = bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True
        bg = next((node for node in world.node_tree.nodes if node.type == "BACKGROUND"), None)
        if bg is None:
            return
        world_color, world_strength = resolve_world_lighting_background(
            self.background,
            transparent_bg=self.transparent_bg,
        )
        bg.inputs[0].default_value = world_color
        bg.inputs[1].default_value = world_strength


def resolve_world_lighting_background(
    background: tuple[float, float, float, float],
    *,
    transparent_bg: bool,
) -> tuple[tuple[float, float, float, float], float]:
    r = max(0.0, min(1.0, float(background[0])))
    g = max(0.0, min(1.0, float(background[1])))
    b = max(0.0, min(1.0, float(background[2])))
    a = max(0.0, min(1.0, float(background[3])))

    if not bool(transparent_bg):
        return (r, g, b, a), 1.0

    # Transparent outputs are often viewed on white. Keep the world neutral to avoid
    # color cast, but do not under-light reflective materials.
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luma >= 0.75:
        return (0.32, 0.32, 0.32, 1.0), 0.55
    if luma >= 0.50:
        scale = 0.60 / max(luma, 1e-8)
        return (
            max(0.0, min(1.0, r * scale)),
            max(0.0, min(1.0, g * scale)),
            max(0.0, min(1.0, b * scale)),
            1.0,
        ), 0.50
    return (r, g, b, 1.0), 0.45
