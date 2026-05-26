from __future__ import annotations

from pathlib import Path
from typing import Any

from atomstudio.config import RenderJobConfig

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


class RenderEffectsBuilder:
    def __init__(
        self,
        cfg: RenderJobConfig,
        *,
        background: tuple[float, float, float, float],
    ) -> None:
        self.cfg = cfg
        self.effects = cfg.render.effects
        self.background = background

    @classmethod
    def from_cfg(
        cls,
        cfg: RenderJobConfig,
        *,
        background: tuple[float, float, float, float],
    ) -> "RenderEffectsBuilder":
        return cls(cfg, background=background)

    def apply(self) -> None:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")
        self._apply_hdri_world()
        self._apply_engine_effects()
        self._apply_world_volume()
        self._apply_compositor()

    def _apply_hdri_world(self) -> None:
        hdri = self.effects.hdri
        if not bool(hdri.enabled):
            return
        if not hdri.path:
            raise ValueError("render.effects.hdri.path is required when HDRI is enabled.")
        path = Path(str(hdri.path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"HDRI file not found: {path}")

        scene = bpy.context.scene
        world = scene.world or bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        nodes.clear()

        output = nodes.new("ShaderNodeOutputWorld")
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(path), check_existing=True)
        env_bg = nodes.new("ShaderNodeBackground")
        env_bg.inputs["Strength"].default_value = max(0.0, float(hdri.strength))
        links.new(env.outputs["Color"], env_bg.inputs["Color"])

        if bool(hdri.visible_to_camera):
            links.new(env_bg.outputs["Background"], output.inputs["Surface"])
        else:
            camera_bg = nodes.new("ShaderNodeBackground")
            camera_bg.inputs["Color"].default_value = self.background
            camera_bg.inputs["Strength"].default_value = 1.0
            light_path = nodes.new("ShaderNodeLightPath")
            mix = nodes.new("ShaderNodeMixShader")
            links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
            links.new(env_bg.outputs["Background"], mix.inputs[1])
            links.new(camera_bg.outputs["Background"], mix.inputs[2])
            links.new(mix.outputs["Shader"], output.inputs["Surface"])

    def _apply_engine_effects(self) -> None:
        scene = bpy.context.scene
        ao = self.effects.ambient_occlusion
        if bool(ao.enabled):
            cycles = getattr(scene, "cycles", None)
            if cycles is not None:
                for attr, value in (
                    ("use_fast_gi", True),
                    ("fast_gi_method", "ADD"),
                    ("ao_bounces", max(0, int(round(float(ao.distance))))),
                    ("ao_bounces_render", max(0, int(round(float(ao.distance))))),
                ):
                    if hasattr(cycles, attr):
                        try:
                            setattr(cycles, attr, value)
                        except Exception:
                            pass
            eevee = getattr(scene, "eevee", None)
            if eevee is not None:
                for attr, value in (
                    ("use_gtao", True),
                    ("gtao_factor", max(0.0, float(ao.factor))),
                    ("gtao_distance", max(0.0, float(ao.distance))),
                ):
                    if hasattr(eevee, attr):
                        try:
                            setattr(eevee, attr, value)
                        except Exception:
                            pass
            world = getattr(scene, "world", None)
            light_settings = getattr(world, "light_settings", None)
            if light_settings is not None:
                for attr, value in (
                    ("use_ambient_occlusion", True),
                    ("ao_factor", max(0.0, float(ao.factor))),
                    ("distance", max(0.0, float(ao.distance))),
                ):
                    if hasattr(light_settings, attr):
                        try:
                            setattr(light_settings, attr, value)
                        except Exception:
                            pass

        ssr = self.effects.ssr
        if bool(ssr.enabled):
            eevee = getattr(scene, "eevee", None)
            if eevee is not None:
                for attr, value in (
                    ("use_ssr", True),
                    ("use_ssr_refraction", bool(ssr.refraction)),
                    ("ssr_quality", max(0.0, min(1.0, float(ssr.quality)))),
                ):
                    if hasattr(eevee, attr):
                        try:
                            setattr(eevee, attr, value)
                        except Exception:
                            pass

    def _apply_world_volume(self) -> None:
        atmosphere = self.effects.atmosphere
        volumetric = self.effects.volumetric_light
        density = 0.0
        color = None
        anisotropy = 0.0
        if bool(atmosphere.enabled):
            density += max(0.0, float(atmosphere.density))
            color = atmosphere.color
        if bool(volumetric.enabled):
            density += max(0.0, float(volumetric.density))
            color = volumetric.color if color is None else color
            anisotropy = max(-1.0, min(1.0, float(volumetric.anisotropy)))
        if density <= 0.0:
            return

        scene = bpy.context.scene
        world = scene.world or bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        output = next((node for node in nodes if node.type == "OUTPUT_WORLD"), None)
        if output is None:
            output = nodes.new("ShaderNodeOutputWorld")
        volume = nodes.new("ShaderNodeVolumeScatter")
        volume.inputs["Color"].default_value = color or (1.0, 1.0, 1.0, 1.0)
        volume.inputs["Density"].default_value = density
        if "Anisotropy" in volume.inputs:
            volume.inputs["Anisotropy"].default_value = anisotropy
        links.new(volume.outputs["Volume"], output.inputs["Volume"])

    def _apply_compositor(self) -> None:
        if not self._needs_compositor():
            return
        scene = bpy.context.scene
        scene.use_nodes = True
        tree = _compositor_tree(scene)
        nodes = tree.nodes
        links = tree.links
        nodes.clear()

        render_layers = nodes.new("CompositorNodeRLayers")
        composite, composite_input = _new_composite_output(tree, nodes)
        current = render_layers.outputs["Image"]

        current = self._add_bloom(nodes, links, current)
        current = self._add_chromatic_aberration(nodes, links, current)
        current = self._add_vignette(nodes, links, current)
        current = self._add_film_grain(nodes, links, current)
        links.new(current, composite_input)

    def _needs_compositor(self) -> bool:
        return any(
            bool(item.enabled)
            for item in (
                self.effects.bloom,
                self.effects.vignette,
                self.effects.chromatic_aberration,
                self.effects.film_grain,
            )
        )

    def _add_bloom(self, nodes: Any, links: Any, current: Any) -> Any:
        bloom = self.effects.bloom
        if not bool(bloom.enabled):
            return current
        glare = nodes.new("CompositorNodeGlare")
        if hasattr(glare, "glare_type"):
            glare.glare_type = "FOG_GLOW"
            glare.quality = "HIGH"
            glare.threshold = max(0.0, float(bloom.threshold))
            glare.size = max(2, min(9, int(bloom.size)))
        else:
            _set_input(glare, "Type", "Fog Glow")
            _set_input(glare, "Quality", "High")
            _set_input(glare, "Threshold", max(0.0, float(bloom.threshold)))
            _set_input(glare, "Size", max(0.0, min(1.0, float(bloom.size) / 9.0)))
            _set_input(glare, "Strength", 1.0)
        mix = _new_mix_rgba(nodes, blend_type="ADD", factor=max(0.0, min(1.0, float(bloom.intensity))))
        links.new(current, glare.inputs["Image"])
        links.new(current, _mix_rgba_a(mix))
        links.new(glare.outputs["Glare"] if "Glare" in glare.outputs else glare.outputs["Image"], _mix_rgba_b(mix))
        return _mix_rgba_output(mix)

    def _add_chromatic_aberration(self, nodes: Any, links: Any, current: Any) -> Any:
        aberration = self.effects.chromatic_aberration
        if not bool(aberration.enabled):
            return current
        lens = nodes.new("CompositorNodeLensdist")
        if hasattr(lens, "use_fit"):
            lens.use_fit = True
        else:
            _set_input(lens, "Fit", True)
        _set_input(lens, "Dispersion", max(0.0, float(aberration.dispersion)))
        links.new(current, lens.inputs["Image"])
        return lens.outputs["Image"]

    def _add_vignette(self, nodes: Any, links: Any, current: Any) -> Any:
        vignette = self.effects.vignette
        if not bool(vignette.enabled):
            return current
        ellipse = nodes.new("CompositorNodeEllipseMask")
        ellipse.width = 0.86
        ellipse.height = 0.86
        blur = nodes.new("CompositorNodeBlur")
        softness = max(0.02, min(0.45, float(vignette.softness)))
        if hasattr(blur, "filter_type"):
            blur.filter_type = "FAST_GAUSS"
            blur.use_relative = True
            blur.factor_x = softness
            blur.factor_y = softness
        else:
            _set_input(blur, "Type", "Gaussian")
            _set_input(blur, "Size", (softness, softness))
        invert = nodes.new("CompositorNodeInvert")
        mix = _new_mix_rgba(nodes, blend_type="MULTIPLY", factor=1.0)
        _mix_rgba_b(mix).default_value = (
            1.0 - max(0.0, min(1.0, float(vignette.intensity))),
            1.0 - max(0.0, min(1.0, float(vignette.intensity))),
            1.0 - max(0.0, min(1.0, float(vignette.intensity))),
            1.0,
        )
        links.new(ellipse.outputs["Mask"], blur.inputs["Image"])
        links.new(blur.outputs["Image"], invert.inputs["Color"])
        links.new(invert.outputs["Color"], _mix_rgba_factor(mix))
        links.new(current, _mix_rgba_a(mix))
        return _mix_rgba_output(mix)

    def _add_film_grain(self, nodes: Any, links: Any, current: Any) -> Any:
        grain = self.effects.film_grain
        if not bool(grain.enabled):
            return current
        try:
            tex_node = nodes.new("ShaderNodeTexNoise")
            tex_node.inputs["Scale"].default_value = max(0.1, float(grain.scale))
            tex_node.inputs["Detail"].default_value = 12.0
            tex_node.inputs["Roughness"].default_value = 0.62
        except Exception:
            return current
        mix = _new_mix_rgba(nodes, blend_type="OVERLAY", factor=max(0.0, min(1.0, float(grain.strength))))
        links.new(current, _mix_rgba_a(mix))
        links.new(tex_node.outputs["Fac"], _mix_rgba_b(mix))
        return _mix_rgba_output(mix)


def _compositor_tree(scene: Any) -> Any:
    tree = getattr(scene, "node_tree", None)
    if tree is not None:
        return tree
    tree = getattr(scene, "compositing_node_group", None)
    if tree is not None:
        return tree
    tree = bpy.data.node_groups.new("AtomStudio_Compositor", "CompositorNodeTree")
    scene.compositing_node_group = tree
    return tree


def _new_composite_output(tree: Any, nodes: Any) -> tuple[Any, Any]:
    try:
        node = nodes.new("CompositorNodeComposite")
        return node, node.inputs["Image"]
    except Exception:
        node = nodes.new("NodeGroupOutput")
        try:
            tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        except Exception:
            pass
        socket = next((item for item in node.inputs if item.name == "Image"), None)
        return node, socket or node.inputs[0]


def _new_mix_rgba(nodes: Any, *, blend_type: str, factor: float) -> Any:
    try:
        node = nodes.new("ShaderNodeMix")
        node.data_type = "RGBA"
        node.blend_type = str(blend_type)
        node.inputs[0].default_value = float(factor)
    except Exception:
        node = nodes.new("CompositorNodeMixRGB")
        node.blend_type = str(blend_type)
        node.inputs["Fac"].default_value = float(factor)
    return node


def _mix_rgba_factor(node: Any) -> Any:
    return node.inputs["Fac"] if "Fac" in node.inputs else node.inputs[0]


def _mix_rgba_a(node: Any) -> Any:
    return node.inputs["Color1"] if "Color1" in node.inputs else node.inputs[6]


def _mix_rgba_b(node: Any) -> Any:
    return node.inputs["Color2"] if "Color2" in node.inputs else node.inputs[7]


def _mix_rgba_output(node: Any) -> Any:
    return node.outputs["Image"] if "Image" in node.outputs else node.outputs[2]


def _set_input(node: Any, name: str, value: Any) -> None:
    if name in node.inputs and hasattr(node.inputs[name], "default_value"):
        node.inputs[name].default_value = value
