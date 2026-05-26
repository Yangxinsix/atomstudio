from __future__ import annotations

from math import sqrt

from atomstudio.scene.materials.builders.base import MaterialBuildContext
from atomstudio.scene.materials.specs import HandDrawnMaterialSpec, MaterialLike, as_handdrawn_spec
from atomstudio.style.material_style import tune_rgba

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


class HanddrawnMaterialBuilder:
    def build(self, mat, material: MaterialLike, *, context: MaterialBuildContext) -> None:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        spec = as_handdrawn_spec(material)

        bsdf, output = _ensure_base_nodes(nodes)
        _clear_handdrawn_nodes(nodes)

        if not _scene_engine_is_eevee():
            # Keep handdrawn look consistent across engines.
            # Cycles defaulted to a different branch and could lose the crescent highlight.
            self._apply_eevee(
                mat=mat,
                spec=spec,
                nodes=nodes,
                links=links,
                output=output,
            )
            return

        self._apply_eevee(
            mat=mat,
            spec=spec,
            nodes=nodes,
            links=links,
            output=output,
        )

    def _apply_eevee(
        self,
        *,
        mat,
        spec: HandDrawnMaterialSpec,
        nodes,
        links,
        output,
    ) -> None:
        shadow_area = max(0.0, min(1.0, float(spec.shadow_area)))
        shadow_strength = max(0.0, min(1.0, float(spec.shadow_strength)))
        shadow_softness = max(0.0, min(1.0, float(spec.shadow_softness)))
        highlight_strength = max(0.0, min(0.9, float(spec.highlight_strength)))
        light_direction = _normalize_vec3(spec.light_direction)
        highlight_direction = _normalize_vec3(spec.highlight_direction)
        highlight_arc_length = max(0.0, min(1.0, float(spec.highlight_arc_length)))
        highlight_band_outer = max(0.0, min(1.0, float(spec.highlight_band_outer)))
        if highlight_band_outer <= 0.0:
            highlight_band_outer = 0.05
        # Old threshold was inverse semantics; keep direct "larger => longer arc".
        highlight_direction_threshold = max(0.0, min(1.0, 1.0 - highlight_arc_length))

        dark_color = _darken_rgba(spec.color, amount=0.22 + 0.55 * shadow_strength)
        highlight_color = tune_rgba(spec.color, desaturate=0.30, lighten=0.66 + 0.18 * highlight_strength)

        geom = nodes.new("ShaderNodeNewGeometry")
        geom.name = "ASE_HAND_GEOMETRY"

        dir_vec = nodes.new("ShaderNodeCombineXYZ")
        dir_vec.name = "ASE_HAND_SHADOW_DIR"
        dir_vec.inputs["X"].default_value = float(light_direction[0])
        dir_vec.inputs["Y"].default_value = float(light_direction[1])
        dir_vec.inputs["Z"].default_value = float(light_direction[2])

        dot = nodes.new("ShaderNodeVectorMath")
        dot.name = "ASE_HAND_DOT"
        dot.operation = "DOT_PRODUCT"

        map01 = nodes.new("ShaderNodeMath")
        map01.name = "ASE_HAND_MAP01"
        map01.operation = "MULTIPLY_ADD"
        map01.inputs[1].default_value = 0.5
        map01.inputs[2].default_value = 0.5

        shadow_ramp = nodes.new("ShaderNodeValToRGB")
        shadow_ramp.name = "ASE_HAND_SHADOW_RAMP"
        shadow_ramp.color_ramp.interpolation = "CONSTANT"
        shadow_cut = max(0.12, min(0.92, 0.22 + 0.68 * shadow_area - 0.12 * shadow_softness))
        shadow_ramp.color_ramp.elements[0].position = shadow_cut
        shadow_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
        shadow_ramp.color_ramp.elements[1].position = min(0.999, shadow_cut + 0.001)
        shadow_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)

        shadow_sep, shadow_sep_input, shadow_sep_output = _new_separate_node(nodes, "ASE_HAND_SHADOW_SEPARATE")

        shadow_strength_node = nodes.new("ShaderNodeMath")
        shadow_strength_node.name = "ASE_HAND_SHADOW_STRENGTH"
        shadow_strength_node.operation = "MULTIPLY"
        shadow_strength_node.inputs[1].default_value = shadow_strength

        shadow_mix = nodes.new("ShaderNodeMixRGB")
        shadow_mix.name = "ASE_HAND_SHADOW_MIX"
        shadow_mix.blend_type = "MIX"
        shadow_mix.inputs["Color1"].default_value = spec.color
        shadow_mix.inputs["Color2"].default_value = dark_color

        normal_sep = nodes.new("ShaderNodeSeparateXYZ")
        normal_sep.name = "ASE_HAND_HIGHLIGHT_NORMAL_SEPARATE"

        x_sq = nodes.new("ShaderNodeMath")
        x_sq.name = "ASE_HAND_HIGHLIGHT_X_SQ"
        x_sq.operation = "MULTIPLY"

        y_sq = nodes.new("ShaderNodeMath")
        y_sq.name = "ASE_HAND_HIGHLIGHT_Y_SQ"
        y_sq.operation = "MULTIPLY"

        r2_sum = nodes.new("ShaderNodeMath")
        r2_sum.name = "ASE_HAND_HIGHLIGHT_R2_SUM"
        r2_sum.operation = "ADD"

        r_gt_high = nodes.new("ShaderNodeMath")
        r_gt_high.name = "ASE_HAND_HIGHLIGHT_R_GT_HIGH"
        r_gt_high.operation = "GREATER_THAN"
        r_gt_high.inputs[1].default_value = highlight_band_outer

        radial_mask = nodes.new("ShaderNodeMath")
        radial_mask.name = "ASE_HAND_HIGHLIGHT_FILLED_MASK"
        radial_mask.operation = "SUBTRACT"
        radial_mask.inputs[0].default_value = 1.0

        highlight_dir_vec = nodes.new("ShaderNodeCombineXYZ")
        highlight_dir_vec.name = "ASE_HAND_HIGHLIGHT_DIR_VEC"
        highlight_dir_vec.inputs["X"].default_value = float(highlight_direction[0])
        highlight_dir_vec.inputs["Y"].default_value = float(highlight_direction[1])
        highlight_dir_vec.inputs["Z"].default_value = float(highlight_direction[2])

        dir_dot = nodes.new("ShaderNodeVectorMath")
        dir_dot.name = "ASE_HAND_HIGHLIGHT_DIR_DOT"
        dir_dot.operation = "DOT_PRODUCT"

        dir_map = nodes.new("ShaderNodeMath")
        dir_map.name = "ASE_HAND_HIGHLIGHT_DIR_MAP01"
        dir_map.operation = "MULTIPLY_ADD"
        dir_map.inputs[1].default_value = 0.5
        dir_map.inputs[2].default_value = 0.5

        dir_gate = nodes.new("ShaderNodeMath")
        dir_gate.name = "ASE_HAND_HIGHLIGHT_DIR_GATE"
        dir_gate.operation = "GREATER_THAN"
        dir_gate.inputs[1].default_value = highlight_direction_threshold

        highlight_dir_mul = nodes.new("ShaderNodeMath")
        highlight_dir_mul.name = "ASE_HAND_HIGHLIGHT_DIR_MUL"
        highlight_dir_mul.operation = "MULTIPLY"

        highlight_strength_node = nodes.new("ShaderNodeMath")
        highlight_strength_node.name = "ASE_HAND_HIGHLIGHT_STRENGTH"
        highlight_strength_node.operation = "MULTIPLY"
        highlight_strength_node.inputs[1].default_value = 0.92 + 0.70 * highlight_strength

        highlight_mix = nodes.new("ShaderNodeMixRGB")
        highlight_mix.name = "ASE_HAND_HIGHLIGHT_MIX"
        highlight_mix.blend_type = "MIX"
        highlight_mix.inputs["Color2"].default_value = highlight_color

        emission = nodes.new("ShaderNodeEmission")
        emission.name = "ASE_HAND_EMISSION"
        emission.inputs["Strength"].default_value = 1.0

        links.new(geom.outputs["Normal"], dot.inputs[0])
        links.new(dir_vec.outputs["Vector"], dot.inputs[1])
        links.new(dot.outputs["Value"], map01.inputs[0])
        links.new(map01.outputs["Value"], shadow_ramp.inputs["Fac"])
        links.new(shadow_ramp.outputs["Color"], shadow_sep.inputs[shadow_sep_input])
        links.new(shadow_sep.outputs[shadow_sep_output], shadow_strength_node.inputs[0])
        links.new(shadow_strength_node.outputs["Value"], shadow_mix.inputs["Fac"])

        links.new(geom.outputs["Normal"], normal_sep.inputs["Vector"])
        links.new(normal_sep.outputs["X"], x_sq.inputs[0])
        links.new(normal_sep.outputs["X"], x_sq.inputs[1])
        links.new(normal_sep.outputs["Y"], y_sq.inputs[0])
        links.new(normal_sep.outputs["Y"], y_sq.inputs[1])
        links.new(x_sq.outputs["Value"], r2_sum.inputs[0])
        links.new(y_sq.outputs["Value"], r2_sum.inputs[1])
        links.new(r2_sum.outputs["Value"], r_gt_high.inputs[0])
        links.new(r_gt_high.outputs["Value"], radial_mask.inputs[1])
        links.new(geom.outputs["Normal"], dir_dot.inputs[0])
        links.new(highlight_dir_vec.outputs["Vector"], dir_dot.inputs[1])
        links.new(dir_dot.outputs["Value"], dir_map.inputs[0])
        links.new(dir_map.outputs["Value"], dir_gate.inputs[0])
        links.new(dir_gate.outputs["Value"], highlight_dir_mul.inputs[0])
        links.new(radial_mask.outputs["Value"], highlight_dir_mul.inputs[1])
        links.new(highlight_dir_mul.outputs["Value"], highlight_strength_node.inputs[0])
        links.new(shadow_mix.outputs["Color"], highlight_mix.inputs["Color1"])
        links.new(highlight_strength_node.outputs["Value"], highlight_mix.inputs["Fac"])
        links.new(highlight_mix.outputs["Color"], emission.inputs["Color"])

        alpha = max(0.0, min(1.0, float(spec.alpha)))
        if alpha < 0.999:
            transparent = nodes.new("ShaderNodeBsdfTransparent")
            transparent.name = "ASE_HAND_TRANSPARENT"

            alpha_inv = nodes.new("ShaderNodeMath")
            alpha_inv.name = "ASE_HAND_ALPHA_INV"
            alpha_inv.operation = "SUBTRACT"
            alpha_inv.inputs[0].default_value = 1.0
            alpha_inv.inputs[1].default_value = alpha

            shader_mix = nodes.new("ShaderNodeMixShader")
            shader_mix.name = "ASE_HAND_SHADER_MIX"

            links.new(alpha_inv.outputs["Value"], shader_mix.inputs["Fac"])
            links.new(emission.outputs["Emission"], shader_mix.inputs[1])
            links.new(transparent.outputs["BSDF"], shader_mix.inputs[2])
            _connect_surface_output(links, output, shader_mix.outputs["Shader"])
            _set_material_blend_mode(mat, alpha=alpha)
        else:
            _connect_surface_output(links, output, emission.outputs["Emission"])
            _set_material_blend_mode(mat, alpha=alpha)

    def _apply_cycles(
        self,
        *,
        mat,
        spec: HandDrawnMaterialSpec,
        nodes,
        links,
        bsdf,
        output,
    ) -> None:
        shadow_area = max(0.0, min(1.0, float(spec.shadow_area)))
        shadow_strength = max(0.0, min(1.0, float(spec.shadow_strength)))
        shadow_softness = max(0.0, min(1.0, float(spec.shadow_softness)))
        highlight_strength = max(0.0, min(0.9, float(spec.highlight_strength)))
        light_direction = _normalize_vec3(spec.light_direction)

        dark_color = _darken_rgba(spec.color, amount=0.30 + 0.45 * shadow_strength)
        edge_color = tune_rgba(spec.color, desaturate=0.06, lighten=0.28 + 0.30 * highlight_strength)

        geom = nodes.new("ShaderNodeNewGeometry")
        geom.name = "ASE_HAND_GEOMETRY"

        dir_vec = nodes.new("ShaderNodeCombineXYZ")
        dir_vec.name = "ASE_HAND_SHADOW_DIR"
        dir_vec.inputs["X"].default_value = float(light_direction[0])
        dir_vec.inputs["Y"].default_value = float(light_direction[1])
        dir_vec.inputs["Z"].default_value = float(light_direction[2])

        dot = nodes.new("ShaderNodeVectorMath")
        dot.name = "ASE_HAND_DOT"
        dot.operation = "DOT_PRODUCT"

        map01 = nodes.new("ShaderNodeMath")
        map01.name = "ASE_HAND_MAP01"
        map01.operation = "MULTIPLY_ADD"
        map01.inputs[1].default_value = 0.5
        map01.inputs[2].default_value = 0.5

        shadow_ramp = nodes.new("ShaderNodeValToRGB")
        shadow_ramp.name = "ASE_HAND_SHADOW_RAMP"
        low = max(0.0, 0.05 + 0.35 * shadow_area - 0.10 * shadow_softness)
        high = min(1.0, 0.70 + 0.25 * shadow_area + 0.08 * shadow_softness)
        shadow_ramp.color_ramp.elements[0].position = low
        shadow_ramp.color_ramp.elements[0].color = (1.0, 1.0, 1.0, 1.0)
        shadow_ramp.color_ramp.elements[1].position = high
        shadow_ramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)

        shadow_strength_node = nodes.new("ShaderNodeMath")
        shadow_strength_node.name = "ASE_HAND_SHADOW_STRENGTH"
        shadow_strength_node.operation = "MULTIPLY"
        shadow_strength_node.inputs[1].default_value = shadow_strength

        shadow_mix = nodes.new("ShaderNodeMixRGB")
        shadow_mix.name = "ASE_HAND_SHADOW_MIX"
        shadow_mix.blend_type = "MIX"
        shadow_mix.inputs["Color1"].default_value = spec.color
        shadow_mix.inputs["Color2"].default_value = dark_color

        layer = nodes.new("ShaderNodeLayerWeight")
        layer.name = "ASE_HAND_LAYER"
        layer.inputs["Blend"].default_value = 0.28

        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.name = "ASE_HAND_RAMP"
        ramp.color_ramp.elements[0].position = 0.30
        ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        ramp.color_ramp.elements[1].position = 0.92
        ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

        mix = nodes.new("ShaderNodeMixRGB")
        mix.name = "ASE_HAND_MIX"
        mix.blend_type = "MIX"
        mix.inputs["Fac"].default_value = min(1.0, 0.35 + 0.65 * highlight_strength)
        mix.inputs["Color1"].default_value = spec.color
        mix.inputs["Color2"].default_value = edge_color

        links.new(geom.outputs["Normal"], dot.inputs[0])
        links.new(dir_vec.outputs["Vector"], dot.inputs[1])
        links.new(dot.outputs["Value"], map01.inputs[0])
        links.new(map01.outputs["Value"], shadow_ramp.inputs["Fac"])
        links.new(shadow_ramp.outputs["Color"], shadow_strength_node.inputs[0])
        links.new(shadow_strength_node.outputs["Value"], shadow_mix.inputs["Fac"])

        links.new(layer.outputs["Facing"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], mix.inputs["Fac"])
        links.new(shadow_mix.outputs["Color"], mix.inputs["Color1"])
        links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])

        bsdf.inputs["Roughness"].default_value = max(0.45, float(spec.roughness))
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = min(0.14, float(spec.specular))
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = min(0.14, float(spec.specular))
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = float(spec.metallic)
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = float(spec.alpha)

        _set_material_blend_mode(mat, alpha=float(spec.alpha))
        _connect_surface_output(links, output, bsdf.outputs["BSDF"])


def _normalize_vec3(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    n = sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return (0.68, 0.36, 0.62)
    return (x / n, y / n, z / n)


def _scene_engine_is_eevee() -> bool:
    if bpy is None:
        return False
    try:
        engine = str(bpy.context.scene.render.engine).upper()
    except Exception:
        return False
    return "EEVEE" in engine


def _new_separate_node(nodes, name: str):
    try:
        node = nodes.new("ShaderNodeSeparateColor")
        node.name = name
        return node, "Color", "Red"
    except Exception:
        pass
    node = nodes.new("ShaderNodeSeparateRGB")
    node.name = name
    return node, "Image", "R"


def _darken_rgba(color: tuple[float, float, float, float], amount: float) -> tuple[float, float, float, float]:
    a = max(0.0, min(0.95, float(amount)))
    return (
        max(0.0, min(1.0, float(color[0]) * (1.0 - a))),
        max(0.0, min(1.0, float(color[1]) * (1.0 - a))),
        max(0.0, min(1.0, float(color[2]) * (1.0 - a))),
        float(color[3]),
    )


def _ensure_base_nodes(nodes):
    bsdf = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")
    return bsdf, output


def _clear_handdrawn_nodes(nodes) -> None:
    for node in list(nodes):
        if str(node.name).startswith("ASE_HAND_"):
            nodes.remove(node)
def _connect_surface_output(links, output, shader_socket) -> None:
    if output is None:
        return
    for link in list(links):
        if link.to_node == output and link.to_socket == output.inputs["Surface"]:
            links.remove(link)
    links.new(shader_socket, output.inputs["Surface"])


def _set_material_blend_mode(mat, *, alpha: float) -> None:
    if float(alpha) < 0.999:
        mat.blend_method = "BLEND"
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = "HASHED"
        return
    mat.blend_method = "OPAQUE"
