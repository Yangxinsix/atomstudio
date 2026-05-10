from __future__ import annotations

from atomstudio.scene.materials.builders.base import MaterialBuildContext
from atomstudio.scene.materials.specs import MaterialLike, as_material_spec


class PrincipledMaterialBuilder:
    def build(self, mat, material: MaterialLike, *, context: MaterialBuildContext) -> None:
        spec = as_material_spec(material)
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
        if bsdf is None:
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        if output is None:
            output = nodes.new("ShaderNodeOutputMaterial")

        bsdf.inputs["Base Color"].default_value = spec.color
        bsdf.inputs["Roughness"].default_value = float(spec.roughness)
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = float(spec.metallic)
        if spec.ior is not None and "IOR" in bsdf.inputs:
            bsdf.inputs["IOR"].default_value = float(spec.ior)
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = float(spec.specular)
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = float(spec.specular)
        if "Specular Tint" in bsdf.inputs:
            try:
                bsdf.inputs["Specular Tint"].default_value = float(spec.specular_tint)
            except Exception:
                t = float(spec.specular_tint)
                bsdf.inputs["Specular Tint"].default_value = (t, t, t, 1.0)
        coat = float(spec.coat)
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = coat
        elif "Coat" in bsdf.inputs:
            bsdf.inputs["Coat"].default_value = coat
        if "Coat Roughness" in bsdf.inputs:
            bsdf.inputs["Coat Roughness"].default_value = float(spec.coat_roughness)
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = float(spec.alpha)

        if float(spec.alpha) < 0.999:
            mat.blend_method = "BLEND"
            if hasattr(mat, "shadow_method"):
                mat.shadow_method = "HASHED"
        else:
            mat.blend_method = "OPAQUE"

        if output is not None and not any(l.from_node == bsdf and l.to_node == output for l in links):
            for link in list(links):
                if link.to_node == output and link.to_socket == output.inputs["Surface"]:
                    links.remove(link)
            links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
