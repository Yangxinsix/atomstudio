from __future__ import annotations

from typing import Any

from atomstudio.scene.materials.builders.base import MaterialBuildContext
from atomstudio.scene.materials.builders.handdrawn import HanddrawnMaterialBuilder
from atomstudio.scene.materials.builders.principled import PrincipledMaterialBuilder
from atomstudio.scene.materials.request import MaterialRequest
from atomstudio.scene.materials.specs import as_handdrawn_spec, as_material_spec

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


class MaterialRegistry:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._principled_builder = PrincipledMaterialBuilder()
        self._handdrawn_builder = HanddrawnMaterialBuilder()

    def get(self, request: MaterialRequest):
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")
        key = self._cache_key(request)
        if key in self._cache:
            return self._cache[key]

        mat_name = self._material_name(request.name, key)
        mat = bpy.data.materials.get(mat_name)
        if mat is None:
            mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True

        pipeline = str(request.pipeline).strip().lower()
        context = MaterialBuildContext(
            style_name=str(request.style_name),
            role=str(request.role),
        )
        if pipeline == "handdrawn":
            builder = self._handdrawn_builder
            material = as_handdrawn_spec(request.material)
        else:
            builder = self._principled_builder
            material = request.material
        builder.build(mat, material, context=context)

        self._cache[key] = mat
        return mat

    @staticmethod
    def assign(obj, mat) -> None:
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

    @staticmethod
    def _material_name(name: str, cache_key: str) -> str:
        suffix = f"{abs(hash(cache_key)) & 0xFFFFFFFF:08x}"
        return f"{name}_{suffix}"

    @staticmethod
    def _cache_key(request: MaterialRequest) -> str:
        spec = as_material_spec(request.material)
        pipeline = str(request.pipeline).strip().lower()
        handdrawn_sig = "none"
        if pipeline == "handdrawn":
            hd = as_handdrawn_spec(request.material)
            light_dir = ",".join(f"{float(v):.4f}" for v in hd.light_direction)
            outline2 = ",".join(f"{float(v):.4f}" for v in hd.outline_secondary_color)
            handdrawn_sig = (
                f"jd={float(hd.jmol_desaturate):.4f}|"
                f"jl={float(hd.jmol_lighten):.4f}|"
                f"ld={light_dir}|sa={float(hd.shadow_area):.4f}|"
                f"ss={float(hd.shadow_strength):.4f}|sf={float(hd.shadow_softness):.4f}|"
                f"hs={float(hd.highlight_strength):.4f}|"
                f"os={float(hd.outline_surface):.4f}|om={float(hd.outline_molecule):.4f}|"
                f"ob={float(hd.outline_bond):.4f}|"
                f"ot={float(hd.outline_secondary_thickness):.4f}|oc={outline2}"
            )
        return (
            f"{request.name}|{request.style_name}|{request.role}|{pipeline}|{handdrawn_sig}|"
            f"{tuple(round(v, 6) for v in spec.color)}|"
            f"{spec.roughness:.6f}|{spec.specular:.6f}|{spec.metallic:.6f}|"
            f"{'none' if spec.ior is None else f'{spec.ior:.6f}'}|"
            f"{spec.coat:.6f}|{spec.coat_roughness:.6f}|{spec.specular_tint:.6f}|{spec.alpha:.6f}"
        )
