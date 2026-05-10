__all__ = [
    "BaseMaterialSpec",
    "HandDrawnMaterialSpec",
    "MaterialLike",
    "MaterialRequest",
    "MaterialRegistry",
    "MaterialSpec",
    "as_handdrawn_spec",
    "as_material_spec",
    "handdrawn_spec_from_any",
    "material_from_dict",
]


def __getattr__(name: str):
    if name in {
        "BaseMaterialSpec",
        "HandDrawnMaterialSpec",
        "MaterialLike",
        "MaterialSpec",
        "as_handdrawn_spec",
        "as_material_spec",
        "handdrawn_spec_from_any",
        "material_from_dict",
    }:
        from atomstudio.scene.materials import specs as _specs

        return getattr(_specs, name)
    if name == "MaterialRequest":
        from atomstudio.scene.materials.request import MaterialRequest

        return MaterialRequest
    if name == "MaterialRegistry":
        from atomstudio.scene.materials.registry import MaterialRegistry

        return MaterialRegistry
    raise AttributeError(name)
