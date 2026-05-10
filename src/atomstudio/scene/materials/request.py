from __future__ import annotations

from dataclasses import dataclass

from atomstudio.scene.materials.specs import MaterialLike


@dataclass(frozen=True)
class MaterialRequest:
    name: str
    pipeline: str
    role: str
    style_name: str
    material: MaterialLike

    @classmethod
    def principled(
        cls,
        *,
        name: str,
        material: MaterialLike,
        role: str = "object",
        style_name: str = "principled",
    ) -> "MaterialRequest":
        return cls(
            name=str(name),
            pipeline="principled",
            role=str(role),
            style_name=str(style_name),
            material=material,
        )

    @classmethod
    def handdrawn(
        cls,
        *,
        name: str,
        material: MaterialLike,
        role: str = "object",
        style_name: str = "handdrawn",
    ) -> "MaterialRequest":
        return cls(
            name=str(name),
            pipeline="handdrawn",
            role=str(role),
            style_name=str(style_name),
            material=material,
        )
