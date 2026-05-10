from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from atomstudio.scene.materials.specs import MaterialLike


@dataclass
class MaterialBuildContext:
    style_name: str = "default"
    role: str = "object"


class MaterialBuilder(Protocol):
    def build(self, mat, material: MaterialLike, *, context: MaterialBuildContext) -> None:
        ...
