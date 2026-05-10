from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atomstudio.color_utils import coerce_color_fields
from atomstudio.scene.materials.specs import MaterialLike
from atomstudio.structure.bond import add_cylinder_between


@coerce_color_fields("color", label_prefix="cell")
@dataclass
class Cell:
    vectors: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0, 0.0] for _ in range(3)])
    pbc: tuple[bool, bool, bool] = (False, False, False)
    show: bool = False
    radius: float = 0.04
    style: str | None = None
    material: MaterialLike | None = None
    color: tuple[float, float, float, float] | None = None

    @classmethod
    def build(
        cls,
        cell: "Cell",
        *,
        radius: float,
        mat=None,
        vertices: int = 12,
        collection=None,
    ) -> list[Any]:
        out: list[Any] = []
        for idx, (p1, p2) in enumerate(cell_edges(cell.vectors)):
            obj = add_cylinder_between(
                p1,
                p2,
                float(radius),
                mat,
                f"CellEdge_{idx}",
                int(vertices),
                collection=collection,
            )
            if obj is not None:
                out.append(obj)
        return out


def cell_edges(cell: list[list[float]]) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    a, b, c = cell
    o = (0.0, 0.0, 0.0)
    p1 = tuple(a)
    p2 = tuple(b)
    p3 = tuple(c)
    p4 = (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    p5 = (a[0] + c[0], a[1] + c[1], a[2] + c[2])
    p6 = (b[0] + c[0], b[1] + c[1], b[2] + c[2])
    p7 = (a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2])
    return [
        (o, p1),
        (o, p2),
        (o, p3),
        (p1, p4),
        (p1, p5),
        (p2, p4),
        (p2, p6),
        (p3, p5),
        (p3, p6),
        (p4, p7),
        (p5, p7),
        (p6, p7),
    ]


def has_cell(cell: list[list[float]]) -> bool:
    return any(abs(float(v)) > 1e-8 for row in cell for v in row)
