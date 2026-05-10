from __future__ import annotations

from dataclasses import dataclass, field

from atomstudio.style.data import CPK_RGBA_COLORS, JMOL_RGBA_COLORS, VESTA_RGBA_COLORS


@dataclass
class ColorStyle:
    name: str
    element_colors: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    fallback_atom: tuple[float, float, float, float] = (0.65, 0.65, 0.68, 1.0)
    surface_palette: list[tuple[float, float, float, float]] = field(default_factory=list)

    def color_for(self, symbol: str) -> tuple[float, float, float, float]:
        return self.element_colors.get(symbol, self.fallback_atom)


JMOL_COLOR_STYLE = ColorStyle(
    name="jmol",
    element_colors=dict(JMOL_RGBA_COLORS),
    fallback_atom=(0.65, 0.65, 0.68, 1.0),
)

JMOL_SOFT_COLOR_STYLE = ColorStyle(
    name="jmol_soft",
    element_colors={**JMOL_COLOR_STYLE.element_colors, "H": (1.0, 0.8, 0.8, 1.0)},
    fallback_atom=(0.65, 0.65, 0.68, 1.0),
)

CPK_COLOR_STYLE = ColorStyle(
    name="cpk",
    element_colors=dict(CPK_RGBA_COLORS),
    fallback_atom=(0.78, 0.78, 0.78, 1.0),
)

VESTA_COLOR_STYLE = ColorStyle(
    name="vesta",
    element_colors=dict(VESTA_RGBA_COLORS),
    fallback_atom=(0.70, 0.70, 0.72, 1.0),
)

COLOR_STYLE_LIBRARY: dict[str, ColorStyle] = {
    "jmol": JMOL_COLOR_STYLE,
    "jmol_soft": JMOL_SOFT_COLOR_STYLE,
    "cpk": CPK_COLOR_STYLE,
    "vesta": VESTA_COLOR_STYLE,
}
