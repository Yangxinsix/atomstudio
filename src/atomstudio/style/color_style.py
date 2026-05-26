from __future__ import annotations

import colorsys
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


def _mono_gray_colors() -> dict[str, tuple[float, float, float, float]]:
    colors = {
        symbol: (
            0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2],
            0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2],
            0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2],
            rgba[3],
        )
        for symbol, rgba in JMOL_COLOR_STYLE.element_colors.items()
    }
    colors.update(
        {
            "H": (0.92, 0.92, 0.92, 1.0),
            "C": (0.24, 0.24, 0.24, 1.0),
            "N": (0.42, 0.42, 0.42, 1.0),
            "O": (0.72, 0.72, 0.72, 1.0),
            "F": (0.66, 0.66, 0.66, 1.0),
            "P": (0.52, 0.52, 0.52, 1.0),
            "S": (0.62, 0.62, 0.62, 1.0),
            "Cl": (0.56, 0.56, 0.56, 1.0),
            "Br": (0.34, 0.34, 0.34, 1.0),
            "I": (0.30, 0.30, 0.30, 1.0),
            "Si": (0.46, 0.46, 0.46, 1.0),
            "Al": (0.58, 0.58, 0.58, 1.0),
            "Fe": (0.40, 0.40, 0.40, 1.0),
            "Cu": (0.48, 0.48, 0.48, 1.0),
            "Zn": (0.68, 0.68, 0.68, 1.0),
        }
    )
    return colors


MONO_GRAY_COLOR_STYLE = ColorStyle(
    name="mono_gray",
    element_colors=_mono_gray_colors(),
    fallback_atom=(0.58, 0.58, 0.58, 1.0),
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

def _smoothstep(value: float) -> float:
    x = max(0.0, min(1.0, float(value)))
    return x * x * (3.0 - 2.0 * x)


def _mix(a: float, b: float, t: float) -> float:
    x = max(0.0, min(1.0, float(t)))
    return (1.0 - x) * float(a) + x * float(b)


def _studio_tone_color(
    rgba: tuple[float, float, float, float],
    *,
    saturation: float,
    lightness_min: float,
    lightness_max: float,
    tint: tuple[float, float, float],
    tint_strength: float,
    white_cap: float,
) -> tuple[float, float, float, float]:
    red, green, blue, alpha = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
    hue, lightness, sat = colorsys.rgb_to_hls(red, green, blue)
    toned_lightness = _mix(float(lightness_min), float(lightness_max), _smoothstep(lightness))
    toned_sat = max(0.0, min(1.0, float(sat) * float(saturation)))
    red, green, blue = colorsys.hls_to_rgb(hue, toned_lightness, toned_sat)
    cap = max(0.0, min(1.0, float(white_cap)))
    red = min(cap, max(0.0, _mix(red, tint[0], tint_strength)))
    green = min(cap, max(0.0, _mix(green, tint[1], tint_strength)))
    blue = min(cap, max(0.0, _mix(blue, tint[2], tint_strength)))
    return (red, green, blue, alpha)


def _studio_tone_colors(
    base: ColorStyle,
    *,
    saturation: float,
    lightness_min: float,
    lightness_max: float,
    tint: tuple[float, float, float],
    tint_strength: float,
    white_cap: float,
) -> dict[str, tuple[float, float, float, float]]:
    return {
        symbol: _studio_tone_color(
            rgba,
            saturation=saturation,
            lightness_min=lightness_min,
            lightness_max=lightness_max,
            tint=tint,
            tint_strength=tint_strength,
            white_cap=white_cap,
        )
        for symbol, rgba in base.element_colors.items()
    }


_STUDIO_HIGHKEY_COLOR_KWARGS = {
    "saturation": 0.78,
    "lightness_min": 0.32,
    "lightness_max": 0.64,
    "tint": (0.58, 0.70, 0.80),
    "tint_strength": 0.18,
    "white_cap": 0.86,
}

STUDIO_HIGHKEY_COLOR_STYLE = ColorStyle(
    name="studio_highkey",
    element_colors=_studio_tone_colors(JMOL_COLOR_STYLE, **_STUDIO_HIGHKEY_COLOR_KWARGS),
    fallback_atom=_studio_tone_color(JMOL_COLOR_STYLE.fallback_atom, **_STUDIO_HIGHKEY_COLOR_KWARGS),
)

STUDIO_SOFT_COLOR_STYLE = ColorStyle(
    name="studio_soft",
    element_colors=dict(STUDIO_HIGHKEY_COLOR_STYLE.element_colors),
    fallback_atom=STUDIO_HIGHKEY_COLOR_STYLE.fallback_atom,
)

STUDIO_PRODUCT_COLOR_STYLE = ColorStyle(
    name="studio_product",
    element_colors=_studio_tone_colors(
        JMOL_COLOR_STYLE,
        saturation=0.66,
        lightness_min=0.38,
        lightness_max=0.80,
        tint=(0.76, 0.82, 0.86),
        tint_strength=0.16,
        white_cap=0.90,
    ),
    fallback_atom=_studio_tone_color(
        JMOL_COLOR_STYLE.fallback_atom,
        saturation=0.66,
        lightness_min=0.38,
        lightness_max=0.80,
        tint=(0.76, 0.82, 0.86),
        tint_strength=0.16,
        white_cap=0.90,
    ),
)

STUDIO_MONO_COLOR_STYLE = ColorStyle(
    name="studio_mono",
    element_colors={
        **MONO_GRAY_COLOR_STYLE.element_colors,
        "H": (0.70, 0.70, 0.68, 1.0),
        "C": (0.42, 0.45, 0.46, 1.0),
        "N": (0.46, 0.50, 0.52, 1.0),
        "O": (0.58, 0.58, 0.56, 1.0),
        "Pt": (0.30, 0.40, 0.50, 1.0),
    },
    fallback_atom=(0.46, 0.48, 0.48, 1.0),
)

COLOR_STYLE_LIBRARY: dict[str, ColorStyle] = {
    "jmol": JMOL_COLOR_STYLE,
    "jmol_soft": JMOL_SOFT_COLOR_STYLE,
    "mono_gray": MONO_GRAY_COLOR_STYLE,
    "cpk": CPK_COLOR_STYLE,
    "vesta": VESTA_COLOR_STYLE,
    "studio_highkey": STUDIO_HIGHKEY_COLOR_STYLE,
    "studio_soft": STUDIO_SOFT_COLOR_STYLE,
    "studio_product": STUDIO_PRODUCT_COLOR_STYLE,
    "studio_mono": STUDIO_MONO_COLOR_STYLE,
}
