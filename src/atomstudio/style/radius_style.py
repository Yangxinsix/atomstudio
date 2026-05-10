from __future__ import annotations

from dataclasses import dataclass, field

from atomstudio.style.data import (
    ATOMIC_RADII,
    COVALENT_RADII,
    IONIC_RADII,
    VDW_RADII,
)


@dataclass
class RadiusStyle:
    name: str
    element_radii: dict[str, float] = field(default_factory=dict)
    fallback_radius: float = 0.5

    def radius_for(self, symbol: str) -> float:
        return float(self.element_radii.get(symbol, self.fallback_radius))


ATOMIC_RADIUS_STYLE = RadiusStyle(
    name="atomic",
    element_radii=dict(ATOMIC_RADII),
    fallback_radius=0.5,
)

IONIC_RADIUS_STYLE = RadiusStyle(
    name="ionic",
    element_radii=dict(IONIC_RADII),
    fallback_radius=0.5,
)

VDW_RADIUS_STYLE = RadiusStyle(
    name="vdw",
    element_radii=dict(VDW_RADII),
    fallback_radius=1.7,
)

COVALENT_RADIUS_STYLE = RadiusStyle(
    name="covalent",
    element_radii=dict(COVALENT_RADII),
    fallback_radius=0.9,
)


RADIUS_STYLE_LIBRARY: dict[str, RadiusStyle] = {
    "atomic": ATOMIC_RADIUS_STYLE,
    "ionic": IONIC_RADIUS_STYLE,
    "vdw": VDW_RADIUS_STYLE,
    "covalent": COVALENT_RADIUS_STYLE,
}
