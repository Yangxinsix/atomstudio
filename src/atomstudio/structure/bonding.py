from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import TYPE_CHECKING

from atomstudio.structure.bond import Bond
from atomstudio.structure.data import PRESET_PAIR_BOND_CUTOFFS, normalize_pair_key
from atomstudio.structure.structure import Structure

try:
    from ase.data import atomic_numbers, covalent_radii
except Exception:  # pragma: no cover
    atomic_numbers = {}
    covalent_radii = []

if TYPE_CHECKING:
    from atomstudio.config import BondingConfig


# NeighborList is intentionally not used in this project path:
# we compute bonds directly from displayed Cartesian coordinates.
ASE_AVAILABLE = False


COVALENT_RADII_FALLBACK = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Si": 1.11,
    "Fe": 1.16,
    "Cu": 1.32,
    "Zn": 1.22,
    "Na": 1.66,
    "K": 2.03,
    "Ca": 1.76,
    "Sr": 1.95,
    "Mg": 1.41,
    "Ti": 1.60,
}


def covalent_radius(symbol: str) -> float:
    number = atomic_numbers.get(symbol)
    if number is not None and number < len(covalent_radii):
        return float(covalent_radii[number])
    return float(COVALENT_RADII_FALLBACK.get(symbol, 0.9))


def _pair_key(symbol_a: str, symbol_b: str) -> str:
    return normalize_pair_key(f"{symbol_a}-{symbol_b}")


def _explicit_pair_cutoffs(bond_config: "BondingConfig") -> dict[str, float]:
    raw = getattr(bond_config, "pair_cutoffs", {})
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        out[normalize_pair_key(str(key))] = float(value)
    return out


def _resolve_cutoff(
    *,
    symbol_a: str,
    symbol_b: str,
    bond_config: "BondingConfig",
    explicit_cutoffs: dict[str, float],
) -> float | None:
    key = _pair_key(symbol_a, symbol_b)
    if explicit_cutoffs:
        # Pair-cutoff mode: unspecified pair does not bond.
        return explicit_cutoffs.get(key)
    preset = PRESET_PAIR_BOND_CUTOFFS.get(key)
    if preset is not None:
        return float(preset)
    # Compatibility fallback for pairs not in preset table.
    return (covalent_radius(symbol_a) + covalent_radius(symbol_b)) * float(bond_config.cutoff_scale)


@dataclass
class BondResult:
    bonds: list[Bond] = field(default_factory=list)


class BondEngine:
    def compute(self, structure: Structure, bond_config: "BondingConfig") -> list[Bond]:
        mode = str(bond_config.mode).lower()
        bonds: list[Bond] = []
        if mode in {"covalent", "mixed", "all"}:
            bonds.extend(self._compute_covalent_pairwise(structure, bond_config))
        if bool(bond_config.hbond.enabled):
            bonds.extend(self._compute_hbond(structure, bond_config))

        out: list[Bond] = []
        for idx, bond in enumerate(bonds):
            bond.id = int(idx)
            out.append(bond)
        return out

    def _compute_covalent_pairwise(self, structure: Structure, bond_config: "BondingConfig") -> list[Bond]:
        # include_periodic_images is intentionally ignored here.
        # Periodic visualization continuity is handled by boundary padding.
        min_distance = float(bond_config.min_distance)
        explicit_cutoffs = _explicit_pair_cutoffs(bond_config)

        symbols = structure.symbols
        positions = structure.positions
        out: list[Bond] = []
        for i in range(len(symbols)):
            si = symbols[i]
            pi = positions[i]
            for j in range(i + 1, len(symbols)):
                sj = symbols[j]
                cutoff = _resolve_cutoff(
                    symbol_a=si,
                    symbol_b=sj,
                    bond_config=bond_config,
                    explicit_cutoffs=explicit_cutoffs,
                )
                if cutoff is None:
                    continue
                pj = positions[j]
                d = sqrt((pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2 + (pi[2] - pj[2]) ** 2)
                if d < min_distance:
                    continue
                if d <= float(cutoff) + 1e-8:
                    out.append(Bond(id=0, a=i, b=j, bond_type="covalent", order=1, distance=float(d)))
        return out

    def _compute_hbond(self, structure: Structure, bond_config: "BondingConfig") -> list[Bond]:
        _ = (structure, bond_config)
        return []
