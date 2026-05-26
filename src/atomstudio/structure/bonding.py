from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import TYPE_CHECKING, Iterable

import numpy as np
from scipy.spatial import cKDTree

from atomstudio.structure.bond import Bond
from atomstudio.structure.data import (
    PairBondDistanceRange,
    VESTA_HYDROGEN_BOND_DISTANCE_RANGES,
    VESTA_PAIR_BOND_DISTANCE_RANGES,
    normalize_pair_key,
)
from atomstudio.structure.structure import Structure

if TYPE_CHECKING:
    from atomstudio.config import BondingConfig


def _pair_key(symbol_a: str, symbol_b: str) -> str:
    return normalize_pair_key(f"{symbol_a}-{symbol_b}")


def _explicit_pair_distance_ranges(bond_config: "BondingConfig") -> dict[str, PairBondDistanceRange]:
    raw = getattr(bond_config, "pair_distances", {})
    out: dict[str, PairBondDistanceRange] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        if isinstance(value, dict):
            raw_min = value.get("min_distance", value.get("min", 0.0))
            raw_max = value.get("max_distance", value.get("max"))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            raw_min, raw_max = value
        else:
            continue
        if raw_max is None:
            continue
        out[normalize_pair_key(str(key))] = (float(raw_min), float(raw_max))
    return out


def _disabled_pair_keys(bond_config: "BondingConfig") -> set[str]:
    out: set[str] = set()
    raw = getattr(bond_config, "disabled_pairs", ())
    if not isinstance(raw, (list, tuple, set)):
        return out
    for key in raw:
        try:
            out.add(normalize_pair_key(str(key)))
        except ValueError:
            continue
    return out


def _explicit_pair_orders(bond_config: "BondingConfig") -> dict[str, int]:
    raw = getattr(bond_config, "order_rules", {})
    out: dict[str, int] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        try:
            order = int(value.get("order") if isinstance(value, dict) else value)
        except (TypeError, ValueError):
            continue
        if order in {1, 2, 3}:
            out[normalize_pair_key(str(key))] = order
    return out


def _resolve_distance_range(
    *,
    symbol_a: str,
    symbol_b: str,
    explicit_ranges: dict[str, PairBondDistanceRange],
    disabled_pairs: set[str],
) -> PairBondDistanceRange | None:
    key = _pair_key(symbol_a, symbol_b)
    if key in disabled_pairs:
        return None
    if key in explicit_ranges:
        return explicit_ranges[key]
    return VESTA_PAIR_BOND_DISTANCE_RANGES.get(key)


@dataclass
class BondResult:
    bonds: list[Bond] = field(default_factory=list)


class BondEngine:
    def compute(self, structure: Structure, bond_config: "BondingConfig") -> list[Bond]:
        mode = str(bond_config.mode).lower()
        bonds: list[Bond] = []
        if mode in {"covalent", "mixed", "all"}:
            bonds.extend(self.compute_covalent(structure, bond_config))
        if bool(bond_config.hbond.enabled):
            bonds.extend(self.compute_hydrogen_bonds(structure, bond_config, covalent_bonds=bonds))

        return self.assign_ids(bonds)

    def compute_covalent(self, structure: Structure, bond_config: "BondingConfig") -> list[Bond]:
        mode = str(bond_config.mode).lower()
        if mode not in {"covalent", "mixed", "all"}:
            return []
        return self._compute_covalent_pairwise(structure, bond_config)

    def compute_covalent_pairs(
        self,
        structure: Structure,
        bond_config: "BondingConfig",
        pair_keys: Iterable[str],
    ) -> list[Bond]:
        mode = str(bond_config.mode).lower()
        if mode not in {"covalent", "mixed", "all"}:
            return []
        normalized = {normalize_pair_key(str(pair_key)) for pair_key in pair_keys}
        if not normalized:
            return []
        return self._compute_covalent_pairwise(structure, bond_config, pair_keys=normalized)

    def compute_hydrogen_bonds(
        self,
        structure: Structure,
        bond_config: "BondingConfig",
        *,
        covalent_bonds: list[Bond],
    ) -> list[Bond]:
        if not bool(bond_config.hbond.enabled):
            return []
        return self._compute_hbond(structure, bond_config, covalent_bonds=covalent_bonds)

    @staticmethod
    def assign_ids(bonds: Iterable[Bond]) -> list[Bond]:
        out: list[Bond] = []
        for idx, bond in enumerate(bonds):
            bond.id = int(idx)
            out.append(bond)
        return out

    def _compute_covalent_pairwise(
        self,
        structure: Structure,
        bond_config: "BondingConfig",
        *,
        pair_keys: set[str] | None = None,
    ) -> list[Bond]:
        # include_periodic_images is intentionally ignored here.
        # Periodic visualization continuity is handled by boundary padding.
        explicit_ranges = _explicit_pair_distance_ranges(bond_config)
        disabled_pairs = _disabled_pair_keys(bond_config)
        pair_orders = _explicit_pair_orders(bond_config)

        symbols = list(structure.symbols)
        if len(symbols) < 2:
            return []
        positions = np.asarray(structure.positions, dtype=float).reshape((-1, 3))
        if positions.shape[0] != len(symbols):
            return []
        ranges_by_symbol_pair: dict[tuple[str, str], PairBondDistanceRange] = {}
        for left in sorted(set(symbols)):
            for right in sorted(set(symbols)):
                if pair_keys is not None and _pair_key(left, right) not in pair_keys:
                    continue
                distance_range = _resolve_distance_range(
                    symbol_a=left,
                    symbol_b=right,
                    explicit_ranges=explicit_ranges,
                    disabled_pairs=disabled_pairs,
                )
                if distance_range is None:
                    continue
                ranges_by_symbol_pair[(left, right)] = (
                    float(distance_range[0]),
                    float(distance_range[1]),
                )
        if not ranges_by_symbol_pair:
            return []
        max_cutoff = max(max_distance for _min_distance, max_distance in ranges_by_symbol_pair.values())
        if max_cutoff <= 0.0:
            return []

        pairs = cKDTree(positions).query_pairs(
            r=float(max_cutoff) + 1e-8,
            output_type="ndarray",
        )
        if pairs.size == 0:
            return []
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]

        out: list[Bond] = []
        for raw_i, raw_j in pairs:
            i = int(raw_i)
            j = int(raw_j)
            distance_range = ranges_by_symbol_pair.get((symbols[i], symbols[j]))
            if distance_range is None:
                continue
            min_distance, max_distance = distance_range
            pi = positions[i]
            pj = positions[j]
            d = sqrt(
                (float(pi[0]) - float(pj[0])) ** 2
                + (float(pi[1]) - float(pj[1])) ** 2
                + (float(pi[2]) - float(pj[2])) ** 2
            )
            if d < min_distance:
                continue
            if d <= float(max_distance) + 1e-8:
                out.append(
                    Bond(
                        id=0,
                        a=i,
                        b=j,
                        bond_type="covalent",
                        order=pair_orders.get(_pair_key(symbols[i], symbols[j]), 1),
                        distance=float(d),
                    )
                )
        return out

    def _compute_hbond(
        self,
        structure: Structure,
        bond_config: "BondingConfig",
        *,
        covalent_bonds: list[Bond],
    ) -> list[Bond]:
        hbond_cfg = bond_config.hbond
        symbols = list(structure.symbols)
        if len(symbols) < 3:
            return []
        positions = np.asarray(structure.positions, dtype=float).reshape((-1, 3))
        if positions.shape[0] != len(symbols):
            return []

        ranges = _explicit_hbond_distance_ranges(hbond_cfg) or dict(VESTA_HYDROGEN_BOND_DISTANCE_RANGES)
        if not ranges:
            max_distance = float(getattr(hbond_cfg, "max_distance", 0.0))
            if max_distance <= 0.0:
                return []
            ranges = {
                _pair_key("H", acceptor): (1.2, max_distance)
                for acceptor in getattr(hbond_cfg, "acceptors", ())
            }
        max_cutoff = max(max_distance for _min_distance, max_distance in ranges.values())
        if max_cutoff <= 0.0:
            return []

        hydrogen_symbols = {"H", "D"}
        donor_symbols = {str(value) for value in getattr(hbond_cfg, "donors", ())}
        acceptor_symbols = {str(value) for value in getattr(hbond_cfg, "acceptors", ())}
        covalent_pairs = {tuple(sorted((int(bond.a), int(bond.b)))) for bond in covalent_bonds}
        donors_by_hydrogen: dict[int, list[int]] = {idx: [] for idx, symbol in enumerate(symbols) if symbol in hydrogen_symbols}
        for bond in covalent_bonds:
            a = int(bond.a)
            b = int(bond.b)
            if a >= len(symbols) or b >= len(symbols):
                continue
            if symbols[a] in hydrogen_symbols and symbols[b] in donor_symbols:
                donors_by_hydrogen.setdefault(a, []).append(b)
            elif symbols[b] in hydrogen_symbols and symbols[a] in donor_symbols:
                donors_by_hydrogen.setdefault(b, []).append(a)
        donors_by_hydrogen = {idx: donors for idx, donors in donors_by_hydrogen.items() if donors}
        if not donors_by_hydrogen:
            return []

        min_angle = float(getattr(hbond_cfg, "min_angle_deg", 120.0))
        pairs = cKDTree(positions).query_pairs(
            r=float(max_cutoff) + 1e-8,
            output_type="ndarray",
        )
        if pairs.size == 0:
            return []
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]

        out: list[Bond] = []
        used: set[tuple[int, int]] = set()
        for raw_i, raw_j in pairs:
            i = int(raw_i)
            j = int(raw_j)
            h_idx, acceptor_idx = _hydrogen_acceptor_indices(i, j, symbols, hydrogen_symbols, acceptor_symbols)
            if h_idx is None or acceptor_idx is None:
                continue
            if tuple(sorted((h_idx, acceptor_idx))) in covalent_pairs:
                continue
            donors = donors_by_hydrogen.get(h_idx)
            if not donors:
                continue
            distance_range = ranges.get(_pair_key(symbols[h_idx], symbols[acceptor_idx]))
            if distance_range is None:
                continue
            min_distance, max_distance = distance_range
            d = _distance(positions[h_idx], positions[acceptor_idx])
            if d < float(min_distance) or d > float(max_distance) + 1e-8:
                continue
            if not any(
                donor_idx != acceptor_idx
                and _angle_degrees(positions[donor_idx], positions[h_idx], positions[acceptor_idx]) >= min_angle
                for donor_idx in donors
            ):
                continue
            key = tuple(sorted((h_idx, acceptor_idx)))
            if key in used:
                continue
            used.add(key)
            out.append(
                Bond(
                    id=0,
                    a=h_idx,
                    b=acceptor_idx,
                    bond_type="hydrogen",
                    order=1,
                    distance=float(d),
                    metadata={"preview_bond_style": "dashed"},
                )
            )
        return out


def _explicit_hbond_distance_ranges(hbond_config: object) -> dict[str, PairBondDistanceRange]:
    raw = getattr(hbond_config, "pair_distances", {})
    out: dict[str, PairBondDistanceRange] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        if isinstance(value, dict):
            raw_min = value.get("min_distance", value.get("min", 0.0))
            raw_max = value.get("max_distance", value.get("max"))
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            raw_min, raw_max = value
        else:
            continue
        if raw_max is None:
            continue
        out[normalize_pair_key(str(key))] = (float(raw_min), float(raw_max))
    return out


def _hydrogen_acceptor_indices(
    i: int,
    j: int,
    symbols: list[str],
    hydrogen_symbols: set[str],
    acceptor_symbols: set[str],
) -> tuple[int | None, int | None]:
    if symbols[i] in hydrogen_symbols and symbols[j] in acceptor_symbols:
        return i, j
    if symbols[j] in hydrogen_symbols and symbols[i] in acceptor_symbols:
        return j, i
    return None, None


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return float(np.linalg.norm(delta))


def _angle_degrees(donor: np.ndarray, hydrogen: np.ndarray, acceptor: np.ndarray) -> float:
    left = np.asarray(donor, dtype=float) - np.asarray(hydrogen, dtype=float)
    right = np.asarray(acceptor, dtype=float) - np.asarray(hydrogen, dtype=float)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))
