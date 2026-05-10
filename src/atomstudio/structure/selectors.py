from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field


def norm_symbol_pair(a: str, b: str) -> str:
    x, y = sorted((str(a), str(b)))
    return f"{x}-{y}"


def norm_index_pair(i: int, j: int) -> str:
    a, b = sorted((int(i), int(j)))
    return f"{a}-{b}"


@dataclass
class AtomSelector:
    symbol: str | None = None
    symbols: list[str] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    index_range: tuple[int, int] | None = None
    z_range: tuple[float, float] | None = None
    tags: list[str] = field(default_factory=list)
    _indices_set: set[int] = field(default_factory=set, init=False, repr=False, compare=False)
    _symbols_set: set[str] = field(default_factory=set, init=False, repr=False, compare=False)
    _tags_set: set[str] = field(default_factory=set, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._symbols_set = {str(v) for v in self.symbols}
        self._indices_set = {int(v) for v in self.indices}
        self._tags_set = {str(v) for v in self.tags}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AtomSelector":
        src = {} if data is None else dict(data)
        return cls(
            symbol=str(src["symbol"]) if src.get("symbol") is not None else None,
            symbols=[str(v) for v in src.get("symbols", [])],
            indices=[int(v) for v in src.get("indices", [])],
            index_range=_int_range(src.get("index_range")),
            z_range=_float_range(src.get("z_range")),
            tags=[str(v) for v in src.get("tags", [])],
        )

    def matches(self, index: int, symbol: str, position: tuple[float, float, float], tag: str | None = None) -> bool:
        if self.symbol is not None and symbol != self.symbol:
            return False
        if self._symbols_set and symbol not in self._symbols_set:
            return False
        if self._indices_set and int(index) not in self._indices_set:
            return False
        if self.index_range is not None:
            lo, hi = self.index_range
            if not (lo <= index <= hi):
                return False
        if self.z_range is not None:
            z = float(position[2])
            if not (self.z_range[0] <= z <= self.z_range[1]):
                return False
        if self._tags_set:
            if tag is None or tag not in self._tags_set:
                return False
        return True


@dataclass
class BondSelector:
    pair: str | None = None
    pairs: list[str] = field(default_factory=list)
    bond_order: int | None = None
    distance_range: tuple[float, float] | None = None
    index_pairs: list[tuple[int, int]] = field(default_factory=list)
    index_range: tuple[int, int] | None = None
    _pair_key: str | None = field(default=None, init=False, repr=False, compare=False)
    _pair_keys: set[str] = field(default_factory=set, init=False, repr=False, compare=False)
    _index_pair_keys: set[str] = field(default_factory=set, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._pair_key = _normalize_symbol_pair_key(self.pair)
        self._pair_keys = {key for key in (_normalize_symbol_pair_key(v) for v in self.pairs) if key is not None}
        self._index_pair_keys = {norm_index_pair(a, b) for a, b in self.index_pairs}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BondSelector":
        src = {} if data is None else dict(data)
        return cls(
            pair=str(src["pair"]) if src.get("pair") is not None else None,
            pairs=[str(v) for v in src.get("pairs", [])],
            bond_order=int(src["bond_order"]) if src.get("bond_order") is not None else None,
            distance_range=_float_range(src.get("distance_range")),
            index_pairs=[
                (int(v[0]), int(v[1]))
                for v in src.get("index_pairs", [])
                if isinstance(v, (list, tuple)) and len(v) == 2
            ],
            index_range=_int_range(src.get("index_range")),
        )

    def matches(
        self,
        bond_index: int,
        atom_i: int,
        atom_j: int,
        symbol_i: str,
        symbol_j: str,
        distance: float,
        bond_order: int = 1,
    ) -> bool:
        pair_key = norm_symbol_pair(symbol_i, symbol_j)
        index_pair = norm_index_pair(atom_i, atom_j)

        if self._pair_key is not None:
            if pair_key != self._pair_key:
                return False

        if self._pair_keys and pair_key not in self._pair_keys:
            return False

        if self.bond_order is not None and int(bond_order) != int(self.bond_order):
            return False

        if self.distance_range is not None:
            lo, hi = self.distance_range
            if not (lo <= distance <= hi):
                return False

        if self._index_pair_keys and index_pair not in self._index_pair_keys:
            return False

        if self.index_range is not None:
            lo, hi = self.index_range
            if not (lo <= bond_index <= hi):
                return False

        return True


@dataclass
class PolyhedraSelector:
    center_symbol: str | None = None
    center_symbols: list[str] = field(default_factory=list)
    center_indices: list[int] = field(default_factory=list)
    index_range: tuple[int, int] | None = None
    neighbor_count_range: tuple[int, int] | None = None
    _center_symbols_set: set[str] = field(default_factory=set, init=False, repr=False, compare=False)
    _center_indices_set: set[int] = field(default_factory=set, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._center_symbols_set = {str(v) for v in self.center_symbols}
        self._center_indices_set = {int(v) for v in self.center_indices}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PolyhedraSelector":
        src = {} if data is None else dict(data)
        return cls(
            center_symbol=str(src["center_symbol"]) if src.get("center_symbol") is not None else None,
            center_symbols=[str(v) for v in src.get("center_symbols", [])],
            center_indices=[int(v) for v in src.get("center_indices", [])],
            index_range=_int_range(src.get("index_range")),
            neighbor_count_range=_int_range(src.get("neighbor_count_range")),
        )

    def matches(
        self,
        polyhedron_index: int,
        center_index: int,
        center_symbol: str,
        neighbor_count: int,
    ) -> bool:
        if self.center_symbol is not None and center_symbol != self.center_symbol:
            return False
        if self._center_symbols_set and center_symbol not in self._center_symbols_set:
            return False
        if self._center_indices_set and int(center_index) not in self._center_indices_set:
            return False
        if self.index_range is not None:
            lo, hi = self.index_range
            if not (lo <= int(polyhedron_index) <= hi):
                return False
        if self.neighbor_count_range is not None:
            lo, hi = self.neighbor_count_range
            if not (lo <= int(neighbor_count) <= hi):
                return False
        return True


def _int_range(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return int(value[0]), int(value[1])


def _float_range(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return float(value[0]), float(value[1])


def _normalize_symbol_pair_key(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value)
    if "-" not in key:
        return key
    left, right = key.split("-", 1)
    return norm_symbol_pair(left, right)
