from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from atomstudio.color_utils import coerce_color_fields, rgba_from_any
from atomstudio.scene.lights.specs import LightSpec as LightConfig
from atomstudio.scene.materials.specs import (
    HandDrawnMaterialSpec,
    MaterialLike,
    as_handdrawn_spec,
    as_material_spec,
    material_from_dict,
)
from atomstudio.structure.selectors import AtomSelector, BondSelector, norm_index_pair, norm_symbol_pair


def _copy_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _jsonable_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable_config(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_jsonable_config(v) for v in value)
    if isinstance(value, tuple):
        return [_jsonable_config(v) for v in value]
    if isinstance(value, list):
        return [_jsonable_config(v) for v in value]
    return value


def _reject_unknown_keys(src: dict[str, Any], *, allowed: set[str], context: str) -> None:
    unknown = sorted(set(src) - allowed)
    if unknown:
        raise ValueError(f"Unknown {context} field(s): {', '.join(unknown)}")


def _to_bool3(value: Any) -> tuple[bool, bool, bool]:
    data = value if isinstance(value, (list, tuple)) else [False, False, False]
    if len(data) != 3:
        data = [False, False, False]
    return tuple(bool(v) for v in data)  # type: ignore[return-value]


def _to_cell3x3(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return [[0.0, 0.0, 0.0] for _ in range(3)]
    out: list[list[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            out.append([0.0, 0.0, 0.0])
        else:
            out.append([float(row[0]), float(row[1]), float(row[2])])
    return out


def _to_rgba(value: Any, fallback: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)) -> tuple[float, float, float, float]:
    return rgba_from_any(value, fallback=fallback)


def _to_float_tuple(value: Any, *, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != int(length):
        return None
    return tuple(float(item) for item in value)


def _normalize_atom_representation(value: Any, *, allow_none: bool = True, label: str = "representation") -> str | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{label} must be one of: ball_stick, space_filling")
    rep = str(value).strip().lower()
    if rep not in {"ball_stick", "space_filling"}:
        raise ValueError(f"{label} must be one of: ball_stick, space_filling")
    return rep


def _parse_space_filling_scale(value: Any) -> float | str:
    if value is None:
        return "auto"
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw == "auto":
            return "auto"
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError("structure.space_filling_scale must be > 0 or 'auto'.") from exc
    else:
        parsed = float(value)
    if parsed <= 0.0:
        raise ValueError("structure.space_filling_scale must be > 0 or 'auto'.")
    return parsed


def _parse_disabled_pair_keys(value: Any, *, context: str) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"{context} must be a list of element pairs like ['O-Ti'].")
    out: set[str] = set()
    for item in value:
        key = str(item).strip()
        if "-" not in key:
            raise ValueError(f"{context} keys must be element pairs like 'O-Ti'.")
        left, right = key.split("-", 1)
        out.add(norm_symbol_pair(left.strip(), right.strip()))
    return sorted(out)


def _parse_pair_order_rules(value: Any, *, context: str) -> dict[str, int]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping like {{'C-O': 2}}.")
    out: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if "-" not in key:
            raise ValueError(f"{context} keys must be element pairs like 'C-O'.")
        left, right = key.split("-", 1)
        if isinstance(raw_value, dict):
            raw_order = raw_value.get("order")
        else:
            raw_order = raw_value
        try:
            order = int(raw_order)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} values must be 1, 2, or 3.") from exc
        if order not in {1, 2, 3}:
            raise ValueError(f"{context} values must be 1, 2, or 3.")
        out[norm_symbol_pair(left.strip(), right.strip())] = order
    return dict(sorted(out.items()))


def _merge_material_override(material: MaterialLike, base_material: MaterialLike) -> MaterialLike:
    if isinstance(base_material, HandDrawnMaterialSpec):
        return as_handdrawn_spec(material, fallback=base_material)
    if isinstance(material, HandDrawnMaterialSpec):
        return as_handdrawn_spec(material)
    return as_material_spec(material, fallback=as_material_spec(base_material))


@dataclass
class MaterialRule:
    selector: AtomSelector | BondSelector
    material: MaterialLike

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, kind: str) -> "MaterialRule":
        src = {} if data is None else dict(data)
        selector: AtomSelector | BondSelector
        if kind == "atom":
            selector = AtomSelector.from_dict(_copy_dict(src.get("selector")))
        else:
            selector = BondSelector.from_dict(_copy_dict(src.get("selector")))
        return cls(selector=selector, material=material_from_dict(_copy_dict(src.get("material"))))


@dataclass
class AtomStylePresetConfig:
    color: tuple[float, float, float, float] | None = None
    radius: float | None = None
    material: MaterialLike | None = None
    representation: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AtomStylePresetConfig":
        src = {} if data is None else dict(data)
        material_data = src.get("material")
        return cls(
            color=_to_rgba(src.get("color")) if src.get("color") is not None else None,
            radius=float(src["radius"]) if src.get("radius") is not None else None,
            material=material_from_dict(_copy_dict(material_data)) if isinstance(material_data, dict) else None,
            representation=_normalize_atom_representation(src.get("representation"), label="style.atom_styles.*.representation"),
        )


@dataclass
class AtomStyleRuleConfig:
    selector: AtomSelector = field(default_factory=AtomSelector)
    style: str | None = None
    color: tuple[float, float, float, float] | None = None
    radius: float | None = None
    material: MaterialLike | None = None
    representation: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AtomStyleRuleConfig":
        src = {} if data is None else dict(data)
        selector = AtomSelector.from_dict(_copy_dict(src.get("selector")))
        material_data = src.get("material")
        return cls(
            selector=selector,
            style=str(src["style"]) if src.get("style") is not None else None,
            color=_to_rgba(src.get("color")) if src.get("color") is not None else None,
            radius=float(src["radius"]) if src.get("radius") is not None else None,
            material=material_from_dict(_copy_dict(material_data)) if isinstance(material_data, dict) else None,
            representation=_normalize_atom_representation(src.get("representation"), label="style.atom_style_rules[].representation"),
        )


@dataclass
class MaterialPolicy:
    atom_defaults: dict[str, MaterialLike] = field(default_factory=dict)
    atom_rules: list[MaterialRule] = field(default_factory=list)
    atom_overrides: dict[int, MaterialLike] = field(default_factory=dict)
    bond_defaults: dict[str, MaterialLike] = field(default_factory=dict)
    bond_rules: list[MaterialRule] = field(default_factory=list)
    bond_overrides_by_index: dict[int, MaterialLike] = field(default_factory=dict)
    bond_overrides_by_pair: dict[str, MaterialLike] = field(default_factory=dict)
    cell: MaterialLike | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MaterialPolicy":
        src = {} if data is None else dict(data)

        atom_defaults = {str(k): material_from_dict(_copy_dict(v)) for k, v in _copy_dict(src.get("atom_defaults")).items()}
        atom_rules = [MaterialRule.from_dict(item if isinstance(item, dict) else {}, kind="atom") for item in src.get("atom_rules", [])]
        atom_overrides = {int(k): material_from_dict(_copy_dict(v)) for k, v in _copy_dict(src.get("atom_overrides")).items()}

        bond_defaults: dict[str, MaterialLike] = {}
        for k, v in _copy_dict(src.get("bond_defaults")).items():
            key = str(k)
            if "-" in key:
                key = norm_symbol_pair(*key.split("-", 1))
            bond_defaults[key] = material_from_dict(_copy_dict(v))

        bond_rules = [MaterialRule.from_dict(item if isinstance(item, dict) else {}, kind="bond") for item in src.get("bond_rules", [])]

        raw_overrides = _copy_dict(src.get("bond_overrides"))
        by_index_raw = _copy_dict(raw_overrides.get("by_index")) if raw_overrides else {}
        by_pair_raw = _copy_dict(raw_overrides.get("by_pair")) if raw_overrides else {}
        if not by_index_raw and not by_pair_raw:
            for k, v in raw_overrides.items():
                key = str(k)
                if key.isdigit():
                    by_index_raw[key] = v
                elif "-" in key:
                    by_pair_raw[key] = v

        bond_overrides_by_index = {int(k): material_from_dict(_copy_dict(v)) for k, v in by_index_raw.items()}

        bond_overrides_by_pair: dict[str, MaterialLike] = {}
        for k, v in by_pair_raw.items():
            key = str(k)
            if "-" in key:
                left, right = key.split("-", 1)
                if left.isdigit() and right.isdigit():
                    key = norm_index_pair(int(left), int(right))
                else:
                    key = norm_symbol_pair(left, right)
            bond_overrides_by_pair[key] = material_from_dict(_copy_dict(v))

        cell_data = src.get("cell")
        cell = material_from_dict(_copy_dict(cell_data)) if isinstance(cell_data, dict) else None

        return cls(
            atom_defaults=atom_defaults,
            atom_rules=atom_rules,
            atom_overrides=atom_overrides,
            bond_defaults=bond_defaults,
            bond_rules=bond_rules,
            bond_overrides_by_index=bond_overrides_by_index,
            bond_overrides_by_pair=bond_overrides_by_pair,
            cell=cell,
        )

    def resolve_atom_material(
        self,
        index: int,
        symbol: str,
        position: tuple[float, float, float],
        tag: str | None,
        base_material: MaterialLike,
        fallback: MaterialLike,
    ) -> MaterialLike:
        if index in self.atom_overrides:
            return _merge_material_override(self.atom_overrides[index], base_material)
        for rule in self.atom_rules:
            selector = rule.selector
            if isinstance(selector, AtomSelector) and selector.matches(index, symbol, position, tag):
                return _merge_material_override(rule.material, base_material)
        if symbol in self.atom_defaults:
            return _merge_material_override(self.atom_defaults[symbol], base_material)
        return fallback

    def resolve_bond_material(
        self,
        bond_index: int,
        atom_i: int,
        atom_j: int,
        symbol_i: str,
        symbol_j: str,
        distance: float,
        base_material: MaterialLike,
        fallback: MaterialLike,
    ) -> MaterialLike:
        if bond_index in self.bond_overrides_by_index:
            return _merge_material_override(self.bond_overrides_by_index[bond_index], base_material)

        idx_key = norm_index_pair(atom_i, atom_j)
        type_key = norm_symbol_pair(symbol_i, symbol_j)

        if idx_key in self.bond_overrides_by_pair:
            return _merge_material_override(self.bond_overrides_by_pair[idx_key], base_material)
        if type_key in self.bond_overrides_by_pair:
            return _merge_material_override(self.bond_overrides_by_pair[type_key], base_material)

        for rule in self.bond_rules:
            selector = rule.selector
            if isinstance(selector, BondSelector) and selector.matches(
                bond_index, atom_i, atom_j, symbol_i, symbol_j, distance, bond_order=1
            ):
                return _merge_material_override(rule.material, base_material)

        if type_key in self.bond_defaults:
            return _merge_material_override(self.bond_defaults[type_key], base_material)

        return fallback


@dataclass
class CellStyleConfig:
    show: bool = False
    radius: float = 0.01
    material: MaterialLike | None = None
    color: tuple[float, float, float, float] | None = None
    transparent: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CellStyleConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"show", "radius", "material", "color", "transparent"}, context="structure.cell_style")
        material_data = src.get("material")
        material = material_from_dict(_copy_dict(material_data)) if isinstance(material_data, dict) else None
        return cls(
            show=bool(src.get("show", False)),
            radius=float(src.get("radius", 0.01)),
            material=material,
            color=_to_rgba(src.get("color")) if src.get("color") is not None else None,
            transparent=bool(src.get("transparent", False)),
        )


@dataclass
class SurfaceOptions:
    symbols: list[str] = field(default_factory=list)
    layer_coloring: bool = False
    layer_tolerance: float = 0.35

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SurfaceOptions":
        src = {} if data is None else dict(data)
        return cls(
            symbols=[str(v) for v in src.get("symbols", [])],
            layer_coloring=bool(src.get("layer_coloring", False)),
            layer_tolerance=float(src.get("layer_tolerance", 0.35)),
        )


@dataclass
class HBondConfig:
    enabled: bool = True
    donors: list[str] = field(default_factory=lambda: ["N", "O"])
    acceptors: list[str] = field(default_factory=lambda: ["N", "O"])
    max_distance: float = 2.5
    min_angle_deg: float = 120.0
    pair_distances: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HBondConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(
            src,
            allowed={"enabled", "donors", "acceptors", "max_distance", "min_angle_deg", "pair_distances"},
            context="structure.bonding.hbond",
        )
        pair_distances_raw = src.get("pair_distances", {})
        if not isinstance(pair_distances_raw, dict):
            raise ValueError("structure.bonding.hbond.pair_distances must be a mapping like {'H-O': [1.2, 2.1]}.")
        pair_distances: dict[str, tuple[float, float]] = {}
        for raw_key, raw_value in pair_distances_raw.items():
            key = str(raw_key).strip()
            if "-" not in key:
                raise ValueError("structure.bonding.hbond.pair_distances keys must be element pairs like 'H-O'.")
            left, right = key.split("-", 1)
            norm_key = norm_symbol_pair(left.strip(), right.strip())
            if isinstance(raw_value, dict):
                raw_min = raw_value.get("min_distance", raw_value.get("min", 0.0))
                raw_max = raw_value.get("max_distance", raw_value.get("max"))
            elif isinstance(raw_value, (list, tuple)) and len(raw_value) == 2:
                raw_min, raw_max = raw_value
            else:
                raise ValueError("structure.bonding.hbond.pair_distances values must be [min, max] or mappings.")
            if raw_max is None:
                raise ValueError("structure.bonding.hbond.pair_distances values require max_distance.")
            min_distance = float(raw_min)
            max_distance = float(raw_max)
            if min_distance < 0.0 or max_distance <= min_distance:
                raise ValueError("structure.bonding.hbond.pair_distances values must satisfy 0 <= min < max.")
            pair_distances[norm_key] = (min_distance, max_distance)
        return cls(
            enabled=bool(src.get("enabled", True)),
            donors=[str(v) for v in src.get("donors", ["N", "O"])],
            acceptors=[str(v) for v in src.get("acceptors", ["N", "O"])],
            max_distance=float(src.get("max_distance", 2.5)),
            min_angle_deg=float(src.get("min_angle_deg", 120.0)),
            pair_distances=dict(sorted(pair_distances.items())),
        )


@dataclass
class BondingConfig:
    mode: str = "covalent"
    include_periodic_images: bool = False
    pair_distances: dict[str, tuple[float, float]] = field(default_factory=dict)
    disabled_pairs: list[str] = field(default_factory=list)
    hbond: HBondConfig = field(default_factory=HBondConfig)
    order_rules: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BondingConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(
            src,
            allowed={"mode", "include_periodic_images", "pair_distances", "disabled_pairs", "hbond", "order_rules"},
            context="structure.bonding",
        )
        pair_distances_raw = src.get("pair_distances", {})
        if not isinstance(pair_distances_raw, dict):
            raise ValueError("structure.bonding.pair_distances must be a mapping like {'O-Ti': [0.0, 2.3]}.")
        pair_distances: dict[str, tuple[float, float]] = {}
        for raw_key, raw_value in pair_distances_raw.items():
            key = str(raw_key).strip()
            if "-" not in key:
                raise ValueError("structure.bonding.pair_distances keys must be element pairs like 'O-Ti'.")
            left, right = key.split("-", 1)
            norm_key = norm_symbol_pair(left.strip(), right.strip())
            if isinstance(raw_value, dict):
                raw_min = raw_value.get("min_distance", raw_value.get("min", 0.0))
                raw_max = raw_value.get("max_distance", raw_value.get("max"))
            elif isinstance(raw_value, (list, tuple)) and len(raw_value) == 2:
                raw_min, raw_max = raw_value
            else:
                raise ValueError("structure.bonding.pair_distances values must be [min, max] or mappings.")
            if raw_max is None:
                raise ValueError("structure.bonding.pair_distances values require max_distance.")
            min_distance = float(raw_min)
            max_distance = float(raw_max)
            if min_distance < 0.0:
                raise ValueError("structure.bonding.pair_distances minimum values must be >= 0 (Angstrom).")
            if max_distance <= min_distance:
                raise ValueError("structure.bonding.pair_distances maximum values must be > minimum values.")
            pair_distances[norm_key] = (min_distance, max_distance)
        disabled_pairs = _parse_disabled_pair_keys(src.get("disabled_pairs", []), context="structure.bonding.disabled_pairs")
        return cls(
            mode=str(src.get("mode", "covalent")),
            include_periodic_images=bool(src.get("include_periodic_images", False)),
            pair_distances=dict(sorted(pair_distances.items())),
            disabled_pairs=disabled_pairs,
            hbond=HBondConfig.from_dict(_copy_dict(src.get("hbond"))),
            order_rules=_parse_pair_order_rules(src.get("order_rules", {}), context="structure.bonding.order_rules"),
        )


@dataclass
class PolyhedraRuleConfig:
    center_symbols: list[str] = field(default_factory=list)
    neighbor_symbols: list[str] = field(default_factory=list)
    min_neighbors: int = 4
    max_neighbors: int | None = None
    max_distance: float | None = None
    material: MaterialLike | None = None
    color: tuple[float, float, float, float] | None = None
    style: str | None = None
    show_edges: bool = False
    edge_radius: float | None = None
    edge_color: tuple[float, float, float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PolyhedraRuleConfig":
        src = {} if data is None else dict(data)
        center_symbols = [str(v) for v in src.get("center_symbols", [])]
        neighbor_symbols = [str(v) for v in src.get("neighbor_symbols", [])]
        material_data = src.get("material")
        edge_color_raw = src.get("edge_color")
        color_raw = src.get("color")

        max_neighbors = None
        if src.get("max_neighbors") is not None:
            max_neighbors = int(src.get("max_neighbors"))
            if max_neighbors <= 0:
                raise ValueError("structure.polyhedra.rules[].max_neighbors must be > 0 when provided.")
        min_neighbors = int(src.get("min_neighbors", 4))
        if min_neighbors <= 0:
            raise ValueError("structure.polyhedra.rules[].min_neighbors must be > 0.")
        if max_neighbors is not None and max_neighbors < min_neighbors:
            raise ValueError("structure.polyhedra.rules[].max_neighbors must be >= min_neighbors.")

        return cls(
            center_symbols=center_symbols,
            neighbor_symbols=neighbor_symbols,
            min_neighbors=min_neighbors,
            max_neighbors=max_neighbors,
            max_distance=float(src.get("max_distance")) if src.get("max_distance") is not None else None,
            material=material_from_dict(_copy_dict(material_data)) if isinstance(material_data, dict) else None,
            color=_to_rgba(color_raw) if color_raw is not None else None,
            style=str(src.get("style")) if src.get("style") is not None else None,
            show_edges=bool(src.get("show_edges", False)),
            edge_radius=float(src.get("edge_radius")) if src.get("edge_radius") is not None else None,
            edge_color=_to_rgba(edge_color_raw) if edge_color_raw is not None else None,
        )


@dataclass
class PolyhedraConfig:
    enabled: bool = False
    include_periodic_images: bool = True
    default_alpha: float = 0.35
    default_edge_radius: float = 0.015
    rules: list[PolyhedraRuleConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PolyhedraConfig":
        src = {} if data is None else dict(data)
        rules_raw = src.get("rules", [])
        if not isinstance(rules_raw, list):
            raise ValueError("structure.polyhedra.rules must be a list.")
        rules = [PolyhedraRuleConfig.from_dict(item if isinstance(item, dict) else {}) for item in rules_raw]
        return cls(
            enabled=bool(src.get("enabled", False)),
            include_periodic_images=bool(src.get("include_periodic_images", True)),
            default_alpha=float(src.get("default_alpha", 0.35)),
            default_edge_radius=float(src.get("default_edge_radius", 0.015)),
            rules=rules,
        )


@dataclass
class BoundaryConfig:
    enabled: bool = False
    window_frac: list[list[float]] = field(
        default_factory=lambda: [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    )
    eps: float = 1e-6

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BoundaryConfig":
        src = {} if data is None else dict(data)
        enabled = bool(src.get("enabled", False))
        eps = float(src.get("eps", 1e-6))
        if eps < 0.0:
            raise ValueError("structure.boundary.eps must be >= 0.")

        window_raw = src.get("window_frac", [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        if not isinstance(window_raw, (list, tuple)) or len(window_raw) != 3:
            raise ValueError("structure.boundary.window_frac must be a 3x2 array in fractional coordinates.")

        window: list[list[float]] = []
        for axis in range(3):
            row = window_raw[axis]
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ValueError("structure.boundary.window_frac must be a 3x2 array in fractional coordinates.")
            lo = float(row[0])
            hi = float(row[1])
            if lo > hi:
                raise ValueError("structure.boundary.window_frac axis range must satisfy min <= max.")
            window.append([lo, hi])
        return cls(enabled=enabled, window_frac=window, eps=eps)


@dataclass
class BoundaryAtomsConfig:
    enabled: bool = True
    sigma: float = 0.03

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BoundaryAtomsConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"enabled", "sigma"}, context="structure.boundary_atoms")
        sigma = float(src.get("sigma", 0.03))
        if sigma < 0.0 or sigma >= 0.5:
            raise ValueError("structure.boundary_atoms.sigma must satisfy 0 <= sigma < 0.5.")
        return cls(enabled=bool(src.get("enabled", True)), sigma=sigma)


@dataclass
class StructureConfig:
    representation: str = "auto"
    atom_scale: float = 1.0
    space_filling_scale: float | str = "auto"
    # Default follows VESTA ball-stick convention (40% of chosen atomic radii).
    # Ref: https://jp-minerals.org/vesta/en/doc/VESTAch5.html
    radii_scale: float = 0.40
    bond_radius: float = 0.08
    draw_bonds: bool | None = None
    draw_surface_bonds: bool = True
    draw_cell: bool = True
    model_rotation: str | None = None
    model_view: str = "top"
    sphere_segments: int = 32
    sphere_rings: int = 16
    bond_vertices: int = 20
    element_scale: dict[str, float] = field(default_factory=lambda: {"H": 1.0})
    cell_style: CellStyleConfig = field(default_factory=CellStyleConfig)
    surface_options: SurfaceOptions = field(default_factory=SurfaceOptions)
    bonding: BondingConfig = field(default_factory=BondingConfig)
    polyhedra: PolyhedraConfig = field(default_factory=PolyhedraConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    boundary_atoms: BoundaryAtomsConfig = field(default_factory=BoundaryAtomsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StructureConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(
            src,
            allowed={
                "representation",
                "atom_scale",
                "space_filling_scale",
                "radii_scale",
                "ball_stick_radius_scale",
                "bond_radius",
                "draw_bonds",
                "draw_surface_bonds",
                "draw_cell",
                "model_rotation",
                "model_view",
                "sphere_segments",
                "sphere_rings",
                "bond_vertices",
                "element_scale",
                "cell_style",
                "surface_options",
                "bonding",
                "polyhedra",
                "boundary",
                "boundary_atoms",
                "bond_cutoff_scale",
            },
            context="structure",
        )
        if "bond_cutoff_scale" in src:
            raise ValueError(
                "structure.bond_cutoff_scale has been removed. Use structure.bonding.pair_distances."
            )
        draw_bonds_raw = src.get("draw_bonds", None)
        draw_bonds = None if draw_bonds_raw is None else bool(draw_bonds_raw)
        bonding_raw = _copy_dict(src.get("bonding"))
        user_element_scale = {str(k): float(v) for k, v in _copy_dict(src.get("element_scale")).items()}
        element_scale = {"H": 1.0, **user_element_scale}
        # Backward-compatible input key: structure.ball_stick_radius_scale
        radii_scale_raw = src.get("radii_scale", src.get("ball_stick_radius_scale", 0.40))
        radii_scale = float(radii_scale_raw)
        if radii_scale <= 0.0:
            raise ValueError("structure.radii_scale must be > 0.")
        return cls(
            representation=str(src.get("representation", "auto")).lower(),
            atom_scale=float(src.get("atom_scale", 1.0)),
            space_filling_scale=_parse_space_filling_scale(src.get("space_filling_scale", "auto")),
            radii_scale=radii_scale,
            bond_radius=float(src.get("bond_radius", 0.08)),
            draw_bonds=draw_bonds,
            draw_surface_bonds=bool(src.get("draw_surface_bonds", True)),
            draw_cell=bool(src.get("draw_cell", True)),
            model_rotation=str(src["model_rotation"]) if src.get("model_rotation") is not None else None,
            model_view=str(src.get("model_view", "top")).strip().lower(),
            sphere_segments=int(src.get("sphere_segments", 32)),
            sphere_rings=int(src.get("sphere_rings", 16)),
            bond_vertices=int(src.get("bond_vertices", 20)),
            element_scale=element_scale,
            cell_style=CellStyleConfig.from_dict(_copy_dict(src.get("cell_style"))),
            surface_options=SurfaceOptions.from_dict(_copy_dict(src.get("surface_options"))),
            bonding=BondingConfig.from_dict(bonding_raw),
            polyhedra=PolyhedraConfig.from_dict(_copy_dict(src.get("polyhedra"))),
            boundary=BoundaryConfig.from_dict(_copy_dict(src.get("boundary"))),
            boundary_atoms=BoundaryAtomsConfig.from_dict(_copy_dict(src.get("boundary_atoms"))),
        )


@dataclass
class OutlineRoleConfig:
    enabled: bool | None = None
    thickness: float | None = None
    color: tuple[float, float, float, float] | None = None
    follow_atom_color: bool | None = None
    color_scale: float | None = None
    secondary_thickness: float | None = None
    secondary_color: tuple[float, float, float, float] | None = None
    ignore_occlusion: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutlineRoleConfig | None":
        if not isinstance(data, dict) or (not data):
            return None
        src = dict(data)
        return cls(
            enabled=bool(src["enabled"]) if "enabled" in src else None,
            thickness=float(src["thickness"]) if src.get("thickness") is not None else None,
            color=_to_rgba(src.get("color")) if src.get("color") is not None else None,
            follow_atom_color=bool(src["follow_atom_color"]) if "follow_atom_color" in src else None,
            color_scale=float(src["color_scale"]) if src.get("color_scale") is not None else None,
            secondary_thickness=(
                float(src["secondary_thickness"]) if src.get("secondary_thickness") is not None else None
            ),
            secondary_color=(
                _to_rgba(src.get("secondary_color")) if src.get("secondary_color") is not None else None
            ),
            ignore_occlusion=bool(src["ignore_occlusion"]) if "ignore_occlusion" in src else None,
        )


@dataclass
class OutlineConfig:
    enabled: bool = False
    thickness: float = 1.2
    color: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    atoms: OutlineRoleConfig | None = None
    bonds: OutlineRoleConfig | None = None
    cell: OutlineRoleConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutlineConfig":
        src = {} if data is None else dict(data)
        return cls(
            enabled=bool(src.get("enabled", False)),
            thickness=float(src.get("thickness", 1.2)),
            color=_to_rgba(src.get("color"), fallback=(0.2, 0.2, 0.2, 1.0)),
            atoms=OutlineRoleConfig.from_dict(_copy_dict(src.get("atoms"))),
            bonds=OutlineRoleConfig.from_dict(_copy_dict(src.get("bonds"))),
            cell=OutlineRoleConfig.from_dict(_copy_dict(src.get("cell"))),
        )


@dataclass
class HanddrawnStyleConfig:
    substrate_symbols: list[str] = field(default_factory=list)
    layer_coloring: bool = True
    layer_tolerance: float = 0.35
    substrate_palette: list[tuple[float, float, float, float]] = field(
        default_factory=lambda: [(0.70, 0.79, 0.66, 1.0), (0.62, 0.79, 0.90, 1.0)]
    )
    molecule_use_jmol: bool = True
    jmol_desaturate: float = 0.10
    jmol_lighten: float = 0.04
    bond_color_mode: str = "atom_pair_avg"
    light_direction: tuple[float, float, float] = (0.68, 0.36, 0.62)
    shadow_area: float = 0.34
    shadow_strength: float = 0.42
    shadow_softness: float = 0.12
    highlight_strength: float = 0.16
    highlight_direction: tuple[float, float, float] = (0.78, 0.62, 0.0)
    # 0..1, larger means longer arc
    highlight_arc_length: float = 0.22
    highlight_band_inner: float = 0.56
    highlight_band_outer: float = 0.90
    outline_surface: float = 2.0
    outline_molecule: float = 2.4
    outline_bond: float = 1.6
    outline_secondary_thickness: float = 0.8
    outline_secondary_color: tuple[float, float, float, float] = (0.76, 0.82, 0.92, 1.0)
    background: tuple[float, float, float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HanddrawnStyleConfig":
        src = {} if data is None else dict(data)

        palette = src.get("substrate_palette")
        if isinstance(palette, (list, tuple)) and palette:
            parsed_palette = [_to_rgba(v, fallback=(0.70, 0.79, 0.66, 1.0)) for v in palette]
        else:
            parsed_palette = [(0.70, 0.79, 0.66, 1.0), (0.62, 0.79, 0.90, 1.0)]

        if "background" not in src:
            background = None
        else:
            bg_value = src.get("background")
            background = None if bg_value is None else _to_rgba(bg_value, fallback=(1.0, 1.0, 1.0, 1.0))

        light_direction_raw = src.get("light_direction")
        if isinstance(light_direction_raw, (list, tuple)) and len(light_direction_raw) == 3:
            light_direction = (
                float(light_direction_raw[0]),
                float(light_direction_raw[1]),
                float(light_direction_raw[2]),
            )
        else:
            light_direction = (0.68, 0.36, 0.62)
        highlight_direction_raw = src.get("highlight_direction")
        if isinstance(highlight_direction_raw, (list, tuple)) and len(highlight_direction_raw) == 3:
            highlight_direction = (
                float(highlight_direction_raw[0]),
                float(highlight_direction_raw[1]),
                float(highlight_direction_raw[2]),
            )
        else:
            highlight_direction = (0.78, 0.62, 0.0)
        highlight_arc_length_raw = src.get("highlight_arc_length")
        if highlight_arc_length_raw is None and src.get("highlight_direction_threshold") is not None:
            highlight_arc_length_raw = 1.0 - float(src.get("highlight_direction_threshold"))
        if highlight_arc_length_raw is None:
            highlight_arc_length_raw = 0.22

        return cls(
            substrate_symbols=[str(v) for v in src.get("substrate_symbols", [])],
            layer_coloring=bool(src.get("layer_coloring", True)),
            layer_tolerance=float(src.get("layer_tolerance", 0.35)),
            substrate_palette=parsed_palette,
            molecule_use_jmol=bool(src.get("molecule_use_jmol", True)),
            jmol_desaturate=float(src.get("jmol_desaturate", 0.10)),
            jmol_lighten=float(src.get("jmol_lighten", 0.04)),
            bond_color_mode=str(src.get("bond_color_mode", "atom_pair_avg")),
            light_direction=light_direction,
            shadow_area=float(src.get("shadow_area", 0.34)),
            shadow_strength=float(src.get("shadow_strength", 0.42)),
            shadow_softness=float(src.get("shadow_softness", 0.12)),
            highlight_strength=float(src.get("highlight_strength", 0.16)),
            highlight_direction=highlight_direction,
            highlight_arc_length=max(0.0, min(1.0, float(highlight_arc_length_raw))),
            outline_surface=float(src.get("outline_surface", 2.0)),
            outline_molecule=float(src.get("outline_molecule", 2.4)),
            outline_bond=float(src.get("outline_bond", 1.6)),
            highlight_band_inner=max(0.0, min(1.0, float(src.get("highlight_band_inner", 0.56)))),
            highlight_band_outer=max(0.0, min(1.0, float(src.get("highlight_band_outer", 0.90)))),
            outline_secondary_thickness=float(src.get("outline_secondary_thickness", 0.8)),
            outline_secondary_color=_to_rgba(src.get("outline_secondary_color"), fallback=(0.76, 0.82, 0.92, 1.0)),
            background=background,
        )

    @classmethod
    def from_material_spec(cls, spec: HandDrawnMaterialSpec) -> "HanddrawnStyleConfig":
        return cls(
            jmol_desaturate=float(spec.jmol_desaturate),
            jmol_lighten=float(spec.jmol_lighten),
            light_direction=(
                float(spec.light_direction[0]),
                float(spec.light_direction[1]),
                float(spec.light_direction[2]),
            ),
            shadow_area=float(spec.shadow_area),
            shadow_strength=float(spec.shadow_strength),
            shadow_softness=float(spec.shadow_softness),
            highlight_strength=float(spec.highlight_strength),
            highlight_direction=(
                float(spec.highlight_direction[0]),
                float(spec.highlight_direction[1]),
                float(spec.highlight_direction[2]),
            ),
            highlight_arc_length=float(spec.highlight_arc_length),
            highlight_band_inner=float(spec.highlight_band_inner),
            highlight_band_outer=float(spec.highlight_band_outer),
            outline_surface=float(spec.outline_surface),
            outline_molecule=float(spec.outline_molecule),
            outline_bond=float(spec.outline_bond),
            outline_secondary_thickness=float(spec.outline_secondary_thickness),
            outline_secondary_color=(
                float(spec.outline_secondary_color[0]),
                float(spec.outline_secondary_color[1]),
                float(spec.outline_secondary_color[2]),
                float(spec.outline_secondary_color[3]),
            ),
            background=None,
        )


@dataclass
class StyleConfig:
    scene_style: str = "default"
    color_style: str | None = None
    material_style: str | None = None
    light_style: str | None = None
    radius_style: str | None = None
    background: tuple[float, float, float, float] | None = None
    outline: OutlineConfig | None = None
    handdrawn: HanddrawnStyleConfig | None = None
    material_policy: MaterialPolicy = field(default_factory=MaterialPolicy)
    atom_styles: dict[str, AtomStylePresetConfig] = field(default_factory=dict)
    atom_style_rules: list[AtomStyleRuleConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StyleConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(
            src,
            allowed={
                "scene_style",
                "color_style",
                "material_style",
                "light_style",
                "radius_style",
                "background",
                "outline",
                "handdrawn",
                "material_policy",
                "atom_styles",
                "atom_style_rules",
                "preset",
            },
            context="style",
        )
        if "preset" in src:
            raise ValueError("style.preset has been removed. Use style.scene_style.")
        background = None if src.get("background") is None else _to_rgba(src.get("background"))
        outline_data = src.get("outline")
        handdrawn_data = src.get("handdrawn")
        scene_style = str(src.get("scene_style", "default"))
        atom_styles_raw = _copy_dict(src.get("atom_styles"))
        atom_styles = {
            str(k): AtomStylePresetConfig.from_dict(v if isinstance(v, dict) else {})
            for k, v in atom_styles_raw.items()
        }
        rules_raw = src.get("atom_style_rules", [])
        if not isinstance(rules_raw, list):
            raise ValueError("style.atom_style_rules must be a list.")
        atom_style_rules = [AtomStyleRuleConfig.from_dict(v if isinstance(v, dict) else {}) for v in rules_raw]
        for idx, rule in enumerate(atom_style_rules):
            if rule.style is None:
                continue
            if rule.style not in atom_styles:
                raise ValueError(
                    f"style.atom_style_rules[{idx}].style references unknown preset '{rule.style}'."
                )
        return cls(
            scene_style=scene_style,
            color_style=str(src["color_style"]) if src.get("color_style") is not None else None,
            material_style=str(src["material_style"]) if src.get("material_style") is not None else None,
            light_style=str(src["light_style"]) if src.get("light_style") is not None else None,
            radius_style=str(src["radius_style"]) if src.get("radius_style") is not None else None,
            background=background,
            outline=OutlineConfig.from_dict(_copy_dict(outline_data)) if isinstance(outline_data, dict) else None,
            handdrawn=HanddrawnStyleConfig.from_dict(_copy_dict(handdrawn_data)) if isinstance(handdrawn_data, dict) else None,
            material_policy=MaterialPolicy.from_dict(_copy_dict(src.get("material_policy"))),
            atom_styles=atom_styles,
            atom_style_rules=atom_style_rules,
        )


@dataclass
class ASEViewConfig:
    rotations: str | None = None
    axes_matrix: list[list[float]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ASEViewConfig":
        src = {} if data is None else dict(data)
        matrix = None
        if isinstance(src.get("axes_matrix"), (list, tuple)) and len(src["axes_matrix"]) == 3:
            matrix = _to_cell3x3(src.get("axes_matrix"))
        return cls(
            rotations=str(src["rotations"]) if src.get("rotations") is not None else None,
            axes_matrix=matrix,
        )


@dataclass
class CameraConfig:
    projection: str = "ORTHOGRAPHIC"
    fit_mode: str = "orbit_origin"
    fit_padding: float = 0.10
    lens_mm: float = 80.0
    clip_start: float = 0.01
    clip_end: float = 5000.0
    center: tuple[float, float, float] | None = None
    right: tuple[float, float, float] | None = None
    up: tuple[float, float, float] | None = None
    forward: tuple[float, float, float] | None = None
    ortho_scale: float | None = None
    distance: float | None = None
    position: tuple[float, float, float] | None = None
    rotation_euler: tuple[float, float, float] | None = None
    rotation: str | None = None
    model_rotation: tuple[float, ...] | None = None
    model_translation: tuple[float, float, float] | None = None
    view: str = "top"
    frame_scale: float = 1.0
    ase_view: ASEViewConfig = field(default_factory=ASEViewConfig)
    dof_enabled: bool = False
    focus_distance: float | None = None
    aperture_fstop: float = 5.6

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CameraConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(
            src,
            allowed={
                "projection",
                "fit_mode",
                "fit_padding",
                "lens_mm",
                "clip_start",
                "clip_end",
                "center",
                "right",
                "up",
                "forward",
                "ortho_scale",
                "distance",
                "position",
                "rotation_euler",
                "rotation",
                "model_rotation",
                "model_translation",
                "view",
                "frame_scale",
                "ase_view",
                "dof_enabled",
                "focus_distance",
                "aperture_fstop",
                "ase_rotation",
            },
            context="camera",
        )
        if "ase_rotation" in src:
            raise ValueError("camera.ase_rotation has been removed. Use camera.rotation.")
        center = src.get("center")
        right = src.get("right")
        up = src.get("up")
        forward = src.get("forward")
        pos = src.get("position")
        rot = src.get("rotation_euler")
        view = str(src.get("view", "top")).strip().lower()
        frame_scale = float(src.get("frame_scale", 1.0))
        if frame_scale <= 0.0:
            raise ValueError("camera.frame_scale must be > 0.")
        return cls(
            projection=str(src.get("projection", "ORTHOGRAPHIC")).upper(),
            fit_mode=str(src.get("fit_mode", "orbit_origin")),
            fit_padding=float(src.get("fit_padding", 0.10)),
            lens_mm=float(src.get("lens_mm", 80.0)),
            clip_start=float(src.get("clip_start", 0.01)),
            clip_end=float(src.get("clip_end", 5000.0)),
            center=tuple(float(v) for v in center) if isinstance(center, (list, tuple)) and len(center) == 3 else None,
            right=tuple(float(v) for v in right) if isinstance(right, (list, tuple)) and len(right) == 3 else None,
            up=tuple(float(v) for v in up) if isinstance(up, (list, tuple)) and len(up) == 3 else None,
            forward=tuple(float(v) for v in forward) if isinstance(forward, (list, tuple)) and len(forward) == 3 else None,
            ortho_scale=float(src["ortho_scale"]) if src.get("ortho_scale") is not None else None,
            distance=float(src["distance"]) if src.get("distance") is not None else None,
            position=tuple(float(v) for v in pos) if isinstance(pos, (list, tuple)) and len(pos) == 3 else None,
            rotation_euler=tuple(float(v) for v in rot) if isinstance(rot, (list, tuple)) and len(rot) == 3 else None,
            rotation=str(src["rotation"]) if src.get("rotation") is not None else None,
            model_rotation=_to_float_tuple(src.get("model_rotation"), length=16),
            model_translation=_to_float_tuple(src.get("model_translation"), length=3),
            view=view,
            frame_scale=frame_scale,
            ase_view=ASEViewConfig.from_dict(_copy_dict(src.get("ase_view"))),
            dof_enabled=bool(src.get("dof_enabled", False)),
            focus_distance=float(src["focus_distance"]) if src.get("focus_distance") is not None else None,
            aperture_fstop=max(0.1, float(src.get("aperture_fstop", 5.6))),
        )


_GROUND_MODES = {"auto", "visible", "shadow_catcher"}


@dataclass
class GroundPlaneConfig:
    enabled: bool = False
    mode: str = "auto"
    size_scale: float = 2.2
    z_offset_scale: float = 0.03
    color: tuple[float, float, float, float] = (0.88, 0.88, 0.88, 1.0)
    roughness: float = 0.82
    specular: float = 0.05
    metallic: float = 0.0
    coat: float = 0.0
    coat_roughness: float = 0.08

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GroundPlaneConfig":
        src = {} if data is None else dict(data)
        mode = str(src.get("mode", "auto")).strip().lower()
        if mode not in _GROUND_MODES:
            raise ValueError(
                f"Unknown lighting.ground.mode '{mode}'. "
                "Valid values: auto, visible, shadow_catcher."
            )
        return cls(
            enabled=bool(src.get("enabled", False)),
            mode=mode,
            size_scale=float(src.get("size_scale", 2.2)),
            z_offset_scale=float(src.get("z_offset_scale", 0.03)),
            color=_to_rgba(src.get("color"), fallback=(0.88, 0.88, 0.88, 1.0)),
            roughness=float(src.get("roughness", 0.82)),
            specular=float(src.get("specular", 0.05)),
            metallic=float(src.get("metallic", 0.0)),
            coat=float(src.get("coat", 0.0)),
            coat_roughness=float(src.get("coat_roughness", 0.08)),
        )


@dataclass
class StudioSweepConfig:
    enabled: bool = False
    width_scale: float = 8.0
    width_segments: int = 32
    floor_depth_scale: float = 6.0
    wall_height_scale: float = 6.0
    radius_scale: float = 1.4
    floor_offset_scale: float = 0.02
    wall_offset_scale: float = 0.40
    segments: int = 32
    color: tuple[float, float, float, float] = (0.76, 0.78, 0.79, 1.0)
    roughness: float = 0.62
    specular: float = 0.10
    metallic: float = 0.0
    coat: float = 0.04
    coat_roughness: float = 0.32
    gradient_enabled: bool = False
    bottom_color: tuple[float, float, float, float] | None = None
    top_color: tuple[float, float, float, float] | None = None
    spot_color: tuple[float, float, float, float] | None = None
    spot_strength: float = 0.0
    spot_x: float = 0.50
    spot_y: float = 0.72
    spot_radius: float = 0.32
    vignette_strength: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StudioSweepConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(
            src,
            allowed={
                "enabled",
                "width_scale",
                "width_segments",
                "floor_depth_scale",
                "wall_height_scale",
                "radius_scale",
                "floor_offset_scale",
                "wall_offset_scale",
                "segments",
                "color",
                "roughness",
                "specular",
                "metallic",
                "coat",
                "coat_roughness",
                "gradient_enabled",
                "bottom_color",
                "top_color",
                "spot_color",
                "spot_strength",
                "spot_x",
                "spot_y",
                "spot_radius",
                "vignette_strength",
            },
            context="lighting.sweep",
        )
        color = _to_rgba(src.get("color"), fallback=(0.76, 0.78, 0.79, 1.0))
        return cls(
            enabled=bool(src.get("enabled", False)),
            width_scale=float(src.get("width_scale", 8.0)),
            width_segments=max(1, int(src.get("width_segments", 32))),
            floor_depth_scale=float(src.get("floor_depth_scale", 6.0)),
            wall_height_scale=float(src.get("wall_height_scale", 6.0)),
            radius_scale=float(src.get("radius_scale", 1.4)),
            floor_offset_scale=float(src.get("floor_offset_scale", 0.02)),
            wall_offset_scale=float(src.get("wall_offset_scale", 0.40)),
            segments=max(4, int(src.get("segments", 32))),
            color=color,
            roughness=float(src.get("roughness", 0.62)),
            specular=float(src.get("specular", 0.10)),
            metallic=float(src.get("metallic", 0.0)),
            coat=float(src.get("coat", 0.04)),
            coat_roughness=float(src.get("coat_roughness", 0.32)),
            gradient_enabled=bool(src.get("gradient_enabled", False)),
            bottom_color=_to_rgba(src.get("bottom_color"), fallback=color) if src.get("bottom_color") is not None else None,
            top_color=_to_rgba(src.get("top_color"), fallback=color) if src.get("top_color") is not None else None,
            spot_color=_to_rgba(src.get("spot_color"), fallback=(1.0, 1.0, 1.0, 1.0)) if src.get("spot_color") is not None else None,
            spot_strength=float(src.get("spot_strength", 0.0)),
            spot_x=float(src.get("spot_x", 0.50)),
            spot_y=float(src.get("spot_y", 0.72)),
            spot_radius=max(0.01, float(src.get("spot_radius", 0.32))),
            vignette_strength=float(src.get("vignette_strength", 0.0)),
        )


@dataclass
class LightingConfig:
    light_style: str | None = None
    intensity: float = 1.0
    lights: list[LightConfig] = field(default_factory=list)
    ground: GroundPlaneConfig = field(default_factory=GroundPlaneConfig)
    sweep: StudioSweepConfig = field(default_factory=StudioSweepConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LightingConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"light_style", "intensity", "lights", "ground", "sweep", "preset"}, context="lighting")
        if "preset" in src:
            raise ValueError("lighting.preset has been removed. Use lighting.light_style.")
        light_style = str(src["light_style"]) if src.get("light_style") is not None else None
        lights = [LightConfig.from_dict(item if isinstance(item, dict) else {}) for item in src.get("lights", [])]
        ground = GroundPlaneConfig.from_dict(_copy_dict(src.get("ground")))
        sweep = StudioSweepConfig.from_dict(_copy_dict(src.get("sweep")))
        return cls(light_style=light_style, intensity=float(src.get("intensity", 1.0)), lights=lights, ground=ground, sweep=sweep)


@dataclass
class HDRIConfig:
    enabled: bool = False
    path: str | None = None
    strength: float = 0.8
    visible_to_camera: bool = False
    rotation_z: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HDRIConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"enabled", "path", "strength", "visible_to_camera", "rotation_z"}, context="render.effects.hdri")
        return cls(
            enabled=bool(src.get("enabled", False)),
            path=str(src["path"]) if src.get("path") is not None else None,
            strength=float(src.get("strength", 0.8)),
            visible_to_camera=bool(src.get("visible_to_camera", False)),
            rotation_z=float(src.get("rotation_z", 0.0)),
        )


@dataclass
class AmbientOcclusionConfig:
    enabled: bool = False
    factor: float = 0.7
    distance: float = 3.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AmbientOcclusionConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"enabled", "factor", "distance"}, context="render.effects.ambient_occlusion")
        return cls(enabled=bool(src.get("enabled", False)), factor=float(src.get("factor", 0.7)), distance=float(src.get("distance", 3.0)))


@dataclass
class BloomConfig:
    enabled: bool = False
    threshold: float = 0.85
    intensity: float = 0.18
    size: int = 6

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BloomConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"enabled", "threshold", "intensity", "size"}, context="render.effects.bloom")
        return cls(
            enabled=bool(src.get("enabled", False)),
            threshold=float(src.get("threshold", 0.85)),
            intensity=float(src.get("intensity", 0.18)),
            size=int(src.get("size", 6)),
        )


@coerce_color_fields("color", label_prefix="render.effects.atmosphere")
@dataclass
class AtmosphereConfig:
    enabled: bool = False
    density: float = 0.0
    color: tuple[float, float, float, float] = (0.82, 0.88, 0.94, 1.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AtmosphereConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"enabled", "density", "color"}, context="render.effects.atmosphere")
        return cls(
            enabled=bool(src.get("enabled", False)),
            density=float(src.get("density", 0.0)),
            color=_to_rgba(src.get("color"), fallback=(0.82, 0.88, 0.94, 1.0)),
        )


@coerce_color_fields("color", label_prefix="render.effects.volumetric_light")
@dataclass
class VolumetricLightConfig:
    enabled: bool = False
    density: float = 0.0
    anisotropy: float = 0.0
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VolumetricLightConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"enabled", "density", "anisotropy", "color"}, context="render.effects.volumetric_light")
        return cls(
            enabled=bool(src.get("enabled", False)),
            density=float(src.get("density", 0.0)),
            anisotropy=float(src.get("anisotropy", 0.0)),
            color=_to_rgba(src.get("color"), fallback=(1.0, 1.0, 1.0, 1.0)),
        )


@coerce_color_fields("color", label_prefix="render.effects.sunbeam")
@dataclass
class SunbeamConfig:
    enabled: bool = False
    color: tuple[float, float, float, float] = (1.0, 0.78, 0.46, 1.0)
    strength: float = 0.18
    width: float = 0.22
    softness: float = 0.65
    start: tuple[float, float] = (0.92, 0.94)
    end: tuple[float, float] = (0.40, 0.44)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SunbeamConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(
            src,
            allowed={"enabled", "color", "strength", "width", "softness", "start", "end"},
            context="render.effects.sunbeam",
        )
        start = _to_float_tuple(src.get("start"), length=2)
        end = _to_float_tuple(src.get("end"), length=2)
        return cls(
            enabled=bool(src.get("enabled", False)),
            color=_to_rgba(src.get("color"), fallback=(1.0, 0.78, 0.46, 1.0)),
            strength=max(0.0, float(src.get("strength", 0.18))),
            width=max(0.01, float(src.get("width", 0.22))),
            softness=max(0.0, min(1.0, float(src.get("softness", 0.65)))),
            start=(0.92, 0.94) if start is None else (float(start[0]), float(start[1])),
            end=(0.40, 0.44) if end is None else (float(end[0]), float(end[1])),
        )


@dataclass
class SSRConfig:
    enabled: bool = False
    refraction: bool = True
    quality: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SSRConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"enabled", "refraction", "quality"}, context="render.effects.ssr")
        return cls(enabled=bool(src.get("enabled", False)), refraction=bool(src.get("refraction", True)), quality=float(src.get("quality", 0.5)))


@dataclass
class VignetteConfig:
    enabled: bool = False
    intensity: float = 0.12
    softness: float = 0.45

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VignetteConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"enabled", "intensity", "softness"}, context="render.effects.vignette")
        return cls(enabled=bool(src.get("enabled", False)), intensity=float(src.get("intensity", 0.12)), softness=float(src.get("softness", 0.45)))


@dataclass
class ChromaticAberrationConfig:
    enabled: bool = False
    dispersion: float = 0.015

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ChromaticAberrationConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"enabled", "dispersion"}, context="render.effects.chromatic_aberration")
        return cls(enabled=bool(src.get("enabled", False)), dispersion=float(src.get("dispersion", 0.015)))


@dataclass
class FilmGrainConfig:
    enabled: bool = False
    strength: float = 0.025
    scale: float = 85.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FilmGrainConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"enabled", "strength", "scale"}, context="render.effects.film_grain")
        return cls(enabled=bool(src.get("enabled", False)), strength=float(src.get("strength", 0.025)), scale=float(src.get("scale", 85.0)))


@dataclass
class RenderEffectsConfig:
    hdri: HDRIConfig = field(default_factory=HDRIConfig)
    ambient_occlusion: AmbientOcclusionConfig = field(default_factory=AmbientOcclusionConfig)
    bloom: BloomConfig = field(default_factory=BloomConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    volumetric_light: VolumetricLightConfig = field(default_factory=VolumetricLightConfig)
    sunbeam: SunbeamConfig = field(default_factory=SunbeamConfig)
    ssr: SSRConfig = field(default_factory=SSRConfig)
    vignette: VignetteConfig = field(default_factory=VignetteConfig)
    chromatic_aberration: ChromaticAberrationConfig = field(default_factory=ChromaticAberrationConfig)
    film_grain: FilmGrainConfig = field(default_factory=FilmGrainConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RenderEffectsConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(
            src,
            allowed={
                "hdri",
                "ambient_occlusion",
                "ao",
                "bloom",
                "atmosphere",
                "volumetric_light",
                "sunbeam",
                "ssr",
                "vignette",
                "chromatic_aberration",
                "film_grain",
            },
            context="render.effects",
        )
        return cls(
            hdri=HDRIConfig.from_dict(_copy_dict(src.get("hdri"))),
            ambient_occlusion=AmbientOcclusionConfig.from_dict(_copy_dict(src.get("ambient_occlusion", src.get("ao")))),
            bloom=BloomConfig.from_dict(_copy_dict(src.get("bloom"))),
            atmosphere=AtmosphereConfig.from_dict(_copy_dict(src.get("atmosphere"))),
            volumetric_light=VolumetricLightConfig.from_dict(_copy_dict(src.get("volumetric_light"))),
            sunbeam=SunbeamConfig.from_dict(_copy_dict(src.get("sunbeam"))),
            ssr=SSRConfig.from_dict(_copy_dict(src.get("ssr"))),
            vignette=VignetteConfig.from_dict(_copy_dict(src.get("vignette"))),
            chromatic_aberration=ChromaticAberrationConfig.from_dict(_copy_dict(src.get("chromatic_aberration"))),
            film_grain=FilmGrainConfig.from_dict(_copy_dict(src.get("film_grain"))),
        )


@dataclass
class RenderSettings:
    engine: str = "cycles"
    device: str = "auto"
    samples: int = 64
    resolution: tuple[int, int] = (1024, 1024)
    transparent_bg: bool = True
    seed: int = 7
    color_management: dict[str, Any] = field(default_factory=dict)
    effects: RenderEffectsConfig = field(default_factory=RenderEffectsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RenderSettings":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(
            src,
            allowed={
                "engine",
                "device",
                "samples",
                "resolution",
                "transparent_bg",
                "seed",
                "color_management",
                "effects",
            },
            context="render",
        )
        resolution = src.get("resolution", [1024, 1024])
        if not isinstance(resolution, (list, tuple)) or len(resolution) != 2:
            resolution = [1024, 1024]
        device = str(src.get("device", "auto")).strip().lower()
        if device not in {"auto", "gpu", "cpu"}:
            raise ValueError("render.device must be one of: auto, gpu, cpu")
        return cls(
            engine=str(src.get("engine", "cycles")),
            device=device,
            samples=int(src.get("samples", 64)),
            resolution=(int(resolution[0]), int(resolution[1])),
            transparent_bg=bool(src.get("transparent_bg", True)),
            seed=int(src.get("seed", 7)),
            color_management=_copy_dict(src.get("color_management")),
            effects=RenderEffectsConfig.from_dict(_copy_dict(src.get("effects"))),
        )


@dataclass
class InputConfig:
    path: str
    frames: str = "last"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "InputConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"path", "frames"}, context="input")
        path = src.get("path")
        if not path:
            raise ValueError("input.path is required")
        return cls(path=str(path), frames=str(src.get("frames", "last")))


@dataclass
class OutputConfig:
    path: str | None = None
    dir: str | None = None
    filename_template: str = "{job_id}_{frame:04d}.png"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OutputConfig":
        src = {} if data is None else dict(data)
        _reject_unknown_keys(src, allowed={"path", "dir", "filename_template"}, context="output")
        return cls(
            path=str(src["path"]) if src.get("path") is not None else None,
            dir=str(src["dir"]) if src.get("dir") is not None else None,
            filename_template=str(src.get("filename_template", "{job_id}_{frame:04d}.png")),
        )

    def resolve_path(self, job_id: str, frame: int, cwd: Path | None = None) -> Path:
        base = Path.cwd() if cwd is None else cwd
        if self.path:
            p = Path(self.path)
            return p if p.is_absolute() else (base / p).resolve()
        if self.dir:
            d = Path(self.dir)
            d = d if d.is_absolute() else (base / d).resolve()
            return d / self.filename_template.format(job_id=job_id, frame=frame)
        return (base / self.filename_template.format(job_id=job_id, frame=frame)).resolve()


@dataclass
class RenderJobConfig:
    id: str
    input: InputConfig
    output: OutputConfig
    structure: StructureConfig = field(default_factory=StructureConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    render: RenderSettings = field(default_factory=RenderSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderJobConfig":
        src = dict(data)
        _reject_unknown_keys(
            src,
            allowed={"id", "input", "output", "structure", "style", "camera", "lighting", "render"},
            context="job",
        )
        if not src.get("id"):
            raise ValueError("job.id is required")
        return cls(
            id=str(src["id"]),
            input=InputConfig.from_dict(_copy_dict(src.get("input"))),
            output=OutputConfig.from_dict(_copy_dict(src.get("output"))),
            structure=StructureConfig.from_dict(_copy_dict(src.get("structure"))),
            style=StyleConfig.from_dict(_copy_dict(src.get("style"))),
            camera=CameraConfig.from_dict(_copy_dict(src.get("camera"))),
            lighting=LightingConfig.from_dict(_copy_dict(src.get("lighting"))),
            render=RenderSettings.from_dict(_copy_dict(src.get("render"))),
        )

    def with_output_path(self, output_path: str) -> "RenderJobConfig":
        return replace(self, output=replace(self.output, path=str(output_path), dir=None))

    def to_dict(self) -> dict[str, Any]:
        data = _jsonable_config(asdict(self))
        data["render"]["resolution"] = list(self.render.resolution)
        return data


@dataclass
class BatchConfig:
    version: int
    jobs: list[RenderJobConfig]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchConfig":
        if not isinstance(data, dict):
            raise ValueError("config must be a mapping")
        _reject_unknown_keys(data, allowed={"version", "jobs", "job"}, context="config")
        if "job" in data:
            raise ValueError("Top-level 'job' has been removed. Use top-level 'jobs' list.")

        version = int(data.get("version", 2))
        if version != 2:
            raise ValueError("Only version: 2 config is supported.")

        jobs_data = data.get("jobs")
        if not isinstance(jobs_data, list) or not jobs_data:
            raise ValueError("jobs must be a non-empty list")
        jobs = [RenderJobConfig.from_dict(job if isinstance(job, dict) else {}) for job in jobs_data]
        return cls(version=2, jobs=jobs)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "jobs": [job.to_dict() for job in self.jobs]}
