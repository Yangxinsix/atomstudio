from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atomstudio.color_utils import coerce_color_fields, parse_rgba

_MISSING = object()


@coerce_color_fields("color", "secondary_color", label_prefix="outline_role")
@dataclass
class OutlineRoleStyle:
    enabled: bool = True
    thickness: float | None = None
    color: tuple[float, float, float, float] | None = None
    follow_atom_color: bool = False
    color_scale: float = 0.52
    secondary_thickness: float | None = None
    secondary_color: tuple[float, float, float, float] | None = None
    ignore_occlusion: bool = True

    def copy(self) -> "OutlineRoleStyle":
        return OutlineRoleStyle(
            enabled=bool(self.enabled),
            thickness=None if self.thickness is None else float(self.thickness),
            color=None if self.color is None else tuple(float(v) for v in self.color),
            follow_atom_color=bool(self.follow_atom_color),
            color_scale=float(self.color_scale),
            secondary_thickness=None if self.secondary_thickness is None else float(self.secondary_thickness),
            secondary_color=None if self.secondary_color is None else tuple(float(v) for v in self.secondary_color),
            ignore_occlusion=bool(self.ignore_occlusion),
        )

    @classmethod
    def from_any(cls, data: Any, fallback: "OutlineRoleStyle | None" = None) -> "OutlineRoleStyle":
        base = OutlineRoleStyle() if fallback is None else fallback.copy()
        if data is None:
            return base
        enabled = _get_value(data, "enabled", _MISSING)
        if enabled is not _MISSING and enabled is not None:
            base.enabled = bool(enabled)
        thickness = _get_value(data, "thickness", _MISSING)
        if thickness is not _MISSING:
            base.thickness = None if thickness is None else float(thickness)
        color = _get_value(data, "color", _MISSING)
        if color is not _MISSING:
            base.color = _to_optional_rgba(color)
        follow_atom_color = _get_value(data, "follow_atom_color", _MISSING)
        if follow_atom_color is not _MISSING and follow_atom_color is not None:
            base.follow_atom_color = bool(follow_atom_color)
        color_scale = _get_value(data, "color_scale", _MISSING)
        if color_scale is not _MISSING and color_scale is not None:
            base.color_scale = max(0.0, min(1.0, float(color_scale)))
        secondary_thickness = _get_value(data, "secondary_thickness", _MISSING)
        if secondary_thickness is not _MISSING:
            base.secondary_thickness = None if secondary_thickness is None else float(secondary_thickness)
        secondary_color = _get_value(data, "secondary_color", _MISSING)
        if secondary_color is not _MISSING:
            base.secondary_color = _to_optional_rgba(secondary_color)
        ignore_occlusion = _get_value(data, "ignore_occlusion", _MISSING)
        if ignore_occlusion is not _MISSING and ignore_occlusion is not None:
            base.ignore_occlusion = bool(ignore_occlusion)
        return base


@coerce_color_fields("color", label_prefix="outline")
@dataclass
class OutlineStyle:
    enabled: bool = False
    thickness: float = 1.2
    color: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    atoms: OutlineRoleStyle = field(default_factory=OutlineRoleStyle)
    bonds: OutlineRoleStyle = field(default_factory=OutlineRoleStyle)
    cell: OutlineRoleStyle = field(default_factory=OutlineRoleStyle)

    def copy(self) -> "OutlineStyle":
        return OutlineStyle(
            enabled=bool(self.enabled),
            thickness=float(self.thickness),
            color=tuple(float(v) for v in self.color),
            atoms=self.atoms.copy(),
            bonds=self.bonds.copy(),
            cell=self.cell.copy(),
        )

    @classmethod
    def from_any(cls, data: Any, fallback: "OutlineStyle | None" = None) -> "OutlineStyle":
        base = OutlineStyle() if fallback is None else fallback.copy()
        if data is None:
            return base
        enabled = _get_value(data, "enabled", _MISSING)
        if enabled is not _MISSING:
            base.enabled = bool(enabled)
        thickness = _get_value(data, "thickness", _MISSING)
        if thickness is not _MISSING:
            base.thickness = float(thickness)
        color = _get_value(data, "color", _MISSING)
        if color is not _MISSING:
            base.color = _to_rgba(color)

        atoms = _get_value(data, "atoms", _MISSING)
        if atoms is not _MISSING and atoms is not None:
            base.atoms = OutlineRoleStyle.from_any(atoms, fallback=base.atoms)
        bonds = _get_value(data, "bonds", _MISSING)
        if bonds is not _MISSING and bonds is not None:
            base.bonds = OutlineRoleStyle.from_any(bonds, fallback=base.bonds)
        cell = _get_value(data, "cell", _MISSING)
        if cell is not _MISSING and cell is not None:
            base.cell = OutlineRoleStyle.from_any(cell, fallback=base.cell)
        return base


DEFAULT_OUTLINE_STYLE = OutlineStyle(
    enabled=False,
    thickness=1.2,
    color=(0.2, 0.2, 0.2, 1.0),
)

HANDDRAWN_OUTLINE_STYLE = OutlineStyle(
    enabled=True,
    thickness=1.6,
    color=(0.41, 0.46, 0.56, 1.0),
    atoms=OutlineRoleStyle(
        enabled=True,
        thickness=3.0,
        color=(0.184, 0.204, 0.235, 1.0),
        follow_atom_color=True,
        color_scale=0.52,
        ignore_occlusion=False,
    ),
    bonds=OutlineRoleStyle(
        enabled=True,
        thickness=2.0,
        color=(0.184, 0.204, 0.235, 1.0),
        ignore_occlusion=False,
    ),
    cell=OutlineRoleStyle(enabled=False, ignore_occlusion=True),
)


def _get_value(data: Any, key: str, fallback: Any) -> Any:
    if isinstance(data, dict):
        return data.get(key, fallback)
    if hasattr(data, key):
        return getattr(data, key)
    return fallback


def _to_optional_rgba(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    return _to_rgba(value)


def _to_rgba(value: Any) -> tuple[float, float, float, float]:
    parsed = parse_rgba(value)
    if parsed is not None:
        return parsed
    raise ValueError("Outline color must be a named color, 3/4-length sequence, or #RRGGBB/#RRGGBBAA.")


__all__ = [
    "OutlineRoleStyle",
    "OutlineStyle",
    "DEFAULT_OUTLINE_STYLE",
    "HANDDRAWN_OUTLINE_STYLE",
]
