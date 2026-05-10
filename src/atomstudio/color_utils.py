from __future__ import annotations

from collections.abc import Callable
from typing import Any

_NAMED_COLOR_CACHE: dict[str, tuple[float, float, float, float]] | None = None


def _named_colors() -> dict[str, tuple[float, float, float, float]]:
    global _NAMED_COLOR_CACHE
    if _NAMED_COLOR_CACHE is None:
        from atomstudio.style.data import MATPLOTLIB_NAMED_COLORS_RGBA

        _NAMED_COLOR_CACHE = MATPLOTLIB_NAMED_COLORS_RGBA
    return _NAMED_COLOR_CACHE


def coerce_color_fields(*field_names: str, label_prefix: str | None = None) -> Callable[[type], type]:
    keys = {str(name) for name in field_names if str(name)}
    if not keys:
        raise ValueError("coerce_color_fields requires at least one field name.")

    prefix = "color" if label_prefix is None else str(label_prefix).strip() or "color"

    def _decorate(cls: type) -> type:
        previous_setattr = getattr(cls, "__setattr__", object.__setattr__)

        def __setattr__(self, key: str, value: Any) -> None:
            if key in keys and value is not None:
                rgba = parse_rgba(value)
                if rgba is None:
                    raise ValueError(f"{prefix}.{key}必须是命名色/3-4序列/#hex。")
                value = rgba
            previous_setattr(self, key, value)

        cls.__setattr__ = __setattr__
        return cls

    return _decorate


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_rgba(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, str):
        text = value.strip()
        named = _named_colors().get(text.lower())
        if named is not None:
            return named
        if not text.startswith("#"):
            return None
        hexv = text[1:]
        if len(hexv) == 6:
            return (
                int(hexv[0:2], 16) / 255.0,
                int(hexv[2:4], 16) / 255.0,
                int(hexv[4:6], 16) / 255.0,
                1.0,
            )
        if len(hexv) == 8:
            return (
                int(hexv[0:2], 16) / 255.0,
                int(hexv[2:4], 16) / 255.0,
                int(hexv[4:6], 16) / 255.0,
                int(hexv[6:8], 16) / 255.0,
            )

    if isinstance(value, (list, tuple)):
        if len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]), 1.0)
        if len(value) == 4:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    return None


def rgba4_or_none(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) != 4:
        return None
    return parse_rgba(value)


def rgba_from_any(
    value: Any,
    *,
    fallback: tuple[float, float, float, float] | None = None,
    error_label: str = "color",
) -> tuple[float, float, float, float]:
    rgba = parse_rgba(value)
    if rgba is not None:
        return rgba
    if fallback is not None:
        return fallback
    raise ValueError(f"{error_label} must be a named color, 3/4-length sequence, or #RRGGBB/#RRGGBBAA.")
