from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atomstudio.color_utils import coerce_color_fields, rgba_from_any


_PLACEMENTS = {"absolute", "fixed_offset", "scaled_offset"}


def _to_rgba(value: Any) -> tuple[float, float, float, float]:
    return rgba_from_any(value, error_label="light.color")


def _to_vector(value: Any, *, field_name: str) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    raise ValueError(f"light.{field_name} must be a 3-length number sequence.")


@coerce_color_fields("color", label_prefix="light")
@dataclass
class LightSpec:
    type: str = "AREA"
    placement: str = "absolute"
    vector: tuple[float, float, float] = (0.0, 0.0, 0.0)
    energy: float = 100.0
    size: float = 1.0
    size_y: float | None = None
    shape: str | None = None
    color: tuple[float, float, float, float] | None = None
    lock_to_camera: bool = False

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        default_placement: str = "absolute",
    ) -> "LightSpec":
        src = {} if data is None else dict(data)
        deprecated = [key for key in ("location", "offset", "mode") if key in src]
        if deprecated:
            raise ValueError(
                "Deprecated light fields are not supported: "
                + ", ".join(deprecated)
                + ". Use 'placement' and 'vector'."
            )

        placement = str(src.get("placement", default_placement)).strip().lower()
        if placement not in _PLACEMENTS:
            raise ValueError(
                f"Unknown light.placement '{placement}'. "
                "Valid values: absolute, fixed_offset, scaled_offset."
            )

        color_raw = src.get("color")
        color = None if color_raw is None else _to_rgba(color_raw)
        vector = _to_vector(src.get("vector", (0.0, 0.0, 0.0)), field_name="vector")

        return cls(
            type=str(src.get("type", "AREA")).upper(),
            placement=placement,
            vector=vector,
            energy=float(src.get("energy", 100.0)),
            size=float(src.get("size", 1.0)),
            size_y=float(src["size_y"]) if src.get("size_y") is not None else None,
            shape=str(src["shape"]).upper() if src.get("shape") is not None else None,
            color=color,
            lock_to_camera=bool(src.get("lock_to_camera", False)),
        )

    def resolve_location(
        self,
        center: tuple[float, float, float],
        extent: float,
    ) -> tuple[float, float, float]:
        cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
        vx, vy, vz = (float(self.vector[0]), float(self.vector[1]), float(self.vector[2]))

        if self.placement == "absolute":
            return (vx, vy, vz)
        if self.placement == "fixed_offset":
            return (cx + vx, cy + vy, cz + vz)
        if self.placement == "scaled_offset":
            scale = float(extent)
            return (cx + vx * scale, cy + vy * scale, cz + vz * scale)
        raise ValueError(f"Unsupported light placement '{self.placement}'.")

    def resolve_size(self, extent: float) -> float:
        if self.placement == "scaled_offset":
            return max(0.1, float(self.size) * float(extent))
        return float(self.size)

    def resolve_size_y(self, extent: float) -> float | None:
        if self.size_y is None:
            return None
        if self.placement == "scaled_offset":
            return max(0.1, float(self.size_y) * float(extent))
        return float(self.size_y)

    def to_runtime_spec(
        self,
        center: tuple[float, float, float],
        extent: float,
        intensity: float,
    ) -> dict[str, Any]:
        return {
            "type": str(self.type).upper(),
            "location": self.resolve_location(center, extent),
            "energy": float(self.energy) * float(intensity),
            "size": self.resolve_size(extent),
            "size_y": self.resolve_size_y(extent),
            "shape": None if self.shape is None else str(self.shape).upper(),
            "color": self.color,
            "lock_to_camera": bool(self.lock_to_camera),
        }


LightParams = LightSpec
