from __future__ import annotations

from atomstudio.config import LightConfig


def _lights(raw: list[dict], *, default_placement: str) -> list[LightConfig]:
    return [LightConfig.from_dict(item, default_placement=default_placement) for item in raw]


LIGHT_STYLE_LIBRARY: dict[str, list[LightConfig]] = {
    "homogeneous": _lights(
        [
            {"type": "AREA", "vector": (-2.8, -0.2, 4.0), "energy": 240, "size": 3.2, "color": (0.84, 0.91, 1.0, 1.0)},
            {"type": "SUN", "vector": (0.0, 0.0, 150.0), "energy": 1.1, "size": 4.0, "lock_to_camera": True},
            {"type": "AREA", "vector": (2.4, -1.2, 3.6), "energy": 160, "size": 3.4, "color": (1.0, 0.97, 0.93, 1.0)},
            {"type": "AREA", "vector": (0.0, 1.8, 3.4), "energy": 120, "size": 4.6, "color": (0.90, 0.93, 1.0, 1.0)},
        ],
        default_placement="fixed_offset",
    ),
    "three_point": _lights(
        [
            {"type": "AREA", "vector": (0.9, -0.9, 1.2), "energy": 1200, "size": 0.75},
            {"type": "AREA", "vector": (-1.0, -0.4, 0.8), "energy": 500, "size": 1.05},
            {"type": "POINT", "vector": (0.0, 1.0, 0.8), "energy": 180, "size": 0.3},
        ],
        default_placement="scaled_offset",
    ),
    "teacher_three_point": _lights(
        [
            {"type": "AREA", "vector": (-1.35, 1.45, 3.15), "energy": 146, "size": 2.1},
            {"type": "AREA", "vector": (1.90, 1.30, 2.90), "energy": 74, "size": 2.6},
            {"type": "POINT", "vector": (-0.25, -2.80, 2.20), "energy": 22, "size": 0.3},
        ],
        default_placement="fixed_offset",
    ),
    "handdrawn_soft": _lights(
        [
            {"type": "AREA", "vector": (-0.90, 0.55, 1.35), "energy": 560, "size": 2.80, "color": (1.0, 1.0, 1.0, 1.0)},
            {"type": "AREA", "vector": (1.25, 0.30, 1.05), "energy": 330, "size": 3.30, "color": (0.96, 0.98, 1.0, 1.0)},
            {"type": "POINT", "vector": (-0.15, -1.10, 0.75), "energy": 35, "size": 0.3, "color": (0.92, 0.96, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "handdrawn_soft_spot": _lights(
        [
            {"type": "AREA", "vector": (-0.85, 0.50, 1.32), "energy": 520, "size": 2.60, "color": (1.0, 1.0, 1.0, 1.0)},
            {"type": "AREA", "vector": (1.40, 0.35, 1.05), "energy": 360, "size": 3.00, "color": (0.97, 0.98, 1.0, 1.0)},
            {"type": "POINT", "vector": (1.20, 1.08, 1.60), "energy": 220, "size": 0.18, "color": (1.0, 1.0, 1.0, 1.0)},
            {"type": "POINT", "vector": (-0.20, -1.10, 0.70), "energy": 28, "size": 0.28, "color": (0.92, 0.96, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "batoms_soft": _lights(
        [
            {"type": "SUN", "vector": (0.0, 0.0, 30.0), "energy": 5.0, "size": 4.0, "lock_to_camera": True},
            {"type": "AREA", "vector": (2.2, -1.8, 1.6), "energy": 45.0, "size": 3.2, "lock_to_camera": False},
        ],
        default_placement="fixed_offset",
    ),
    "preview_softbox": _lights(
        [
            {"type": "AREA", "vector": (-0.40, 0.45, 1.85), "energy": 520, "size": 2.80, "color": (1.0, 1.0, 1.0, 1.0), "lock_to_camera": True},
            {"type": "POINT", "vector": (0.0, 0.0, 1.25), "energy": 55, "size": 0.25, "color": (1.0, 1.0, 1.0, 1.0), "lock_to_camera": True},
        ],
        default_placement="scaled_offset",
    ),
    "three_point_balanced": _lights(
        [
            {"type": "AREA", "vector": (1.05, -0.95, 1.25), "energy": 1000, "size": 0.95, "color": (1.0, 0.96, 0.92, 1.0)},
            {"type": "AREA", "vector": (-1.15, -0.35, 0.95), "energy": 320, "size": 1.65, "color": (1.0, 1.0, 1.0, 1.0)},
            {"type": "AREA", "vector": (-0.15, 1.25, 1.10), "energy": 220, "size": 0.75, "color": (0.86, 0.92, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "product_softbox": _lights(
        [
            {"type": "AREA", "vector": (1.60, -1.20, 1.40), "energy": 720, "size": 2.70, "color": (1.0, 0.98, 0.95, 1.0)},
            {"type": "AREA", "vector": (-1.60, -1.10, 1.20), "energy": 300, "size": 3.10, "color": (0.97, 0.98, 1.0, 1.0)},
            {"type": "AREA", "vector": (0.00, 0.20, 2.40), "energy": 220, "size": 4.00, "color": (1.0, 1.0, 1.0, 1.0)},
            {"type": "AREA", "vector": (0.00, 1.50, 1.10), "energy": 140, "size": 1.40, "color": (0.86, 0.90, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "interior_window": _lights(
        [
            {"type": "AREA", "vector": (-18.40, 2.40, -10.20), "energy": 250, "size": 2.1, "color": (0.84, 0.91, 1.0, 1.0)},
            {"type": "SUN", "vector": (0.0, 0.0, 150.0), "energy": 2.0, "size": 3.0, "lock_to_camera": True},
            {"type": "AREA", "vector": (18.80, 2.80, -10.90), "energy": 180, "size": 2.60, "color": (1.0, 0.97, 0.93, 1.0)},
            {"type": "POINT", "vector": (-0.20, 2.40, 2.00), "energy": 40, "size": 0.25, "color": (0.84, 0.89, 1.0, 1.0)},
        ],
        default_placement="fixed_offset",
    ),
}
