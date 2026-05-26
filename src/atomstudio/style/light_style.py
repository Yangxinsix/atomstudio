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
    "darklab_rim": _lights(
        [
            {"type": "AREA", "vector": (1.05, -1.05, 1.25), "energy": 1120, "size": 0.72, "color": (1.0, 0.96, 0.90, 1.0)},
            {"type": "AREA", "vector": (-1.25, -0.25, 0.85), "energy": 185, "size": 1.80, "color": (0.86, 0.92, 1.0, 1.0)},
            {"type": "AREA", "vector": (-0.10, 1.35, 1.05), "energy": 420, "size": 0.58, "color": (0.74, 0.84, 1.0, 1.0)},
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
    "style_sphere_showcase": _lights(
        [
            {"type": "AREA", "vector": (-1.20, -1.375, 1.80), "energy": 620.0, "size": 1.45},
            {"type": "AREA", "vector": (1.20, -1.55, 0.70), "energy": 80.0, "size": 1.875},
            {"type": "AREA", "vector": (0.0, 0.75, 1.70), "energy": 360.0, "size": 0.70},
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
    "ceramic_softbox": _lights(
        [
            {"type": "AREA", "vector": (1.25, -1.10, 1.45), "energy": 520, "size": 4.60, "color": (1.0, 0.98, 0.94, 1.0)},
            {"type": "AREA", "vector": (-1.40, -0.45, 1.20), "energy": 330, "size": 5.40, "color": (0.96, 0.98, 1.0, 1.0)},
            {"type": "AREA", "vector": (0.00, 0.40, 2.30), "energy": 210, "size": 6.20, "color": (1.0, 1.0, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "glass_lab_rim": _lights(
        [
            {"type": "AREA", "vector": (1.10, -1.05, 1.25), "energy": 620, "size": 0.55, "color": (1.0, 0.98, 0.94, 1.0)},
            {"type": "AREA", "vector": (-1.30, -0.25, 0.80), "energy": 90, "size": 2.20, "color": (0.70, 0.82, 1.0, 1.0)},
            {"type": "AREA", "vector": (-0.15, 1.38, 1.10), "energy": 980, "size": 0.38, "color": (0.58, 0.76, 1.0, 1.0)},
            {"type": "POINT", "vector": (0.80, 0.80, 1.25), "energy": 110, "size": 0.12, "color": (1.0, 1.0, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "monochrome_softbox": _lights(
        [
            {"type": "AREA", "vector": (1.20, -1.00, 1.30), "energy": 760, "size": 1.45, "color": (1.0, 1.0, 1.0, 1.0)},
            {"type": "AREA", "vector": (-1.35, -0.55, 1.05), "energy": 420, "size": 2.45, "color": (1.0, 1.0, 1.0, 1.0)},
            {"type": "AREA", "vector": (0.00, 0.20, 2.20), "energy": 240, "size": 4.20, "color": (1.0, 1.0, 1.0, 1.0)},
            {"type": "AREA", "vector": (-0.15, 1.35, 1.10), "energy": 150, "size": 1.10, "color": (1.0, 1.0, 1.0, 1.0)},
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
    "studio_highkey_softbox": _lights(
        [
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-0.85, -0.78, 1.38), "energy": 520, "size": 2.80, "size_y": 1.35, "color": (1.0, 0.985, 0.955, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (1.10, -0.15, 1.15), "energy": 155, "size": 3.50, "size_y": 2.20, "color": (0.90, 0.95, 1.0, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-0.20, 1.35, 0.95), "energy": 105, "size": 2.60, "size_y": 0.52, "color": (0.82, 0.90, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "studio_highkey_clean_softbox": _lights(
        [
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-0.82, -0.78, 1.42), "energy": 260, "size": 3.60, "size_y": 1.80, "color": (1.0, 0.992, 0.975, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (1.10, -0.12, 1.18), "energy": 105, "size": 4.40, "size_y": 2.80, "color": (0.94, 0.970, 1.0, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (0.00, 0.88, 1.05), "energy": 58, "size": 4.20, "size_y": 0.80, "color": (0.90, 0.945, 1.0, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (0.00, -0.55, 2.10), "energy": 42, "size": 5.20, "size_y": 3.20, "color": (1.0, 1.0, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "studio_warm_softbox": _lights(
        [
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-0.92, -0.72, 1.32), "energy": 330, "size": 3.20, "size_y": 1.70, "color": (1.0, 0.890, 0.740, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (1.12, -0.18, 1.02), "energy": 105, "size": 4.40, "size_y": 2.60, "color": (1.0, 0.945, 0.860, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-0.18, 1.22, 0.92), "energy": 85, "size": 3.40, "size_y": 0.62, "color": (1.0, 0.830, 0.660, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (0.72, 0.86, 0.55), "energy": 42, "size": 2.80, "size_y": 0.82, "color": (1.0, 0.760, 0.540, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "studio_cool_rim_softbox": _lights(
        [
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-0.95, -0.76, 1.20), "energy": 190, "size": 2.55, "size_y": 1.10, "color": (0.88, 0.940, 1.0, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (1.18, -0.20, 1.04), "energy": 46, "size": 3.80, "size_y": 2.20, "color": (0.72, 0.840, 1.0, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-0.18, 1.30, 1.08), "energy": 230, "size": 2.20, "size_y": 0.42, "color": (0.50, 0.700, 1.0, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (0.85, 0.95, 0.72), "energy": 135, "size": 1.75, "size_y": 0.36, "color": (0.58, 0.780, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "studio_macro_softbox": _lights(
        [
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-0.72, -0.55, 1.02), "energy": 430, "size": 2.25, "size_y": 1.15, "color": (1.0, 0.985, 0.955, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (1.05, -0.25, 0.92), "energy": 95, "size": 3.20, "size_y": 1.90, "color": (0.90, 0.96, 1.0, 1.0)},
            {"type": "POINT", "vector": (0.30, 0.85, 0.75), "energy": 22, "size": 0.18, "color": (0.86, 0.92, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "studio_crystal_tabletop": _lights(
        [
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-1.05, -0.68, 1.18), "energy": 610, "size": 3.10, "size_y": 1.10, "color": (1.0, 0.985, 0.960, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (1.25, -0.40, 0.78), "energy": 120, "size": 3.80, "size_y": 2.40, "color": (0.88, 0.94, 1.0, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (0.00, 1.40, 0.72), "energy": 160, "size": 3.00, "size_y": 0.36, "color": (0.76, 0.86, 1.0, 1.0)},
        ],
        default_placement="scaled_offset",
    ),
    "studio_softmetal_strip": _lights(
        [
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-0.95, -0.72, 1.22), "energy": 480, "size": 2.40, "size_y": 0.62, "color": (1.0, 0.98, 0.95, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (1.16, -0.36, 1.02), "energy": 170, "size": 3.20, "size_y": 0.75, "color": (0.86, 0.93, 1.0, 1.0)},
            {"type": "AREA", "shape": "RECTANGLE", "vector": (-0.12, 1.28, 0.92), "energy": 190, "size": 2.20, "size_y": 0.30, "color": (0.72, 0.84, 1.0, 1.0)},
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
