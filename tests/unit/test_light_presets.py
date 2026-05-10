from __future__ import annotations

import pytest

from atomstudio.scene.lights.builder import resolve_lighting_specs
from atomstudio.config import LightConfig, RenderJobConfig
from atomstudio.style.light_style import LIGHT_STYLE_LIBRARY


def _cfg_with_light_style(light_style: str) -> RenderJobConfig:
    return RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
            "lighting": {"light_style": light_style},
        }
    )


def test_new_light_presets_registered():
    assert "three_point_balanced" in LIGHT_STYLE_LIBRARY
    assert "product_softbox" in LIGHT_STYLE_LIBRARY
    assert "interior_window" in LIGHT_STYLE_LIBRARY
    assert "preview_softbox" in LIGHT_STYLE_LIBRARY
    assert isinstance(LIGHT_STYLE_LIBRARY["three_point"], list)
    assert all(isinstance(item, LightConfig) for item in LIGHT_STYLE_LIBRARY["three_point"])


def test_three_point_balanced_keeps_key_fill_rim_ratio():
    cfg = _cfg_with_light_style("three_point_balanced")
    lights = resolve_lighting_specs(cfg, center=(0.0, 0.0, 0.0), extent=1.0, default_light_style="batoms_soft")
    assert len(lights) == 3
    assert all(item["type"] == "AREA" for item in lights)
    assert lights[0]["energy"] > lights[1]["energy"] > lights[2]["energy"]
    assert lights[1]["energy"] == pytest.approx(lights[0]["energy"] * 0.32)


def test_interior_window_uses_fixed_size_and_location():
    cfg = _cfg_with_light_style("interior_window")
    lights_small = resolve_lighting_specs(cfg, center=(0.0, 0.0, 0.0), extent=1.0, default_light_style="batoms_soft")
    lights_large = resolve_lighting_specs(cfg, center=(0.0, 0.0, 0.0), extent=12.0, default_light_style="batoms_soft")
    assert len(lights_small) == len(lights_large) == 4
    for left, right in zip(lights_small, lights_large):
        assert left["location"] == pytest.approx(right["location"])
        assert left["size"] == pytest.approx(right["size"])


def test_handdrawn_soft_is_area_area_point_with_descending_energy():
    cfg = _cfg_with_light_style("handdrawn_soft")
    lights = resolve_lighting_specs(cfg, center=(0.0, 0.0, 0.0), extent=1.0, default_light_style="handdrawn_soft")
    assert len(lights) == 3
    assert [item["type"] for item in lights] == ["AREA", "AREA", "POINT"]
    assert lights[0]["energy"] > lights[1]["energy"] > lights[2]["energy"]


def test_preview_softbox_uses_camera_locked_area_and_point_light():
    cfg = _cfg_with_light_style("preview_softbox")
    lights = resolve_lighting_specs(cfg, center=(0.0, 0.0, 0.0), extent=2.0, default_light_style="batoms_soft")

    assert [item["type"] for item in lights] == ["AREA", "POINT"]
    assert all(item["lock_to_camera"] is True for item in lights)
    assert lights[0]["location"] == pytest.approx((-0.8, 0.9, 3.7))
    assert lights[1]["location"] == pytest.approx((0.0, 0.0, 2.5))
    assert lights[0]["energy"] > lights[1]["energy"]
