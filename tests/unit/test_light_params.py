from __future__ import annotations

import pytest

from atomstudio.config import LightConfig, RenderJobConfig
from atomstudio.scene.lights.builder import resolve_lighting_specs


def test_light_config_from_dict_accepts_new_placement_values():
    absolute = LightConfig.from_dict({"placement": "absolute", "vector": [1, 2, 3]})
    fixed = LightConfig.from_dict({"placement": "fixed_offset", "vector": [1, 2, 3]})
    scaled = LightConfig.from_dict({"placement": "scaled_offset", "vector": [1, 2, 3]})
    assert absolute.placement == "absolute"
    assert fixed.placement == "fixed_offset"
    assert scaled.placement == "scaled_offset"
    assert scaled.vector == (1.0, 2.0, 3.0)


def test_light_config_rejects_legacy_location_offset_mode_fields():
    with pytest.raises(ValueError, match="Deprecated light fields"):
        LightConfig.from_dict({"location": [0, 0, 10]})
    with pytest.raises(ValueError, match="Deprecated light fields"):
        LightConfig.from_dict({"offset": [0, 0, 10]})
    with pytest.raises(ValueError, match="Deprecated light fields"):
        LightConfig.from_dict({"mode": "fixed"})


def test_resolve_lighting_specs_respects_placement_and_intensity():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
            "lighting": {
                "intensity": 3.0,
                "lights": [
                    {"type": "AREA", "placement": "scaled_offset", "vector": [1, 0, 0], "energy": 10, "size": 2.0},
                    {"type": "AREA", "placement": "fixed_offset", "vector": [1, 0, 0], "energy": 10, "size": 2.0},
                ],
            },
        }
    )
    lights = resolve_lighting_specs(cfg, center=(10.0, 0.0, 0.0), extent=2.0)
    assert lights[0]["location"] == pytest.approx((12.0, 0.0, 0.0))
    assert lights[0]["size"] == pytest.approx(4.0)
    assert lights[0]["energy"] == pytest.approx(30.0)
    assert lights[1]["location"] == pytest.approx((11.0, 0.0, 0.0))
    assert lights[1]["size"] == pytest.approx(2.0)
    assert lights[1]["energy"] == pytest.approx(30.0)


def test_resolve_lighting_specs_keeps_preset_in_world_space():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
            "camera": {"view": "front"},
            "lighting": {"light_style": "three_point"},
        }
    )
    lights = resolve_lighting_specs(cfg, center=(0.0, 0.0, 0.0), extent=2.0, default_light_style="three_point")
    assert lights[0]["location"] == pytest.approx((1.8, -1.8, 2.4))
    assert all(item["lock_to_camera"] is False for item in lights)

    cfg_side = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
            "camera": {"view": "side"},
            "lighting": {"light_style": "three_point"},
        }
    )
    lights_side = resolve_lighting_specs(cfg_side, center=(0.0, 0.0, 0.0), extent=2.0, default_light_style="three_point")
    assert lights_side[0]["location"] == pytest.approx((1.8, -1.8, 2.4))

    cfg_rot = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
            "camera": {"view": "30x,0y,0z"},
            "lighting": {"light_style": "three_point"},
        }
    )
    lights_rot = resolve_lighting_specs(
        cfg_rot,
        center=(0.0, 0.0, 0.0),
        extent=2.0,
        default_light_style="three_point",
    )
    assert lights_rot[0]["location"] == pytest.approx((1.8, -1.8, 2.4))


def test_resolve_lighting_specs_keeps_custom_lights_not_camera_rotated():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
            "camera": {"view": "side"},
            "lighting": {
                "intensity": 3.0,
                "lights": [
                    {"type": "AREA", "placement": "scaled_offset", "vector": [1, 0, 0], "energy": 10, "size": 2.0},
                ],
            },
        }
    )
    lights = resolve_lighting_specs(cfg, center=(10.0, 0.0, 0.0), extent=2.0)
    assert lights[0]["location"] == pytest.approx((12.0, 0.0, 0.0))
    assert lights[0]["lock_to_camera"] is False


def test_resolve_lighting_specs_forces_preset_lights_world_fixed():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
            "lighting": {"light_style": "homogeneous"},
        }
    )
    lights = resolve_lighting_specs(cfg, center=(0.0, 0.0, 0.0), extent=2.0, default_light_style="homogeneous")
    assert all(item["lock_to_camera"] is False for item in lights)


def test_light_config_accepts_named_color_assignment():
    light = LightConfig()
    light.color = "red"
    assert light.color == pytest.approx((1.0, 0.0, 0.0, 1.0))


def test_light_config_rejects_invalid_named_color_assignment():
    light = LightConfig()
    with pytest.raises(ValueError, match="命名色/3-4序列/#hex"):
        light.color = "not_a_color"
