from __future__ import annotations

import pytest

from atomstudio.config import GroundPlaneConfig, RenderJobConfig
from atomstudio.scene.ground_builder import GroundBuilder, resolve_ground_spec


def _base_job() -> dict:
    return {
        "id": "x",
        "input": {"path": "tests/data/water.xyz", "frames": "last"},
        "output": {"path": "/tmp/x.png"},
        "style": {"scene_style": "default"},
    }


def test_ground_plane_config_defaults():
    cfg = RenderJobConfig.from_dict(_base_job())
    ground = cfg.lighting.ground
    assert ground.enabled is False
    assert ground.mode == "auto"
    assert ground.size_scale == pytest.approx(2.2)
    assert ground.z_offset_scale == pytest.approx(0.03)
    assert ground.color == pytest.approx((0.88, 0.88, 0.88, 1.0))
    assert ground.roughness == pytest.approx(0.82)
    assert ground.specular == pytest.approx(0.05)


def test_ground_builder_from_cfg_accepts_registry_reference():
    cfg = RenderJobConfig.from_dict(_base_job())
    token = object()
    builder = GroundBuilder.from_cfg(cfg, registry=token)
    assert builder.registry is token


def test_ground_builder_from_cfg_requires_registry():
    cfg = RenderJobConfig.from_dict(_base_job())
    with pytest.raises(TypeError):
        GroundBuilder.from_cfg(cfg)


def test_ground_plane_config_parses_explicit_values_and_rgb_color():
    cfg = RenderJobConfig.from_dict(
        {
            **_base_job(),
            "lighting": {
                "ground": {
                    "enabled": True,
                    "mode": "visible",
                    "size_scale": 3.4,
                    "z_offset_scale": 0.12,
                    "color": [0.1, 0.2, 0.3],
                    "roughness": 0.66,
                    "specular": 0.11,
                }
            },
        }
    )
    ground = cfg.lighting.ground
    assert ground.enabled is True
    assert ground.mode == "visible"
    assert ground.size_scale == pytest.approx(3.4)
    assert ground.z_offset_scale == pytest.approx(0.12)
    assert ground.color == pytest.approx((0.1, 0.2, 0.3, 1.0))
    assert ground.roughness == pytest.approx(0.66)
    assert ground.specular == pytest.approx(0.11)


def test_ground_plane_config_parses_hex_color():
    cfg = RenderJobConfig.from_dict(
        {
            **_base_job(),
            "lighting": {"ground": {"color": "#808080cc"}},
        }
    )
    ground = cfg.lighting.ground
    assert ground.color == pytest.approx((128 / 255.0, 128 / 255.0, 128 / 255.0, 204 / 255.0))


def test_ground_plane_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="lighting.ground.mode"):
        GroundPlaneConfig.from_dict({"mode": "mystery"})


def test_resolve_ground_spec_mode_decisions():
    points = [(-1.0, -1.0, -0.5), (1.0, 1.0, 0.5)]

    auto_cycles = resolve_ground_spec(
        points=points,
        render_engine="CYCLES",
        transparent_bg=True,
        enabled=True,
        mode="auto",
        size_scale=2.2,
        z_offset_scale=0.03,
        color=(0.88, 0.88, 0.88, 1.0),
        roughness=0.82,
        specular=0.05,
    )
    assert auto_cycles["effective_mode"] == "shadow_catcher"

    auto_eevee = resolve_ground_spec(
        points=points,
        render_engine="BLENDER_EEVEE",
        transparent_bg=True,
        enabled=True,
        mode="auto",
        size_scale=2.2,
        z_offset_scale=0.03,
        color=(0.88, 0.88, 0.88, 1.0),
        roughness=0.82,
        specular=0.05,
    )
    assert auto_eevee["effective_mode"] == "visible"

    forced_shadow_eevee = resolve_ground_spec(
        points=points,
        render_engine="eevee",
        transparent_bg=True,
        enabled=True,
        mode="shadow_catcher",
        size_scale=2.2,
        z_offset_scale=0.03,
        color=(0.88, 0.88, 0.88, 1.0),
        roughness=0.82,
        specular=0.05,
    )
    assert forced_shadow_eevee["effective_mode"] == "visible"


def test_resolve_ground_spec_uses_bbox_bottom_and_scale():
    spec = resolve_ground_spec(
        points=[(0.0, 0.0, 1.0), (2.0, 4.0, -1.0)],
        render_engine="cycles",
        transparent_bg=True,
        enabled=True,
        mode="visible",
        size_scale=2.2,
        z_offset_scale=0.03,
        color=(0.88, 0.88, 0.88, 1.0),
        roughness=0.82,
        specular=0.05,
    )
    assert spec["extent"] == pytest.approx(4.0)
    assert spec["location"] == pytest.approx((1.0, 2.0, -1.12))
    assert spec["plane_size"] == pytest.approx(8.8)


def test_resolve_ground_spec_handles_empty_points():
    spec = resolve_ground_spec(
        points=[],
        render_engine="eevee",
        transparent_bg=True,
        enabled=True,
        mode="visible",
        size_scale=2.2,
        z_offset_scale=0.03,
        color=(0.88, 0.88, 0.88, 1.0),
        roughness=0.82,
        specular=0.05,
    )
    assert spec["extent"] == pytest.approx(1.0)
    assert spec["location"] == pytest.approx((0.0, 0.0, -0.03))
    assert spec["plane_size"] == pytest.approx(2.2)
