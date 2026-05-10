from __future__ import annotations

import pytest

from atomstudio.config import RenderJobConfig
from atomstudio.scene.camera_resolver import resolve_scene_camera
from atomstudio.scene.light_resolver import resolve_scene_lights
from atomstudio.scene.model import SceneBounds


def _cfg(**overrides) -> RenderJobConfig:
    payload = {
        "id": "scene-resolvers",
        "input": {"path": "tests/data/water.xyz", "frames": "last"},
        "output": {"path": "/tmp/x.png"},
        "style": {"scene_style": "default"},
    }
    payload.update(overrides)
    return RenderJobConfig.from_dict(payload)


def test_resolve_scene_camera_fits_orthographic_view():
    bounds = SceneBounds(
        minimum=(-1.0, -1.0, -1.0),
        maximum=(1.0, 1.0, 1.0),
        center=(0.0, 0.0, 0.0),
        radius=1.732,
    )
    cfg = _cfg(camera={"view": "front", "frame_scale": 1.1})

    camera = resolve_scene_camera(cfg, bounds=bounds, points=[(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)])

    assert camera.projection == "ORTHOGRAPHIC"
    assert camera.up == pytest.approx((0.0, 0.0, 1.0))
    assert camera.forward == pytest.approx((0.0, 1.0, 0.0))
    assert camera.ortho_scale is not None and camera.ortho_scale >= 1.0
    assert camera.distance is not None and camera.distance > 0.0


def test_resolve_scene_camera_keeps_explicit_position():
    bounds = SceneBounds(center=(1.0, 2.0, 3.0), radius=2.0)
    cfg = _cfg(camera={"position": [4.0, 5.0, 6.0]})

    camera = resolve_scene_camera(cfg, bounds=bounds)

    assert camera.position == pytest.approx((4.0, 5.0, 6.0))
    assert camera.target == pytest.approx((1.0, 2.0, 3.0))


def test_resolve_scene_lights_uses_runtime_specs_as_pure_data():
    bounds = SceneBounds(center=(0.0, 0.0, 0.0), radius=1.0)
    cfg = _cfg(lighting={"light_style": "three_point", "intensity": 2.0})

    lights = resolve_scene_lights(cfg, bounds=bounds, default_light_style="three_point")

    assert len(lights) == 3
    assert lights[0].type == "AREA"
    assert lights[0].energy > lights[1].energy > lights[2].energy
    assert lights[0].direction is not None
