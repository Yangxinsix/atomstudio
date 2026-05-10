from __future__ import annotations

import pytest

from atomstudio.preview.lighting import PreviewLightingSettings, configure_shading_filter, preview_light_dir, screen_space_light_dir
from atomstudio.preview.renderer import PreviewCameraState


class _Filter:
    pass


class _Visual:
    def __init__(self) -> None:
        self.shading_filter = _Filter()


def test_screen_space_light_direction_tracks_camera_basis():
    settings = PreviewLightingSettings(light_dir_screen=(1.0, 0.0, 0.0))
    camera = PreviewCameraState(right=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0), forward=(-1.0, 0.0, 0.0))

    assert preview_light_dir(settings) == pytest.approx((1.0, 0.0, 0.0))
    assert screen_space_light_dir(camera, settings) == pytest.approx((0.0, 1.0, 0.0))


def test_configure_shading_filter_sets_stable_preview_light_values():
    visual = _Visual()
    camera = PreviewCameraState()
    settings = PreviewLightingSettings(
        light_dir_screen=(0.0, 0.0, 1.0),
        ambient_light=(1.0, 1.0, 1.0, 0.5),
        diffuse_light=(1.0, 1.0, 1.0, 0.4),
        specular_light=(1.0, 1.0, 1.0, 0.1),
        shininess=32.0,
    )

    assert configure_shading_filter(visual, camera, settings) is True
    assert visual.shading_filter.light_dir == pytest.approx(camera.forward)
    assert visual.shading_filter.ambient_light == settings.ambient_light
    assert visual.shading_filter.diffuse_light == settings.diffuse_light
    assert visual.shading_filter.specular_light == settings.specular_light
    assert visual.shading_filter.shininess == pytest.approx(32.0)
