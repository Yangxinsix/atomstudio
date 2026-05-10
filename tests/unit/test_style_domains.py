from __future__ import annotations

import pytest

from atomstudio.config import RenderJobConfig
from atomstudio.render.config_resolver import ConfigError, apply_scene_style_defaults_to_job_payload
from atomstudio.style.registry import (
    color_style_choices,
    get_light_style_name,
    get_material_style,
    material_style_choices,
    scene_style_choices,
)
from atomstudio.style.resolver import resolve_style_bundle


def _job_payload() -> dict:
    return {
        "id": "x",
        "input": {"path": "tests/data/water.xyz", "frames": "last"},
        "output": {"path": "/tmp/x.png"},
    }


def test_scene_style_choices_cover_core_styles():
    assert set(scene_style_choices()) == {"default", "darklab", "monochrome", "handdrawn", "handdrawn_v2"}


def test_style_defaults_fill_domain_styles():
    payload = _job_payload()
    payload["style"] = {"scene_style": "default"}
    out = apply_scene_style_defaults_to_job_payload(payload)
    assert out["style"]["color_style"] == "jmol_soft"
    assert out["style"]["material_style"] == "clean"
    assert out["style"]["light_style"] == "homogeneous"


def test_style_subdomain_override_beats_scene_style():
    cfg = RenderJobConfig.from_dict(
        {
            **_job_payload(),
            "style": {
                "scene_style": "darklab",
                "material_style": "handdrawn",
                "light_style": "batoms_soft",
            },
        }
    )
    bundle = resolve_style_bundle(cfg.style)
    assert bundle.scene_style_name == "darklab"
    assert bundle.material_style_name == "handdrawn"
    assert bundle.light_style_name == "batoms_soft"
    assert bundle.material_style.pipeline == "handdrawn"


def test_light_style_accepts_alias_and_native_keys():
    assert get_light_style_name("default") == "batoms_soft"
    assert get_light_style_name("three_point_balanced") == "three_point_balanced"


def test_bad_substyle_name_raises_config_error():
    payload = _job_payload()
    payload["style"] = {"scene_style": "default", "material_style": "not_real"}
    with pytest.raises(ConfigError, match="material_style"):
        apply_scene_style_defaults_to_job_payload(payload)


def test_color_style_choices_only_jmol_and_cpk():
    assert set(color_style_choices()) == {"jmol", "jmol_soft", "cpk", "vesta"}


def test_material_style_choices_cover_new_profiles():
    assert set(material_style_choices()) == {"clean", "glass", "ceramic", "metallic", "emissive", "marble", "handdrawn"}


def test_marble_material_style_matches_teacher_parameters():
    marble = get_material_style("marble")
    assert marble.atom_default.roughness == pytest.approx(0.10)
    assert marble.atom_default.specular == pytest.approx(0.65)
    assert marble.atom_default.coat == pytest.approx(0.65)
    assert marble.atom_default.coat_roughness == pytest.approx(0.05)
    assert marble.atom_default.specular_tint == pytest.approx(0.10)
    assert marble.bond_default.color == pytest.approx((0.20, 0.20, 0.20, 1.0))
    assert marble.bond_default.roughness == pytest.approx(0.78)
    assert marble.bond_default.specular == pytest.approx(0.08)
    assert marble.bond_default.coat == pytest.approx(0.0)


def test_handdrawn_scene_defaults_to_handdrawn_material():
    payload = _job_payload()
    payload["style"] = {"scene_style": "handdrawn"}
    out = apply_scene_style_defaults_to_job_payload(payload)
    assert out["style"]["material_style"] == "handdrawn"
    assert out["style"]["color_style"] == "jmol"


def test_old_material_style_name_default_is_rejected():
    payload = _job_payload()
    payload["style"] = {"scene_style": "default", "material_style": "default"}
    with pytest.raises(ConfigError, match="material_style"):
        apply_scene_style_defaults_to_job_payload(payload)
