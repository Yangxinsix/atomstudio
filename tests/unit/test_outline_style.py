from __future__ import annotations

import pytest

from atomstudio.color_utils import parse_rgba
from atomstudio.config import StyleConfig
from atomstudio.style.outline_style import OutlineRoleStyle, OutlineStyle
from atomstudio.style.registry import get_scene_style
from atomstudio.style.resolver import resolve_style_bundle


def test_handdrawn_scene_style_has_object_level_outline_defaults():
    style = get_scene_style("handdrawn")
    assert style.outline.enabled is True
    assert style.outline.atoms.enabled is True
    assert isinstance(style.outline.bonds.enabled, bool)
    assert isinstance(style.outline.cell.enabled, bool)


def test_resolver_merges_outline_role_overrides_with_scene_profile():
    base = get_scene_style("handdrawn")
    style_cfg = StyleConfig.from_dict(
        {
            "scene_style": "handdrawn",
            "outline": {
                "enabled": True,
                "bonds": {"enabled": False},
            },
        }
    )
    bundle = resolve_style_bundle(style_cfg, base)
    assert bundle.outline.enabled is True
    assert bundle.outline.atoms.enabled == base.outline.atoms.enabled
    assert bundle.outline.bonds.enabled is False
    assert bundle.outline.cell.enabled == base.outline.cell.enabled


def test_resolver_parses_outline_ignore_occlusion():
    style_cfg = StyleConfig.from_dict(
        {
            "scene_style": "handdrawn",
            "outline": {
                "enabled": True,
                "atoms": {"ignore_occlusion": True},
            },
        }
    )
    bundle = resolve_style_bundle(style_cfg, get_scene_style("handdrawn"))
    assert bundle.outline.atoms.ignore_occlusion is True


def test_outline_objects_accept_named_color_assignment():
    role = OutlineRoleStyle()
    role.color = "tab:blue"
    role.secondary_color = "xkcd:charcoal"
    assert role.color == parse_rgba("tab:blue")
    assert role.secondary_color == parse_rgba("xkcd:charcoal")

    outline = OutlineStyle()
    outline.color = "gold"
    assert outline.color == parse_rgba("gold")


def test_outline_objects_reject_invalid_named_color_assignment():
    role = OutlineRoleStyle()
    with pytest.raises(ValueError, match="命名色/3-4序列/#hex"):
        role.color = "not_a_color"

    outline = OutlineStyle()
    with pytest.raises(ValueError, match="命名色/3-4序列/#hex"):
        outline.color = "not_a_color"
