from __future__ import annotations

import pytest

from atomstudio.style.color_style import JMOL_COLOR_STYLE, JMOL_SOFT_COLOR_STYLE, VESTA_COLOR_STYLE
from atomstudio.style.radius_style import (
    ATOMIC_RADIUS_STYLE,
    COVALENT_RADIUS_STYLE,
    IONIC_RADIUS_STYLE,
    VDW_RADIUS_STYLE,
)
from atomstudio.style.registry import get_color_style, get_scene_style, radius_style_choices


def test_vesta_color_style_is_registered_and_has_hydrogen_color():
    style = get_color_style("vesta")
    h = style.color_for("H")
    assert h[0] == pytest.approx(1.0)
    assert h[1] == pytest.approx(0.8)
    assert h[2] == pytest.approx(0.8)
    assert h[3] == pytest.approx(1.0)
    assert VESTA_COLOR_STYLE.name == "vesta"


def test_vesta_radius_styles_have_expected_hydrogen_values():
    assert ATOMIC_RADIUS_STYLE.radius_for("H") == pytest.approx(0.46)
    assert COVALENT_RADIUS_STYLE.radius_for("H") == pytest.approx(0.31)
    assert IONIC_RADIUS_STYLE.radius_for("H") == pytest.approx(1.20)
    assert VDW_RADIUS_STYLE.radius_for("H") == pytest.approx(0.200)


def test_radius_style_choices_include_covalent():
    assert "covalent" in radius_style_choices()


def test_jmol_soft_only_changes_hydrogen_color():
    assert JMOL_SOFT_COLOR_STYLE.color_for("H") != JMOL_COLOR_STYLE.color_for("H")
    assert JMOL_SOFT_COLOR_STYLE.color_for("C") == JMOL_COLOR_STYLE.color_for("C")


def test_default_scene_style_has_outline_disabled():
    style = get_scene_style("default")
    assert style.outline.enabled is False
