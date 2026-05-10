from __future__ import annotations

import pytest

from atomstudio.config import RenderJobConfig
from atomstudio.preview import PreviewSettings, build_preview_scene
from atomstudio.preview.types import PreviewMaterialPayload
from atomstudio.scene.materials.specs import HandDrawnMaterialSpec, MaterialSpec
from atomstudio.structure.atom import Atom
from atomstudio.structure.cell import Cell
from atomstudio.structure.structure import Structure


def test_preview_builder_uses_clean_default_material_profile():
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        cell=Cell(),
    )
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
        }
    )

    scene = build_preview_scene(structure, cfg, PreviewSettings())

    assert tuple(scene.atoms.colors[0]) == (1.0, 0.051, 0.051, 1.0)
    assert scene.metadata["material_style"] == "clean"


def test_preview_builder_prefers_atom_color_override():
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0), color=(0.2, 0.3, 0.4, 1.0))],
        cell=Cell(),
    )
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
        }
    )

    scene = build_preview_scene(structure, cfg, PreviewSettings())

    assert tuple(scene.atoms.colors[0]) == (0.2, 0.3, 0.4, 1.0)


def test_preview_builder_applies_atom_style_rules_to_representation_and_radius():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0)),
        ],
        cell=Cell(),
    )
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {
                "scene_style": "default",
                "atom_styles": {"focus": {"representation": "ball_stick", "radius": 0.33}},
                "atom_style_rules": [{"selector": {"symbol": "O"}, "style": "focus"}],
            },
            "structure": {"representation": "space_filling"},
        }
    )

    scene = build_preview_scene(structure, cfg, PreviewSettings())

    assert scene.atoms.representations[0] == "ball_stick"
    assert scene.atoms.radii[0] == 0.33


def test_preview_material_payload_from_principled_spec_keeps_principled_fields_only():
    payload = PreviewMaterialPayload.from_material_like(
        MaterialSpec(color=(0.1, 0.2, 0.3, 0.4), roughness=0.51, specular=0.26, metallic=0.18, ior=1.45)
    )

    assert payload.pipeline == "principled"
    assert payload.color == (0.1, 0.2, 0.3, 0.4)
    assert payload.roughness == pytest.approx(0.51)
    assert payload.specular == pytest.approx(0.26)
    assert payload.metallic == pytest.approx(0.18)
    assert payload.ior == pytest.approx(1.45)
    assert payload.shadow_area is None
    assert payload.highlight_direction is None


def test_preview_material_payload_from_handdrawn_spec_keeps_handdrawn_fields():
    payload = PreviewMaterialPayload.from_material_like(
        HandDrawnMaterialSpec(
            color=(0.3, 0.4, 0.5, 1.0),
            roughness=0.88,
            specular=0.05,
            shadow_area=0.41,
            shadow_strength=0.37,
            highlight_strength=0.22,
            highlight_direction=(0.7, 0.6, 0.1),
            outline_surface=2.4,
            outline_bond=1.7,
        )
    )

    assert payload.pipeline == "handdrawn"
    assert payload.color == (0.3, 0.4, 0.5, 1.0)
    assert payload.shadow_area == pytest.approx(0.41)
    assert payload.shadow_strength == pytest.approx(0.37)
    assert payload.highlight_strength == pytest.approx(0.22)
    assert payload.highlight_direction == (0.7, 0.6, 0.1)
    assert payload.outline_surface == pytest.approx(2.4)
    assert payload.outline_bond == pytest.approx(1.7)
