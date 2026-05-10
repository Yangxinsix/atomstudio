from __future__ import annotations

import pytest

from atomstudio.color_utils import parse_rgba
from atomstudio.cli import _build_debug_payload, _build_parser, _job_from_flags
from atomstudio.render.config_resolver import apply_scene_style_defaults_to_job_payload
from atomstudio.scene.lights.builder import resolve_lighting_specs
from atomstudio.scene.materials.request import MaterialRequest
from atomstudio.scene.materials.registry import MaterialRegistry
from atomstudio.scene.style_helpers import (
    build_surface_layer_map,
    material_specs_equivalent,
    render_atom_radius,
    resolve_atom_representations,
    resolve_atom_style_state,
    resolve_draw_bonds,
    resolve_draw_bonds_with_atom_representations,
    resolve_handdrawn_profile,
    resolve_representation as resolve_global_representation,
    use_atom_matched_split_bonds,
)
from atomstudio.scene.styling import resolve_atom_scene_styles, resolve_bond_scene_styles
from atomstudio.scene.world_builder import resolve_world_lighting_background
from atomstudio.config import HanddrawnStyleConfig, LightConfig, MaterialPolicy, RenderJobConfig, StyleConfig
from atomstudio.structure.atom import Atom
from atomstudio.structure.bond import Bond
from atomstudio.structure.cell import Cell
from atomstudio.structure.structure import Structure
from atomstudio.scene.materials.specs import HandDrawnMaterialSpec, MaterialSpec, handdrawn_spec_from_any
from atomstudio.style.material_style import tune_rgba
from atomstudio.style.registry import get_material_style, get_radius_style, get_scene_style
from atomstudio.style.resolver import resolve_style_bundle


def test_style_config_parses_handdrawn_block():
    style = StyleConfig.from_dict(
        {
            "scene_style": "handdrawn",
            "handdrawn": {
                "substrate_symbols": ["Mg", "O"],
                "layer_coloring": True,
                "substrate_palette": [
                    [0.70, 0.79, 0.66, 1.0],
                    [0.62, 0.79, 0.90, 1.0],
                ],
            },
        }
    )
    assert style.scene_style == "handdrawn"
    assert style.handdrawn is not None
    assert style.handdrawn.substrate_symbols == ["Mg", "O"]
    assert len(style.handdrawn.substrate_palette) == 2


def test_style_config_parses_atom_styles_and_rules():
    style = StyleConfig.from_dict(
        {
            "atom_styles": {
                "mol_ball": {"representation": "ball_stick", "radius": 0.4},
            },
            "atom_style_rules": [
                {
                    "selector": {"symbols": ["C", "O"], "indices": [0, 2]},
                    "style": "mol_ball",
                    "color": [0.2, 0.3, 0.4, 1.0],
                }
            ],
        }
    )
    assert "mol_ball" in style.atom_styles
    assert style.atom_styles["mol_ball"].representation == "ball_stick"
    assert style.atom_style_rules[0].style == "mol_ball"
    assert style.atom_style_rules[0].selector.indices == [0, 2]


def test_style_config_rejects_atom_style_rule_with_unknown_style():
    with pytest.raises(ValueError, match="unknown preset"):
        StyleConfig.from_dict(
            {
                "atom_style_rules": [
                    {"selector": {"symbol": "O"}, "style": "missing"},
                ]
            }
        )


def test_style_config_rejects_invalid_atom_representation():
    with pytest.raises(ValueError, match="ball_stick, space_filling"):
        StyleConfig.from_dict({"atom_styles": {"x": {"representation": "foo"}}})


def test_handdrawn_defaults_match_flat_cartoon_profile():
    cfg = HanddrawnStyleConfig.from_dict({})
    assert cfg.jmol_desaturate == pytest.approx(0.10)
    assert cfg.jmol_lighten == pytest.approx(0.04)
    assert cfg.light_direction == pytest.approx((0.68, 0.36, 0.62))
    assert cfg.shadow_area == pytest.approx(0.34)
    assert cfg.shadow_strength == pytest.approx(0.42)
    assert cfg.shadow_softness == pytest.approx(0.12)
    assert cfg.highlight_strength == pytest.approx(0.16)
    assert cfg.outline_surface == pytest.approx(2.0)
    assert cfg.outline_molecule == pytest.approx(2.4)
    assert cfg.outline_bond == pytest.approx(1.6)
    assert cfg.outline_secondary_thickness == pytest.approx(0.8)
    assert cfg.outline_secondary_color == pytest.approx((0.76, 0.82, 0.92, 1.0))
    assert cfg.background is None


def test_handdrawn_material_style_exposes_tunable_shadow_and_outline_spec():
    style = get_material_style("handdrawn")
    assert style.handdrawn_spec is not None
    spec = style.handdrawn_spec
    assert spec.shadow_area == pytest.approx(0.34)
    assert spec.shadow_strength == pytest.approx(0.42)
    assert spec.shadow_softness == pytest.approx(0.12)
    assert spec.outline_surface == pytest.approx(2.0)
    assert spec.outline_molecule == pytest.approx(2.4)
    assert spec.outline_bond == pytest.approx(1.6)


def test_handdrawn_spec_collects_outline_settings():
    cfg = HanddrawnStyleConfig.from_dict(
        {
            "outline_surface": 2.8,
            "outline_molecule": 3.2,
            "outline_bond": 1.4,
            "outline_secondary_thickness": 0.6,
            "outline_secondary_color": [0.2, 0.3, 0.4, 1.0],
        }
    )
    spec = handdrawn_spec_from_any(cfg)
    assert spec.outline_surface == pytest.approx(2.8)
    assert spec.outline_molecule == pytest.approx(3.2)
    assert spec.outline_bond == pytest.approx(1.4)
    assert spec.outline_secondary_thickness == pytest.approx(0.6)
    assert spec.outline_secondary_color == pytest.approx((0.2, 0.3, 0.4, 1.0))


def test_handdrawn_scene_defaults_do_not_force_bonds_and_hide_surface_bonds():
    style = get_scene_style("handdrawn")
    assert style.structure_tokens["representation"] == "space_filling"
    assert "draw_bonds" not in style.structure_tokens
    assert style.structure_tokens["draw_surface_bonds"] is False
    assert style.background == pytest.approx((1.0, 1.0, 1.0, 1.0))


def test_surface_layer_map_groups_by_z_tolerance():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=12, symbol="Mg", position=(0.0, 0.0, 0.00)),
            Atom(index=1, atomic_number=8, symbol="O", position=(1.0, 0.0, 0.02)),
            Atom(index=2, atomic_number=12, symbol="Mg", position=(0.5, 0.5, 0.55)),
            Atom(index=3, atomic_number=6, symbol="C", position=(0.5, 0.5, 1.10)),
        ],
        bonds=[],
        cell=Cell(),
    )
    layers = build_surface_layer_map(structure, {"Mg", "O"}, tolerance=0.10, enabled=True)
    assert layers[0] == layers[1]
    assert layers[2] > layers[1]
    assert 3 not in layers


def test_tune_rgba_stays_in_bounds():
    c = tune_rgba((1.0, 0.05, 0.05, 1.0), desaturate=0.3, lighten=0.2)
    assert all(0.0 <= v <= 1.0 for v in c)
    assert c[3] == 1.0


def test_material_cache_key_includes_handdrawn_signature():
    key_a = MaterialRegistry._cache_key(
        MaterialRequest.handdrawn(
            name="Atom_C",
            material=HandDrawnMaterialSpec(highlight_strength=0.20),
            role="atom",
            style_name="handdrawn",
        )
    )
    key_b = MaterialRegistry._cache_key(
        MaterialRequest.handdrawn(
            name="Atom_C",
            material=HandDrawnMaterialSpec(highlight_strength=0.45),
            role="atom",
            style_name="handdrawn",
        )
    )
    assert key_a != key_b


def test_representation_auto_defaults_to_space_filling_for_handdrawn():
    assert resolve_global_representation("auto", "handdrawn") == "space_filling"
    assert resolve_global_representation("auto", "default") == "ball_stick"


def test_handdrawn_light_direction_is_world_fixed():
    material_style = get_material_style("handdrawn")
    cfg = HanddrawnStyleConfig.from_dict({"light_direction": [0.0, 0.0, 1.0]})
    profile = resolve_handdrawn_profile(
        "handdrawn",
        material_style,
        cfg,
    )
    assert profile is not None
    assert profile.light_direction == pytest.approx((0.0, 0.0, 1.0))


def test_draw_bonds_default_depends_on_representation():
    assert resolve_draw_bonds(None, "space_filling") is False
    assert resolve_draw_bonds(None, "ball_stick") is True
    assert resolve_draw_bonds(True, "space_filling") is True


def test_draw_bonds_with_atom_representations_auto_enables_when_mixed():
    result = resolve_draw_bonds_with_atom_representations(
        draw_bonds=None,
        representation="space_filling",
        atom_representations={0: "space_filling", 1: "ball_stick"},
    )
    assert result is True


def test_draw_bonds_with_atom_representations_respects_explicit_flag():
    result = resolve_draw_bonds_with_atom_representations(
        draw_bonds=False,
        representation="space_filling",
        atom_representations={0: "space_filling", 1: "ball_stick"},
    )
    assert result is False


def test_resolve_atom_style_state_applies_rule_order_and_atom_override():
    style = StyleConfig.from_dict(
        {
            "atom_styles": {
                "ball": {
                    "representation": "ball_stick",
                    "radius": 0.45,
                    "color": [0.1, 0.2, 0.3, 1.0],
                }
            },
            "atom_style_rules": [
                {"selector": {"symbol": "O"}, "style": "ball"},
                {"selector": {"symbol": "O"}, "representation": "space_filling", "radius": 0.50},
            ],
        }
    )
    atom = Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0), radius=0.7, representation="ball_stick")
    state = resolve_atom_style_state(atom=atom, style_cfg=style, default_representation="space_filling")
    assert state.style == "ball"
    assert state.representation == "ball_stick"
    assert state.radius == pytest.approx(0.7)
    assert state.color == pytest.approx((0.1, 0.2, 0.3, 1.0))


def test_resolve_atom_representations_supports_selector_indices():
    style = StyleConfig.from_dict(
        {
            "atom_styles": {
                "ball": {"representation": "ball_stick"},
            },
            "atom_style_rules": [
                {"selector": {"indices": [1]}, "style": "ball"},
            ],
        }
    )
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0)),
        ],
        bonds=[],
        cell=Cell(),
    )
    reps = resolve_atom_representations(structure, style_cfg=style, default_representation="space_filling")
    assert reps[0] == "space_filling"
    assert reps[1] == "ball_stick"


def test_atom_style_rule_color_overrides_material_policy():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {
                "scene_style": "default",
                "material_policy": {
                    "atom_defaults": {
                        "O": {"color": [1.0, 0.0, 0.0, 1.0]},
                    }
                },
                "atom_styles": {
                    "blue": {"color": [0.0, 0.0, 1.0, 1.0]},
                },
                "atom_style_rules": [
                    {"selector": {"symbol": "O"}, "style": "blue"},
                ],
            },
        }
    )
    style_bundle = resolve_style_bundle(cfg.style)

    structure = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    atoms, _, _ = resolve_atom_scene_styles(
        structure,
        cfg,
        style_bundle=style_bundle,
        representation="space_filling",
    )
    mat = atoms[0].material
    assert mat.color == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_mixed_representation_only_keeps_ball_stick_ball_stick_bonds():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
        }
    )
    style_bundle = resolve_style_bundle(cfg.style)

    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0), representation="space_filling"),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0), representation="ball_stick"),
        ],
        bonds=[Bond(id=0, a=0, b=1, distance=1.0)],
        cell=Cell(),
    )
    atom_styles, atom_reps, _ = resolve_atom_scene_styles(
        structure,
        cfg,
        style_bundle=style_bundle,
        representation="space_filling",
    )
    bond_styles = resolve_bond_scene_styles(
        structure,
        cfg,
        style_bundle=style_bundle,
        atom_styles=atom_styles,
        atom_representations=atom_reps,
        base_representation="space_filling",
        draw_bonds=True,
    )
    assert bond_styles[0].visible is False

    structure.atoms[0].representation = "ball_stick"
    atom_styles, atom_reps, _ = resolve_atom_scene_styles(
        structure,
        cfg,
        style_bundle=style_bundle,
        representation="space_filling",
    )
    bond_styles = resolve_bond_scene_styles(
        structure,
        cfg,
        style_bundle=style_bundle,
        atom_styles=atom_styles,
        atom_representations=atom_reps,
        base_representation="space_filling",
        draw_bonds=True,
    )
    assert bond_styles[0].visible is True


def test_structure_default_hydrogen_element_scale_is_1p0():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
        }
    )
    assert cfg.structure.element_scale["H"] == pytest.approx(1.0)


def test_use_atom_matched_split_bonds_default_enabled_without_custom_bond_policy():
    policy = MaterialPolicy.from_dict({})
    assert use_atom_matched_split_bonds("principled", policy) is True


def test_use_atom_matched_split_bonds_enabled_for_handdrawn_without_custom_bond_policy():
    policy = MaterialPolicy.from_dict({})
    assert use_atom_matched_split_bonds("handdrawn", policy) is True


def test_use_atom_matched_split_bonds_disabled_for_custom_bond_policy():
    custom_policy = MaterialPolicy.from_dict({"bond_defaults": {"O-H": {"color": [1.0, 0.0, 0.0, 1.0]}}})
    assert use_atom_matched_split_bonds("principled", custom_policy) is False


def test_render_atom_radius_uses_radius_style():
    vdw = get_radius_style("vdw")
    r_vdw = render_atom_radius(
        "O",
        radius_style=vdw,
        atom_scale=1.0,
        element_scale={},
        representation="space_filling",
        space_filling_scale=1.0,
        radii_scale=0.65,
    )
    assert r_vdw == pytest.approx(vdw.radius_for("O"))

    atomic = get_radius_style("atomic")
    r_ball = render_atom_radius(
        "O",
        radius_style=atomic,
        atom_scale=1.0,
        element_scale={},
        representation="ball_stick",
        space_filling_scale=1.0,
        radii_scale=0.65,
    )
    assert r_ball == pytest.approx(atomic.radius_for("O") * 0.65)


def test_render_atom_radius_ball_stick_uses_space_filling_base_with_0p65_ratio():
    covalent = get_radius_style("covalent")
    r_space = render_atom_radius(
        "Si",
        radius_style=covalent,
        atom_scale=1.0,
        element_scale={},
        representation="space_filling",
        space_filling_scale=1.0,
        radii_scale=0.65,
    )
    r_ball = render_atom_radius(
        "Si",
        radius_style=covalent,
        atom_scale=1.0,
        element_scale={},
        representation="ball_stick",
        space_filling_scale=1.0,
        radii_scale=0.65,
    )
    assert r_ball / r_space == pytest.approx(0.65)


def test_covalent_radius_style_matches_expected_ase_samples():
    covalent = get_radius_style("covalent")
    assert covalent.radius_for("H") == pytest.approx(0.31)
    assert covalent.radius_for("C") == pytest.approx(0.76)
    assert covalent.radius_for("N") == pytest.approx(0.71)
    assert covalent.radius_for("O") == pytest.approx(0.66)
    assert covalent.radius_for("Si") == pytest.approx(1.11)
    assert covalent.radius_for("Fe") == pytest.approx(1.32)
    assert covalent.radius_for("Cu") == pytest.approx(1.32)
    assert covalent.radius_for("Zn") == pytest.approx(1.22)


def test_render_atom_radius_scales_with_atom_and_space_filling_scale():
    vdw = get_radius_style("vdw")
    r_base = render_atom_radius(
        "K",
        radius_style=vdw,
        atom_scale=1.0,
        element_scale={},
        representation="space_filling",
        space_filling_scale=1.0,
        radii_scale=0.65,
    )
    r_big = render_atom_radius(
        "K",
        radius_style=vdw,
        atom_scale=5.0,
        element_scale={},
        representation="space_filling",
        space_filling_scale=5.0,
        radii_scale=0.65,
    )
    assert r_big > r_base


def test_render_atom_radius_ball_stick_allows_custom_scale():
    atomic = get_radius_style("atomic")
    r_ball = render_atom_radius(
        "O",
        radius_style=atomic,
        atom_scale=1.0,
        element_scale={},
        representation="ball_stick",
        space_filling_scale=1.0,
        radii_scale=0.50,
    )
    assert r_ball == pytest.approx(atomic.radius_for("O") * 0.50)


def test_default_scene_style_has_full_profiles():
    style = get_scene_style("default")
    assert style.light_style == "homogeneous"
    assert style.material_style.pipeline == "principled"
    atom = style.material_style.atom_default
    assert atom.metallic == pytest.approx(0.05)
    assert atom.roughness == pytest.approx(0.32)
    assert atom.ior == pytest.approx(1.45)


def test_style_defaults_merged_into_render_job_payload():
    payload = {
        "id": "x",
        "input": {"path": "tests/data/water.xyz", "frames": "last"},
        "output": {"path": "/tmp/x.png"},
        "style": {"scene_style": "default"},
    }
    merged = apply_scene_style_defaults_to_job_payload(payload)
    cfg = RenderJobConfig.from_dict(merged)
    assert cfg.style.light_style == "homogeneous"
    assert cfg.lighting.light_style is None


def test_space_filling_defaults_to_no_bonds_when_not_explicitly_set():
    payload = {
        "id": "x",
        "input": {"path": "tests/data/water.xyz", "frames": "last"},
        "output": {"path": "/tmp/x.png"},
        "style": {"scene_style": "default"},
        "structure": {"representation": "space_filling"},
    }
    merged = apply_scene_style_defaults_to_job_payload(payload)
    cfg = RenderJobConfig.from_dict(merged)
    assert cfg.structure.draw_bonds is None
    assert resolve_draw_bonds(cfg.structure.draw_bonds, cfg.structure.representation) is False


def test_cli_rejects_removed_styles():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["render", "--style", "handdrawn_legacy", "--input", "x.xyz"])
    with pytest.raises(SystemExit):
        parser.parse_args(["render", "--style", "publication", "--input", "x.xyz"])


def test_cli_accepts_rotation_view_and_frame_scale_flags():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "render",
            "--input",
            "tests/data/water.xyz",
            "--rotation=-90x,-90y,0z",
            "--view",
            "side",
            "--camera-view",
            "front",
            "--frame-scale",
            "1.1",
        ]
    )
    assert args.rotation == "-90x,-90y,0z"
    assert args.view == "side"
    assert args.camera_view == "front"
    assert args.frame_scale == pytest.approx(1.1)


def test_cli_accepts_view_as_rotation_string():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "render",
            "--input",
            "tests/data/water.xyz",
            "--view=-90x,-90y,0z",
        ]
    )
    assert args.view == "-90x,-90y,0z"


def test_job_from_flags_writes_model_rotation_model_view_and_camera_view():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "render",
            "--input",
            "tests/data/water.xyz",
            "--rotation=-90x,-90y,0z",
            "--view",
            "front",
            "--camera-view",
            "side",
            "--frame-scale",
            "1.1",
        ]
    )
    cfg = _job_from_flags(args, "/tmp/out.png")
    assert cfg.structure.model_rotation == "-90x,-90y,0z"
    assert cfg.structure.model_view == "front"
    assert cfg.camera.rotation is None
    assert cfg.camera.view == "side"
    assert cfg.camera.frame_scale == pytest.approx(1.1)


def test_job_from_flags_writes_model_view_rotation_string():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "render",
            "--input",
            "tests/data/water.xyz",
            "--view=-90x,-90y,0z",
        ]
    )
    cfg = _job_from_flags(args, "/tmp/out.png")
    assert cfg.structure.model_view == "-90x,-90y,0z"
    assert cfg.camera.view == "top"


def test_cli_accepts_draw_bonds_toggle_flags():
    parser = _build_parser()
    args_off = parser.parse_args(["render", "--input", "tests/data/water.xyz", "--no-bonds"])
    args_on = parser.parse_args(["render", "--input", "tests/data/water.xyz", "--draw-bonds"])
    assert args_off.draw_bonds is False
    assert args_on.draw_bonds is True


def test_job_from_flags_writes_draw_bonds_and_radius_style():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "render",
            "--input",
            "tests/data/water.xyz",
            "--no-bonds",
            "--radius-style",
            "vdw",
        ]
    )
    cfg = _job_from_flags(args, "/tmp/out.png")
    assert cfg.structure.draw_bonds is False
    assert cfg.style.radius_style == "vdw"
    assert cfg.structure.radii_scale == pytest.approx(0.40)


def test_job_from_flags_accepts_covalent_radius_style():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "render",
            "--input",
            "tests/data/water.xyz",
            "--radius-style",
            "covalent",
        ]
    )
    cfg = _job_from_flags(args, "/tmp/out.png")
    assert cfg.style.radius_style == "covalent"


def test_job_from_flags_writes_radii_scale():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "render",
            "--input",
            "tests/data/water.xyz",
            "--radii-scale",
            "0.52",
        ]
    )
    cfg = _job_from_flags(args, "/tmp/out.png")
    assert cfg.structure.radii_scale == pytest.approx(0.52)


def test_job_from_flags_applies_quality_preset():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "render",
            "--input",
            "tests/data/water.xyz",
            "--quality",
            "very_high",
        ]
    )
    cfg = _job_from_flags(args, "/tmp/out.png")
    assert cfg.render.samples == 256
    assert cfg.render.resolution == (2048, 2048)


def test_job_from_flags_quality_preset_respects_explicit_render_overrides():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "render",
            "--input",
            "tests/data/water.xyz",
            "--quality",
            "high",
            "--samples",
            "72",
            "--res-x",
            "880",
            "--res-y",
            "660",
        ]
    )
    cfg = _job_from_flags(args, "/tmp/out.png")
    assert cfg.render.samples == 72
    assert cfg.render.resolution == (880, 660)


def test_structure_config_accepts_legacy_ball_stick_radius_scale_key():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "structure": {"ball_stick_radius_scale": 0.33},
        }
    )
    assert cfg.structure.radii_scale == pytest.approx(0.33)


def test_material_spec_ior_roundtrip():
    spec = MaterialSpec.from_dict({"color": [0.1, 0.2, 0.3, 1.0], "ior": 1.4, "metallic": 0.10})
    assert spec.ior == pytest.approx(1.4)
    payload = spec.to_dict()
    assert payload["ior"] == pytest.approx(1.4)


def test_material_specs_accept_named_color_assignment():
    spec = MaterialSpec()
    spec.color = "red"
    assert spec.color == parse_rgba("red")

    hand = HandDrawnMaterialSpec()
    hand.outline_secondary_color = "tab:blue"
    assert hand.outline_secondary_color == parse_rgba("tab:blue")


def test_material_specs_reject_invalid_named_color_assignment():
    spec = MaterialSpec()
    with pytest.raises(ValueError, match="命名色/3-4序列/#hex"):
        spec.color = "not_a_color"

    hand = HandDrawnMaterialSpec()
    with pytest.raises(ValueError, match="命名色/3-4序列/#hex"):
        hand.outline_secondary_color = "not_a_color"


def test_material_specs_equivalent_detects_equal_and_different_specs():
    a = MaterialSpec.from_dict({"color": [0.2, 0.3, 0.4, 1.0], "roughness": 0.5, "specular": 0.2, "ior": 1.45})
    b = MaterialSpec.from_dict({"color": [0.2, 0.3, 0.4, 1.0], "roughness": 0.5, "specular": 0.2, "ior": 1.45})
    c = MaterialSpec.from_dict({"color": [0.2, 0.31, 0.4, 1.0], "roughness": 0.5, "specular": 0.2, "ior": 1.45})
    assert material_specs_equivalent(a, b) is True
    assert material_specs_equivalent(a, c) is False


def test_resolve_world_lighting_background_tones_white_when_transparent():
    color, strength = resolve_world_lighting_background((1.0, 1.0, 1.0, 1.0), transparent_bg=True)
    assert color[:3] == pytest.approx((0.32, 0.32, 0.32))
    assert strength == pytest.approx(0.55)


def test_resolve_world_lighting_background_keeps_input_when_solid():
    color, strength = resolve_world_lighting_background((1.0, 1.0, 1.0, 1.0), transparent_bg=False)
    assert color == pytest.approx((1.0, 1.0, 1.0, 1.0))
    assert strength == pytest.approx(1.0)


def test_light_config_parses_lock_to_camera():
    light = LightConfig.from_dict(
        {
            "type": "SUN",
            "placement": "absolute",
            "vector": [0, 0, 30],
            "energy": 5,
            "lock_to_camera": True,
        }
    )
    assert light.type == "SUN"
    assert light.placement == "absolute"
    assert light.vector == (0.0, 0.0, 30.0)
    assert light.lock_to_camera is True


def test_batoms_soft_lighting_is_world_fixed():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
            "lighting": {"light_style": "batoms_soft"},
        }
    )
    lights = resolve_lighting_specs(cfg, center=(0.0, 0.0, 0.0), extent=1.0, default_light_style="batoms_soft")
    assert len(lights) >= 2
    sun = next(item for item in lights if item["type"] == "SUN")
    assert sun["energy"] == pytest.approx(5.0)
    assert sun["lock_to_camera"] is False


def test_debug_payload_includes_material_and_lighting():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
            "lighting": {"light_style": "batoms_soft"},
        }
    )
    payload = _build_debug_payload(
        cfg,
        symbols=["O", "H"],
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
    )
    assert payload["debug_material"]["material_pipeline"] == "principled"
    assert payload["debug_material"]["symbol_materials"]["O"]["ior"] == pytest.approx(1.45)
    assert payload["debug_lighting"]["effective_light_style"] == "batoms_soft"
