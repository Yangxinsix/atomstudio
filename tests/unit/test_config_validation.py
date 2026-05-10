from pathlib import Path

import pytest

from atomstudio.render.config import ConfigError, load_batch_config, validate_config_file


ROOT = Path(__file__).resolve().parents[2]


def test_validate_ok_v2():
    cfg = validate_config_file(str(ROOT / "configs/examples/water_single.yaml"))
    assert cfg["version"] == 2
    assert len(cfg["jobs"]) == 1


def test_validate_handdrawn_v2():
    cfg = validate_config_file(str(ROOT / "configs/examples/handdrawn_surface_v2.yaml"))
    assert cfg["version"] == 2
    assert cfg["jobs"][0]["style"]["scene_style"] == "handdrawn"


def test_load_batch_config_applies_scene_style_tokens_before_dataclass_defaults(tmp_path: Path):
    p = tmp_path / "darklab_defaults.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/water.xyz, frames: last}\n"
        "    output: {path: /tmp/a.png}\n"
        "    style: {scene_style: darklab}\n",
        encoding="utf-8",
    )
    cfg = load_batch_config(str(p))
    job = cfg.jobs[0]

    assert job.structure.representation == "ball_stick"
    assert job.camera.fit_mode == "ase_gui"
    assert job.camera.fit_padding == pytest.approx(0.12)
    assert job.render.color_management["exposure"] == pytest.approx(-0.15)


def test_validate_reject_unknown_structure_field(tmp_path: Path):
    p = tmp_path / "bad_unknown_structure_field.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/water.xyz, frames: last}\n"
        "    output: {path: /tmp/a.png}\n"
        "    style: {scene_style: default}\n"
        "    structure:\n"
        "      space_filling_radius_basis: schematic\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Unknown structure field"):
        validate_config_file(str(p))


def test_validate_reject_removed_render_output_path(tmp_path: Path):
    p = tmp_path / "render_output_path_only.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/water.xyz, frames: last}\n"
        "    output: {}\n"
        "    render: {output_path: /tmp/a.png}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="render.output_path"):
        validate_config_file(str(p))


def test_validate_reject_v1(tmp_path: Path):
    p = tmp_path / "bad_v1.yaml"
    p.write_text(
        "version: 1\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/water.xyz, frames: last}\n"
        "    render: {engine: eevee, samples: 4, resolution: [320,320], transparent_bg: true}\n"
        "    output: {dir: outputs/x}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        validate_config_file(str(p))


def test_validate_missing_required_field(tmp_path: Path):
    p = tmp_path / "bad_missing.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {frames: last}\n"
        "    output: {path: /tmp/a.png}\n"
        "    render: {engine: eevee, samples: 4, resolution: [320,320], transparent_bg: true}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        validate_config_file(str(p))


def test_validate_reject_removed_publication(tmp_path: Path):
    p = tmp_path / "bad_preset_publication.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/water.xyz, frames: last}\n"
        "    output: {path: /tmp/a.png}\n"
        "    style: {scene_style: publication}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Unknown style"):
        validate_config_file(str(p))


def test_validate_reject_removed_handdrawn_legacy(tmp_path: Path):
    p = tmp_path / "bad_preset_handdrawn_legacy.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/water.xyz, frames: last}\n"
        "    output: {path: /tmp/a.png}\n"
        "    style: {scene_style: handdrawn_legacy}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Unknown style"):
        validate_config_file(str(p))


def test_validate_reject_style_preset_key(tmp_path: Path):
    p = tmp_path / "bad_style_preset.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/water.xyz, frames: last}\n"
        "    output: {path: /tmp/a.png}\n"
        "    style: {preset: default}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="style.preset"):
        validate_config_file(str(p))


def test_validate_reject_lighting_preset_key(tmp_path: Path):
    p = tmp_path / "bad_lighting_preset.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/water.xyz, frames: last}\n"
        "    output: {path: /tmp/a.png}\n"
        "    style: {scene_style: default}\n"
        "    lighting: {preset: three_point}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="lighting.preset"):
        validate_config_file(str(p))


def test_validate_reject_bond_cutoff_scale_key(tmp_path: Path):
    p = tmp_path / "bad_bond_cutoff_scale.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/water.xyz, frames: last}\n"
        "    output: {path: /tmp/a.png}\n"
        "    style: {scene_style: default}\n"
        "    structure: {bond_cutoff_scale: 1.2}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="bond_cutoff_scale"):
        validate_config_file(str(p))


def test_validate_reject_top_level_job_key(tmp_path: Path):
    p = tmp_path / "bad_job_key.yaml"
    p.write_text(
        "version: 2\n"
        "job:\n"
        "  id: x\n"
        "  input: {path: tests/data/water.xyz, frames: last}\n"
        "  output: {path: /tmp/a.png}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Top-level 'job'"):
        validate_config_file(str(p))


def test_validate_polyhedra_config_ok(tmp_path: Path):
    p = tmp_path / "poly_ok.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/silicon.cif, frames: last}\n"
        "    output: {path: /tmp/a.png}\n"
        "    style: {scene_style: default}\n"
        "    structure:\n"
        "      polyhedra:\n"
        "        enabled: true\n"
        "        include_periodic_images: true\n"
        "        default_alpha: 0.35\n"
        "        default_edge_radius: 0.015\n"
        "        rules:\n"
        "          - center_symbols: [Si]\n"
        "            neighbor_symbols: [Si]\n"
        "            min_neighbors: 4\n",
        encoding="utf-8",
    )
    cfg = validate_config_file(str(p))
    poly = cfg["jobs"][0]["structure"]["polyhedra"]
    assert poly["enabled"] is True
    assert len(poly["rules"]) == 1


def test_validate_polyhedra_reject_rules_not_list(tmp_path: Path):
    p = tmp_path / "poly_bad.yaml"
    p.write_text(
        "version: 2\n"
        "jobs:\n"
        "  - id: x\n"
        "    input: {path: tests/data/silicon.cif, frames: last}\n"
        "    output: {path: /tmp/a.png}\n"
        "    style: {scene_style: default}\n"
        "    structure:\n"
        "      polyhedra:\n"
        "        enabled: true\n"
        "        rules: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="polyhedra.rules"):
        validate_config_file(str(p))
