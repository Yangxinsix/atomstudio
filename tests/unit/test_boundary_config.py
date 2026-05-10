from __future__ import annotations

import pytest

from atomstudio.config import BoundaryConfig, RenderJobConfig


def test_boundary_config_defaults():
    cfg = BoundaryConfig.from_dict({})
    assert cfg.enabled is False
    assert cfg.window_frac == [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    assert cfg.eps == pytest.approx(1e-6)


def test_boundary_config_parses_valid_payload():
    cfg = BoundaryConfig.from_dict(
        {
            "enabled": True,
            "window_frac": [[-0.1, 1.1], [0.2, 0.8], [0.0, 1.0]],
            "eps": 1e-5,
        }
    )
    assert cfg.enabled is True
    assert cfg.window_frac == [[-0.1, 1.1], [0.2, 0.8], [0.0, 1.0]]
    assert cfg.eps == pytest.approx(1e-5)


def test_boundary_config_rejects_invalid_shapes_and_ranges():
    with pytest.raises(ValueError, match="window_frac"):
        BoundaryConfig.from_dict({"window_frac": [0.0, 1.0]})

    with pytest.raises(ValueError, match="window_frac"):
        BoundaryConfig.from_dict({"window_frac": [[0.0, 1.0], [0.0, 1.0], [0.0]]})

    with pytest.raises(ValueError, match="min <= max"):
        BoundaryConfig.from_dict({"window_frac": [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]})


def test_boundary_config_rejects_negative_eps():
    with pytest.raises(ValueError, match="boundary.eps"):
        BoundaryConfig.from_dict({"eps": -1e-6})


def test_render_job_config_parses_structure_boundary_block():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "structure": {
                "boundary": {
                    "enabled": True,
                    "window_frac": [[0.1, 0.9], [0.0, 1.0], [0.0, 1.0]],
                    "eps": 1e-4,
                }
            },
        }
    )
    assert cfg.structure.boundary.enabled is True
    assert cfg.structure.boundary.window_frac == [[0.1, 0.9], [0.0, 1.0], [0.0, 1.0]]
    assert cfg.structure.boundary.eps == pytest.approx(1e-4)


def test_render_job_config_parses_pair_cutoffs():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "structure": {
                "bonding": {"pair_cutoffs": {"Ti-O": 2.30, " Sr-O ": 3.20}},
            },
        }
    )
    assert cfg.structure.bonding.pair_cutoffs == {"O-Sr": 3.20, "O-Ti": 2.30}


def test_pair_cutoffs_rejects_invalid_key_and_non_positive_value():
    with pytest.raises(ValueError, match="pair_cutoffs keys"):
        RenderJobConfig.from_dict(
            {
                "id": "x",
                "input": {"path": "tests/data/water.xyz", "frames": "last"},
                "output": {"path": "/tmp/x.png"},
                "structure": {"bonding": {"pair_cutoffs": {"TiO": 2.3}}},
            }
        )

    with pytest.raises(ValueError, match="pair_cutoffs values"):
        RenderJobConfig.from_dict(
            {
                "id": "x",
                "input": {"path": "tests/data/water.xyz", "frames": "last"},
                "output": {"path": "/tmp/x.png"},
                "structure": {"bonding": {"pair_cutoffs": {"Ti-O": 0.0}}},
            }
        )
