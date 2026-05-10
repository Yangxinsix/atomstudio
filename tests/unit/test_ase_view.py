import numpy as np
import pytest

from ase.utils import rotate

from atomstudio.scene.camera_builder import _rotate_axes, resolve_camera_basis
from atomstudio.scene.transforms import resolve_model_rotation_matrix
from atomstudio.config import CameraConfig, RenderJobConfig


def test_camera_config_parses_ase_view_matrix_and_rotations():
    cfg = CameraConfig.from_dict(
        {
            "projection": "orthographic",
            "fit_mode": "ase_gui",
            "rotation": "45x,10y,0z",
            "view": "FRONT",
            "frame_scale": 1.08,
            "ase_view": {
                "rotations": "-90x,-90y,0z",
                "axes_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        }
    )
    assert cfg.projection == "ORTHOGRAPHIC"
    assert cfg.rotation == "45x,10y,0z"
    assert cfg.view == "front"
    assert cfg.frame_scale == pytest.approx(1.08)
    assert cfg.ase_view.rotations == "-90x,-90y,0z"
    assert cfg.ase_view.axes_matrix == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_rotate_axes_matches_ase_rotate():
    s = "90x,0y,45z"
    a = _rotate_axes(s)
    b = np.array(rotate(s), dtype=float)
    assert np.allclose(a, b)


def test_camera_config_rejects_non_positive_frame_scale():
    with pytest.raises(ValueError, match="camera.frame_scale"):
        CameraConfig.from_dict({"frame_scale": 0.0})
    with pytest.raises(ValueError, match="camera.frame_scale"):
        CameraConfig.from_dict({"frame_scale": -1.0})


def test_camera_config_rejects_removed_ase_rotation_field():
    with pytest.raises(ValueError, match="camera.ase_rotation"):
        CameraConfig.from_dict({"ase_rotation": "-90x,-90y,0z"})


def test_camera_config_keeps_unknown_view_string_for_runtime_resolution():
    cfg = CameraConfig.from_dict({"view": "diagonal"})
    assert cfg.view == "diagonal"


def test_camera_config_accepts_rotation_only_in_view():
    cfg_rotation_only = CameraConfig.from_dict({"view": "-90x,-90y,0z"})
    assert cfg_rotation_only.view == "-90x,-90y,0z"


def test_resolve_camera_basis_accepts_rotation_in_view():
    right_a, up_a, forward_a = resolve_camera_basis(view="30x,0y,0z")
    _, up_b, forward_b = resolve_camera_basis(view="top")
    assert not np.allclose(up_a, up_b)
    assert not np.allclose(forward_a, forward_b)
    assert np.allclose(np.linalg.norm(np.array(right_a)), 1.0)
    assert np.allclose(np.linalg.norm(np.array(up_a)), 1.0)
    assert np.allclose(np.linalg.norm(np.array(forward_a)), 1.0)


def test_resolve_camera_basis_colon_view_is_not_supported_and_falls_back_top():
    right_a, up_a, forward_a = resolve_camera_basis(view="foo:90z")
    right_b, up_b, forward_b = resolve_camera_basis(view="top")
    assert np.allclose(right_a, right_b)
    assert np.allclose(up_a, up_b)
    assert np.allclose(forward_a, forward_b)


def test_resolve_camera_basis_prefers_rotation_over_view():
    right, up, forward = resolve_camera_basis(rotation="90x,0y,0z", view="side")
    axes = _rotate_axes("90x,0y,0z")
    expected_right = (axes[0, 0], axes[1, 0], axes[2, 0])
    expected_up = (-axes[0, 1], -axes[1, 1], -axes[2, 1])
    expected_forward = (-axes[0, 2], -axes[1, 2], -axes[2, 2])
    assert np.allclose(right, expected_right)
    assert np.allclose(up, expected_up)
    assert np.allclose(forward, expected_forward)


def test_structure_config_parses_model_rotation_and_model_view():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "structure": {"model_rotation": "-90x,-90y,0z", "model_view": "front"},
        }
    )
    assert cfg.structure.model_rotation == "-90x,-90y,0z"
    assert cfg.structure.model_view == "front"


def test_structure_config_defaults_model_rotation_and_model_view():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
        }
    )
    assert cfg.structure.model_rotation is None
    assert cfg.structure.model_view == "top"


def test_model_rotation_string_uses_inverse_of_ase_rotate_matrix():
    s = "-90x,-90y,0z"
    resolved = resolve_model_rotation_matrix(model_rotation=s, model_view="top")
    assert resolved is not None
    assert np.allclose(resolved, _rotate_axes(s).T)


def test_model_view_rotation_string_uses_inverse_of_ase_rotate_matrix():
    s = "30x,20y,10z"
    resolved = resolve_model_rotation_matrix(model_rotation=None, model_view=s)
    assert resolved is not None
    assert np.allclose(resolved, _rotate_axes(s).T)
