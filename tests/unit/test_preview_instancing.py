from __future__ import annotations

import numpy as np
import pytest

from atomstudio.app.summary import build_structure_summary
from atomstudio.preview.mesh_builder import build_atom_instance_payload, build_bond_instance_payload


def test_atom_instance_payload_uses_one_instance_per_atom() -> None:
    payload = build_atom_instance_payload(
        (
            {"position": (1.0, 2.0, 3.0), "radius": 0.5, "face_color": (1.0, 0.0, 0.0, 1.0)},
            {"position": (4.0, 5.0, 6.0), "radius": 0.25, "face_color": (0.0, 1.0, 0.0, 1.0)},
        )
    )

    assert payload["instance_positions"].shape == (2, 3)
    assert payload["instance_transforms"].shape == (2, 3, 3)
    assert payload["instance_colors"].shape == (2, 4)
    assert payload["instance_positions"][0] == pytest.approx((1.0, 2.0, 3.0))
    assert payload["instance_transforms"][0] == pytest.approx(np.eye(3) * 0.5)


def test_bond_instance_payload_uses_one_instance_per_segment() -> None:
    payload = build_bond_instance_payload(
        (
            {
                "segments": (
                    {"start": (0.0, 0.0, 0.0), "end": (1.0, 0.0, 0.0), "width": 1.8, "color": (1.0, 0.0, 0.0, 1.0)},
                    {"start": (1.0, 0.0, 0.0), "end": (2.0, 0.0, 0.0), "width": 1.8, "color": (0.0, 0.0, 1.0, 1.0)},
                )
            },
        ),
        bond_scale=18.0,
    )

    assert payload["instance_positions"].shape == (2, 3)
    assert payload["instance_transforms"].shape == (2, 3, 3)
    assert payload["instance_colors"].shape == (2, 4)
    assert payload["instance_positions"][0] == pytest.approx((0.0, 0.0, 0.0))
    assert np.linalg.norm(payload["instance_transforms"][0][:, 2]) == pytest.approx(1.0)


def test_structure_summary_includes_preview_diagnostics() -> None:
    text = build_structure_summary(
        structure=None,
        preview_scene=None,
        graphics={
            "opengl_version": "4.6",
            "renderer": "GPU",
            "qt_platform": "xcb",
            "vispy_backend": "PySide6",
            "gl_backend": "vispy.gloo.gl.glplus",
            "instancing_requested": True,
            "instancing_supported": True,
            "instancing_reason": "available",
            "preview_renderer": "instanced",
            "display": ":0",
            "preview_instances": {"mode": "instanced", "atoms": 200, "bond_segments": 496},
        },
    )

    assert "Qt platform: xcb" in text
    assert "GL backend: vispy.gloo.gl.glplus" in text
    assert "Instancing requested: True" in text
    assert "Instancing supported: True" in text
    assert "Preview renderer: instanced" in text
    assert "Preview instances: mode=instanced, atoms=200, bond_segments=496" in text
