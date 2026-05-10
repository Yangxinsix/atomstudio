from __future__ import annotations

from atomstudio.backend.blender import scene_writer
from atomstudio.backend.blender import renderer
from atomstudio.backend.blender.renderer import build_render_scene_payload
from atomstudio.backend.blender.scene_writer import BlenderSceneWriter
from atomstudio.config import RenderJobConfig
from atomstudio.structure.atom import Atom
from atomstudio.structure.cell import Cell
from atomstudio.structure.structure import Structure


def _water_like_structure() -> Structure:
    return Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(0.96, 0.0, 0.0)),
            Atom(index=2, atomic_number=1, symbol="H", position=(-0.24, 0.93, 0.0)),
        ],
        bonds=[],
        cell=Cell(),
        source_path="tests/data/water.xyz",
        frame_index=0,
    )


def _cfg() -> RenderJobConfig:
    return RenderJobConfig.from_dict(
        {
            "id": "water",
            "input": {"path": "tests/data/water.xyz", "frames": "0"},
            "output": {"path": "/tmp/water.png"},
            "structure": {"representation": "ball_stick"},
            "style": {"scene_style": "default"},
        }
    )


def test_build_render_scene_payload_uses_scene_builder():
    payload = build_render_scene_payload(_water_like_structure(), _cfg())

    assert payload["schema"] == "atomstudio.render_scene.v1"
    assert payload["source"] == "scene_builder"
    assert payload["config"]["output"]["path"] == "/tmp/water.png"
    assert len(payload["render_scene"]["atoms"]) == 3
    assert len(payload["render_scene"]["bonds"]) == 2
    assert payload["render_scene"]["atoms"][0]["material"]["color"] == [1.0, 0.051, 0.051, 1.0]


def test_build_render_scene_payload_uses_style_background():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "water",
            "input": {"path": "tests/data/water.xyz", "frames": "0"},
            "output": {"path": "/tmp/water.png"},
            "style": {"scene_style": "default", "background": [0.1, 0.2, 0.3, 1.0]},
        }
    )

    payload = build_render_scene_payload(_water_like_structure(), cfg)

    assert payload["render_scene"]["background"] == [0.1, 0.2, 0.3, 1.0]


def test_animation_payload_uses_blender_animation_renderer(monkeypatch):
    payload = build_render_scene_payload(_water_like_structure(), _cfg())
    animation_payload = {
        "schema": "atomstudio.animation.v1",
        "config": payload["config"],
        "output_dir": "/tmp/water_frames",
        "frames": [payload],
    }
    seen = {}

    class FakeAnimationRenderer:
        def __init__(self, cfg):
            seen["cfg"] = cfg

        def render_frames(self, frames, *, output_dir):
            seen["frames"] = frames
            seen["output_dir"] = output_dir
            return {"success": True, "output_dir": output_dir, "outputs": ["/tmp/water_frames/frame_0000.png"]}

    def fail_blender_renderer(_cfg):
        raise AssertionError("animation rendering must not use per-frame BlenderRenderer")

    monkeypatch.setattr(renderer, "BlenderAnimationRenderer", FakeAnimationRenderer)
    monkeypatch.setattr(renderer, "BlenderRenderer", fail_blender_renderer)

    result = renderer.render_animation_payload(animation_payload)

    assert result["success"] is True
    assert seen["cfg"].id == "water"
    assert seen["frames"] == [payload]
    assert seen["output_dir"] == "/tmp/water_frames"


def test_blender_scene_writer_batches_bond_segments(monkeypatch):
    cfg = _cfg()
    writer = BlenderSceneWriter.__new__(BlenderSceneWriter)
    writer.cfg = cfg
    writer.default_material_pipeline = "principled"
    writer.default_material_style_name = "default"
    writer.collections = {"Bonds": object()}

    class FakeMaterials:
        def resolve(self, payload, **_kwargs):
            return f"material:{payload['color'][0]}"

    writer.materials = FakeMaterials()
    captured = {}

    def fake_build_bond_mesh_batches(segments, *, vertices, collection, name_prefix):
        captured["segments"] = segments
        captured["vertices"] = vertices
        captured["collection"] = collection
        captured["name_prefix"] = name_prefix
        return ["bond_batch_object"]

    monkeypatch.setattr(scene_writer, "build_bond_mesh_batches", fake_build_bond_mesh_batches)
    bonds = [
        {
            "id": 0,
            "segments": [
                {
                    "start": (0.0, 0.0, 0.0),
                    "end": (0.5, 0.0, 0.0),
                    "radius": 0.08,
                    "side": "left",
                    "material": {"color": (1.0, 0.0, 0.0, 1.0)},
                },
                {
                    "start": (0.5, 0.0, 0.0),
                    "end": (1.0, 0.0, 0.0),
                    "radius": 0.08,
                    "side": "right",
                    "material": {"color": (0.0, 0.0, 1.0, 1.0)},
                },
            ],
        },
        {
            "id": 1,
            "segments": [
                {
                    "start": (0.0, 1.0, 0.0),
                    "end": (1.0, 1.0, 0.0),
                    "radius": 0.08,
                    "side": "uniform",
                    "material": {"color": (1.0, 0.0, 0.0, 1.0)},
                },
            ],
        },
    ]

    objects, segment_count = writer._write_bonds(bonds)

    assert objects == ["bond_batch_object"]
    assert segment_count == 3
    assert len(captured["segments"]) == 3
    assert captured["vertices"] == cfg.structure.bond_vertices


def test_blender_scene_writer_batches_atoms(monkeypatch):
    cfg = _cfg()
    writer = BlenderSceneWriter.__new__(BlenderSceneWriter)
    writer.cfg = cfg
    writer.default_material_pipeline = "principled"
    writer.default_material_style_name = "default"
    writer.collections = {"Atoms": object()}

    class FakeMaterials:
        def resolve(self, payload, **_kwargs):
            return f"material:{payload['color'][0]}"

    writer.materials = FakeMaterials()
    captured = {}

    def fake_build_atom_mesh_batches(atoms, *, collection, name_prefix):
        captured["atoms"] = atoms
        captured["collection"] = collection
        captured["name_prefix"] = name_prefix
        return ["atom_batch_object"]

    monkeypatch.setattr(scene_writer, "build_atom_mesh_batches", fake_build_atom_mesh_batches)
    atoms = [
        {
            "index": 0,
            "symbol": "O",
            "position": (0.0, 0.0, 0.0),
            "radius": 0.6,
            "segments": 32,
            "rings": 16,
            "material": {"color": (1.0, 0.0, 0.0, 1.0)},
        },
        {
            "index": 3,
            "symbol": "H",
            "position": (1.0, 0.0, 0.0),
            "radius": 0.2,
            "segments": 16,
            "rings": 8,
            "material": {"color": (0.0, 0.0, 1.0, 1.0)},
        },
    ]

    objects, positions, atom_count = writer._write_atoms(atoms)

    assert objects == ["atom_batch_object"]
    assert atom_count == 2
    assert len(captured["atoms"]) == 2
    assert positions[0] == (0.0, 0.0, 0.0)
    assert positions[3] == (1.0, 0.0, 0.0)
