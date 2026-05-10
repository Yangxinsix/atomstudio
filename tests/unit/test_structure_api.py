from pathlib import Path
import sys
from types import ModuleType

import pytest
from ase import Atoms
from ase.build import molecule

import atomstudio.structure.structure as structure_module
from atomstudio.color_utils import parse_rgba
from atomstudio.render.results import RenderResult
from atomstudio.config import PolyhedraConfig, PolyhedraRuleConfig, RenderJobConfig
from atomstudio.structure import (
    BondingConfig,
    Structure,
    apply_style,
    compute_bonds,
    compute_polyhedra,
    render_structure_image,
)
from atomstudio.structure.atom import Atom
from atomstudio.structure.bond import Bond
from atomstudio.structure.cell import Cell
from atomstudio.structure.polyhedron import Polyhedron
from atomstudio.structure.selectors import AtomSelector
from atomstudio.scene.materials.specs import HandDrawnMaterialSpec, MaterialSpec
from atomstudio.scene.styling import resolve_atom_scene_styles
from atomstudio.style.resolver import resolve_style_bundle


def test_compute_bonds_covalent_water_like():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(0.96, 0.0, 0.0)),
            Atom(index=2, atomic_number=1, symbol="H", position=(-0.24, 0.93, 0.0)),
        ],
        bonds=[],
        cell=Cell(),
    )
    out = compute_bonds(structure, BondingConfig(cutoff_scale=1.2, min_distance=0.2))
    pairs = sorted((min(b.a, b.b), max(b.a, b.b)) for b in out.bonds)
    assert pairs == [(0, 1), (0, 2)]


def test_structure_supports_atom_index_shortcut():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0), material=MaterialSpec(color=(0.2, 0.3, 0.4, 1.0))),
            Atom(index=1, atomic_number=1, symbol="H", position=(0.96, 0.0, 0.0)),
        ],
        bonds=[],
        cell=Cell(),
    )

    assert structure[0] is structure.atoms[0]
    assert structure[0].symbol == "O"
    assert structure[0].material is not None
    assert structure[0].material.color == (0.2, 0.3, 0.4, 1.0)
    assert len(structure) == 2
    assert [atom.symbol for atom in structure[:2]] == ["O", "H"]


def test_structure_assign_atom_style_accepts_named_color():
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    structure.assign_atom_style(AtomSelector(symbol="O"), color="tomato")
    assert structure.atoms[0].material is None
    assert structure.atoms[0].color == parse_rgba("tomato")


def test_atom_color_override_preserves_style_material_properties():
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
            "style": {"scene_style": "default"},
        }
    )
    style_bundle = resolve_style_bundle(cfg.style)

    atom = Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))
    atom.color = "tomato"
    structure = Structure(atoms=[atom], bonds=[], cell=Cell())
    atoms, _, _ = resolve_atom_scene_styles(
        structure,
        cfg,
        style_bundle=style_bundle,
        representation="space_filling",
    )
    mat = atoms[0].material

    assert mat.color == parse_rgba("tomato")
    assert mat.roughness == pytest.approx(style_bundle.material_style.atom_default.roughness)
    assert mat.specular == pytest.approx(style_bundle.material_style.atom_default.specular)
    assert mat.coat == pytest.approx(style_bundle.material_style.atom_default.coat)


def test_structure_assign_atom_style_supports_representation():
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    structure.assign_atom_style(AtomSelector(symbol="O"), representation="ball_stick")
    assert structure.atoms[0].representation == "ball_stick"


def test_atom_selector_indices_filter_matches_exact_atoms():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0)),
            Atom(index=2, atomic_number=1, symbol="H", position=(-1.0, 0.0, 0.0)),
        ],
        bonds=[],
        cell=Cell(),
    )
    structure.assign_atom_style(AtomSelector(indices=[1]), color="gold")
    assert structure.atoms[0].color is None
    assert structure.atoms[1].color == parse_rgba("gold")
    assert structure.atoms[2].color is None


def test_atom_and_outline_direct_assignment_accept_named_colors():
    structure = Structure.from_ase(molecule("NH3"))
    structure[0].color = "red"
    structure[0].outline.thickness = 3.0
    structure[0].outline.color = "xkcd:charcoal"
    structure[1].outline.enabled = False

    assert structure[0].color == parse_rgba("red")
    assert structure[0].outline.color == parse_rgba("xkcd:charcoal")
    assert structure[0].outline.thickness == pytest.approx(3.0)
    assert structure[1].outline.enabled is False


def test_runtime_objects_accept_named_color_assignment():
    atom = Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))
    atom.color = "red"
    assert atom.color == parse_rgba("red")

    bond = Bond(id=0, a=0, b=1)
    bond.color = "tomato"
    bond.color_a = "tab:blue"
    bond.color_b = "xkcd:charcoal"
    assert bond.color == parse_rgba("tomato")
    assert bond.color_a == parse_rgba("tab:blue")
    assert bond.color_b == parse_rgba("xkcd:charcoal")

    cell = Cell()
    cell.color = "gold"
    assert cell.color == parse_rgba("gold")

    poly = Polyhedron(id=0, center=0, center_symbol="O")
    poly.color = "salmon"
    poly.edge_color = "black"
    assert poly.color == parse_rgba("salmon")
    assert poly.edge_color == parse_rgba("black")


def test_atom_constructor_color_does_not_create_material_override():
    atom = Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0), color="tomato")
    assert atom.material is None
    assert atom.color == parse_rgba("tomato")


def test_runtime_objects_reject_invalid_named_color_assignment():
    atom = Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="命名色/3-4序列/#hex"):
        atom.color = "not_a_color"

    bond = Bond(id=0, a=0, b=1)
    with pytest.raises(ValueError, match="命名色/3-4序列/#hex"):
        bond.color = "not_a_color"

    cell = Cell()
    with pytest.raises(ValueError, match="命名色/3-4序列/#hex"):
        cell.color = "not_a_color"

    poly = Polyhedron(id=0, center=0, center_symbol="O")
    with pytest.raises(ValueError, match="命名色/3-4序列/#hex"):
        poly.edge_color = "not_a_color"


def test_apply_style_with_object_overrides():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0)),
        ],
        bonds=[Bond(id=0, a=0, b=1, distance=1.0)],
        cell=Cell(),
    )

    apply_style(
        structure,
        "default",
        overrides={
            "atoms": [
                {
                    "selector": {"symbol": "O"},
                    "color": [0.1, 0.2, 0.3, 1.0],
                    "style": "oxygen_custom",
                }
            ],
            "bonds": [
                {
                    "selector": {"pair": "O-H"},
                    "color": [0.4, 0.5, 0.6, 1.0],
                    "style": "bond_custom",
                }
            ],
            "cell": {"color": [0.7, 0.8, 0.9, 1.0], "style": "cell_custom"},
        },
    )

    assert structure.metadata["scene_style"] == "default"
    assert structure.atoms[0].style == "oxygen_custom"
    assert structure.atoms[0].material is None
    assert structure.atoms[0].color == (0.1, 0.2, 0.3, 1.0)
    assert structure.bonds[0].style == "bond_custom"
    assert structure.bonds[0].color == (0.4, 0.5, 0.6, 1.0)
    assert structure.cell.style == "cell_custom"
    assert structure.cell.color == (0.7, 0.8, 0.9, 1.0)


def test_apply_style_accepts_named_colors():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0)),
        ],
        bonds=[Bond(id=0, a=0, b=1, distance=1.0)],
        cell=Cell(),
    )
    apply_style(
        structure,
        "default",
        overrides={
            "atoms": [{"selector": {"symbol": "O"}, "color": "tomato"}],
            "bonds": [{"selector": {"pair": "O-H"}, "color": "tab:blue"}],
            "cell": {"color": "rebeccapurple"},
        },
    )

    assert structure.atoms[0].material is None
    assert structure.atoms[0].color == parse_rgba("tomato")
    assert structure.bonds[0].color == parse_rgba("tab:blue")
    assert structure.cell.color == parse_rgba("rebeccapurple")


def test_apply_style_accepts_atom_representation_override():
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    apply_style(
        structure,
        "default",
        overrides={
            "atoms": [{"selector": {"symbol": "O"}, "representation": "space_filling"}],
        },
    )
    assert structure.atoms[0].representation == "space_filling"


def test_apply_style_supports_split_bond_segments():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0)),
        ],
        bonds=[Bond(id=0, a=0, b=1, distance=1.0)],
        cell=Cell(),
    )

    apply_style(
        structure,
        "default",
        overrides={
            "bonds": [
                {
                    "selector": {"pair": "O-H"},
                    "color_a": [0.9, 0.1, 0.1, 1.0],
                    "color_b": [0.1, 0.1, 0.9, 1.0],
                    "split_ratio": 0.35,
                }
            ]
        },
    )

    assert structure.bonds[0].color_a == (0.9, 0.1, 0.1, 1.0)
    assert structure.bonds[0].color_b == (0.1, 0.1, 0.9, 1.0)
    assert structure.bonds[0].split_ratio == 0.35


def test_apply_style_accepts_handdrawn_material_fields():
    structure = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    apply_style(
        structure,
        "handdrawn",
        overrides={
            "atoms": [
                {
                    "selector": {"symbol": "O"},
                    "material": {"color": [0.2, 0.3, 0.4, 1.0], "shadow_softness": 0.45},
                }
            ]
        },
    )
    assert isinstance(structure.atoms[0].material, HandDrawnMaterialSpec)
    assert structure.atoms[0].material is not None
    assert structure.atoms[0].material.shadow_softness == pytest.approx(0.45)


def test_structure_from_ase_maps_basic_fields():
    atoms = Atoms(
        symbols=["O", "H"],
        positions=[(0.0, 0.0, 0.0), (0.95, 0.0, 0.0)],
        cell=[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        pbc=[True, False, True],
    )
    atoms.set_tags([7, 9])
    s = Structure.from_ase(atoms, source_path="x.xyz", frame_index=3)
    assert [a.symbol for a in s.atoms] == ["O", "H"]
    assert [a.atomic_number for a in s.atoms] == [8, 1]
    assert s.positions[1] == (0.95, 0.0, 0.0)
    assert s.tags == ["7", "9"]
    assert s.pbc == (True, False, True)
    assert s.cell_vectors[2][2] == 4.0
    assert s.source_path == "x.xyz"
    assert s.frame_index == 3


def test_structure_from_dict_accepts_named_colors():
    s = Structure.from_dict(
        {
            "atoms": [{"index": 0, "symbol": "O", "position": [0.0, 0.0, 0.0], "color": "gold"}],
            "bonds": [{"id": 0, "a": 0, "b": 0, "distance": 0.0, "color": "tab:orange"}],
            "polyhedra": [{"id": 0, "center": 0, "edge_color": "xkcd:water blue"}],
            "cell": {"color": "black"},
        }
    )

    assert s.atoms[0].color == parse_rgba("gold")
    assert s.bonds[0].color == parse_rgba("tab:orange")
    assert s.polyhedra[0].edge_color == parse_rgba("xkcd:water blue")
    assert s.cell.color == parse_rgba("black")


def test_structure_dict_roundtrip_preserves_atom_representation():
    s = Structure.from_dict(
        {
            "atoms": [
                {
                    "index": 0,
                    "symbol": "O",
                    "position": [0.0, 0.0, 0.0],
                    "representation": "ball_stick",
                }
            ],
            "bonds": [],
            "cell": {},
        }
    )
    payload = s.to_dict()
    assert payload["atoms"][0]["representation"] == "ball_stick"


def test_structure_ensure_bonds_force_behavior():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(0.96, 0.0, 0.0)),
            Atom(index=2, atomic_number=1, symbol="H", position=(-0.24, 0.93, 0.0)),
        ],
        bonds=[],
        cell=Cell(),
    )
    structure.ensure_bonds(BondingConfig(cutoff_scale=1.2, min_distance=0.2))
    first = list(structure.bond_pairs)
    assert first == [(0, 1), (0, 2)]

    structure.set_bonds_from_pairs([(1, 2)])
    assert structure.bond_pairs == [(1, 2)]
    structure.ensure_bonds(BondingConfig(cutoff_scale=1.2, min_distance=0.2), force=False)
    assert structure.bond_pairs == [(1, 2)]
    structure.ensure_bonds(BondingConfig(cutoff_scale=1.2, min_distance=0.2), force=True)
    assert structure.bond_pairs == [(0, 1), (0, 2)]


def test_structure_get_image_calls_pipeline_and_returns_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_render(structure, cfg, **kwargs):
        captured["structure"] = structure
        captured["cfg"] = cfg
        captured["kwargs"] = kwargs
        out = tmp_path / "img.png"
        return RenderResult(success=True, output_path=str(out), frame_index=0, message="ok", elapsed_seconds=0.01)

    monkeypatch.setattr("atomstudio.render.pipeline.render_structure", _fake_render)

    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    out = s.get_image(
        str(tmp_path / "target.png"),
        return_type="path",
        overrides={"render": {"samples": 8}},
        blender_path="/x/blender",
        timeout_seconds=33,
    )

    assert out == str((tmp_path / "img.png").resolve())
    assert captured["structure"] is s
    cfg = captured["cfg"]
    assert getattr(cfg, "output").path == str(tmp_path / "target.png")
    assert getattr(cfg, "render").samples == 8
    kwargs = captured["kwargs"]
    assert kwargs["blender_path"] == "/x/blender"
    assert kwargs["timeout_seconds"] == 33


def test_structure_get_image_auto_returns_display_in_notebook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def _fake_render(structure, cfg, **kwargs):
        out = tmp_path / "img.png"
        return RenderResult(success=True, output_path=str(out), frame_index=0, message="ok", elapsed_seconds=0.01)

    monkeypatch.setattr("atomstudio.render.pipeline.render_structure", _fake_render)
    monkeypatch.setattr(structure_module, "_is_notebook_runtime", lambda: True)
    monkeypatch.setattr(
        structure_module,
        "_display_image",
        lambda path, *, width=None, height=None: {"kind": "display", "path": path, "width": width, "height": height},
    )

    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    out = s.get_image(str(tmp_path / "target.png"), return_type="auto")
    assert out == {"kind": "display", "path": str((tmp_path / "img.png").resolve()), "width": 480, "height": None}


def test_structure_get_image_auto_returns_path_outside_notebook(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def _fake_render(structure, cfg, **kwargs):
        out = tmp_path / "img.png"
        return RenderResult(success=True, output_path=str(out), frame_index=0, message="ok", elapsed_seconds=0.01)

    monkeypatch.setattr("atomstudio.render.pipeline.render_structure", _fake_render)
    monkeypatch.setattr(structure_module, "_is_notebook_runtime", lambda: False)

    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    out = s.get_image(str(tmp_path / "target.png"), return_type="auto")
    assert out == str((tmp_path / "img.png").resolve())


def test_structure_get_image_display_return_type_uses_ipython_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class FakeImage:
        def __init__(self, *, filename, width=None, height=None):
            self.filename = str(filename)
            self.width = width
            self.height = height

    fake_ipython = ModuleType("IPython")
    fake_ipython.get_ipython = lambda: None  # type: ignore[attr-defined]
    fake_display = ModuleType("IPython.display")
    fake_display.Image = FakeImage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)

    def _fake_render(structure, cfg, **kwargs):
        out = tmp_path / "img.png"
        return RenderResult(success=True, output_path=str(out), frame_index=0, message="ok", elapsed_seconds=0.01)

    monkeypatch.setattr("atomstudio.render.pipeline.render_structure", _fake_render)

    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    out = s.get_image(str(tmp_path / "target.png"), return_type="display")
    assert isinstance(out, FakeImage)
    assert out.filename == str((tmp_path / "img.png").resolve())
    assert out.width == 480
    assert out.height is None


def test_structure_get_image_display_requires_ipython(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "IPython", None)
    monkeypatch.setitem(sys.modules, "IPython.display", None)
    with pytest.raises(RuntimeError, match="IPython/Jupyter"):
        structure_module._display_image("/tmp/nonexistent.png")


def test_structure_get_image_display_size_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class FakeImage:
        def __init__(self, *, filename, width=None, height=None):
            self.filename = str(filename)
            self.width = width
            self.height = height

    fake_ipython = ModuleType("IPython")
    fake_ipython.get_ipython = lambda: None  # type: ignore[attr-defined]
    fake_display = ModuleType("IPython.display")
    fake_display.Image = FakeImage  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)

    def _fake_render(structure, cfg, **kwargs):
        out = tmp_path / "img.png"
        return RenderResult(success=True, output_path=str(out), frame_index=0, message="ok", elapsed_seconds=0.01)

    monkeypatch.setattr("atomstudio.render.pipeline.render_structure", _fake_render)
    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    out = s.get_image(str(tmp_path / "target.png"), return_type="display", display_width=420, display_height=280)
    assert isinstance(out, FakeImage)
    assert out.width == 420
    assert out.height == 280


def test_structure_get_image_rejects_unknown_return_type():
    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    with pytest.raises(ValueError, match="return_type"):
        s.get_image("x.png", return_type="unknown")


def test_structure_get_image_rejects_non_positive_display_size():
    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    with pytest.raises(ValueError, match="display_width"):
        s.get_image("x.png", return_type="path", display_width=0)
    with pytest.raises(ValueError, match="display_height"):
        s.get_image("x.png", return_type="path", display_height=0)


def test_structure_get_image_accepts_cli_like_kwargs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_render(structure, cfg, **kwargs):
        captured["cfg"] = cfg
        out = tmp_path / "img.png"
        return RenderResult(success=True, output_path=str(out), frame_index=0, message="ok", elapsed_seconds=0.01)

    monkeypatch.setattr("atomstudio.render.pipeline.render_structure", _fake_render)
    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    out = s.get_image(
        str(tmp_path / "target.png"),
        return_type="path",
        style="handdrawn",
        color_style="jmol",
        material_style="handdrawn",
        light_style="handdrawn_soft",
        radius_style="atomic",
        light_intensity=1.7,
        representation="ball_stick",
        engine="eevee",
        device="cpu",
        samples=12,
        res_x=640,
        res_y=480,
        seed=11,
        atom_scale=1.2,
        radii_scale=0.33,
        bond_radius=0.05,
        draw_bonds=False,
        draw_cell=False,
        rotation="-90x,-90y,0z",
        view="side",
        camera_view="front",
        frame_scale=1.1,
        transparent_bg=False,
    )

    assert out == str((tmp_path / "img.png").resolve())
    cfg = captured["cfg"]
    assert getattr(cfg, "style").scene_style == "handdrawn"
    assert getattr(cfg, "style").color_style == "jmol"
    assert getattr(cfg, "style").material_style == "handdrawn"
    assert getattr(cfg, "style").light_style == "handdrawn_soft"
    assert getattr(cfg, "style").radius_style == "atomic"
    assert getattr(cfg, "lighting").intensity == pytest.approx(1.7)
    assert getattr(cfg, "structure").representation == "ball_stick"
    assert getattr(cfg, "structure").space_filling_scale == "auto"
    assert getattr(cfg, "structure").draw_bonds is False
    assert getattr(cfg, "structure").draw_cell is False
    assert getattr(cfg, "structure").model_rotation == "-90x,-90y,0z"
    assert getattr(cfg, "structure").model_view == "side"
    assert getattr(cfg, "camera").rotation is None
    assert getattr(cfg, "camera").view == "front"
    assert getattr(cfg, "camera").frame_scale == pytest.approx(1.1)
    assert getattr(cfg, "render").engine == "eevee"
    assert getattr(cfg, "render").device == "cpu"
    assert getattr(cfg, "render").samples == 12
    assert getattr(cfg, "render").resolution == (640, 480)
    assert getattr(cfg, "render").transparent_bg is False
    assert getattr(cfg, "render").seed == 11


def test_structure_get_image_applies_quality_preset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    def _fake_render(structure, cfg, **kwargs):
        captured["cfg"] = cfg
        out = tmp_path / "img.png"
        return RenderResult(success=True, output_path=str(out), frame_index=0, message="ok", elapsed_seconds=0.01)

    monkeypatch.setattr("atomstudio.render.pipeline.render_structure", _fake_render)
    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    out = s.get_image(str(tmp_path / "target.png"), return_type="path", quality="high")

    assert out == str((tmp_path / "img.png").resolve())
    cfg = captured["cfg"]
    assert getattr(cfg, "render").samples == 128
    assert getattr(cfg, "render").resolution == (1536, 1536)


def test_structure_get_image_quality_preset_respects_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured: dict[str, object] = {}

    def _fake_render(structure, cfg, **kwargs):
        captured["cfg"] = cfg
        out = tmp_path / "img.png"
        return RenderResult(success=True, output_path=str(out), frame_index=0, message="ok", elapsed_seconds=0.01)

    monkeypatch.setattr("atomstudio.render.pipeline.render_structure", _fake_render)
    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    out = s.get_image(
        str(tmp_path / "target.png"),
        return_type="path",
        quality="very_high",
        samples=40,
        res_x=900,
        res_y=700,
    )

    assert out == str((tmp_path / "img.png").resolve())
    cfg = captured["cfg"]
    assert getattr(cfg, "render").samples == 40
    assert getattr(cfg, "render").resolution == (900, 700)


def test_structure_get_image_rejects_unknown_quality_preset():
    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    with pytest.raises(ValueError, match="render quality"):
        s.get_image("x.png", return_type="path", quality="ultra")


def test_structure_get_image_rejects_unknown_cli_like_kwargs():
    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    with pytest.raises(ValueError, match="Unknown CLI-like get_image args"):
        s.get_image("x.png", return_type="path", not_a_real_cli_flag=1)


def test_structure_get_image_rejects_cfg_and_cli_kwargs_together():
    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
        }
    )
    with pytest.raises(ValueError, match="either cfg"):
        s.get_image("x.png", cfg=cfg, style="handdrawn")


def test_structure_get_image_rejects_legacy_positional_cfg():
    s = Structure(
        atoms=[Atom(index=0, atomic_number=8, symbol="O", position=(0.0, 0.0, 0.0))],
        bonds=[],
        cell=Cell(),
    )
    cfg = RenderJobConfig.from_dict(
        {
            "id": "x",
            "input": {"path": "tests/data/water.xyz", "frames": "last"},
            "output": {"path": "/tmp/x.png"},
        }
    )
    with pytest.raises(TypeError, match="cfg="):
        s.get_image(cfg)


def test_render_structure_image_wrapper_uses_new_get_image_signature(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_get_image(
        self,
        output_path=None,
        *,
        cfg=None,
        return_type="auto",
        display_width=None,
        display_height=None,
        overrides=None,
        blender_path=None,
        timeout_seconds=1800,
        **cli_kwargs,
    ):
        captured["output_path"] = output_path
        captured["cfg"] = cfg
        captured["return_type"] = return_type
        captured["display_width"] = display_width
        captured["display_height"] = display_height
        captured["overrides"] = overrides
        captured["blender_path"] = blender_path
        captured["timeout_seconds"] = timeout_seconds
        captured["cli_kwargs"] = dict(cli_kwargs)
        return "ok"

    monkeypatch.setattr(Structure, "get_image", _fake_get_image)
    s = Structure(atoms=[], bonds=[], cell=Cell())
    cfg = object()
    out = render_structure_image(
        s,
        "/tmp/out.png",
        cfg=cfg,
        return_type="path",
        display_width=333,
        display_height=222,
        overrides={"render": {"samples": 4}},
        blender_path="/x/blender",
        timeout_seconds=7,
        style="handdrawn",
    )

    assert out == "ok"
    assert captured["output_path"] == "/tmp/out.png"
    assert captured["cfg"] is cfg
    assert captured["return_type"] == "path"
    assert captured["display_width"] == 333
    assert captured["display_height"] == 222
    assert captured["overrides"] == {"render": {"samples": 4}}
    assert captured["blender_path"] == "/x/blender"
    assert captured["timeout_seconds"] == 7
    assert captured["cli_kwargs"] == {"style": "handdrawn"}


def test_structure_no_longer_exposes_blender_build_methods():
    assert not hasattr(Structure, "build_atoms")
    assert not hasattr(Structure, "build_bonds")


def test_structure_polyhedra_roundtrip_and_style_override():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=6, symbol="C", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(1.0, 0.0, 0.0)),
            Atom(index=2, atomic_number=1, symbol="H", position=(-1.0, 0.0, 0.0)),
            Atom(index=3, atomic_number=1, symbol="H", position=(0.0, 1.0, 0.0)),
            Atom(index=4, atomic_number=1, symbol="H", position=(0.0, -1.0, 0.0)),
        ],
        bonds=[],
        polyhedra=[
            Polyhedron(
                id=0,
                center=0,
                center_symbol="C",
                vertex_positions=[(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)],
                neighbor_indices=[1, 2, 3, 4],
                neighbor_offsets=[(0, 0, 0)] * 4,
            )
        ],
        cell=Cell(),
    )

    payload = structure.to_dict()
    restored = Structure.from_dict(payload)
    assert len(restored.polyhedra) == 1
    assert restored.polyhedra[0].center == 0
    assert restored.polyhedra[0].neighbor_indices == [1, 2, 3, 4]

    apply_style(
        restored,
        "default",
        overrides={
            "polyhedra": [
                {
                    "selector": {"center_symbol": "C"},
                    "color": [0.2, 0.4, 0.8, 0.35],
                    "show_edges": True,
                    "edge_radius": 0.02,
                    "edge_color": [0.1, 0.1, 0.1, 1.0],
                }
            ]
        },
    )
    poly = restored.polyhedra[0]
    assert poly.color == (0.2, 0.4, 0.8, 0.35)
    assert poly.show_edges is True
    assert poly.edge_radius == 0.02
    assert poly.edge_color == (0.1, 0.1, 0.1, 1.0)


def test_compute_polyhedra_api_function():
    structure = Structure(
        atoms=[
            Atom(index=0, atomic_number=6, symbol="C", position=(0.0, 0.0, 0.0)),
            Atom(index=1, atomic_number=1, symbol="H", position=(0.7, 0.7, 0.7)),
            Atom(index=2, atomic_number=1, symbol="H", position=(-0.7, -0.7, 0.7)),
            Atom(index=3, atomic_number=1, symbol="H", position=(-0.7, 0.7, -0.7)),
            Atom(index=4, atomic_number=1, symbol="H", position=(0.7, -0.7, -0.7)),
        ],
        bonds=[],
        cell=Cell(),
    )
    cfg = PolyhedraConfig(
        enabled=True,
        rules=[PolyhedraRuleConfig(center_symbols=["C"], neighbor_symbols=["H"], min_neighbors=4, max_neighbors=4)],
    )
    out = compute_polyhedra(structure, cfg, BondingConfig(cutoff_scale=1.3))
    assert out is structure
    assert len(structure.polyhedra) == 1
