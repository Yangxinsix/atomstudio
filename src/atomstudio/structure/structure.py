from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

from atomstudio.color_utils import parse_rgba
from atomstudio.scene.materials.specs import MaterialLike, material_from_dict
from atomstudio.structure.atom import Atom, infer_atomic_number
from atomstudio.structure.bond import Bond
from atomstudio.structure.cell import Cell
from atomstudio.structure.polyhedron import Polyhedron
from atomstudio.structure.selectors import AtomSelector, BondSelector, PolyhedraSelector
from atomstudio.style.outline_style import OutlineRoleStyle

_DEFAULT_NOTEBOOK_DISPLAY_WIDTH = 480


@dataclass
class Structure:
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)
    polyhedra: list[Polyhedron] = field(default_factory=list)
    cell: Cell = field(default_factory=Cell)
    frame_index: int = 0
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms)

    def __getitem__(self, index: int | slice) -> Atom | list[Atom]:
        return self.atoms[index]

    @property
    def symbols(self) -> list[str]:
        return [a.symbol for a in self.atoms]

    @property
    def positions(self) -> list[tuple[float, float, float]]:
        return [a.position for a in self.atoms]

    @property
    def tags(self) -> list[str]:
        return [a.tag for a in self.atoms]

    @property
    def pbc(self) -> tuple[bool, bool, bool]:
        return self.cell.pbc

    @property
    def cell_vectors(self) -> list[list[float]]:
        return self.cell.vectors

    @property
    def bond_pairs(self) -> list[tuple[int, int]]:
        return [(b.a, b.b) for b in self.bonds]

    @classmethod
    def from_ase(
        cls,
        atoms: Any,
        *,
        source_path: str = "",
        frame_index: int = 0,
    ) -> "Structure":
        tags = []
        try:
            tags = [str(v) for v in atoms.get_tags().tolist()]
        except Exception:
            tags = []

        symbols = list(atoms.get_chemical_symbols())
        numbers = [int(v) for v in atoms.get_atomic_numbers().tolist()]
        positions = [tuple(map(float, p)) for p in atoms.get_positions()]
        atom_objs = [
            Atom(
                index=i,
                atomic_number=numbers[i] if i < len(numbers) else 0,
                symbol=symbol,
                position=positions[i],
                tag=tags[i] if i < len(tags) else "",
            )
            for i, symbol in enumerate(symbols)
        ]

        cell = Cell(
            vectors=[[float(v) for v in row] for row in atoms.get_cell().tolist()],
            pbc=tuple(bool(v) for v in atoms.get_pbc().tolist()),
            show=False,
            radius=0.04,
        )

        return cls(
            atoms=atom_objs,
            bonds=[],
            cell=cell,
            source_path=str(source_path),
            frame_index=int(frame_index),
            metadata={},
        )

    def compute_bonds(self, bonding_config: Any = None) -> "Structure":
        from atomstudio.structure.bonding import BondEngine
        from atomstudio.config import BondingConfig

        cfg = bonding_config if isinstance(bonding_config, BondingConfig) else BondingConfig.from_dict(bonding_config or {})
        self.bonds = BondEngine().compute(self, cfg)
        return self

    def ensure_bonds(self, bonding_config: Any = None, *, force: bool = False) -> "Structure":
        if self.bonds and (not force):
            return self
        return self.compute_bonds(bonding_config)

    def compute_polyhedra(self, polyhedra_config: Any, bonding_config: Any = None) -> "Structure":
        from atomstudio.config import BondingConfig, PolyhedraConfig
        from atomstudio.structure.polyhedra_engine import PolyhedraEngine

        pcfg = polyhedra_config if isinstance(polyhedra_config, PolyhedraConfig) else PolyhedraConfig.from_dict(polyhedra_config or {})
        bcfg = bonding_config if isinstance(bonding_config, BondingConfig) else BondingConfig.from_dict(bonding_config or {})
        if not pcfg.enabled or not pcfg.rules:
            self.polyhedra = []
            return self
        self.polyhedra = PolyhedraEngine().compute(self, pcfg, bcfg)
        return self

    def ensure_polyhedra(
        self,
        polyhedra_config: Any,
        bonding_config: Any = None,
        *,
        force: bool = False,
    ) -> "Structure":
        if self.polyhedra and (not force):
            return self
        return self.compute_polyhedra(polyhedra_config, bonding_config)

    def get_image(
        self,
        output_path: str | os.PathLike[str] | None = None,
        *,
        cfg: Any | None = None,
        return_type: str = "auto",
        display_width: int | None = None,
        display_height: int | None = None,
        overrides: dict[str, Any] | None = None,
        blender_path: str | None = None,
        timeout_seconds: int = 1800,
        **cli_kwargs: Any,
    ) -> Any:
        """Render this structure through the standard Blender backend pipeline."""
        from atomstudio.config import RenderJobConfig
        from atomstudio.render.pipeline import render_structure

        output = _normalize_output_path(output_path)
        mode = str(return_type).strip().lower()
        if mode not in {"auto", "path", "display"}:
            raise ValueError("return_type must be one of: 'auto', 'path', 'display'")
        if display_width is not None and int(display_width) <= 0:
            raise ValueError("display_width must be > 0")
        if display_height is not None and int(display_height) <= 0:
            raise ValueError("display_height must be > 0")
        width = None if display_width is None else int(display_width)
        height = None if display_height is None else int(display_height)
        if mode in {"auto", "display"} and width is None and height is None:
            width = _DEFAULT_NOTEBOOK_DISPLAY_WIDTH

        if cfg is not None and cli_kwargs:
            raise ValueError("Pass either cfg=... or CLI-like keyword arguments, not both.")

        config = cfg if isinstance(cfg, RenderJobConfig) else None
        if config is None:
            if cli_kwargs:
                from atomstudio.render.cli_like import build_render_job_config_from_cli_like_kwargs

                config = build_render_job_config_from_cli_like_kwargs(
                    output_path=str(output or _default_render_payload(self)["output"]["path"]),
                    input_path=self.source_path or "<memory>",
                    kwargs=cli_kwargs,
                    job_id="structure_get_image_cli",
                    frames="last",
                )
            else:
                config = RenderJobConfig.from_dict(_default_render_payload(self, output_path=output))
        if overrides:
            payload = config.to_dict()
            _deep_merge(payload, dict(overrides))
            config = RenderJobConfig.from_dict(payload)
        resolved_output = output
        if resolved_output is None:
            resolved_output = str(config.output.resolve_path(config.id, self.frame_index))
        config = config.with_output_path(str(resolved_output))

        result = render_structure(
            self,
            config,
            blender_path=blender_path,
            timeout_seconds=int(timeout_seconds),
        )
        if not result.success:
            raise RuntimeError(result.message or "render failed")
        resolved = str(Path(result.output_path).expanduser().resolve())

        if mode == "path":
            return resolved
        if mode == "display":
            return _display_image(resolved, width=width, height=height)
        if _is_notebook_runtime():
            return _display_image(resolved, width=width, height=height)
        return resolved

    def set_bonds_from_pairs(self, pairs: list[tuple[int, int]], *, bond_type: str = "covalent", order: int = 1) -> None:
        out: list[Bond] = []
        for i, (a, b) in enumerate(pairs):
            p1 = self.atoms[a].position
            p2 = self.atoms[b].position
            d = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
            out.append(Bond(id=i, a=int(a), b=int(b), bond_type=bond_type, order=int(order), distance=float(d)))
        self.bonds = out

    def assign_atom_style(
        self,
        selector: AtomSelector,
        *,
        style: str | None = None,
        representation: str | None = None,
        material: MaterialLike | None = None,
        color: Any | None = None,
        radius: float | None = None,
    ) -> None:
        parsed_color = _parse_optional_rgba(color, label="atom color")
        for atom in self.atoms:
            if selector.matches(atom.index, atom.symbol, atom.position, atom.tag):
                if style is not None:
                    atom.style = style
                if representation is not None:
                    atom.representation = str(representation)
                if material is not None:
                    atom.material = material
                if parsed_color is not None:
                    atom.color = parsed_color
                if radius is not None:
                    atom.radius = float(radius)
                atom.sync_color_to_material()

    def assign_bond_style(
        self,
        selector: BondSelector,
        *,
        style: str | None = None,
        material: MaterialLike | None = None,
        color: Any | None = None,
        material_a: MaterialLike | None = None,
        color_a: Any | None = None,
        material_b: MaterialLike | None = None,
        color_b: Any | None = None,
        split_ratio: float | None = None,
    ) -> None:
        parsed_color = _parse_optional_rgba(color, label="bond color")
        parsed_color_a = _parse_optional_rgba(color_a, label="bond color_a")
        parsed_color_b = _parse_optional_rgba(color_b, label="bond color_b")
        for bond in self.bonds:
            a = self.atoms[bond.a]
            b = self.atoms[bond.b]
            if selector.matches(bond.id, bond.a, bond.b, a.symbol, b.symbol, bond.distance, bond.order):
                if style is not None:
                    bond.style = style
                if material is not None:
                    bond.material = material
                if parsed_color is not None:
                    bond.color = parsed_color
                if material_a is not None:
                    bond.material_a = material_a
                if parsed_color_a is not None:
                    bond.color_a = parsed_color_a
                if material_b is not None:
                    bond.material_b = material_b
                if parsed_color_b is not None:
                    bond.color_b = parsed_color_b
                if split_ratio is not None:
                    bond.split_ratio = float(split_ratio)

    def assign_cell_style(
        self,
        *,
        style: str | None = None,
        material: MaterialLike | None = None,
        color: Any | None = None,
    ) -> None:
        parsed_color = _parse_optional_rgba(color, label="cell color")
        if style is not None:
            self.cell.style = style
        if material is not None:
            self.cell.material = material
        if parsed_color is not None:
            self.cell.color = parsed_color

    def assign_polyhedra_style(
        self,
        selector: PolyhedraSelector,
        *,
        style: str | None = None,
        material: MaterialLike | None = None,
        color: Any | None = None,
        show_edges: bool | None = None,
        edge_radius: float | None = None,
        edge_color: Any | None = None,
    ) -> None:
        parsed_color = _parse_optional_rgba(color, label="polyhedra color")
        parsed_edge_color = _parse_optional_rgba(edge_color, label="polyhedra edge_color")
        for poly in self.polyhedra:
            neighbor_count = len(poly.neighbor_indices)
            if not selector.matches(poly.id, poly.center, poly.center_symbol, neighbor_count):
                continue
            if style is not None:
                poly.style = style
            if material is not None:
                poly.material = material
            if parsed_color is not None:
                poly.color = parsed_color
            if show_edges is not None:
                poly.show_edges = bool(show_edges)
            if edge_radius is not None:
                poly.edge_radius = float(edge_radius)
            if parsed_edge_color is not None:
                poly.edge_color = parsed_edge_color
            poly.sync_color_to_material()

    def to_dict(self) -> dict[str, Any]:
        return {
            "atoms": [
                {
                    "index": a.index,
                    "atomic_number": a.atomic_number,
                    "symbol": a.symbol,
                    "position": list(a.position),
                    "radius": a.radius,
                    "segments": a.segments,
                    "rings": a.rings,
                    "material": None if a.material is None else a.material.to_dict(),
                    "color": None if a.color is None else list(a.color),
                    "style": a.style,
                    "representation": a.representation,
                    "outline": {
                        "enabled": bool(a.outline.enabled),
                        "thickness": a.outline.thickness,
                        "color": None if a.outline.color is None else list(a.outline.color),
                        "secondary_thickness": a.outline.secondary_thickness,
                        "secondary_color": (
                            None if a.outline.secondary_color is None else list(a.outline.secondary_color)
                        ),
                        "ignore_occlusion": bool(a.outline.ignore_occlusion),
                    },
                    "tag": a.tag,
                    "metadata": a.metadata,
                }
                for a in self.atoms
            ],
            "bonds": [
                {
                    "id": b.id,
                    "a": b.a,
                    "b": b.b,
                    "bond_type": b.bond_type,
                    "order": b.order,
                    "distance": b.distance,
                    "metadata": b.metadata,
                    "style": b.style,
                    "material": None if b.material is None else b.material.to_dict(),
                    "color": None if b.color is None else list(b.color),
                    "material_a": None if b.material_a is None else b.material_a.to_dict(),
                    "color_a": None if b.color_a is None else list(b.color_a),
                    "material_b": None if b.material_b is None else b.material_b.to_dict(),
                    "color_b": None if b.color_b is None else list(b.color_b),
                    "split_ratio": b.split_ratio,
                }
                for b in self.bonds
            ],
            "polyhedra": [
                {
                    "id": p.id,
                    "center": p.center,
                    "center_symbol": p.center_symbol,
                    "vertex_positions": [list(v) for v in p.vertex_positions],
                    "neighbor_indices": [int(v) for v in p.neighbor_indices],
                    "neighbor_offsets": [list(v) for v in p.neighbor_offsets],
                    "metadata": p.metadata,
                    "style": p.style,
                    "material": None if p.material is None else p.material.to_dict(),
                    "color": None if p.color is None else list(p.color),
                    "show_edges": bool(p.show_edges),
                    "edge_radius": p.edge_radius,
                    "edge_color": None if p.edge_color is None else list(p.edge_color),
                }
                for p in self.polyhedra
            ],
            "cell": {
                "vectors": self.cell.vectors,
                "pbc": list(self.cell.pbc),
                "show": self.cell.show,
                "radius": self.cell.radius,
                "style": self.cell.style,
                "material": None if self.cell.material is None else self.cell.material.to_dict(),
                "color": None if self.cell.color is None else list(self.cell.color),
            },
            "frame_index": self.frame_index,
            "source_path": self.source_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Structure":
        if "atoms" not in data:
            raise ValueError("Structure payload must contain an 'atoms' list.")

        atoms = [
            Atom(
                index=int(item.get("index", item.get("id", i))),
                atomic_number=infer_atomic_number(
                    str(item.get("symbol", item.get("element", "C"))),
                    item.get("atomic_number"),
                ),
                symbol=str(item.get("symbol", item.get("element", "C"))),
                position=(
                    float(item.get("position", [0.0, 0.0, 0.0])[0]),
                    float(item.get("position", [0.0, 0.0, 0.0])[1]),
                    float(item.get("position", [0.0, 0.0, 0.0])[2]),
                ),
                radius=float(item["radius"]) if item.get("radius") is not None else None,
                segments=int(item["segments"]) if item.get("segments") is not None else None,
                rings=int(item["rings"]) if item.get("rings") is not None else None,
                material=material_from_dict(item.get("material")) if isinstance(item.get("material"), dict) else None,
                color=parse_rgba(item.get("color")),
                style=str(item["style"]) if item.get("style") is not None else None,
                representation=(
                    str(item["representation"]) if item.get("representation") is not None else None
                ),
                outline=OutlineRoleStyle.from_any(
                    item.get("outline"),
                    fallback=OutlineRoleStyle(),
                ),
                tag=str(item.get("tag", "")),
                metadata=dict(item.get("metadata", {})),
            )
            for i, item in enumerate(data.get("atoms", []))
        ]
        bonds = [
            Bond(
                id=int(item.get("id", i)),
                a=int(item.get("a", 0)),
                b=int(item.get("b", 0)),
                bond_type=str(item.get("bond_type", "covalent")),
                order=int(item.get("order", 1)),
                distance=float(item.get("distance", 0.0)),
                metadata=dict(item.get("metadata", {})),
                style=item.get("style"),
                material=material_from_dict(item.get("material")) if isinstance(item.get("material"), dict) else None,
                color=parse_rgba(item.get("color")),
                material_a=material_from_dict(item.get("material_a")) if isinstance(item.get("material_a"), dict) else None,
                color_a=parse_rgba(item.get("color_a")),
                material_b=material_from_dict(item.get("material_b")) if isinstance(item.get("material_b"), dict) else None,
                color_b=parse_rgba(item.get("color_b")),
                split_ratio=float(item.get("split_ratio", 0.5)),
            )
            for i, item in enumerate(data.get("bonds", []))
        ]
        polyhedra = [
            Polyhedron(
                id=int(item.get("id", i)),
                center=int(item.get("center", 0)),
                center_symbol=str(item.get("center_symbol", "")),
                vertex_positions=[
                    (float(v[0]), float(v[1]), float(v[2]))
                    for v in item.get("vertex_positions", [])
                    if isinstance(v, (list, tuple)) and len(v) == 3
                ],
                neighbor_indices=[int(v) for v in item.get("neighbor_indices", [])],
                neighbor_offsets=[
                    (int(v[0]), int(v[1]), int(v[2]))
                    for v in item.get("neighbor_offsets", [])
                    if isinstance(v, (list, tuple)) and len(v) == 3
                ],
                metadata=dict(item.get("metadata", {})),
                style=str(item["style"]) if item.get("style") is not None else None,
                material=material_from_dict(item.get("material")) if isinstance(item.get("material"), dict) else None,
                color=parse_rgba(item.get("color")),
                show_edges=bool(item.get("show_edges", False)),
                edge_radius=float(item.get("edge_radius")) if item.get("edge_radius") is not None else None,
                edge_color=parse_rgba(item.get("edge_color")),
            )
            for i, item in enumerate(data.get("polyhedra", []))
            if isinstance(item, dict)
        ]
        for item in data.get("bonds", []):
            if not isinstance(item, dict):
                continue
            _reject_legacy_keys(item, {"style_ref", "material_override", "color_override"}, "bond")
        cell_data = data.get("cell", {}) if isinstance(data.get("cell"), dict) else {}
        _reject_legacy_keys(cell_data, {"style_ref", "material_override", "color_override"}, "cell")
        cell = Cell(
            vectors=[[float(v) for v in row] for row in cell_data.get("vectors", [[0.0, 0.0, 0.0] for _ in range(3)])],
            pbc=tuple(bool(v) for v in cell_data.get("pbc", [False, False, False])),
            show=bool(cell_data.get("show", False)),
            radius=float(cell_data.get("radius", 0.04)),
            style=cell_data.get("style"),
            material=material_from_dict(cell_data.get("material")) if isinstance(cell_data.get("material"), dict) else None,
            color=parse_rgba(cell_data.get("color")),
        )
        out = cls(
            atoms=atoms,
            bonds=bonds,
            polyhedra=polyhedra,
            cell=cell,
            frame_index=int(data.get("frame_index", 0)),
            source_path=str(data.get("source_path", "")),
            metadata=dict(data.get("metadata", {})),
        )

        for bond in out.bonds:
            if float(bond.distance) > 0.0:
                continue
            p1 = out.atoms[int(bond.a)].position
            p2 = out.atoms[int(bond.b)].position
            bond.distance = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
        return out


def _reject_legacy_keys(payload: dict[str, Any], legacy_keys: set[str], context: str) -> None:
    found = sorted(k for k in legacy_keys if k in payload)
    if found:
        raise ValueError(f"Legacy {context} keys are not supported: {', '.join(found)}")


def _parse_optional_rgba(value: Any, *, label: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    rgba = parse_rgba(value)
    if rgba is not None:
        return rgba
    raise ValueError(f"{label} must be a named color, 3/4-length sequence, or #RRGGBB/#RRGGBBAA.")


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
            continue
        base[k] = v


def _default_render_payload(structure: Structure, *, output_path: str | None = None) -> dict[str, Any]:
    out = output_path
    if out is None:
        fd, tmp = tempfile.mkstemp(prefix="atomstudio_", suffix=".png")
        os.close(fd)
        out = tmp
    return {
        "id": "structure_get_image",
        "input": {"path": structure.source_path or "<memory>", "frames": "last"},
        "output": {"path": str(out)},
        "style": {"scene_style": "default"},
        "render": {"engine": "cycles"},
    }


def _normalize_output_path(output_path: str | os.PathLike[str] | None) -> str | None:
    if output_path is None:
        return None
    if isinstance(output_path, os.PathLike):
        return str(output_path)
    if isinstance(output_path, str):
        return output_path
    raise TypeError("output_path must be str | os.PathLike | None. Pass RenderJobConfig via cfg=...")


def _is_notebook_runtime() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False
    try:
        shell = get_ipython()
    except Exception:
        return False
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def _display_image(path: str, *, width: int | None = None, height: int | None = None):
    try:
        from IPython.display import Image as IPyImage  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "return_type='display' requires IPython/Jupyter. Install IPython or use return_type='path'."
        ) from exc
    return IPyImage(filename=path, width=width, height=height)
