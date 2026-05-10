from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from atomstudio.scene.materials.specs import HandDrawnMaterialSpec, MaterialLike, as_handdrawn_spec, as_material_spec


def _normalize_selection_kind(kind: str | None) -> str:
    value = str(kind or "atom").strip().lower()
    if value not in {"atom", "bond"}:
        raise ValueError("Preview selection kind must be 'atom' or 'bond'.")
    return value


def _empty_f32(*shape: int) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _empty_i32(*shape: int) -> np.ndarray:
    return np.zeros(shape, dtype=np.int32)


def _empty_b(*shape: int) -> np.ndarray:
    return np.zeros(shape, dtype=np.bool_)


@dataclass(slots=True)
class PreviewSettings:
    show_atoms: bool = True
    show_bonds: bool = True
    show_cell: bool = True
    show_polyhedra: bool = True
    fit_padding: float = 0.10
    atom_size_scale: float = 1.0
    highlight_color: tuple[float, float, float, float] = (1.0, 0.78, 0.24, 1.0)


@dataclass(slots=True, init=False)
class PreviewSelection:
    kind: str = "atom"
    index: int | None = None

    def __init__(
        self,
        kind: str = "atom",
        index: int | None = None,
        *,
        atom_index: int | None = None,
    ) -> None:
        resolved_kind = _normalize_selection_kind(kind)
        resolved_index = index if index is not None else atom_index
        if atom_index is not None and index is not None and int(atom_index) != int(index):
            raise ValueError("PreviewSelection atom_index and index must match when both are provided.")
        object.__setattr__(self, "kind", resolved_kind)
        object.__setattr__(self, "index", None if resolved_index is None else int(resolved_index))

    @property
    def atom_index(self) -> int | None:
        if self.kind != "atom":
            return None
        return self.index

    @atom_index.setter
    def atom_index(self, value: int | None) -> None:
        object.__setattr__(self, "kind", "atom")
        object.__setattr__(self, "index", None if value is None else int(value))

    @property
    def bond_index(self) -> int | None:
        if self.kind != "bond":
            return None
        return self.index

    @bond_index.setter
    def bond_index(self, value: int | None) -> None:
        object.__setattr__(self, "kind", "bond")
        object.__setattr__(self, "index", None if value is None else int(value))

    @property
    def empty(self) -> bool:
        return self.index is None


@dataclass(slots=True)
class PreviewSelectionTarget:
    kind: str
    index: int
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = _normalize_selection_kind(self.kind)
        self.index = int(self.index)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class PreviewMaterialPayload:
    pipeline: str = "principled"
    color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    alpha: float = 1.0
    roughness: float = 0.35
    specular: float = 0.30
    metallic: float = 0.0
    ior: float | None = None
    coat: float = 0.0
    coat_roughness: float = 0.08
    specular_tint: float = 0.0
    jmol_desaturate: float | None = None
    jmol_lighten: float | None = None
    light_direction: tuple[float, float, float] | None = None
    shadow_area: float | None = None
    shadow_strength: float | None = None
    shadow_softness: float | None = None
    highlight_strength: float | None = None
    highlight_direction: tuple[float, float, float] | None = None
    highlight_arc_length: float | None = None
    highlight_band_inner: float | None = None
    highlight_band_outer: float | None = None
    outline_surface: float | None = None
    outline_molecule: float | None = None
    outline_bond: float | None = None
    outline_secondary_thickness: float | None = None
    outline_secondary_color: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        self.pipeline = str(self.pipeline).strip().lower() or "principled"
        self.color = tuple(float(v) for v in self.color)
        if len(self.color) != 4:
            raise ValueError("PreviewMaterialPayload.color must contain 4 RGBA values.")
        self.alpha = float(self.alpha)
        self.roughness = float(self.roughness)
        self.specular = float(self.specular)
        self.metallic = float(self.metallic)
        self.ior = None if self.ior is None else float(self.ior)
        self.coat = float(self.coat)
        self.coat_roughness = float(self.coat_roughness)
        self.specular_tint = float(self.specular_tint)
        self.jmol_desaturate = None if self.jmol_desaturate is None else float(self.jmol_desaturate)
        self.jmol_lighten = None if self.jmol_lighten is None else float(self.jmol_lighten)
        self.light_direction = None if self.light_direction is None else tuple(float(v) for v in self.light_direction)
        self.shadow_area = None if self.shadow_area is None else float(self.shadow_area)
        self.shadow_strength = None if self.shadow_strength is None else float(self.shadow_strength)
        self.shadow_softness = None if self.shadow_softness is None else float(self.shadow_softness)
        self.highlight_strength = None if self.highlight_strength is None else float(self.highlight_strength)
        self.highlight_direction = None if self.highlight_direction is None else tuple(
            float(v) for v in self.highlight_direction
        )
        self.highlight_arc_length = None if self.highlight_arc_length is None else float(self.highlight_arc_length)
        self.highlight_band_inner = None if self.highlight_band_inner is None else float(self.highlight_band_inner)
        self.highlight_band_outer = None if self.highlight_band_outer is None else float(self.highlight_band_outer)
        self.outline_surface = None if self.outline_surface is None else float(self.outline_surface)
        self.outline_molecule = None if self.outline_molecule is None else float(self.outline_molecule)
        self.outline_bond = None if self.outline_bond is None else float(self.outline_bond)
        self.outline_secondary_thickness = (
            None if self.outline_secondary_thickness is None else float(self.outline_secondary_thickness)
        )
        self.outline_secondary_color = (
            None if self.outline_secondary_color is None else tuple(float(v) for v in self.outline_secondary_color)
        )
        if self.outline_secondary_color is not None and len(self.outline_secondary_color) != 4:
            raise ValueError("PreviewMaterialPayload.outline_secondary_color must contain 4 RGBA values.")

    @classmethod
    def from_material_like(cls, material: MaterialLike) -> "PreviewMaterialPayload":
        if isinstance(material, HandDrawnMaterialSpec):
            spec = as_handdrawn_spec(material)
            return cls(
                pipeline="handdrawn",
                color=tuple(float(v) for v in spec.color),
                alpha=float(spec.alpha),
                roughness=float(spec.roughness),
                specular=float(spec.specular),
                metallic=0.0,
                ior=None,
                coat=0.0,
                coat_roughness=0.0,
                specular_tint=0.0,
                jmol_desaturate=float(spec.jmol_desaturate),
                jmol_lighten=float(spec.jmol_lighten),
                light_direction=tuple(float(v) for v in spec.light_direction),
                shadow_area=float(spec.shadow_area),
                shadow_strength=float(spec.shadow_strength),
                shadow_softness=float(spec.shadow_softness),
                highlight_strength=float(spec.highlight_strength),
                highlight_direction=tuple(float(v) for v in spec.highlight_direction),
                highlight_arc_length=float(spec.highlight_arc_length),
                highlight_band_inner=float(spec.highlight_band_inner),
                highlight_band_outer=float(spec.highlight_band_outer),
                outline_surface=float(spec.outline_surface),
                outline_molecule=float(spec.outline_molecule),
                outline_bond=float(spec.outline_bond),
                outline_secondary_thickness=float(spec.outline_secondary_thickness),
                outline_secondary_color=tuple(float(v) for v in spec.outline_secondary_color),
            )

        spec = as_material_spec(material)
        return cls(
            pipeline="principled",
            color=tuple(float(v) for v in spec.color),
            alpha=float(spec.alpha),
            roughness=float(spec.roughness),
            specular=float(spec.specular),
            metallic=float(spec.metallic),
            ior=spec.ior,
            coat=float(spec.coat),
            coat_roughness=float(spec.coat_roughness),
            specular_tint=float(spec.specular_tint),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "pipeline": self.pipeline,
            "color": list(self.color),
            "alpha": float(self.alpha),
            "roughness": float(self.roughness),
            "specular": float(self.specular),
            "metallic": float(self.metallic),
            "ior": self.ior,
            "coat": float(self.coat),
            "coat_roughness": float(self.coat_roughness),
            "specular_tint": float(self.specular_tint),
        }
        for key in (
            "jmol_desaturate",
            "jmol_lighten",
            "light_direction",
            "shadow_area",
            "shadow_strength",
            "shadow_softness",
            "highlight_strength",
            "highlight_direction",
            "highlight_arc_length",
            "highlight_band_inner",
            "highlight_band_outer",
            "outline_surface",
            "outline_molecule",
            "outline_bond",
            "outline_secondary_thickness",
            "outline_secondary_color",
        ):
            value = getattr(self, key)
            if value is None:
                continue
            payload[key] = list(value) if isinstance(value, tuple) else float(value)
        return payload


@dataclass(slots=True)
class PreviewAtomRecord:
    index: int
    symbol: str
    atomic_number: int
    position: tuple[float, float, float]
    radius: float
    representation: str
    style: str | None
    tag: str
    material: PreviewMaterialPayload
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = int(self.index)
        self.atomic_number = int(self.atomic_number)
        self.position = tuple(float(v) for v in self.position)
        self.radius = float(self.radius)
        self.representation = str(self.representation)
        self.style = None if self.style is None else str(self.style)
        self.tag = str(self.tag)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class PreviewBondRecord:
    id: int
    a: int
    b: int
    bond_type: str
    order: int
    distance: float
    split_ratio: float
    material_left: PreviewMaterialPayload
    material_right: PreviewMaterialPayload
    material_uniform: PreviewMaterialPayload
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.a = int(self.a)
        self.b = int(self.b)
        self.bond_type = str(self.bond_type)
        self.order = int(self.order)
        self.distance = float(self.distance)
        self.split_ratio = float(self.split_ratio)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class PreviewBounds:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    center: tuple[float, float, float]
    radius: float


@dataclass(slots=True)
class AtomPreviewBuffer:
    positions: np.ndarray = field(default_factory=lambda: _empty_f32(0, 3))
    colors: np.ndarray = field(default_factory=lambda: _empty_f32(0, 4))
    radii: np.ndarray = field(default_factory=lambda: _empty_f32(0))
    atom_indices: np.ndarray = field(default_factory=lambda: _empty_i32(0))
    atomic_numbers: np.ndarray = field(default_factory=lambda: _empty_i32(0))
    symbols: tuple[str, ...] = ()
    representations: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def empty(self) -> bool:
        return self.count == 0

    @property
    def indices(self) -> np.ndarray:
        return self.atom_indices


@dataclass(slots=True)
class BondPreviewBuffer:
    positions: np.ndarray = field(default_factory=lambda: _empty_f32(0, 2, 3))
    colors: np.ndarray = field(default_factory=lambda: _empty_f32(0, 2, 4))
    connect: np.ndarray = field(default_factory=lambda: _empty_i32(0, 2))
    bond_ids: np.ndarray = field(default_factory=lambda: _empty_i32(0))
    atom_indices: np.ndarray = field(default_factory=lambda: _empty_i32(0, 2))
    radii: np.ndarray = field(default_factory=lambda: _empty_f32(0))
    orders: np.ndarray = field(default_factory=lambda: _empty_i32(0))
    split_ratios: np.ndarray = field(default_factory=lambda: _empty_f32(0))
    split_mask: np.ndarray = field(default_factory=lambda: _empty_b(0))
    bond_types: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return self.segment_count

    @property
    def segment_count(self) -> int:
        return int(self.connect.shape[0])

    @property
    def empty(self) -> bool:
        return self.segment_count == 0


@dataclass(slots=True)
class CellPreviewBuffer:
    positions: np.ndarray = field(default_factory=lambda: _empty_f32(0, 2, 3))
    colors: np.ndarray = field(default_factory=lambda: _empty_f32(0, 4))
    connect: np.ndarray = field(default_factory=lambda: _empty_i32(0, 2))
    edge_indices: np.ndarray = field(default_factory=lambda: _empty_i32(0, 2))
    radii: np.ndarray = field(default_factory=lambda: _empty_f32(0))

    @property
    def count(self) -> int:
        return self.segment_count

    @property
    def segment_count(self) -> int:
        return int(self.connect.shape[0])

    @property
    def empty(self) -> bool:
        return self.segment_count == 0


@dataclass(slots=True)
class PolyhedraPreviewBuffer:
    vertices: np.ndarray = field(default_factory=lambda: _empty_f32(0, 3))
    faces: np.ndarray = field(default_factory=lambda: _empty_i32(0, 3))
    colors: np.ndarray = field(default_factory=lambda: _empty_f32(0, 4))
    polyhedron_ids: np.ndarray = field(default_factory=lambda: _empty_i32(0))
    center_indices: np.ndarray = field(default_factory=lambda: _empty_i32(0))
    centers: np.ndarray = field(default_factory=lambda: _empty_f32(0, 3))
    edge_colors: np.ndarray = field(default_factory=lambda: _empty_f32(0, 4))
    show_edges: np.ndarray = field(default_factory=lambda: _empty_b(0))
    edge_radii: np.ndarray = field(default_factory=lambda: _empty_f32(0))
    vertex_atom_indices: np.ndarray = field(default_factory=lambda: _empty_i32(0))
    vertex_offsets: np.ndarray = field(default_factory=lambda: _empty_i32(1))
    face_offsets: np.ndarray = field(default_factory=lambda: _empty_i32(1))

    @property
    def count(self) -> int:
        return int(self.polyhedron_ids.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    @property
    def empty(self) -> bool:
        return self.face_count == 0

    @property
    def vertex_positions(self) -> np.ndarray:
        return self.vertices


@dataclass(slots=True)
class PreviewScene:
    atoms: AtomPreviewBuffer
    bonds: BondPreviewBuffer
    cell: CellPreviewBuffer
    polyhedra: PolyhedraPreviewBuffer
    atom_records: tuple[PreviewAtomRecord, ...] = ()
    bond_records: tuple[PreviewBondRecord, ...] = ()
    selection_targets: tuple[PreviewSelectionTarget, ...] = ()
    config: Any | None = None
    style_name: str = "default"
    representation: str = "ball_stick"
    draw_bonds: bool = True
    draw_cell: bool = False
    space_filling_scale: float = 1.0
    bounds: PreviewBounds | None = None
    frame_index: int = 0
    source_path: str = ""
    bounds_min: np.ndarray = field(default_factory=lambda: _empty_f32(3))
    bounds_max: np.ndarray = field(default_factory=lambda: _empty_f32(3))
    center: np.ndarray = field(default_factory=lambda: _empty_f32(3))
    extent: float = 1.0
    selection: PreviewSelection = field(default_factory=PreviewSelection)
    metadata: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.atoms.empty and self.bonds.empty and self.cell.empty and self.polyhedra.empty

    @property
    def atom_count(self) -> int:
        return self.atoms.count

    @property
    def bond_count(self) -> int:
        return self.bonds.segment_count

    @property
    def cell_edge_count(self) -> int:
        return self.cell.segment_count

    @property
    def polyhedra_count(self) -> int:
        return self.polyhedra.face_count


@dataclass(frozen=True, slots=True)
class RenderLineSegment:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    color: tuple[float, float, float, float]
    width_px: float


@dataclass(frozen=True, slots=True)
class RenderAtom:
    index: int
    symbol: str
    position: tuple[float, float, float]
    radius: float
    representation: str
    color: tuple[float, float, float, float]
    size_px: float
    material: PreviewMaterialPayload | None = None
    record: PreviewAtomRecord | None = None


@dataclass(frozen=True, slots=True)
class RenderBond:
    index: int
    a: int
    b: int
    order: int
    color: tuple[float, float, float, float]
    width_px: float
    bond_type: str = "covalent"
    distance: float | None = None
    split_ratio: float = 0.5
    material_uniform: PreviewMaterialPayload | None = None
    material_left: PreviewMaterialPayload | None = None
    material_right: PreviewMaterialPayload | None = None
    record: PreviewBondRecord | None = None
    segments: tuple[RenderLineSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class RenderCellEdge:
    index: int
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    color: tuple[float, float, float, float]
    width_px: float


@dataclass(frozen=True, slots=True)
class RenderPolyhedron:
    index: int
    center_index: int
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    face_color: tuple[float, float, float, float]
    edge_color: tuple[float, float, float, float]
    show_edges: bool
    edge_segments: tuple[RenderLineSegment, ...]


@dataclass(frozen=True, slots=True)
class PreviewRenderScene:
    atoms: tuple[RenderAtom, ...]
    bonds: tuple[RenderBond, ...]
    cell_edges: tuple[RenderCellEdge, ...]
    polyhedra: tuple[RenderPolyhedron, ...]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    center: tuple[float, float, float]
    radius: float
    background: tuple[float, float, float, float]
    style_name: str
    render_mode: str = "principled-preview"
    atom_records: tuple[PreviewAtomRecord, ...] = ()
    bond_records: tuple[PreviewBondRecord, ...] = ()
    selection_targets: tuple[dict[str, Any], ...] = ()
    report: dict[str, Any] | None = None

    @property
    def empty(self) -> bool:
        return not self.atoms and not self.bonds and not self.cell_edges and not self.polyhedra


__all__ = [
    "AtomPreviewBuffer",
    "BondPreviewBuffer",
    "CellPreviewBuffer",
    "PreviewAtomRecord",
    "PolyhedraPreviewBuffer",
    "PreviewBounds",
    "PreviewBondRecord",
    "PreviewMaterialPayload",
    "PreviewScene",
    "PreviewSelection",
    "PreviewSelectionTarget",
    "PreviewSettings",
    "RenderAtom",
    "RenderBond",
    "RenderCellEdge",
    "RenderLineSegment",
    "RenderPolyhedron",
    "PreviewRenderScene",
]
