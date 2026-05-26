from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atomstudio.config import RenderJobConfig
from atomstudio.scene.materials.specs import MaterialLike


Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]


def _vec3(value: tuple[float, float, float] | list[float]) -> Vec3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _vec4(value: tuple[float, float, float, float] | list[float]) -> Vec4:
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


@dataclass(slots=True)
class SceneBounds:
    minimum: Vec3 = (0.0, 0.0, 0.0)
    maximum: Vec3 = (0.0, 0.0, 0.0)
    center: Vec3 = (0.0, 0.0, 0.0)
    radius: float = 0.0

    @property
    def empty(self) -> bool:
        return float(self.radius) <= 1e-12


@dataclass(slots=True)
class SceneAtom:
    index: int
    symbol: str
    atomic_number: int
    position: Vec3
    radius: float
    representation: str
    material: MaterialLike
    selection_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    style: str | None = None
    tag: str = ""

    def __post_init__(self) -> None:
        self.index = int(self.index)
        self.symbol = str(self.symbol)
        self.atomic_number = int(self.atomic_number)
        self.position = _vec3(self.position)
        self.radius = float(self.radius)
        self.representation = str(self.representation)
        self.selection_payload = dict(self.selection_payload)
        self.metadata = dict(self.metadata)
        self.style = None if self.style is None else str(self.style)
        self.tag = str(self.tag)


@dataclass(slots=True)
class SceneBondSegment:
    start: Vec3
    end: Vec3
    radius: float
    material: MaterialLike
    order_index: int = 0
    side: str = "uniform"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.start = _vec3(self.start)
        self.end = _vec3(self.end)
        self.radius = float(self.radius)
        self.order_index = int(self.order_index)
        self.side = str(self.side)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class SceneBond:
    id: int
    a: int
    b: int
    order: int
    bond_type: str
    distance: float
    radius: float
    segments: tuple[SceneBondSegment, ...] = ()
    material_uniform: MaterialLike | None = None
    material_left: MaterialLike | None = None
    material_right: MaterialLike | None = None
    selection_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    split_ratio: float = 0.5
    visible: bool = True

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.a = int(self.a)
        self.b = int(self.b)
        self.order = int(self.order)
        self.bond_type = str(self.bond_type)
        self.distance = float(self.distance)
        self.radius = float(self.radius)
        self.segments = tuple(self.segments)
        self.selection_payload = dict(self.selection_payload)
        self.metadata = dict(self.metadata)
        self.split_ratio = float(self.split_ratio)
        self.visible = bool(self.visible)


@dataclass(slots=True)
class SceneCellEdge:
    index: int
    start: Vec3
    end: Vec3
    radius: float
    material: MaterialLike
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = int(self.index)
        self.start = _vec3(self.start)
        self.end = _vec3(self.end)
        self.radius = float(self.radius)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class ScenePolyhedron:
    id: int
    center: int
    center_symbol: str
    vertex_positions: tuple[Vec3, ...]
    faces: tuple[tuple[int, int, int], ...]
    material: MaterialLike
    edge_material: MaterialLike | None = None
    selection_payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    show_edges: bool = False
    edge_radius: float | None = None

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.center = int(self.center)
        self.center_symbol = str(self.center_symbol)
        self.vertex_positions = tuple(_vec3(item) for item in self.vertex_positions)
        self.faces = tuple(tuple(int(v) for v in face) for face in self.faces)
        self.selection_payload = dict(self.selection_payload)
        self.metadata = dict(self.metadata)
        self.show_edges = bool(self.show_edges)
        self.edge_radius = None if self.edge_radius is None else float(self.edge_radius)


@dataclass(slots=True)
class SceneCamera:
    projection: str
    center: Vec3
    right: Vec3
    up: Vec3
    forward: Vec3
    scale_factor: float
    lens_mm: float
    clip_start: float
    clip_end: float
    target: Vec3
    position: Vec3
    metadata: dict[str, Any] = field(default_factory=dict)
    fit_mode: str = "orbit_origin"
    fit_padding: float = 0.10
    ortho_scale: float | None = None
    distance: float | None = None
    rotation: str | None = None
    rotation_euler: Vec3 | None = None
    model_rotation: tuple[float, ...] | None = None
    model_translation: Vec3 | None = None
    view: str = "top"
    frame_scale: float = 1.0
    ase_view_rotations: str | None = None
    ase_view_axes_matrix: tuple[tuple[float, float, float], ...] | None = None
    dof_enabled: bool = False
    focus_distance: float | None = None
    aperture_fstop: float = 5.6

    def __post_init__(self) -> None:
        self.projection = str(self.projection).upper()
        self.center = _vec3(self.center)
        self.right = _vec3(self.right)
        self.up = _vec3(self.up)
        self.forward = _vec3(self.forward)
        self.scale_factor = float(self.scale_factor)
        self.lens_mm = float(self.lens_mm)
        self.clip_start = float(self.clip_start)
        self.clip_end = float(self.clip_end)
        self.target = _vec3(self.target)
        self.position = _vec3(self.position)
        self.metadata = dict(self.metadata)
        self.fit_mode = str(self.fit_mode)
        self.fit_padding = float(self.fit_padding)
        self.ortho_scale = None if self.ortho_scale is None else float(self.ortho_scale)
        self.distance = None if self.distance is None else float(self.distance)
        self.rotation = None if self.rotation is None else str(self.rotation)
        self.rotation_euler = None if self.rotation_euler is None else _vec3(self.rotation_euler)
        self.model_rotation = None if self.model_rotation is None else tuple(float(v) for v in self.model_rotation)
        self.model_translation = None if self.model_translation is None else _vec3(self.model_translation)
        self.view = str(self.view)
        self.frame_scale = float(self.frame_scale)
        if self.ase_view_axes_matrix is not None:
            self.ase_view_axes_matrix = tuple(_vec3(row) for row in self.ase_view_axes_matrix)
        self.dof_enabled = bool(self.dof_enabled)
        self.focus_distance = None if self.focus_distance is None else float(self.focus_distance)
        self.aperture_fstop = float(self.aperture_fstop)


@dataclass(slots=True)
class SceneLight:
    type: str
    location: Vec3
    energy: float
    size: float
    size_y: float | None = None
    shape: str | None = None
    color: Vec4 | None = None
    direction: Vec3 | None = None
    lock_to_camera: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = str(self.type).upper()
        self.location = _vec3(self.location)
        self.energy = float(self.energy)
        self.size = float(self.size)
        self.size_y = None if self.size_y is None else float(self.size_y)
        self.shape = None if self.shape is None else str(self.shape).upper()
        self.color = None if self.color is None else _vec4(self.color)
        self.direction = None if self.direction is None else _vec3(self.direction)
        self.lock_to_camera = bool(self.lock_to_camera)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class RenderScene:
    config: RenderJobConfig
    structure_source: str
    frame_index: int
    representation: str
    draw_bonds: bool
    draw_cell: bool
    atoms: tuple[SceneAtom, ...]
    bonds: tuple[SceneBond, ...]
    polyhedra: tuple[ScenePolyhedron, ...]
    cell_edges: tuple[SceneCellEdge, ...]
    camera: SceneCamera
    lights: tuple[SceneLight, ...]
    background: Vec4
    bounds: SceneBounds
    metadata: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    space_filling_scale: float = 1.0
    style_name: str = "default"
    color_style_name: str = "default"
    material_style_name: str = "default"
    light_style_name: str = "three_point"

    def __post_init__(self) -> None:
        self.structure_source = str(self.structure_source)
        self.frame_index = int(self.frame_index)
        self.representation = str(self.representation)
        self.draw_bonds = bool(self.draw_bonds)
        self.draw_cell = bool(self.draw_cell)
        self.atoms = tuple(self.atoms)
        self.bonds = tuple(self.bonds)
        self.polyhedra = tuple(self.polyhedra)
        self.cell_edges = tuple(self.cell_edges)
        self.lights = tuple(self.lights)
        self.background = _vec4(self.background)
        self.metadata = dict(self.metadata)
        self.report = dict(self.report)
        self.space_filling_scale = float(self.space_filling_scale)
        self.style_name = str(self.style_name)
        self.color_style_name = str(self.color_style_name)
        self.material_style_name = str(self.material_style_name)
        self.light_style_name = str(self.light_style_name)

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    @property
    def bond_count(self) -> int:
        return len(self.bonds)

    @property
    def visible_bond_count(self) -> int:
        return sum(1 for bond in self.bonds if bond.visible)

    @property
    def cell_edge_count(self) -> int:
        return len(self.cell_edges)

    @property
    def polyhedra_count(self) -> int:
        return len(self.polyhedra)
