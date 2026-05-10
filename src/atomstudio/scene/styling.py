from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

from atomstudio.config import HanddrawnStyleConfig, MaterialPolicy, RenderJobConfig
from atomstudio.scene.materials.specs import HandDrawnMaterialSpec, MaterialLike, as_handdrawn_spec, as_material_spec
from atomstudio.scene.style_helpers import (
    material_specs_equivalent,
    render_atom_radius,
    resolve_atom_style_state,
    resolve_draw_bonds_with_atom_representations,
    resolve_handdrawn_config,
    resolve_handdrawn_profile,
    resolve_representation,
    use_atom_matched_split_bonds,
)
from atomstudio.style.material_style import tune_rgba
from atomstudio.style.registry import get_radius_style
from atomstudio.style.resolver import ResolvedStyleBundle, resolve_style_bundle
from atomstudio.structure.structure import Structure


@dataclass(slots=True)
class ResolvedAtomSceneStyle:
    atom_index: int
    symbol: str
    representation: str
    radius: float
    material: MaterialLike
    color: tuple[float, float, float, float]
    tag: str
    style_name: str | None
    atomic_number: int
    position: tuple[float, float, float]
    metadata: dict[str, object]


@dataclass(slots=True)
class ResolvedBondSceneStyle:
    bond_id: int
    a: int
    b: int
    order: int
    bond_type: str
    distance: float
    radius: float
    material_uniform: MaterialLike
    material_left: MaterialLike
    material_right: MaterialLike
    split_colors: bool
    split_ratio: float
    visible: bool
    metadata: dict[str, object]


def resolve_scene_style_bundle(cfg: RenderJobConfig) -> ResolvedStyleBundle:
    return resolve_style_bundle(cfg.style)


def resolve_scene_representation(cfg: RenderJobConfig, style_bundle: ResolvedStyleBundle) -> str:
    return resolve_representation(cfg.structure.representation, style_bundle.scene_style_name)


def resolve_atom_scene_styles(
    structure: Structure,
    cfg: RenderJobConfig,
    *,
    style_bundle: ResolvedStyleBundle,
    representation: str,
) -> tuple[list[ResolvedAtomSceneStyle], dict[int, str], float]:
    radius_style = get_radius_style(str(cfg.style.radius_style or "").strip().lower() or "atomic")
    raw_space_filling_scale = cfg.structure.space_filling_scale
    space_filling_scale = float(raw_space_filling_scale) if not isinstance(raw_space_filling_scale, str) else 1.0
    tags = structure.tags if len(structure.tags) == len(structure.symbols) else [""] * len(structure.symbols)
    policy = cfg.style.material_policy
    handdrawn_cfg = resolve_handdrawn_config(
        style_bundle.material_style.pipeline,
        cfg,
        material_style=style_bundle.material_style,
    )

    resolved: list[ResolvedAtomSceneStyle] = []
    atom_representations: dict[int, str] = {}
    for idx, atom in enumerate(structure.atoms):
        state = resolve_atom_style_state(atom=atom, style_cfg=cfg.style, default_representation=representation)
        atom_representations[int(atom.index)] = str(state.representation)
        if int(atom.index) != idx and idx not in atom_representations:
            atom_representations[idx] = str(state.representation)
        material = _resolve_atom_material_spec(
            index=idx,
            atom=atom,
            tag=str(tags[idx]),
            policy=policy,
            state=state,
            style_bundle=style_bundle,
            handdrawn_cfg=handdrawn_cfg,
        )
        radius = float(
            state.radius
            if state.radius is not None
            else render_atom_radius(
                atom.symbol,
                radius_style=radius_style,
                atom_scale=cfg.structure.atom_scale,
                element_scale=cfg.structure.element_scale,
                representation=state.representation,
                space_filling_scale=space_filling_scale,
                radii_scale=cfg.structure.radii_scale,
            )
        )
        resolved.append(
            ResolvedAtomSceneStyle(
                atom_index=int(atom.index),
                symbol=str(atom.symbol),
                representation=str(state.representation),
                radius=radius,
                material=material,
                color=tuple(float(v) for v in as_material_spec(material).color),
                tag=str(tags[idx]),
                style_name=None if state.style is None else str(state.style),
                atomic_number=int(atom.atomic_number),
                position=tuple(float(v) for v in atom.position),
                metadata={
                    "tag": str(tags[idx]),
                    "resolved_representation": str(state.representation),
                    "material_pipeline": "handdrawn"
                    if isinstance(material, HandDrawnMaterialSpec)
                    else "principled",
                    **({"resolved_style": str(state.style)} if state.style is not None else {}),
                    **dict(getattr(atom, "metadata", {}) or {}),
                },
            )
        )
    return resolved, atom_representations, space_filling_scale


def resolve_bond_scene_styles(
    structure: Structure,
    cfg: RenderJobConfig,
    *,
    style_bundle: ResolvedStyleBundle,
    atom_styles: list[ResolvedAtomSceneStyle],
    atom_representations: dict[int, str],
    base_representation: str,
    draw_bonds: bool,
) -> list[ResolvedBondSceneStyle]:
    if not structure.bonds:
        return []

    atom_materials = [item.material for item in atom_styles]
    policy = cfg.style.material_policy
    split_by_atom = use_atom_matched_split_bonds(style_bundle.material_style.pipeline, policy)
    handdrawn_cfg = resolve_handdrawn_config(
        style_bundle.material_style.pipeline,
        cfg,
        material_style=style_bundle.material_style,
    )
    mixed_representation = any(rep != base_representation for rep in atom_representations.values())

    resolved: list[ResolvedBondSceneStyle] = []
    for idx, bond in enumerate(structure.bonds):
        i = int(bond.a)
        j = int(bond.b)
        if i >= len(structure.atoms) or j >= len(structure.atoms):
            continue

        visible = bool(draw_bonds)
        if mixed_representation:
            rep_i = atom_representations.get(i, base_representation)
            rep_j = atom_representations.get(j, base_representation)
            if rep_i != "ball_stick" or rep_j != "ball_stick":
                visible = False

        distance = float(bond.distance) if float(bond.distance) > 0.0 else _distance(structure.atoms[i].position, structure.atoms[j].position)
        material_uniform = _resolve_bond_material_spec(
            bond=bond,
            i=i,
            j=j,
            si=structure.symbols[i],
            sj=structure.symbols[j],
            distance=distance,
            atom_materials=atom_materials,
            style_bundle=style_bundle,
            policy=policy,
            handdrawn_cfg=handdrawn_cfg,
        )
        material_left, material_right = _resolve_bond_side_specs(
            bond=bond,
            i=i,
            j=j,
            material_uniform=material_uniform,
            atom_materials=atom_materials,
            split_by_atom=split_by_atom,
        )
        split_colors = not material_specs_equivalent(material_left, material_right)
        resolved.append(
            ResolvedBondSceneStyle(
                bond_id=int(bond.id),
                a=i,
                b=j,
                order=max(1, int(getattr(bond, "order", 1))),
                bond_type=str(getattr(bond, "bond_type", "covalent")),
                distance=distance,
                radius=float(cfg.structure.bond_radius),
                material_uniform=material_uniform,
                material_left=material_left,
                material_right=material_right,
                split_colors=bool(split_colors),
                split_ratio=float(getattr(bond, "split_ratio", 0.5)),
                visible=visible,
                metadata={
                    **dict(getattr(bond, "metadata", {}) or {}),
                    "atom_indices": [i, j],
                    "bond_type": str(getattr(bond, "bond_type", "covalent")),
                    "order": max(1, int(getattr(bond, "order", 1))),
                    "distance": float(distance),
                    "split_ratio": float(getattr(bond, "split_ratio", 0.5)),
                    "split_colors": bool(split_colors),
                },
            )
        )
    return resolved


def resolve_draw_bonds_for_scene(
    cfg: RenderJobConfig,
    *,
    representation: str,
    atom_representations: dict[int, str],
) -> bool:
    return resolve_draw_bonds_with_atom_representations(
        draw_bonds=cfg.structure.draw_bonds,
        representation=representation,
        atom_representations=atom_representations,
    )


def resolve_cell_material(
    structure: Structure,
    cfg: RenderJobConfig,
    *,
    style_bundle: ResolvedStyleBundle,
) -> MaterialLike:
    material = (
        structure.cell.material
        or cfg.style.material_policy.cell
        or cfg.structure.cell_style.material
        or style_bundle.material_style.cell_default
    )
    if structure.cell.color is not None:
        material = replace(material, color=structure.cell.color)
    return _normalize_material(material, style_bundle=style_bundle, cfg=cfg)


def resolve_polyhedron_materials(
    polyhedron,
    cfg: RenderJobConfig,
    *,
    style_bundle: ResolvedStyleBundle,
) -> tuple[MaterialLike, MaterialLike | None]:
    poly_cfg = cfg.structure.polyhedra
    default_alpha = float(poly_cfg.default_alpha)
    if default_alpha < 0.0 or default_alpha > 1.0:
        default_alpha = 0.35
    face_spec = style_bundle.material_style.polyhedra_default
    face_color = (float(face_spec.color[0]), float(face_spec.color[1]), float(face_spec.color[2]), default_alpha)
    face_spec = replace(face_spec, color=face_color, alpha=default_alpha)
    if polyhedron.material is not None:
        face_spec = polyhedron.material
    if polyhedron.color is not None:
        face_spec = replace(face_spec, color=polyhedron.color, alpha=float(polyhedron.color[3]))
    face_spec = _normalize_material(face_spec, style_bundle=style_bundle, cfg=cfg)

    edge_spec = None
    if bool(polyhedron.show_edges):
        edge_color = polyhedron.edge_color
        if edge_color is None:
            base = as_material_spec(face_spec).color
            edge_color = (
                max(0.0, min(1.0, base[0] * 0.8)),
                max(0.0, min(1.0, base[1] * 0.8)),
                max(0.0, min(1.0, base[2] * 0.8)),
                1.0,
            )
        edge_spec = replace(
            style_bundle.material_style.bond_default,
            color=edge_color,
            alpha=float(edge_color[3]),
        )
        edge_spec = _normalize_material(edge_spec, style_bundle=style_bundle, cfg=cfg)
    return face_spec, edge_spec


def _resolve_atom_material_spec(
    *,
    index: int,
    atom,
    tag: str,
    policy: MaterialPolicy,
    state,
    style_bundle: ResolvedStyleBundle,
    handdrawn_cfg: HanddrawnStyleConfig | None,
) -> MaterialLike:
    base_material = replace(style_bundle.material_style.atom_for(atom.symbol), color=style_bundle.color_style.color_for(atom.symbol))
    fallback = _fallback_atom_material(atom.symbol, style_bundle=style_bundle, handdrawn_cfg=handdrawn_cfg)
    material = policy.resolve_atom_material(index, atom.symbol, atom.position, tag, base_material, fallback)
    if state.material is not None:
        material = state.material
    if state.color is not None:
        material = replace(material, color=state.color)
    return _normalize_material(material, style_bundle=style_bundle)


def _resolve_bond_material_spec(
    *,
    bond,
    i: int,
    j: int,
    si: str,
    sj: str,
    distance: float,
    atom_materials: list[MaterialLike],
    style_bundle: ResolvedStyleBundle,
    policy: MaterialPolicy,
    handdrawn_cfg: HanddrawnStyleConfig | None,
) -> MaterialLike:
    base_material = style_bundle.material_style.bond_default
    fallback = _fallback_bond_material(
        i,
        j,
        atom_materials=atom_materials,
        style_bundle=style_bundle,
        handdrawn_cfg=handdrawn_cfg,
    )
    material = policy.resolve_bond_material(int(bond.id), i, j, si, sj, distance, base_material, fallback)
    if bond.material is not None:
        material = bond.material
    if bond.color is not None:
        material = replace(material, color=bond.color)
    return _normalize_material(material, style_bundle=style_bundle)


def _resolve_bond_side_specs(
    *,
    bond,
    i: int,
    j: int,
    material_uniform: MaterialLike,
    atom_materials: list[MaterialLike],
    split_by_atom: bool,
) -> tuple[MaterialLike, MaterialLike]:
    if split_by_atom and not _bond_has_object_override(bond):
        left = atom_materials[i] if i < len(atom_materials) else material_uniform
        right = atom_materials[j] if j < len(atom_materials) else material_uniform
    else:
        left = material_uniform
        right = material_uniform
    if bond.material_a is not None:
        left = bond.material_a
    if bond.color_a is not None:
        left = replace(left, color=bond.color_a)
    if bond.material_b is not None:
        right = bond.material_b
    if bond.color_b is not None:
        right = replace(right, color=bond.color_b)
    return left, right


def _fallback_atom_material(
    symbol: str,
    *,
    style_bundle: ResolvedStyleBundle,
    handdrawn_cfg: HanddrawnStyleConfig | None,
) -> MaterialLike:
    base = replace(style_bundle.material_style.atom_for(symbol), color=style_bundle.color_style.color_for(symbol))
    if str(style_bundle.material_style.pipeline).strip().lower() != "handdrawn":
        return base
    fallback = style_bundle.material_style.handdrawn_spec or HandDrawnMaterialSpec()
    if handdrawn_cfg is not None and bool(getattr(handdrawn_cfg, "molecule_use_jmol", True)):
        tuned = tune_rgba(
            as_material_spec(base).color,
            desaturate=float(getattr(handdrawn_cfg, "jmol_desaturate", fallback.jmol_desaturate)),
            lighten=float(getattr(handdrawn_cfg, "jmol_lighten", fallback.jmol_lighten)),
        )
        return replace(
            fallback,
            color=tuned,
            roughness=max(float(fallback.roughness), 0.56),
            specular=min(float(fallback.specular), 0.12),
        )
    return replace(
        fallback,
        color=as_material_spec(base).color,
        roughness=float(fallback.roughness),
        specular=float(fallback.specular),
    )


def _fallback_bond_material(
    i: int,
    j: int,
    *,
    atom_materials: list[MaterialLike],
    style_bundle: ResolvedStyleBundle,
    handdrawn_cfg: HanddrawnStyleConfig | None,
) -> MaterialLike:
    fallback = style_bundle.material_style.bond_default
    if str(style_bundle.material_style.pipeline).strip().lower() != "handdrawn":
        return fallback
    if (
        handdrawn_cfg is not None
        and str(getattr(handdrawn_cfg, "bond_color_mode", "atom_pair_avg")).lower() in {"atom_pair_avg", "atom_average"}
        and i < len(atom_materials)
        and j < len(atom_materials)
    ):
        c1 = as_material_spec(atom_materials[i]).color
        c2 = as_material_spec(atom_materials[j]).color
        avg = ((c1[0] + c2[0]) * 0.5, (c1[1] + c2[1]) * 0.5, (c1[2] + c2[2]) * 0.5, 1.0)
        toned = tune_rgba(avg, desaturate=0.35, lighten=0.02)
        return replace(
            fallback,
            color=toned,
            roughness=max(float(fallback.roughness), 0.74),
            specular=min(float(fallback.specular), 0.06),
        )
    return replace(
        fallback,
        roughness=max(float(fallback.roughness), 0.74),
        specular=min(float(fallback.specular), 0.06),
    )


def _normalize_material(
    material: MaterialLike,
    *,
    style_bundle: ResolvedStyleBundle,
    cfg: RenderJobConfig | None = None,
) -> MaterialLike:
    pipeline = str(style_bundle.material_style.pipeline).strip().lower()
    if pipeline != "handdrawn":
        return as_material_spec(material)
    handdrawn_cfg = None
    if cfg is not None:
        handdrawn_cfg = resolve_handdrawn_config(
            style_bundle.material_style.pipeline,
            cfg,
            material_style=style_bundle.material_style,
        )
    profile = resolve_handdrawn_profile(
        style_bundle.material_style.pipeline,
        style_bundle.material_style,
        handdrawn_cfg,
    )
    return as_handdrawn_spec(material, fallback=profile)


def _bond_has_object_override(bond) -> bool:
    return (
        bond.material is not None
        or bond.color is not None
        or bond.material_a is not None
        or bond.color_a is not None
        or bond.material_b is not None
        or bond.color_b is not None
    )


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
