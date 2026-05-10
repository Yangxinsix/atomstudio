from __future__ import annotations

from typing import Any

from atomstudio.color_utils import rgba4_or_none
from atomstudio.config import BondingConfig, PolyhedraConfig, RenderJobConfig
from atomstudio.scene.materials.specs import material_from_dict
from atomstudio.structure.selectors import AtomSelector, BondSelector, PolyhedraSelector
from atomstudio.structure.structure import Structure
from atomstudio.style.registry import get_scene_style
from atomstudio.style.scene_style import SceneStyle


def compute_bonds(structure: Structure, bonding_config: BondingConfig | dict[str, Any] | None = None) -> Structure:
    return structure.compute_bonds(bonding_config)


def compute_polyhedra(
    structure: Structure,
    polyhedra_config: PolyhedraConfig | dict[str, Any] | None = None,
    bonding_config: BondingConfig | dict[str, Any] | None = None,
) -> Structure:
    return structure.compute_polyhedra(polyhedra_config, bonding_config)


def render_structure_image(
    structure: Structure,
    output_path: str | None = None,
    *,
    cfg: RenderJobConfig | None = None,
    return_type: str = "auto",
    display_width: int | None = None,
    display_height: int | None = None,
    overrides: dict[str, Any] | None = None,
    blender_path: str | None = None,
    timeout_seconds: int = 1800,
    **cli_kwargs: Any,
) -> Any:
    return structure.get_image(
        output_path,
        cfg=cfg,
        return_type=return_type,
        display_width=display_width,
        display_height=display_height,
        overrides=overrides,
        blender_path=blender_path,
        timeout_seconds=timeout_seconds,
        **cli_kwargs,
    )


def apply_style(
    structure: Structure,
    scene_style: SceneStyle | str,
    overrides: dict[str, Any] | None = None,
) -> Structure:
    resolved = get_scene_style(scene_style) if isinstance(scene_style, str) else scene_style
    structure.metadata["scene_style"] = resolved.name

    ov = overrides or {}

    for item in ov.get("atoms", []):
        if not isinstance(item, dict):
            continue
        selector = AtomSelector.from_dict(dict(item.get("selector", {})))
        material = material_from_dict(item.get("material")) if isinstance(item.get("material"), dict) else None
        color = rgba4_or_none(item.get("color"))
        structure.assign_atom_style(
            selector,
            style=str(item["style"]) if item.get("style") is not None else None,
            representation=str(item["representation"]) if item.get("representation") is not None else None,
            material=material,
            color=color,
            radius=float(item["radius"]) if item.get("radius") is not None else None,
        )

    for item in ov.get("bonds", []):
        if not isinstance(item, dict):
            continue
        _reject_legacy_style_keys(item, {"style_ref", "material_override", "color_override"}, "bond override")
        selector = BondSelector.from_dict(dict(item.get("selector", {})))
        material = material_from_dict(item.get("material")) if isinstance(item.get("material"), dict) else None
        color = rgba4_or_none(item.get("color"))
        structure.assign_bond_style(
            selector,
            style=item.get("style"),
            material=material,
            color=color,
            material_a=material_from_dict(item.get("material_a")) if isinstance(item.get("material_a"), dict) else None,
            color_a=rgba4_or_none(item.get("color_a")),
            material_b=material_from_dict(item.get("material_b")) if isinstance(item.get("material_b"), dict) else None,
            color_b=rgba4_or_none(item.get("color_b")),
            split_ratio=float(item["split_ratio"]) if item.get("split_ratio") is not None else None,
        )

    for item in ov.get("polyhedra", []):
        if not isinstance(item, dict):
            continue
        selector = PolyhedraSelector.from_dict(dict(item.get("selector", {})))
        material = material_from_dict(item.get("material")) if isinstance(item.get("material"), dict) else None
        color = rgba4_or_none(item.get("color"))
        edge_color = rgba4_or_none(item.get("edge_color"))
        structure.assign_polyhedra_style(
            selector,
            style=item.get("style"),
            material=material,
            color=color,
            show_edges=bool(item["show_edges"]) if item.get("show_edges") is not None else None,
            edge_radius=float(item["edge_radius"]) if item.get("edge_radius") is not None else None,
            edge_color=edge_color,
        )

    cell = ov.get("cell")
    if isinstance(cell, dict):
        _reject_legacy_style_keys(cell, {"style_ref", "material_override", "color_override"}, "cell override")
        material = material_from_dict(cell.get("material")) if isinstance(cell.get("material"), dict) else None
        color = rgba4_or_none(cell.get("color"))
        structure.assign_cell_style(style=cell.get("style"), material=material, color=color)

    return structure

def _reject_legacy_style_keys(payload: dict[str, Any], legacy_keys: set[str], context: str) -> None:
    found = sorted(k for k in legacy_keys if k in payload)
    if found:
        raise ValueError(f"Legacy {context} keys are not supported: {', '.join(found)}")
