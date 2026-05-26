from __future__ import annotations

_BOUNDARY_EXPORTS = {
    "build_boundary_expanded_structure",
    "enumerate_offsets",
    "fractional_positions",
    "normalize_window",
}
_API_EXPORTS = {"apply_style", "compute_bonds", "compute_polyhedra", "render_structure_image"}
_SELECTOR_EXPORTS = {"AtomSelector", "BondSelector", "PolyhedraSelector", "norm_index_pair", "norm_symbol_pair"}
_CONFIG_EXPORTS = {
    "ASEViewConfig",
    "AtomStylePresetConfig",
    "AtomStyleRuleConfig",
    "BatchConfig",
    "BoundaryAtomsConfig",
    "BoundaryConfig",
    "BondingConfig",
    "CameraConfig",
    "CellStyleConfig",
    "HBondConfig",
    "HanddrawnStyleConfig",
    "InputConfig",
    "LightConfig",
    "LightingConfig",
    "MaterialPolicy",
    "MaterialRule",
    "OutputConfig",
    "OutlineConfig",
    "OutlineRoleConfig",
    "PolyhedraConfig",
    "PolyhedraRuleConfig",
    "RenderJobConfig",
    "RenderSettings",
    "StructureConfig",
    "StyleConfig",
    "SurfaceOptions",
}


def __getattr__(name: str):
    if name == "Atom":
        from atomstudio.structure.atom import Atom

        return Atom
    if name == "Bond":
        from atomstudio.structure.bond import Bond

        return Bond
    if name == "Polyhedron":
        from atomstudio.structure.polyhedron import Polyhedron

        return Polyhedron
    if name == "Cell":
        from atomstudio.structure.cell import Cell

        return Cell
    if name == "Structure":
        from atomstudio.structure.structure import Structure

        return Structure
    if name in {"BondEngine", "BondResult"}:
        from atomstudio.structure import bonding

        return getattr(bonding, name)
    if name in _BOUNDARY_EXPORTS:
        from atomstudio.structure import boundary

        return getattr(boundary, name)
    if name in _API_EXPORTS:
        from atomstudio.structure import api

        return getattr(api, name)
    if name in _SELECTOR_EXPORTS:
        from atomstudio.structure import selectors

        return getattr(selectors, name)
    if name in _CONFIG_EXPORTS:
        from atomstudio import config

        return getattr(config, name)
    raise AttributeError(name)


__all__ = [
    "Atom",
    "Bond",
    "Polyhedron",
    "Cell",
    "Structure",
    "BondEngine",
    "BondResult",
    "normalize_window",
    "fractional_positions",
    "enumerate_offsets",
    "build_boundary_expanded_structure",
    "apply_style",
    "compute_bonds",
    "compute_polyhedra",
    "render_structure_image",
    "AtomSelector",
    "BondSelector",
    "PolyhedraSelector",
    "norm_symbol_pair",
    "norm_index_pair",
    "ASEViewConfig",
    "AtomStylePresetConfig",
    "AtomStyleRuleConfig",
    "BatchConfig",
    "BoundaryAtomsConfig",
    "BoundaryConfig",
    "BondingConfig",
    "CameraConfig",
    "CellStyleConfig",
    "HBondConfig",
    "HanddrawnStyleConfig",
    "InputConfig",
    "LightConfig",
    "LightingConfig",
    "MaterialPolicy",
    "MaterialRule",
    "OutputConfig",
    "OutlineConfig",
    "OutlineRoleConfig",
    "PolyhedraConfig",
    "PolyhedraRuleConfig",
    "RenderJobConfig",
    "RenderSettings",
    "StructureConfig",
    "StyleConfig",
    "SurfaceOptions",
]
