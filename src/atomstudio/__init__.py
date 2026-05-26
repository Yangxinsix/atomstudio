from __future__ import annotations

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "Structure":
        from atomstudio.structure.structure import Structure

        return Structure
    if name == "RenderJobConfig":
        from atomstudio.config import RenderJobConfig

        return RenderJobConfig
    if name in {"BatchResult", "RenderResult"}:
        from atomstudio.render import results

        return getattr(results, name)
    if name in {"apply_style", "compute_bonds", "compute_polyhedra", "render_structure_image"}:
        from atomstudio.structure import api

        return getattr(api, name)
    raise AttributeError(name)


def load_structure(path: str, frame: int | str = "last") -> Structure:
    from atomstudio.io.ase_loader import load_structure as _load_structure

    return _load_structure(path, frame)


def load_trajectory(path: str, frame_selector: str = "all") -> list[Structure]:
    from atomstudio.io.ase_loader import load_trajectory as _load_trajectory

    return _load_trajectory(path, frame_selector)


def render_structure(
    structure: Structure,
    cfg: RenderJobConfig,
    *,
    blender_path: str | None = None,
    timeout_seconds: int = 1800,
) -> RenderResult:
    from atomstudio.render.pipeline import render_structure as _render_structure

    return _render_structure(
        structure,
        cfg,
        blender_path=blender_path,
        timeout_seconds=timeout_seconds,
    )


def render_batch(config_path: str) -> BatchResult:
    from atomstudio.render.pipeline import render_batch as _render_batch

    return _render_batch(config_path)


__all__ = [
    "Structure",
    "RenderJobConfig",
    "RenderResult",
    "BatchResult",
    "load_structure",
    "load_trajectory",
    "apply_style",
    "compute_bonds",
    "compute_polyhedra",
    "render_structure_image",
    "render_structure",
    "render_batch",
]
