from atomstudio.render.results import BatchResult, JobReport, RenderResult


def render_structure(*args, **kwargs):
    from atomstudio.render.pipeline import render_structure as _render_structure

    return _render_structure(*args, **kwargs)


def render_batch(*args, **kwargs):
    from atomstudio.render.pipeline import render_batch as _render_batch

    return _render_batch(*args, **kwargs)


def render_single_from_config(*args, **kwargs):
    from atomstudio.render.pipeline import render_single_from_config as _render_single_from_config

    return _render_single_from_config(*args, **kwargs)

__all__ = [
    "render_structure",
    "render_batch",
    "render_single_from_config",
    "RenderResult",
    "JobReport",
    "BatchResult",
]
