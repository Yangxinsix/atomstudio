__all__ = ["LightingBuilder", "LightParams", "LightSpec", "resolve_lighting_specs"]


def __getattr__(name: str):
    if name in {"LightParams", "LightSpec"}:
        from atomstudio.scene.lights import specs as _specs

        return getattr(_specs, name)
    if name in {"LightingBuilder", "resolve_lighting_specs"}:
        from atomstudio.scene.lights import builder as _builder

        return getattr(_builder, name)
    raise AttributeError(name)
