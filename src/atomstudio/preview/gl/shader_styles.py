from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpenGLShaderStyle:
    name: str
    ambient: float
    diffuse: float
    wrap: float
    specular: float
    shininess: float
    exposure: float = 1.0
    mode: int = 0


SHADER_STYLES: dict[str, OpenGLShaderStyle] = {
    "studio_glossy": OpenGLShaderStyle(
        "studio_glossy", ambient=0.44, diffuse=0.46, wrap=0.25, specular=0.080, shininess=64.0, mode=3
    ),
    "toon_rim": OpenGLShaderStyle(
        "toon_rim", ambient=0.54, diffuse=0.42, wrap=0.15, specular=0.020, shininess=32.0, mode=5
    ),
    "matcap": OpenGLShaderStyle(
        "matcap", ambient=0.56, diffuse=0.34, wrap=0.45, specular=0.055, shininess=48.0, mode=1
    ),
    "soft_clay": OpenGLShaderStyle(
        "soft_clay", ambient=0.62, diffuse=0.24, wrap=0.85, specular=0.000, shininess=1.0, mode=6
    ),
    "technical_gooch": OpenGLShaderStyle(
        "technical_gooch", ambient=0.48, diffuse=0.38, wrap=0.35, specular=0.018, shininess=24.0, mode=4
    ),
}

SCENE_STYLE_SHADER_ALIASES: dict[str, str] = {
    "default": "studio_glossy",
    "studio": "studio_glossy",
    "glossy": "studio_glossy",
}

DEFAULT_SHADER_STYLE = "studio_glossy"


def resolve_shader_style(name: str | OpenGLShaderStyle | None) -> OpenGLShaderStyle:
    if isinstance(name, OpenGLShaderStyle):
        return name
    key = str(name or DEFAULT_SHADER_STYLE).strip().lower()
    key = SCENE_STYLE_SHADER_ALIASES.get(key, key)
    return SHADER_STYLES.get(key, SHADER_STYLES[DEFAULT_SHADER_STYLE])


def shader_style_choices() -> tuple[str, ...]:
    return tuple(SHADER_STYLES)


__all__ = [
    "DEFAULT_SHADER_STYLE",
    "OpenGLShaderStyle",
    "SHADER_STYLES",
    "resolve_shader_style",
    "shader_style_choices",
]
