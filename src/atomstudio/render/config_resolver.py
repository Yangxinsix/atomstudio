from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from atomstudio.config import BatchConfig, RenderJobConfig
from atomstudio.style.registry import (
    get_color_style,
    get_light_style_name,
    get_material_style,
    get_radius_style,
    get_scene_style,
)


class ConfigError(ValueError):
    pass


def _deep_fill_missing(target: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in target:
            target[key] = deepcopy(value)
            continue
        if isinstance(target[key], dict) and isinstance(value, dict):
            _deep_fill_missing(target[key], value)


def apply_scene_style_defaults_to_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    src = deepcopy(job)
    style = src.get("style")
    style_dict = style if isinstance(style, dict) else {}
    if "preset" in style_dict:
        raise ConfigError("style.preset has been removed. Use style.scene_style.")

    try:
        style_name = str(style_dict.get("scene_style", "default")).strip().lower()
        scene_style = get_scene_style(style_name)
    except Exception as exc:
        raise ConfigError(str(exc)) from exc
    style_dict["scene_style"] = scene_style.name

    if not style_dict.get("color_style"):
        style_dict["color_style"] = scene_style.color_style.name
    if not style_dict.get("material_style"):
        style_dict["material_style"] = scene_style.material_style.name
    if not style_dict.get("light_style"):
        style_dict["light_style"] = scene_style.light_style

    try:
        style_dict["color_style"] = str(style_dict["color_style"]).strip().lower()
        style_dict["material_style"] = str(style_dict["material_style"]).strip().lower()
        style_dict["light_style"] = get_light_style_name(str(style_dict["light_style"]).strip().lower())
        if style_dict.get("radius_style") is not None:
            style_dict["radius_style"] = str(style_dict["radius_style"]).strip().lower()
        get_color_style(style_dict["color_style"])
        get_material_style(style_dict["material_style"])
        if style_dict.get("radius_style") is not None:
            get_radius_style(style_dict["radius_style"])
    except Exception as exc:
        raise ConfigError(str(exc)) from exc

    structure = src.get("structure")
    structure_dict = structure if isinstance(structure, dict) else {}
    if "bond_cutoff_scale" in structure_dict:
        raise ConfigError(
            "structure.bond_cutoff_scale has been removed. Use structure.bonding.cutoff_scale."
        )
    camera = src.get("camera")
    camera_dict = camera if isinstance(camera, dict) else {}
    lighting = src.get("lighting")
    lighting_dict = lighting if isinstance(lighting, dict) else {}
    if "preset" in lighting_dict:
        raise ConfigError("lighting.preset has been removed. Use lighting.light_style.")
    if lighting_dict.get("light_style") is not None:
        try:
            lighting_dict["light_style"] = get_light_style_name(str(lighting_dict["light_style"]))
        except Exception as exc:
            raise ConfigError(str(exc)) from exc
    render = src.get("render")
    render_dict = render if isinstance(render, dict) else {}

    _deep_fill_missing(structure_dict, scene_style.structure_tokens)
    _deep_fill_missing(camera_dict, scene_style.camera_tokens)
    _deep_fill_missing(lighting_dict, {"intensity": 1.0})
    _deep_fill_missing(render_dict, scene_style.render_tokens)

    src["structure"] = structure_dict
    src["camera"] = camera_dict
    src["lighting"] = lighting_dict
    src["render"] = render_dict
    src["style"] = style_dict
    return src


def _reject_legacy_job_output_fields(job: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(job)
    render_cfg = out.get("render")
    render_dict = render_cfg if isinstance(render_cfg, dict) else {}
    render_output = render_dict.get("output_path")

    if render_output is not None:
        raise ConfigError("render.output_path has been removed. Use output.path.")
    return out


def normalize_batch_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    if "job" in payload:
        raise ConfigError("Top-level 'job' has been removed. Use top-level 'jobs' list.")
    jobs = payload.get("jobs")
    if isinstance(jobs, list):
        payload["jobs"] = [
            apply_scene_style_defaults_to_job_payload(
                _reject_legacy_job_output_fields(job if isinstance(job, dict) else {})
            )
            for job in jobs
        ]
    return payload


def load_yaml_config(path: str) -> dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError("Top-level config must be a mapping")
    return data


def _resolve_config_relative_paths(payload: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    out = deepcopy(payload)
    jobs = out.get("jobs")
    if not isinstance(jobs, list):
        return out

    for job in jobs:
        if not isinstance(job, dict):
            continue
        input_cfg = job.get("input")
        if isinstance(input_cfg, dict) and isinstance(input_cfg.get("path"), str):
            ip = Path(input_cfg["path"]).expanduser()
            if not ip.is_absolute():
                rel_to_cfg = (cfg_path.parent / ip).resolve()
                rel_to_cwd = (Path.cwd() / ip).resolve()
                input_cfg["path"] = str(rel_to_cfg if rel_to_cfg.exists() or not rel_to_cwd.exists() else rel_to_cwd)

        output_cfg = job.get("output")
        if not isinstance(output_cfg, dict):
            continue
        if isinstance(output_cfg.get("path"), str):
            op = Path(output_cfg["path"]).expanduser()
            if not op.is_absolute():
                output_cfg["path"] = str((cfg_path.parent / op).resolve())
        if isinstance(output_cfg.get("dir"), str):
            od = Path(output_cfg["dir"]).expanduser()
            if not od.is_absolute():
                output_cfg["dir"] = str((cfg_path.parent / od).resolve())

    return out


def validate_config_dict(data: dict[str, Any]) -> None:
    try:
        BatchConfig.from_dict(normalize_batch_payload(data))
    except Exception as exc:
        raise ConfigError(str(exc)) from exc


def validate_config_file(path: str) -> dict[str, Any]:
    try:
        cfg = load_batch_config(path)
    except Exception as exc:
        raise ConfigError(str(exc)) from exc
    return cfg.to_dict()


def load_batch_config(path: str) -> BatchConfig:
    try:
        cfg_path = Path(path).expanduser().resolve()
        raw = load_yaml_config(path)
        resolved = _resolve_config_relative_paths(raw, cfg_path)
        normalized = normalize_batch_payload(resolved)
        return BatchConfig.from_dict(normalized)
    except Exception as exc:
        raise ConfigError(str(exc)) from exc


def job_to_render_config(job: dict[str, Any], output_path: str) -> RenderJobConfig:
    cfg = RenderJobConfig.from_dict(apply_scene_style_defaults_to_job_payload(job))
    return cfg.with_output_path(output_path)
