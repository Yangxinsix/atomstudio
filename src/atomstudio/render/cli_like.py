from __future__ import annotations

from atomstudio.config import RenderJobConfig
from atomstudio.render.config_resolver import apply_scene_style_defaults_to_job_payload

CLI_RENDER_KWARG_DEFAULTS: dict[str, object] = {
    "style": "default",
    "color_style": None,
    "material_style": None,
    "light_style": None,
    "radius_style": None,
    "quality": None,
    "light_intensity": 1.0,
    "representation": "auto",
    "engine": "cycles",
    "device": "auto",
    "samples": 64,
    "res_x": 1024,
    "res_y": 1024,
    "seed": 7,
    "atom_scale": 1.0,
    "radii_scale": 0.40,
    "bond_radius": 0.08,
    "draw_bonds": None,
    "draw_cell": True,
    "rotation": None,
    "view": "top",
    "camera_view": "top",
    "frame_scale": 1.0,
    "transparent_bg": True,
}

RENDER_QUALITY_PRESETS: dict[str, dict[str, int]] = {
    "low": {"samples": 32, "res_x": 768, "res_y": 768},
    "medium": {"samples": 64, "res_x": 1024, "res_y": 1024},
    "high": {"samples": 128, "res_x": 1536, "res_y": 1536},
    "very_high": {"samples": 256, "res_x": 2048, "res_y": 2048},
}


def normalize_engine(value: str) -> str:
    v = str(value)
    lv = v.lower()
    if lv == "cycles":
        return "cycles"
    if lv == "eevee":
        return "eevee"
    if v in {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"}:
        return v
    return "eevee"


def build_render_job_config_from_cli_like_kwargs(
    *,
    output_path: str,
    input_path: str,
    kwargs: dict[str, object] | None = None,
    job_id: str = "cli_render",
    frames: str = "last",
) -> RenderJobConfig:
    cli_kwargs = {} if kwargs is None else dict(kwargs)
    allowed = set(CLI_RENDER_KWARG_DEFAULTS.keys())
    unknown = sorted(set(cli_kwargs) - allowed)
    if unknown:
        raise ValueError(f"Unknown CLI-like get_image args: {', '.join(unknown)}")

    values: dict[str, object] = dict(CLI_RENDER_KWARG_DEFAULTS)
    values.update(cli_kwargs)
    quality = values.get("quality")
    if quality is not None:
        quality_name = str(quality).strip().lower()
        if quality_name not in RENDER_QUALITY_PRESETS:
            choices = ", ".join(sorted(RENDER_QUALITY_PRESETS))
            raise ValueError(f"render quality must be one of: {choices}")
        values["quality"] = quality_name
        for key, val in RENDER_QUALITY_PRESETS[quality_name].items():
            if key not in cli_kwargs:
                values[key] = val

    style_payload = {"scene_style": str(values["style"] or "default")}
    if values["color_style"] is not None:
        style_payload["color_style"] = str(values["color_style"])
    if values["material_style"] is not None:
        style_payload["material_style"] = str(values["material_style"])
    if values["light_style"] is not None:
        style_payload["light_style"] = str(values["light_style"])
    if values["radius_style"] is not None:
        style_payload["radius_style"] = str(values["radius_style"])

    payload = {
        "id": str(job_id),
        "input": {"path": str(input_path), "frames": str(frames)},
        "output": {"path": str(output_path)},
        "structure": {
            "representation": str(values["representation"]),
            "atom_scale": float(values["atom_scale"]),
            "radii_scale": float(values["radii_scale"]),
            "bond_radius": float(values["bond_radius"]),
            "draw_cell": bool(values["draw_cell"]),
            "model_rotation": str(values["rotation"]) if values["rotation"] is not None else None,
            "model_view": str(values["view"]).lower(),
            "cell_style": {"show": bool(values["draw_cell"])},
        },
        "style": style_payload,
        "camera": {
            "view": str(values["camera_view"]).lower(),
            "frame_scale": float(values["frame_scale"]),
        },
        "lighting": {"intensity": float(values["light_intensity"])},
        "render": {
            "engine": normalize_engine(str(values["engine"])),
            "device": str(values["device"]).lower(),
            "samples": int(values["samples"]),
            "resolution": [int(values["res_x"]), int(values["res_y"])],
            "transparent_bg": bool(values["transparent_bg"]),
            "seed": int(values["seed"]),
        },
    }
    if values["draw_bonds"] is not None:
        payload["structure"]["draw_bonds"] = bool(values["draw_bonds"])

    return RenderJobConfig.from_dict(apply_scene_style_defaults_to_job_payload(payload))
