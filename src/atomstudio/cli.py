from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from atomstudio.style.registry import (
    color_style_choices,
    light_style_choices,
    material_style_choices,
    radius_style_choices,
    scene_style_choices,
)

_SCENE_STYLE_CHOICES = tuple(scene_style_choices())
_COLOR_STYLE_CHOICES = tuple(color_style_choices())
_MATERIAL_STYLE_CHOICES = tuple(material_style_choices())
_LIGHT_STYLE_CHOICES = tuple(light_style_choices())
_RADIUS_STYLE_CHOICES = tuple(radius_style_choices())
_QUALITY_CHOICES = ("high", "low", "medium", "very_high")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atomstudio", description="ASE + Blender rendering CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Render one structure or frame sequence")
    render.add_argument("--config", default=None, help="v2 YAML/JSON config path (single job)")
    render.add_argument("--input", default=None, help="Input structure/trajectory path")
    render.add_argument("--frame", default="last", help="Single frame selector, e.g. last or 0")
    render.add_argument("--frames", default=None, help="Multi-frame selector, e.g. all or 0:100:1")
    render.add_argument("--out", default=None, help="Output PNG path for single frame")
    render.add_argument("--out-dir", default=None, help="Output directory for frame sequence")
    render.add_argument("--filename-template", default="frame_{frame:04d}.png", help="Template for sequence")
    render.add_argument("--style", default="default", choices=_SCENE_STYLE_CHOICES)
    render.add_argument("--color-style", default=None, choices=_COLOR_STYLE_CHOICES)
    render.add_argument("--material-style", default=None, choices=_MATERIAL_STYLE_CHOICES)
    render.add_argument("--light-style", default=None, choices=_LIGHT_STYLE_CHOICES)
    render.add_argument("--radius-style", default=None, choices=_RADIUS_STYLE_CHOICES)
    render.add_argument("--quality", default=None, choices=_QUALITY_CHOICES)
    render.add_argument("--light-intensity", default=1.0, type=float)
    render.add_argument("--representation", default="auto", choices=["auto", "space_filling", "ball_stick"])
    render.add_argument("--engine", default="cycles", choices=["cycles", "eevee", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"])
    render.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"])
    render.add_argument("--samples", default=64, type=int)
    render.add_argument("--res-x", default=1024, type=int)
    render.add_argument("--res-y", default=1024, type=int)
    render.add_argument("--seed", default=7, type=int)
    render.add_argument("--atom-scale", default=1.0, type=float)
    # References for radii-scale presets:
    # VESTA 40%: https://jp-minerals.org/vesta/en/doc/VESTAch5.html
    # Jmol compact style around 20%: https://jmol.sourceforge.net/demo/atoms
    render.add_argument(
        "--radii-scale",
        dest="radii_scale",
        default=0.40,
        type=float,
        help="Atom radii multiplier for ball_stick (default 0.40; Jmol-like compact often ~0.20).",
    )
    render.add_argument("--bond-radius", default=0.08, type=float)
    bond_toggle = render.add_mutually_exclusive_group()
    bond_toggle.add_argument("--draw-bonds", dest="draw_bonds", action="store_true")
    bond_toggle.add_argument("--no-bonds", dest="draw_bonds", action="store_false")
    render.add_argument(
        "--rotation",
        default=None,
        help="Model rotation with ASE GUI-compatible semantics (rotation string), e.g. --rotation=-90x,-90y,0z",
    )
    render.add_argument(
        "--view",
        default="top",
        help="Model view/rotation: top/front/side, or ASE GUI-compatible rotation string (e.g. -90x,-90y,0z)",
    )
    render.add_argument(
        "--camera-view",
        default="top",
        help="Camera semantic view: top/front/side",
    )
    render.add_argument("--frame-scale", default=1.0, type=float, help="Camera framing scale (>1 zoom out, <1 zoom in)")
    render.add_argument("--draw-cell", action="store_true")
    render.add_argument("--no-cell", dest="draw_cell", action="store_false")
    render.set_defaults(draw_cell=True, transparent_bg=True, draw_bonds=None)
    render.add_argument("--debug-config", action="store_true", help="Print effective render config before rendering")
    render.add_argument("--transparent-bg", dest="transparent_bg", action="store_true")
    render.add_argument("--solid-bg", dest="transparent_bg", action="store_false")

    batch = sub.add_parser("batch", help="Run YAML batch rendering")
    batch.add_argument("--config", required=True, help="YAML config path")

    val = sub.add_parser("validate-config", help="Validate YAML config")
    val.add_argument("--config", required=True, help="YAML config path")

    app = sub.add_parser("app", help="Launch the desktop OpenGL preview app")
    app.add_argument("--config", default=None, help="Optional v2 YAML/JSON config path")
    app.add_argument("--input", default=None, help="Optional input structure/trajectory path")
    app.add_argument("--frame", default="all", help="Initial frame selector, e.g. all, last, 0, or 0:100:1")
    app.add_argument(
        "--preview-backend",
        default="opengl",
        choices=["vispy", "opengl", "opengl-window", "opengl-widget", "opengl-detached"],
        help="Preview widget backend to use in the desktop app",
    )

    return parser


def _job_from_flags(args: argparse.Namespace, output_path: str) -> RenderJobConfig:
    from atomstudio.render.cli_like import CLI_RENDER_KWARG_DEFAULTS, build_render_job_config_from_cli_like_kwargs

    cli_kwargs = {}
    for key, default in CLI_RENDER_KWARG_DEFAULTS.items():
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if value == default:
            continue
        cli_kwargs[key] = value
    return build_render_job_config_from_cli_like_kwargs(
        output_path=str(output_path),
        input_path=str(args.input),
        kwargs=cli_kwargs,
        job_id="cli_render",
        frames="last",
    )


def _center_extent_from_positions(positions: list[tuple[float, float, float]] | None) -> tuple[tuple[float, float, float], float]:
    if not positions:
        return (0.0, 0.0, 0.0), 1.0
    xs = [float(p[0]) for p in positions]
    ys = [float(p[1]) for p in positions]
    zs = [float(p[2]) for p in positions]
    center = ((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, (min(zs) + max(zs)) * 0.5)
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)
    return center, extent


def _build_debug_payload(
    cfg: RenderJobConfig,
    symbols: list[str] | None = None,
    positions: list[tuple[float, float, float]] | None = None,
) -> dict:
    from atomstudio.scene.lights.builder import resolve_lighting_specs
    from atomstudio.style.resolver import resolve_style_bundle

    payload = {"debug_config": cfg.to_dict()}
    try:
        style_bundle = resolve_style_bundle(cfg.style)
        unique_symbols = sorted(set(symbols or []))
        symbol_materials = {}
        for symbol in unique_symbols:
            base = style_bundle.material_style.atom_for(symbol)
            merged = replace(base, color=style_bundle.color_style.color_for(symbol))
            symbol_materials[symbol] = merged.to_dict()
        payload["debug_material"] = {
            "scene_style": style_bundle.scene_style_name,
            "color_style": style_bundle.color_style_name,
            "material_style": style_bundle.material_style_name,
            "material_pipeline": style_bundle.material_style.pipeline,
            "atom_default": style_bundle.material_style.atom_default.to_dict(),
            "bond_default": style_bundle.material_style.bond_default.to_dict(),
            "cell_default": style_bundle.material_style.cell_default.to_dict(),
            "symbols": unique_symbols,
            "symbol_materials": symbol_materials,
        }
        center, extent = _center_extent_from_positions(positions)
        payload["debug_lighting"] = {
            "style_light_style": style_bundle.light_style_name,
            "lighting_light_style": cfg.lighting.light_style,
            "effective_light_style": cfg.lighting.light_style or style_bundle.light_style_name,
            "lights": resolve_lighting_specs(
                cfg,
                center=center,
                extent=extent,
                default_light_style=style_bundle.light_style_name,
            ),
        }
    except Exception as exc:
        payload["debug_material_error"] = str(exc)
    return payload


def _cmd_render(args: argparse.Namespace) -> int:
    from atomstudio.io.ase_loader import load_structure, load_trajectory
    from atomstudio.render.config_resolver import load_batch_config
    from atomstudio.render.pipeline import render_single_from_config, render_structure

    if args.config:
        if bool(getattr(args, "debug_config", False)):
            batch = load_batch_config(args.config)
            if len(batch.jobs) != 1:
                raise ValueError("render --config expects exactly one job for --debug-config")
            dbg_structure = load_structure(batch.jobs[0].input.path, batch.jobs[0].input.frames)
            print(
                json.dumps(
                    _build_debug_payload(batch.jobs[0], symbols=dbg_structure.symbols, positions=dbg_structure.positions),
                    indent=2,
                )
            )
        result = render_single_from_config(args.config)
        print(json.dumps(asdict(result), indent=2))
        return 0 if result.success else 2

    if not args.input:
        raise ValueError("--input is required when --config is not provided")

    if args.frames is not None:
        selector = args.frames
        structures = load_trajectory(args.input, selector)
        out_dir = Path(args.out_dir or Path.cwd() / "frames").expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for structure in structures:
            output = out_dir / args.filename_template.format(frame=structure.frame_index)
            cfg = _job_from_flags(args, str(output)).with_output_path(str(output))
            if bool(getattr(args, "debug_config", False)):
                print(json.dumps(_build_debug_payload(cfg, symbols=structure.symbols, positions=structure.positions), indent=2))
                args.debug_config = False
            results.append(asdict(render_structure(structure, cfg)))

        ok = all(r["success"] for r in results)
        print(json.dumps({"success": ok, "results": results}, indent=2))
        return 0 if ok else 2

    structure = load_structure(args.input, args.frame)
    output = args.out or str((Path.cwd() / f"render_{structure.frame_index:04d}.png").resolve())
    cfg = _job_from_flags(args, output).with_output_path(output)
    if bool(getattr(args, "debug_config", False)):
        print(json.dumps(_build_debug_payload(cfg, symbols=structure.symbols, positions=structure.positions), indent=2))
    result = render_structure(structure, cfg)
    print(json.dumps(asdict(result), indent=2))
    return 0 if result.success else 2


def _cmd_batch(args: argparse.Namespace) -> int:
    from atomstudio.render.pipeline import render_batch

    result = render_batch(args.config)
    payload = {"success": result.success, "reports": [asdict(r) for r in result.reports]}
    print(json.dumps(payload, indent=2))
    return 0 if result.success else 2


def _cmd_validate(args: argparse.Namespace) -> int:
    from atomstudio.render.config_resolver import validate_config_file

    cfg = validate_config_file(args.config)
    print(json.dumps({"success": True, "version": cfg["version"], "jobs": len(cfg["jobs"])}, indent=2))
    return 0


def _cmd_app(args: argparse.Namespace) -> int:
    try:
        from atomstudio.app.main import run_app
    except ModuleNotFoundError as exc:
        missing = str(exc.name or "")
        if missing in {"PySide6", "vispy"}:
            raise RuntimeError(
                "Desktop UI dependencies are not installed. Install with `pip install 'atomstudio[gui]'`."
            ) from exc
        raise

    kwargs = {
        "input_path": args.input,
        "config_path": args.config,
        "frame": args.frame,
        "preview_backend": args.preview_backend,
    }
    return int(run_app(**kwargs))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            return _cmd_render(args)
        if args.command == "batch":
            return _cmd_batch(args)
        if args.command == "validate-config":
            return _cmd_validate(args)
        if args.command == "app":
            return _cmd_app(args)
        parser.print_help()
        return 1
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
