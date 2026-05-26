from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atomstudio.config import RenderJobConfig
from atomstudio.paths import normalize_host_path
from atomstudio.structure.structure import Structure


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def default_render_script_path(source_path: str | None, *, cwd: Path | None = None) -> Path:
    base = Path.cwd() if cwd is None else Path(cwd)
    source = Path(normalize_host_path(source_path or "")).expanduser()
    stem = source.stem or source.name or "atomstudio"
    return (base / f"{stem}_render_script.py").resolve()


def default_batch_output_spec(
    output_path: str | None,
    source_path: str | None = None,
    *,
    cwd: Path | None = None,
) -> tuple[str, str]:
    base = Path.cwd() if cwd is None else Path(cwd)
    raw = normalize_host_path(output_path or "")
    if not raw:
        source = Path(normalize_host_path(source_path or "")).expanduser()
        stem = source.stem or source.name or "atomstudio_render"
        parent = source.parent if str(source.parent) not in {"", "."} else base
        raw = str(parent / f"{stem}.png")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        return str((path.parent / f"{path.stem}_frames").resolve()), f"{path.stem}_{{frame:04d}}.png"
    return str(path.resolve()), "frame_{frame:04d}.png"


def build_render_script_text(
    *,
    render_config: RenderJobConfig,
    reference_structure: Structure,
    default_input_path: str,
    default_frames: str = "all",
    default_out_dir: str,
    default_filename_template: str,
) -> str:
    config_json = json.dumps(render_config.to_dict(), indent=2, sort_keys=False, ensure_ascii=True)
    reference_json = json.dumps(
        _reference_visual_payload(reference_structure),
        indent=2,
        sort_keys=False,
        ensure_ascii=True,
    )
    config_payload = _json_loads_block("CONFIG_PAYLOAD", config_json)
    reference_payload = _json_loads_block("REFERENCE_VISUALS", reference_json)
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from atomstudio.config import RenderJobConfig
from atomstudio.io.ase_loader import load_trajectory
from atomstudio.render.pipeline import render_structure
from atomstudio.scene.materials.specs import material_from_dict
from atomstudio.style.outline_style import OutlineRoleStyle


{config_payload}
{reference_payload}
DEFAULT_INPUT = {str(default_input_path)!r}
DEFAULT_FRAMES = {str(default_frames or "all")!r}
DEFAULT_OUT_DIR = {str(default_out_dir)!r}
DEFAULT_FILENAME_TEMPLATE = {str(default_filename_template)!r}


def _rgba(value):
    if value is None:
        return None
    return tuple(float(item) for item in value)


def _material(value):
    return material_from_dict(value) if isinstance(value, dict) else None


def _apply_reference_visuals(structure, reference):
    atoms_by_index = {{int(atom.index): atom for atom in structure.atoms}}
    for data in reference.get("atoms", []):
        atom = atoms_by_index.get(int(data.get("index", -1)))
        if atom is None:
            continue
        for attr in ("radius", "segments", "rings", "style", "representation"):
            if attr in data:
                setattr(atom, attr, data[attr])
        if "color" in data:
            atom.color = _rgba(data["color"])
        if "material" in data:
            atom.material = _material(data["material"])
        if "outline" in data:
            atom.outline = OutlineRoleStyle.from_any(data["outline"], fallback=atom.outline)
        atom.sync_color_to_material()

    bonds_by_id = {{int(bond.id): bond for bond in structure.bonds}}
    for data in reference.get("bonds", []):
        bond = bonds_by_id.get(int(data.get("id", -1)))
        if bond is None:
            continue
        if "style" in data:
            bond.style = data["style"]
        if "split_ratio" in data:
            bond.split_ratio = float(data["split_ratio"])
        for attr in ("material", "material_a", "material_b"):
            if attr in data:
                setattr(bond, attr, _material(data[attr]))
        for attr in ("color", "color_a", "color_b"):
            if attr in data:
                setattr(bond, attr, _rgba(data[attr]))
        if "metadata" in data and isinstance(data["metadata"], dict):
            bond.metadata.update(data["metadata"])


def _parse_args():
    parser = argparse.ArgumentParser(description="Render an AtomStudio trajectory with exported GUI styling.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input structure or trajectory path.")
    parser.add_argument("--frames", default=DEFAULT_FRAMES, help="Frame selector, e.g. all, last, 0:100:5.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for rendered frames.")
    parser.add_argument("--filename-template", default=DEFAULT_FILENAME_TEMPLATE)
    parser.add_argument("--blender-path", default=None)
    parser.add_argument("--timeout", default=1800, type=int)
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = RenderJobConfig.from_dict(CONFIG_PAYLOAD)
    frames = load_trajectory(args.input, frame_selector=args.frames)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for structure in frames:
        structure.compute_bonds(cfg.structure.bonding)
        _apply_reference_visuals(structure, REFERENCE_VISUALS)
        output_path = out_dir / args.filename_template.format(job_id=cfg.id, frame=structure.frame_index)
        result = render_structure(
            structure,
            cfg.with_output_path(str(output_path)),
            blender_path=args.blender_path,
            timeout_seconds=int(args.timeout),
        )
        results.append(asdict(result))
        print(json.dumps(results[-1], indent=2), flush=True)

    success = bool(results) and all(bool(item.get("success")) for item in results)
    print(json.dumps({{"success": success, "frames": len(results), "results": results}}, indent=2))
    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
'''


def _json_loads_block(name: str, payload: str) -> str:
    return f'{name} = json.loads("""\\\n{payload}\n""")'


def _reference_visual_payload(structure: Structure) -> dict[str, Any]:
    payload = structure.to_dict()
    return {
        "atoms": [
            atom
            for atom in (_atom_visual_payload(item) for item in payload.get("atoms", []))
            if len(atom) > 1
        ],
        "bonds": [
            bond
            for bond in (_bond_visual_payload(item) for item in payload.get("bonds", []))
            if len(bond) > 1
        ],
    }


def _atom_visual_payload(atom: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"index": int(atom.get("index", 0))}
    for key in ("radius", "segments", "rings", "material", "color", "style", "representation"):
        if atom.get(key) is not None:
            out[key] = atom[key]
    outline = atom.get("outline")
    if isinstance(outline, dict) and _outline_has_override(outline):
        out["outline"] = outline
    return out


def _bond_visual_payload(bond: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": int(bond.get("id", 0))}
    for key in ("style", "material", "color", "material_a", "color_a", "material_b", "color_b"):
        if bond.get(key) is not None:
            out[key] = bond[key]
    if float(bond.get("split_ratio", 0.5)) != 0.5:
        out["split_ratio"] = float(bond.get("split_ratio", 0.5))
    metadata = bond.get("metadata")
    if isinstance(metadata, dict) and metadata.get("preview_bond_style") is not None:
        out["metadata"] = {"preview_bond_style": str(metadata["preview_bond_style"])}
    return out


def _outline_has_override(outline: dict[str, Any]) -> bool:
    return bool(outline.get("enabled")) or any(
        outline.get(key) is not None
        for key in ("thickness", "color", "secondary_thickness", "secondary_color")
    ) or bool(outline.get("ignore_occlusion"))
