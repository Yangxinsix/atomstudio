import argparse
import json
import re
import sys
from copy import deepcopy
from math import tan
from pathlib import Path

import bpy
from mathutils import Vector

try:
    from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS, covalent_radii as ASE_COVALENT_RADII
    from ase.io import read as ase_read
    from ase.neighborlist import NeighborList, natural_cutoffs
    ASE_AVAILABLE = True
except Exception:
    ASE_ATOMIC_NUMBERS = {}
    ASE_COVALENT_RADII = []
    ase_read = None
    NeighborList = None
    natural_cutoffs = None
    ASE_AVAILABLE = False


CPK_COLORS = {
    "H": (1.0, 1.0, 1.0, 1.0),
    "C": (0.31, 0.37, 0.48, 1.0),
    "N": (0.24, 0.44, 0.87, 1.0),
    "O": (0.78, 0.18, 0.22, 1.0),
    "Mg": (0.66, 0.80, 0.64, 1.0),
}


COVALENT_RADII_FALLBACK = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Mg": 1.41,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Na": 1.66,
    "K": 2.03,
    "Ca": 1.76,
    "Al": 1.21,
    "Ti": 1.60,
    "Fe": 1.16,
    "Cu": 1.32,
    "Zn": 1.22,
}


STYLE_PRESETS = {
    "handdrawn": {
        "background": (0.95, 0.98, 1.0, 1.0),
        "fallback_atom": (0.62, 0.67, 0.74, 1.0),
        "element_colors": CPK_COLORS,
        "surface_palette": [
            (0.72, 0.81, 0.69, 1.0),
            (0.64, 0.79, 0.90, 1.0),
        ],
        "atom_material": {
            "roughness": 0.42,
            "specular": 0.20,
            "metallic": 0.0,
            "coat": 0.08,
            "coat_roughness": 0.30,
            "specular_tint": 0.00,
        },
        "bond_material": {
            "color": (0.39, 0.44, 0.53, 1.0),
            "roughness": 0.68,
            "specular": 0.06,
            "metallic": 0.0,
            "coat": 0.00,
            "coat_roughness": 0.10,
            "specular_tint": 0.00,
        },
        "outline": {
            "enabled": True,
            "thickness": 1.5,
            "color": (0.34, 0.39, 0.48, 1.0),
        },
    },
    "teacher_glossy": {
        "background": (1.0, 1.0, 1.0, 1.0),
        "fallback_atom": (0.38, 0.42, 0.47, 1.0),
        "element_colors": CPK_COLORS,
        "surface_palette": [
            (0.72, 0.81, 0.69, 1.0),
            (0.64, 0.79, 0.90, 1.0),
        ],
        "atom_material": {
            "roughness": 0.10,
            "specular": 0.65,
            "metallic": 0.0,
            "coat": 0.65,
            "coat_roughness": 0.05,
            "specular_tint": 0.10,
        },
        "bond_material": {
            "color": (0.20, 0.20, 0.20, 1.0),
            "roughness": 0.78,
            "specular": 0.08,
            "metallic": 0.0,
            "coat": 0.00,
            "coat_roughness": 0.10,
            "specular_tint": 0.00,
        },
        "outline": {
            "enabled": False,
            "thickness": 1.0,
            "color": (0.2, 0.2, 0.2, 1.0),
        },
    },
    "publication": {
        "background": (1.0, 1.0, 1.0, 1.0),
        "fallback_atom": (0.65, 0.65, 0.68, 1.0),
        "element_colors": CPK_COLORS,
        "surface_palette": [
            (0.73, 0.83, 0.72, 1.0),
            (0.66, 0.82, 0.92, 1.0),
        ],
        "atom_material": {
            "roughness": 0.34,
            "specular": 0.28,
            "metallic": 0.0,
            "coat": 0.03,
            "coat_roughness": 0.20,
            "specular_tint": 0.00,
        },
        "bond_material": {
            "color": (0.45, 0.45, 0.48, 1.0),
            "roughness": 0.50,
            "specular": 0.20,
            "metallic": 0.0,
            "coat": 0.00,
            "coat_roughness": 0.10,
            "specular_tint": 0.00,
        },
        "outline": {
            "enabled": False,
            "thickness": 1.0,
            "color": (0.2, 0.2, 0.2, 1.0),
        },
    },
}


CAMERA_PRESETS = {
    "iso_ortho": {
        "type": "ORTHO",
        "forward": (0.65, -0.45, -0.62),
    },
    "top_ortho": {
        "type": "ORTHO",
        "forward": (0.0, 0.0, -1.0),
    },
    "teacher_iso": {
        "type": "PERSP",
        "forward": (0.00, 0.88, -0.48),
    },
}


LIGHTING_PRESETS = {
    "toon_three_point": [
        {"type": "AREA", "offset": (-0.95, 0.90, 1.25), "energy": 900, "size": 0.55},
        {"type": "AREA", "offset": (1.05, 0.55, 0.80), "energy": 480, "size": 0.80},
        {"type": "POINT", "offset": (-0.25, -1.20, 0.95), "energy": 120},
    ],
    "teacher_three_point": [
        {"type": "AREA", "offset": (-0.70, 0.70, 1.15), "energy": 650, "size": 0.55},
        {"type": "AREA", "offset": (0.95, 0.65, 0.80), "energy": 360, "size": 0.75},
        {"type": "POINT", "offset": (-0.15, -1.20, 0.90), "energy": 110},
    ],
    "studio_three_point": [
        {"type": "AREA", "offset": (-1.00, 0.80, 1.10), "energy": 1200, "size": 0.85},
        {"type": "AREA", "offset": (1.20, 0.55, 0.75), "energy": 680, "size": 1.10},
        {"type": "POINT", "offset": (0.00, -1.10, 0.90), "energy": 180},
    ],
}


DEFAULT_CONFIG = {
    "input": {
        "path": "mgo111_dense_row_rect_3ads.xyz",
        "frame": "last",
    },
    "output": {
        "path": "render_stylized.png",
    },
    "style": "handdrawn",
    "model": {
        "atom_scale": 1.0,
        "bond_radius": 0.09,
        "bond_cutoff_scale": 1.10,
        "draw_bonds": True,
        "draw_surface_bonds": False,
        "draw_cell": False,
        "surface_symbols": ["Mg", "O"],
        "surface_layer_coloring": True,
        "surface_layer_tolerance": 0.35,
        "element_scale": {
            "H": 0.90,
            "C": 1.00,
            "N": 1.00,
            "O": 1.05,
            "Mg": 1.45,
        },
        "element_colors": {},
        "sphere_segments": 48,
        "sphere_rings": 24,
        "bond_vertices": 18,
    },
    "camera": {
        "preset": "iso_ortho",
        "fit_padding": 0.10,
        "lens_mm": 80.0,
        "clip_start": 0.01,
        "clip_end": 5000.0,
    },
    "lighting": {
        "preset": "toon_three_point",
        "intensity": 1.0,
    },
    "render": {
        "engine": "BLENDER_EEVEE",
        "samples": 64,
        "resolution": [1600, 1000],
        "transparent_bg": False,
        "background": None,
    },
}


class StructureFrame:
    def __init__(self, symbols, positions, cell=None, pbc=(False, False, False)):
        self._symbols = list(symbols)
        self._positions = [tuple(float(v) for v in p) for p in positions]
        self._cell = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] if cell is None else cell
        self._pbc = tuple(bool(v) for v in pbc)

    def __len__(self):
        return len(self._symbols)

    def get_chemical_symbols(self):
        return self._symbols

    def get_positions(self):
        return self._positions

    def get_cell(self):
        return self._cell

    def get_pbc(self):
        return self._pbc


def _parse_args():
    parser = argparse.ArgumentParser(description="One-click stylized ASE -> Blender renderer")
    parser.add_argument("--config", help="JSON config path", default=None)
    parser.add_argument("--input", help="Structure file path (ASE-readable)", default=None)
    parser.add_argument("--frame", help="Frame selector: last or integer", default=None)
    parser.add_argument("--out", help="Output PNG path", default=None)
    parser.add_argument("--style", choices=sorted(STYLE_PRESETS.keys()), default=None)
    parser.add_argument("--camera-preset", choices=sorted(CAMERA_PRESETS.keys()), default=None)
    parser.add_argument("--lighting-preset", choices=sorted(LIGHTING_PRESETS.keys()), default=None)
    parser.add_argument("--engine", choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"], default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--res-x", type=int, default=None)
    parser.add_argument("--res-y", type=int, default=None)
    parser.add_argument("--fit-padding", type=float, default=None)
    parser.add_argument("--transparent-bg", action="store_true")
    parser.add_argument("--solid-bg", action="store_true")
    parser.add_argument("--no-bonds", action="store_true")
    parser.add_argument("--draw-surface-bonds", action="store_true")
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    return parser.parse_args(argv)


def _deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


def _load_json(path):
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def _as_rgba(value):
    if isinstance(value, str) and value.startswith("#"):
        hexv = value[1:]
        if len(hexv) == 6:
            r = int(hexv[0:2], 16) / 255.0
            g = int(hexv[2:4], 16) / 255.0
            b = int(hexv[4:6], 16) / 255.0
            return (r, g, b, 1.0)
        if len(hexv) == 8:
            r = int(hexv[0:2], 16) / 255.0
            g = int(hexv[2:4], 16) / 255.0
            b = int(hexv[4:6], 16) / 255.0
            a = int(hexv[6:8], 16) / 255.0
            return (r, g, b, a)
    if isinstance(value, (list, tuple)):
        if len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]), 1.0)
        if len(value) == 4:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    raise ValueError(f"Cannot parse color: {value}")


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _get_or_create_material(name, color, cfg):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return mat

    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = float(cfg.get("roughness", 0.35))
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = float(cfg.get("metallic", 0.0))
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = float(cfg.get("specular", 0.3))
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = float(cfg.get("specular", 0.3))
    tint = float(cfg.get("specular_tint", 0.0))
    if "Specular Tint" in bsdf.inputs:
        try:
            bsdf.inputs["Specular Tint"].default_value = tint
        except Exception:
            bsdf.inputs["Specular Tint"].default_value = (tint, tint, tint, 1.0)
    coat = float(cfg.get("coat", 0.0))
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = coat
    elif "Coat" in bsdf.inputs:
        bsdf.inputs["Coat"].default_value = coat
    if "Coat Roughness" in bsdf.inputs:
        bsdf.inputs["Coat Roughness"].default_value = float(cfg.get("coat_roughness", 0.08))
    return mat


def _assign_mat(obj, mat):
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def _add_sphere(location, radius, mat, segments, rings, name):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=int(segments),
        ring_count=int(rings),
        radius=float(radius),
        location=location,
    )
    obj = bpy.context.active_object
    obj.name = name
    for poly in obj.data.polygons:
        poly.use_smooth = True
    _assign_mat(obj, mat)
    return obj


def _add_cylinder_between(p1, p2, radius, mat, vertices, name):
    v1, v2 = Vector(p1), Vector(p2)
    vec = v2 - v1
    length = vec.length
    if length < 1e-8:
        return None
    mid = (v1 + v2) / 2.0
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=int(vertices),
        radius=float(radius),
        depth=float(length),
        location=mid,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(vec.normalized())
    for poly in obj.data.polygons:
        poly.use_smooth = True
    _assign_mat(obj, mat)
    return obj


def _build_bond_pairs(atoms, cutoff_scale):
    if not ASE_AVAILABLE or NeighborList is None or natural_cutoffs is None:
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        pairs = []
        for i in range(len(symbols)):
            ri = COVALENT_RADII_FALLBACK.get(symbols[i], 0.9)
            xi, yi, zi = positions[i]
            for j in range(i + 1, len(symbols)):
                rj = COVALENT_RADII_FALLBACK.get(symbols[j], 0.9)
                xj, yj, zj = positions[j]
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                d2 = dx * dx + dy * dy + dz * dz
                cutoff = float(cutoff_scale) * (ri + rj)
                if 0.04 < d2 <= cutoff * cutoff:
                    pairs.append((i, j))
        return pairs

    cutoffs = [c * float(cutoff_scale) for c in natural_cutoffs(atoms)]
    nlist = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nlist.update(atoms)
    pairs = set()
    for i in range(len(atoms)):
        idx, _ = nlist.get_neighbors(i)
        for j in idx:
            a, b = sorted((int(i), int(j)))
            if a != b:
                pairs.add((a, b))
    return sorted(pairs)


def _surface_layer_map(symbols, positions, surface_symbols, tol):
    surface_symbols = set(surface_symbols)
    candidates = [(i, float(positions[i][2])) for i, sym in enumerate(symbols) if sym in surface_symbols]
    if not candidates:
        return {}

    sorted_z = sorted(z for _, z in candidates)
    cut_idx = max(0, int(len(sorted_z) * 0.85) - 1)
    z_cut = sorted_z[cut_idx]
    candidates = [(i, z) for i, z in candidates if z <= z_cut]
    candidates.sort(key=lambda x: x[1])

    layer_map = {}
    layers = []
    for i, z in candidates:
        if not layers or abs(z - layers[-1]) > tol:
            layers.append(z)
        layer_map[i] = len(layers) - 1
    return layer_map


def _radius_for_symbol(symbol, atom_scale, element_scale):
    if ASE_AVAILABLE:
        number = ASE_ATOMIC_NUMBERS.get(symbol)
        base = ASE_COVALENT_RADII[number] if number is not None and number < len(ASE_COVALENT_RADII) else 0.9
    else:
        base = COVALENT_RADII_FALLBACK.get(symbol, 0.9)
    local = float(element_scale.get(symbol, 1.0))
    return max(0.05, float(base) * float(atom_scale) * local * 0.35)


def _cell_edges(cell):
    a = Vector(cell[0])
    b = Vector(cell[1])
    c = Vector(cell[2])
    origin = Vector((0.0, 0.0, 0.0))
    p = {
        "o": origin, "a": a, "b": b, "c": c,
        "ab": a + b, "ac": a + c, "bc": b + c, "abc": a + b + c,
    }
    edges = [
        ("o", "a"), ("o", "b"), ("o", "c"),
        ("a", "ab"), ("a", "ac"),
        ("b", "ab"), ("b", "bc"),
        ("c", "ac"), ("c", "bc"),
        ("ab", "abc"), ("ac", "abc"), ("bc", "abc"),
    ]
    return [(tuple(p[s]), tuple(p[t])) for s, t in edges]


def _has_cell(cell):
    return sum(v * v for row in cell for v in row) > 1e-12


def _setup_render(cfg, style):
    scene = bpy.context.scene
    engine = cfg["render"]["engine"]
    supported = scene.render.bl_rna.properties["engine"].enum_items.keys()
    if engine not in supported:
        engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in supported else list(supported)[0]
    scene.render.engine = engine
    scene.render.resolution_x = int(cfg["render"]["resolution"][0])
    scene.render.resolution_y = int(cfg["render"]["resolution"][1])
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = bool(cfg["render"]["transparent_bg"])
    scene.render.image_settings.color_mode = "RGBA" if cfg["render"]["transparent_bg"] else "RGB"

    samples = max(1, int(cfg["render"]["samples"]))
    if hasattr(scene, "cycles"):
        scene.cycles.samples = samples
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0

    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = next((node for node in world.node_tree.nodes if node.type == "BACKGROUND"), None)
    bg_color = cfg["render"]["background"]
    bg_color = style["background"] if bg_color is None else _as_rgba(bg_color)
    if bg:
        bg.inputs[0].default_value = bg_color
        bg.inputs[1].default_value = 1.0

    outline = style.get("outline", {})
    scene.render.use_freestyle = bool(outline.get("enabled", False))
    if scene.render.use_freestyle:
        view_layer = bpy.context.view_layer
        fs = view_layer.freestyle_settings
        if len(fs.linesets) == 0:
            bpy.ops.scene.freestyle_lineset_add()
        line_set = fs.linesets[0]
        line_set.select_silhouette = True
        line_set.select_border = True
        line_set.select_crease = False
        line_set.select_ridge_valley = False
        line_set.select_external_contour = True
        line_style = line_set.linestyle
        line_style.color = outline.get("color", (0.2, 0.2, 0.2, 1.0))[:3]
        line_style.thickness = float(outline.get("thickness", 1.2))


def _add_lights(points, cfg):
    preset_name = cfg["lighting"]["preset"]
    preset = LIGHTING_PRESETS[preset_name]
    intensity = float(cfg["lighting"].get("intensity", 1.0))

    xs = [p.x for p in points] or [0.0]
    ys = [p.y for p in points] or [0.0]
    zs = [p.z for p in points] or [0.0]
    center = Vector(((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, (min(zs) + max(zs)) * 0.5))
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)

    for i, spec in enumerate(preset):
        loc = center + Vector(spec["offset"]) * extent
        bpy.ops.object.light_add(type=spec["type"], location=loc)
        light_obj = bpy.context.active_object
        light_obj.name = f"Light_{preset_name}_{i}"
        light_obj.data.energy = float(spec["energy"]) * intensity
        if spec["type"] == "AREA":
            light_obj.data.size = max(0.1, float(spec.get("size", 0.5)) * extent)


def _setup_camera(points, cfg):
    cam_cfg = cfg["camera"]
    preset = CAMERA_PRESETS[cam_cfg["preset"]]
    forward = Vector(preset["forward"])
    if forward.length < 1e-8:
        forward = Vector((0.0, 0.0, -1.0))
    forward.normalize()

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "RenderCamera"
    cam.data.type = preset["type"]
    cam.data.lens = float(cam_cfg["lens_mm"])
    cam.data.clip_start = float(cam_cfg["clip_start"])
    cam.data.clip_end = float(cam_cfg["clip_end"])

    margin = max(0.01, min(0.35, float(cam_cfg["fit_padding"])))
    usable = max(0.05, 1.0 - 2.0 * margin)

    center = sum(points, Vector((0.0, 0.0, 0.0))) / max(1, len(points))
    quat = forward.to_track_quat("-Z", "Y")
    rot_inv = quat.to_matrix().inverted()
    local = [rot_inv @ (p - center) for p in points] if points else [Vector((0.0, 0.0, 0.0))]

    max_abs_x = max(abs(v.x) for v in local)
    max_abs_y = max(abs(v.y) for v in local)
    max_z = max(v.z for v in local)
    min_z = min(v.z for v in local)

    aspect = bpy.context.scene.render.resolution_x / max(1.0, bpy.context.scene.render.resolution_y)
    if cam.data.type == "ORTHO":
        half_w = max_abs_x / usable
        half_h = max_abs_y / usable
        cam.data.ortho_scale = max(2.0 * half_w, 2.0 * half_h * aspect, 1.0)
        distance = max(max_z - min_z + 1.0, 3.5)
    else:
        tan_x = tan(cam.data.angle_x * 0.5) * usable
        tan_y = tan(cam.data.angle_y * 0.5) * usable
        need_d = 0.0
        for v in local:
            need_d = max(
                need_d,
                v.z + abs(v.x) / max(1e-6, tan_x),
                v.z + abs(v.y) / max(1e-6, tan_y),
            )
        distance = max(need_d + 1.2, 4.0)

    cam.location = center - forward * distance
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = quat
    bpy.context.scene.camera = cam
    return cam


def _read_atoms(input_path, frame):
    frame_key = str(frame).lower()
    frame_index = -1 if frame_key == "last" else int(frame)

    if ASE_AVAILABLE and ase_read is not None:
        atoms = ase_read(str(input_path), index=frame_index)
        if isinstance(atoms, list):
            atoms = atoms[-1]
        return atoms

    if input_path.suffix.lower() != ".xyz":
        raise RuntimeError(
            "ASE is not available in Blender Python. "
            "Fallback parser only supports .xyz. "
            "Install ASE in Blender Python for other formats."
        )

    lines = input_path.read_text(encoding="utf-8").splitlines()
    frames = []
    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        natoms = int(lines[i].strip())
        i += 1
        comment = lines[i] if i < len(lines) else ""
        i += 1

        cell = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        pbc = (False, False, False)
        m = re.search(r'Lattice=\"([^\"]+)\"', comment)
        if m:
            vals = [float(x) for x in m.group(1).split()]
            if len(vals) == 9:
                cell = [vals[0:3], vals[3:6], vals[6:9]]
        m = re.search(r'pbc=\"([^\"]+)\"', comment)
        if m:
            bits = m.group(1).split()
            if len(bits) == 3:
                pbc = tuple(v.lower() in {"t", "true", "1"} for v in bits)

        symbols = []
        positions = []
        for _ in range(natoms):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            if len(parts) < 4:
                continue
            symbols.append(parts[0])
            positions.append((float(parts[1]), float(parts[2]), float(parts[3])))

        frames.append(StructureFrame(symbols, positions, cell=cell, pbc=pbc))

    if not frames:
        raise RuntimeError(f"Failed to parse XYZ file: {input_path}")
    return frames[frame_index]


def _collect_bbox_points(objs):
    points = []
    for obj in objs:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    return points


def _build_model(atoms, cfg, style):
    model_cfg = cfg["model"]
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell_raw = atoms.get_cell()
    cell = cell_raw.tolist() if hasattr(cell_raw, "tolist") else cell_raw

    surface_symbols = set(model_cfg["surface_symbols"])
    layer_map = {}
    if model_cfg.get("surface_layer_coloring", False):
        layer_map = _surface_layer_map(
            symbols,
            positions,
            surface_symbols,
            float(model_cfg.get("surface_layer_tolerance", 0.35)),
        )

    mat_cache = {}
    objs = []

    def atom_color(i, symbol):
        override = model_cfg.get("element_colors", {}).get(symbol)
        if override is not None:
            return _as_rgba(override)
        if i in layer_map:
            palette = style["surface_palette"]
            return palette[layer_map[i] % len(palette)]
        return style["element_colors"].get(symbol, style["fallback_atom"])

    atom_mat_cfg = style["atom_material"]
    for i, (symbol, pos) in enumerate(zip(symbols, positions)):
        color = atom_color(i, symbol)
        key = ("atom", symbol, tuple(round(c, 5) for c in color), cfg["style"])
        if key not in mat_cache:
            name = f"AtomMat_{symbol}_{len(mat_cache)}"
            mat_cache[key] = _get_or_create_material(name, color, atom_mat_cfg)
        radius = _radius_for_symbol(
            symbol,
            model_cfg["atom_scale"],
            model_cfg.get("element_scale", {}),
        )
        objs.append(
            _add_sphere(
                location=tuple(float(v) for v in pos),
                radius=radius,
                mat=mat_cache[key],
                segments=model_cfg["sphere_segments"],
                rings=model_cfg["sphere_rings"],
                name=f"Atom_{i}_{symbol}",
            )
        )

    bonds = _build_bond_pairs(atoms, model_cfg["bond_cutoff_scale"])
    bond_mat = _get_or_create_material("BondMat", style["bond_material"]["color"], style["bond_material"])
    if model_cfg["draw_bonds"]:
        for i, j in bonds:
            if not model_cfg["draw_surface_bonds"] and symbols[i] in surface_symbols and symbols[j] in surface_symbols:
                continue
            obj = _add_cylinder_between(
                p1=tuple(float(v) for v in positions[i]),
                p2=tuple(float(v) for v in positions[j]),
                radius=float(model_cfg["bond_radius"]),
                mat=bond_mat,
                vertices=model_cfg["bond_vertices"],
                name=f"Bond_{i}_{j}",
            )
            if obj:
                objs.append(obj)

    pbc_raw = atoms.get_pbc()
    pbc = pbc_raw.tolist() if hasattr(pbc_raw, "tolist") else pbc_raw
    if model_cfg["draw_cell"] and any(bool(v) for v in pbc) and _has_cell(cell):
        cell_mat = _get_or_create_material(
            "CellMat",
            (0.18, 0.18, 0.20, 1.0),
            {"roughness": 0.55, "specular": 0.10, "metallic": 0.0, "coat": 0.0},
        )
        for idx, (p1, p2) in enumerate(_cell_edges(cell)):
            obj = _add_cylinder_between(
                p1=p1,
                p2=p2,
                radius=max(0.03, float(model_cfg["bond_radius"]) * 0.6),
                mat=cell_mat,
                vertices=12,
                name=f"CellEdge_{idx}",
            )
            if obj:
                objs.append(obj)
    return objs, len(bonds)


def _merge_config(args):
    cfg = deepcopy(DEFAULT_CONFIG)
    config_dir = None
    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        config_dir = config_path.parent
        _deep_update(cfg, _load_json(config_path))
    if args.input:
        cfg["input"]["path"] = args.input
    if args.frame:
        cfg["input"]["frame"] = args.frame
    if args.out:
        cfg["output"]["path"] = args.out
    if args.style:
        cfg["style"] = args.style
    if args.camera_preset:
        cfg["camera"]["preset"] = args.camera_preset
    if args.lighting_preset:
        cfg["lighting"]["preset"] = args.lighting_preset
    if args.engine:
        cfg["render"]["engine"] = args.engine
    if args.samples is not None:
        cfg["render"]["samples"] = int(args.samples)
    if args.res_x is not None:
        cfg["render"]["resolution"][0] = int(args.res_x)
    if args.res_y is not None:
        cfg["render"]["resolution"][1] = int(args.res_y)
    if args.fit_padding is not None:
        cfg["camera"]["fit_padding"] = float(args.fit_padding)
    if args.transparent_bg:
        cfg["render"]["transparent_bg"] = True
    if args.solid_bg:
        cfg["render"]["transparent_bg"] = False
    if args.no_bonds:
        cfg["model"]["draw_bonds"] = False
    if args.draw_surface_bonds:
        cfg["model"]["draw_surface_bonds"] = True

    if config_dir is not None:
        input_path = Path(cfg["input"]["path"]).expanduser()
        if not input_path.is_absolute():
            cfg["input"]["path"] = str((config_dir / input_path).resolve())
        output_path = Path(cfg["output"]["path"]).expanduser()
        if not output_path.is_absolute():
            cfg["output"]["path"] = str((config_dir / output_path).resolve())
    return cfg


def main():
    args = _parse_args()
    cfg = _merge_config(args)

    if cfg["style"] not in STYLE_PRESETS:
        raise ValueError(f"Unknown style: {cfg['style']}")
    if cfg["camera"]["preset"] not in CAMERA_PRESETS:
        raise ValueError(f"Unknown camera preset: {cfg['camera']['preset']}")
    if cfg["lighting"]["preset"] not in LIGHTING_PRESETS:
        raise ValueError(f"Unknown lighting preset: {cfg['lighting']['preset']}")

    input_path = Path(cfg["input"]["path"]).expanduser().resolve()
    output_path = Path(cfg["output"]["path"]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atoms = _read_atoms(input_path, cfg["input"]["frame"])

    style = STYLE_PRESETS[cfg["style"]]
    _clear_scene()
    _setup_render(cfg, style)
    objs, bond_count = _build_model(atoms, cfg, style)

    bbox_points = _collect_bbox_points(objs)
    _add_lights(bbox_points, cfg)
    _setup_camera(bbox_points, cfg)

    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    print(
        json.dumps(
            {
                "success": True,
                "style": cfg["style"],
                "atoms": len(atoms),
                "bonds_detected": bond_count,
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
