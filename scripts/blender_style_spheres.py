#!/usr/bin/env python3
"""Render standalone Blender material/style candidate spheres.

This script intentionally does not import atomstudio. Run with Blender:

    blender --background --python scripts/blender_style_spheres.py -- --output /tmp/style_spheres.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/atomstudio_style_spheres.png")
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--res-x", type=int, default=2200)
    parser.add_argument("--res-y", type=int, default=1800)
    parser.add_argument("--save-blend", default="")
    return parser.parse_args(args)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def set_input(node, names: tuple[str, ...], value) -> None:
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return


def principled_material(
    name: str,
    *,
    base: tuple[float, float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.5,
    alpha: float = 1.0,
    transmission: float = 0.0,
    ior: float = 1.45,
    coat: float = 0.0,
    coat_roughness: float = 0.15,
    specular: float = 0.5,
    sheen: float = 0.0,
    subsurface: float = 0.0,
    anisotropic: float = 0.0,
    emission: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        return mat

    set_input(bsdf, ("Base Color",), base)
    set_input(bsdf, ("Metallic",), metallic)
    set_input(bsdf, ("Roughness",), roughness)
    set_input(bsdf, ("Alpha",), alpha)
    set_input(bsdf, ("Transmission Weight", "Transmission"), transmission)
    set_input(bsdf, ("IOR",), ior)
    set_input(bsdf, ("Coat Weight", "Coat"), coat)
    set_input(bsdf, ("Coat Roughness",), coat_roughness)
    set_input(bsdf, ("Specular IOR Level", "Specular"), specular)
    set_input(bsdf, ("Sheen Weight", "Sheen"), sheen)
    set_input(bsdf, ("Subsurface Weight", "Subsurface"), subsurface)
    set_input(bsdf, ("Anisotropic Rotation", "Anisotropic"), anisotropic)
    if emission is not None:
        set_input(bsdf, ("Emission Color", "Emission"), emission)
        set_input(bsdf, ("Emission Strength",), emission_strength)

    if alpha < 0.999 or transmission > 0.0:
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = True if hasattr(mat, "use_screen_refraction") else False
        if hasattr(mat, "show_transparent_back"):
            mat.show_transparent_back = True
    return mat


def add_noise_bump(mat: bpy.types.Material, *, scale: float, strength: float, detail: float = 8.0) -> None:
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    noise.inputs["Roughness"].default_value = 0.58
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = 0.08
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def pearl_material() -> bpy.types.Material:
    mat = principled_material(
        "pearl",
        base=(0.94, 0.90, 0.82, 1.0),
        roughness=0.18,
        specular=0.8,
        coat=0.8,
        coat_roughness=0.055,
        sheen=0.45,
        subsurface=0.12,
    )
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat
    layer = nodes.new("ShaderNodeLayerWeight")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (0.96, 0.88, 0.74, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.72, 0.86, 1.0, 1.0)
    mid = ramp.color_ramp.elements.new(0.58)
    mid.color = (1.0, 0.78, 0.92, 1.0)
    links.new(layer.outputs["Facing"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    add_noise_bump(mat, scale=55.0, strength=0.018, detail=10.0)
    return mat


def holographic_material() -> bpy.types.Material:
    mat = principled_material(
        "holographic",
        base=(0.55, 0.80, 1.0, 1.0),
        roughness=0.12,
        specular=0.9,
        coat=0.9,
        coat_roughness=0.025,
        emission=(0.25, 0.55, 1.0, 1.0),
        emission_strength=0.035,
    )
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat
    layer = nodes.new("ShaderNodeLayerWeight")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.00
    ramp.color_ramp.elements[0].color = (0.36, 0.18, 1.00, 1.0)
    ramp.color_ramp.elements[1].position = 1.00
    ramp.color_ramp.elements[1].color = (1.00, 0.22, 0.72, 1.0)
    for pos, color in [
        (0.24, (0.00, 0.86, 1.00, 1.0)),
        (0.48, (0.26, 1.00, 0.58, 1.0)),
        (0.72, (1.00, 0.92, 0.20, 1.0)),
    ]:
        elem = ramp.color_ramp.elements.new(pos)
        elem.color = color
    links.new(layer.outputs["Facing"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, ("Emission Color", "Emission"), (0.18, 0.45, 0.95, 1.0))
    return mat


def make_materials() -> list[tuple[str, bpy.types.Material]]:
    materials = [
        (
            "clean glossy",
            principled_material(
                "clean_glossy",
                base=(0.95, 0.12, 0.12, 1.0),
                roughness=0.16,
                specular=0.72,
                coat=0.28,
                coat_roughness=0.08,
            ),
        ),
        (
            "porcelain",
            principled_material(
                "porcelain",
                base=(0.93, 0.90, 0.84, 1.0),
                roughness=0.58,
                specular=0.2,
                coat=0.24,
                coat_roughness=0.22,
            ),
        ),
        (
            "matte ceramic",
            principled_material(
                "matte_ceramic",
                base=(0.52, 0.60, 0.66, 1.0),
                roughness=0.88,
                specular=0.04,
            ),
        ),
        (
            "frosted glass",
            principled_material(
                "frosted_glass",
                base=(0.70, 0.88, 1.00, 0.58),
                roughness=0.28,
                alpha=0.58,
                transmission=0.65,
                ior=1.46,
                specular=0.7,
            ),
        ),
        (
            "solid glass",
            principled_material(
                "solid_glass",
                base=(0.72, 0.83, 1.00, 1.0),
                roughness=0.045,
                specular=0.88,
                ior=1.52,
                coat=0.72,
                coat_roughness=0.035,
            ),
        ),
        (
            "clear crystal",
            principled_material(
                "clear_crystal",
                base=(0.74, 0.94, 1.00, 0.36),
                roughness=0.005,
                alpha=0.36,
                transmission=0.92,
                ior=1.55,
                specular=0.9,
                coat=0.2,
                coat_roughness=0.015,
            ),
        ),
        (
            "jade",
            principled_material(
                "jade",
                base=(0.18, 0.58, 0.42, 1.0),
                roughness=0.20,
                specular=0.55,
                subsurface=0.28,
                coat=0.38,
                coat_roughness=0.075,
            ),
        ),
        (
            "obsidian",
            principled_material(
                "obsidian",
                base=(0.010, 0.012, 0.018, 1.0),
                roughness=0.06,
                specular=0.82,
                coat=0.95,
                coat_roughness=0.025,
            ),
        ),
        ("pearl", pearl_material()),
        (
            "brushed metal",
            principled_material(
                "brushed_metal",
                base=(0.72, 0.68, 0.60, 1.0),
                metallic=1.0,
                roughness=0.34,
                specular=0.65,
                anisotropic=0.6,
            ),
        ),
        (
            "chrome",
            principled_material(
                "chrome",
                base=(0.92, 0.94, 0.96, 1.0),
                metallic=1.0,
                roughness=0.035,
                specular=1.0,
            ),
        ),
        ("holographic", holographic_material()),
        (
            "velvet",
            principled_material(
                "velvet",
                base=(0.35, 0.04, 0.22, 1.0),
                roughness=0.82,
                specular=0.18,
                sheen=1.0,
            ),
        ),
        (
            "soft rubber",
            principled_material(
                "soft_rubber",
                base=(0.025, 0.028, 0.032, 1.0),
                roughness=0.92,
                specular=0.08,
            ),
        ),
        (
            "warm clay",
            principled_material(
                "warm_clay",
                base=(0.72, 0.36, 0.19, 1.0),
                roughness=0.95,
                specular=0.025,
            ),
        ),
    ]

    for label, mat in materials:
        if label in {"matte ceramic", "soft rubber", "warm clay"}:
            add_noise_bump(mat, scale=42.0, strength=0.035, detail=9.0)
        if label == "brushed metal":
            add_noise_bump(mat, scale=72.0, strength=0.018, detail=14.0)
    return materials


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_label(label: str, x: float, z: float, mat: bpy.types.Material) -> None:
    bpy.ops.object.text_add(location=(x, -0.82, z), rotation=(math.radians(90.0), 0.0, 0.0))
    text = bpy.context.object
    text.name = f"label_{label.replace(' ', '_')}"
    text.data.body = label
    text.data.align_x = "CENTER"
    text.data.align_y = "CENTER"
    text.data.size = 0.22
    text.data.resolution_u = 16
    text.data.materials.append(mat)


def add_sphere(label: str, mat: bpy.types.Material, x: float, z: float) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=0.72, location=(x, 0.0, z))
    sphere = bpy.context.object
    sphere.name = f"sphere_{label.replace(' ', '_')}"
    sphere.data.materials.append(mat)
    bpy.ops.object.shade_smooth()


def add_lights() -> None:
    bpy.ops.object.light_add(type="AREA", location=(-4.8, -5.5, 7.2))
    key = bpy.context.object
    key.name = "Key_softbox"
    key.data.energy = 620.0
    key.data.size = 5.8
    look_at(key, (0.0, 0.0, 0.0))

    bpy.ops.object.light_add(type="AREA", location=(4.8, -6.2, 2.8))
    fill = bpy.context.object
    fill.name = "Fill_softbox"
    fill.data.energy = 80.0
    fill.data.size = 7.5
    look_at(fill, (0.0, 0.0, 0.0))

    bpy.ops.object.light_add(type="AREA", location=(0.0, 3.0, 6.8))
    rim = bpy.context.object
    rim.name = "Back_rim"
    rim.data.energy = 360.0
    rim.data.size = 2.8
    look_at(rim, (0.0, 0.0, 0.5))


def add_camera() -> None:
    bpy.ops.object.camera_add(location=(0.0, -13.5, 0.0))
    camera = bpy.context.object
    camera.name = "Style_contact_sheet_camera"
    look_at(camera, (0.0, 0.0, 0.0))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 12.4
    bpy.context.scene.camera = camera


def make_text_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("label_text")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        set_input(bsdf, ("Base Color",), (0.93, 0.95, 0.98, 1.0))
        set_input(bsdf, ("Roughness",), 0.65)
        set_input(bsdf, ("Emission Color", "Emission"), (0.70, 0.78, 0.92, 1.0))
        set_input(bsdf, ("Emission Strength",), 0.12)
    return mat


def add_background() -> None:
    mat = principled_material("background_panel", base=(0.055, 0.063, 0.075, 1.0), roughness=0.78)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.95, 0.0), rotation=(math.radians(90.0), 0.0, 0.0))
    panel = bpy.context.object
    panel.name = "dark_background_panel"
    panel.scale = (14.6, 13.2, 1.0)
    panel.data.materials.append(mat)


def setup_render(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(args.samples)
    scene.cycles.use_denoising = True
    scene.render.resolution_x = int(args.res_x)
    scene.render.resolution_y = int(args.res_y)
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("style_world")
    scene.world.color = (0.045, 0.052, 0.062)
    try:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "Medium High Contrast"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass


def main() -> None:
    args = parse_args()
    clear_scene()
    setup_render(args)
    add_background()
    add_lights()
    add_camera()

    text_mat = make_text_material()
    materials = make_materials()
    cols = 4
    spacing_x = 2.8
    spacing_z = 2.6
    for index, (label, mat) in enumerate(materials):
        row = index // cols
        col = index % cols
        x = (col - (cols - 1) / 2.0) * spacing_x
        z = ((len(materials) // cols - 1) / 2.0 - row) * spacing_z
        add_sphere(label, mat, x, z + 0.22)
        add_label(label, x, z - 0.72, text_mat)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output)
    if args.save_blend:
        blend_path = Path(args.save_blend).expanduser()
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.render.render(write_still=True)
    print(f"Rendered style candidates to {output}")


if __name__ == "__main__":
    main()
