from __future__ import annotations

from math import cos, exp, pi, sin, tan
from typing import Any, Sequence

from atomstudio.backend.blender.collections import ensure_collection
from atomstudio.config import RenderJobConfig
from atomstudio.scene.materials.request import MaterialRequest
from atomstudio.scene.materials.specs import MaterialSpec

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    Vector = None


class GroundBuilder:
    def __init__(
        self,
        *,
        enabled: bool = False,
        mode: str = "auto",
        size_scale: float = 2.2,
        z_offset_scale: float = 0.03,
        color: tuple[float, float, float, float] = (0.88, 0.88, 0.88, 1.0),
        roughness: float = 0.82,
        specular: float = 0.05,
        metallic: float = 0.0,
        coat: float = 0.0,
        coat_roughness: float = 0.08,
        sweep_enabled: bool = False,
        sweep_width_scale: float = 8.0,
        sweep_width_segments: int = 32,
        sweep_floor_depth_scale: float = 6.0,
        sweep_wall_height_scale: float = 6.0,
        sweep_radius_scale: float = 1.4,
        sweep_floor_offset_scale: float = 0.02,
        sweep_wall_offset_scale: float = 0.40,
        sweep_segments: int = 32,
        sweep_color: tuple[float, float, float, float] = (0.76, 0.78, 0.79, 1.0),
        sweep_roughness: float = 0.62,
        sweep_specular: float = 0.10,
        sweep_metallic: float = 0.0,
        sweep_coat: float = 0.04,
        sweep_coat_roughness: float = 0.32,
        sweep_gradient_enabled: bool = False,
        sweep_bottom_color: tuple[float, float, float, float] | None = None,
        sweep_top_color: tuple[float, float, float, float] | None = None,
        sweep_spot_color: tuple[float, float, float, float] | None = None,
        sweep_spot_strength: float = 0.0,
        sweep_spot_x: float = 0.50,
        sweep_spot_y: float = 0.72,
        sweep_spot_radius: float = 0.32,
        sweep_vignette_strength: float = 0.0,
        registry: Any,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = str(mode).strip().lower()
        self.size_scale = float(size_scale)
        self.z_offset_scale = float(z_offset_scale)
        self.color = (
            float(color[0]),
            float(color[1]),
            float(color[2]),
            float(color[3]),
        )
        self.roughness = float(roughness)
        self.specular = float(specular)
        self.metallic = float(metallic)
        self.coat = float(coat)
        self.coat_roughness = float(coat_roughness)
        self.sweep_enabled = bool(sweep_enabled)
        self.sweep_width_scale = float(sweep_width_scale)
        self.sweep_width_segments = max(1, int(sweep_width_segments))
        self.sweep_floor_depth_scale = float(sweep_floor_depth_scale)
        self.sweep_wall_height_scale = float(sweep_wall_height_scale)
        self.sweep_radius_scale = float(sweep_radius_scale)
        self.sweep_floor_offset_scale = float(sweep_floor_offset_scale)
        self.sweep_wall_offset_scale = float(sweep_wall_offset_scale)
        self.sweep_segments = max(4, int(sweep_segments))
        self.sweep_color = (
            float(sweep_color[0]),
            float(sweep_color[1]),
            float(sweep_color[2]),
            float(sweep_color[3]),
        )
        self.sweep_roughness = float(sweep_roughness)
        self.sweep_specular = float(sweep_specular)
        self.sweep_metallic = float(sweep_metallic)
        self.sweep_coat = float(sweep_coat)
        self.sweep_coat_roughness = float(sweep_coat_roughness)
        self.sweep_gradient_enabled = bool(sweep_gradient_enabled)
        self.sweep_bottom_color = sweep_bottom_color
        self.sweep_top_color = sweep_top_color
        self.sweep_spot_color = sweep_spot_color
        self.sweep_spot_strength = float(sweep_spot_strength)
        self.sweep_spot_x = float(sweep_spot_x)
        self.sweep_spot_y = float(sweep_spot_y)
        self.sweep_spot_radius = max(0.01, float(sweep_spot_radius))
        self.sweep_vignette_strength = float(sweep_vignette_strength)
        self.registry = registry

    @classmethod
    def from_cfg(
        cls,
        cfg: RenderJobConfig,
        *,
        registry: Any,
    ) -> "GroundBuilder":
        ground = cfg.lighting.ground
        sweep = cfg.lighting.sweep
        return cls(
            enabled=ground.enabled,
            mode=ground.mode,
            size_scale=ground.size_scale,
            z_offset_scale=ground.z_offset_scale,
            color=ground.color,
            roughness=ground.roughness,
            specular=ground.specular,
            metallic=ground.metallic,
            coat=ground.coat,
            coat_roughness=ground.coat_roughness,
            sweep_enabled=sweep.enabled,
            sweep_width_scale=sweep.width_scale,
            sweep_width_segments=sweep.width_segments,
            sweep_floor_depth_scale=sweep.floor_depth_scale,
            sweep_wall_height_scale=sweep.wall_height_scale,
            sweep_radius_scale=sweep.radius_scale,
            sweep_floor_offset_scale=sweep.floor_offset_scale,
            sweep_wall_offset_scale=sweep.wall_offset_scale,
            sweep_segments=sweep.segments,
            sweep_color=sweep.color,
            sweep_roughness=sweep.roughness,
            sweep_specular=sweep.specular,
            sweep_metallic=sweep.metallic,
            sweep_coat=sweep.coat,
            sweep_coat_roughness=sweep.coat_roughness,
            sweep_gradient_enabled=sweep.gradient_enabled,
            sweep_bottom_color=sweep.bottom_color,
            sweep_top_color=sweep.top_color,
            sweep_spot_color=sweep.spot_color,
            sweep_spot_strength=sweep.spot_strength,
            sweep_spot_x=sweep.spot_x,
            sweep_spot_y=sweep.spot_y,
            sweep_spot_radius=sweep.spot_radius,
            sweep_vignette_strength=sweep.vignette_strength,
            registry=registry,
        )

    def build(self, points: Sequence) -> tuple[object | None, dict[str, Any]]:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")
        if self.registry is None or not hasattr(self.registry, "get"):
            raise RuntimeError("GroundBuilder requires a MaterialRegistry with get().")
        scene = bpy.context.scene

        spec = resolve_ground_spec(
            points=points,
            render_engine=str(getattr(scene.render, "engine", "BLENDER_EEVEE")),
            transparent_bg=bool(getattr(scene.render, "film_transparent", False)),
            enabled=self.enabled,
            mode=self.mode,
            size_scale=self.size_scale,
            z_offset_scale=self.z_offset_scale,
            color=self.color,
            roughness=self.roughness,
            specular=self.specular,
            metallic=self.metallic,
            coat=self.coat,
            coat_roughness=self.coat_roughness,
        )
        ground_obj = None

        if bool(spec["enabled"]):
            target_xy = _camera_plane_intersection_xy(float(spec["location"][2]))
            if target_xy is not None:
                spec = {
                    **spec,
                    "location": (float(target_xy[0]), float(target_xy[1]), float(spec["location"][2])),
                }

            camera_size = _camera_cover_size(spec["location"])
            if camera_size is not None and camera_size > float(spec["plane_size"]):
                spec = {**spec, "plane_size": float(camera_size)}

            bpy.ops.mesh.primitive_plane_add(size=float(spec["plane_size"]), location=spec["location"])
            obj = bpy.context.active_object
            if obj is None:
                return None, spec

            obj.name = "GroundPlane"
            collection = ensure_collection("Ground")
            if collection is not None:
                for owner in list(obj.users_collection):
                    if owner != collection:
                        owner.objects.unlink(obj)
                if collection not in obj.users_collection:
                    collection.objects.link(obj)

            _assign_ground_material(
                obj,
                name="GroundPlane",
                color=spec["color"],
                roughness=float(spec["roughness"]),
                specular=float(spec["specular"]),
                metallic=float(spec["metallic"]),
                coat=float(spec["coat"]),
                coat_roughness=float(spec["coat_roughness"]),
                registry=self.registry,
            )

            effective = str(spec["effective_mode"])
            if effective == "shadow_catcher":
                if hasattr(obj, "is_shadow_catcher"):
                    try:
                        obj.is_shadow_catcher = True
                    except Exception:
                        effective = "visible"
                else:
                    effective = "visible"

            if effective == "visible" and hasattr(obj, "is_shadow_catcher"):
                try:
                    obj.is_shadow_catcher = False
                except Exception:
                    pass

            if effective != spec["effective_mode"]:
                spec = {**spec, "effective_mode": effective}
            ground_obj = obj

        sweep_obj, sweep_spec = self._build_studio_sweep(points)
        spec = {**spec, "studio_sweep": sweep_spec}
        return ground_obj or sweep_obj, spec

    def _build_studio_sweep(self, points: Sequence) -> tuple[object | None, dict[str, Any]]:
        spec = resolve_studio_sweep_spec(
            points=points,
            enabled=self.sweep_enabled,
            width_scale=self.sweep_width_scale,
            width_segments=self.sweep_width_segments,
            floor_depth_scale=self.sweep_floor_depth_scale,
            wall_height_scale=self.sweep_wall_height_scale,
            radius_scale=self.sweep_radius_scale,
            floor_offset_scale=self.sweep_floor_offset_scale,
            wall_offset_scale=self.sweep_wall_offset_scale,
            segments=self.sweep_segments,
            color=self.sweep_color,
            roughness=self.sweep_roughness,
            specular=self.sweep_specular,
            metallic=self.sweep_metallic,
            coat=self.sweep_coat,
            coat_roughness=self.sweep_coat_roughness,
            gradient_enabled=self.sweep_gradient_enabled,
            bottom_color=self.sweep_bottom_color,
            top_color=self.sweep_top_color,
            spot_color=self.sweep_spot_color,
            spot_strength=self.sweep_spot_strength,
            spot_x=self.sweep_spot_x,
            spot_y=self.sweep_spot_y,
            spot_radius=self.sweep_spot_radius,
            vignette_strength=self.sweep_vignette_strength,
        )
        if not bool(spec["enabled"]):
            return None, spec

        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")

        vertices = spec.get("vertices")
        faces = spec.get("faces")
        if not isinstance(vertices, list) or not isinstance(faces, list) or not faces:
            return None, {**spec, "enabled": False, "reason": "missing_vertices"}

        mesh = bpy.data.meshes.new("StudioSweepMesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new("StudioSweep", mesh)
        collection = ensure_collection("Ground")
        if collection is None:
            bpy.context.collection.objects.link(obj)
        else:
            collection.objects.link(obj)

        for polygon in mesh.polygons:
            polygon.use_smooth = True
        _assign_sweep_vertex_colors(mesh, spec.get("vertex_colors"))

        _assign_sweep_material(
            obj,
            name="StudioSweep",
            color=spec["color"],
            roughness=float(spec["roughness"]),
            specular=float(spec["specular"]),
            metallic=float(spec["metallic"]),
            coat=float(spec["coat"]),
            coat_roughness=float(spec["coat_roughness"]),
            use_vertex_colors=bool(spec.get("gradient_enabled")),
            registry=self.registry,
        )
        return obj, spec


def resolve_ground_spec(
    *,
    points: Sequence,
    render_engine: str,
    transparent_bg: bool,
    enabled: bool,
    mode: str,
    size_scale: float,
    z_offset_scale: float,
    color: tuple[float, float, float, float],
    roughness: float,
    specular: float,
    metallic: float = 0.0,
    coat: float = 0.0,
    coat_roughness: float = 0.08,
) -> dict[str, Any]:
    requested_mode = str(mode).strip().lower()
    effective_mode = _resolve_effective_mode(
        requested_mode=requested_mode,
        render_engine=render_engine,
        transparent_bg=transparent_bg,
    )

    min_x, max_x, min_y, max_y, min_z, max_z = _bounds(points)
    extent = max(max_x - min_x, max_y - min_y, max_z - min_z, 1.0)
    center_xy = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
    z = min_z - float(z_offset_scale) * extent
    plane_size = max(1.0, float(size_scale) * extent)
    is_enabled = bool(enabled)

    return {
        "enabled": is_enabled,
        "requested_mode": requested_mode,
        "effective_mode": effective_mode if is_enabled else "disabled",
        "location": (float(center_xy[0]), float(center_xy[1]), float(z)),
        "plane_size": float(plane_size),
        "extent": float(extent),
        "color": tuple(float(v) for v in color),
        "roughness": max(0.0, min(1.0, float(roughness))),
        "specular": max(0.0, min(1.0, float(specular))),
        "metallic": max(0.0, min(1.0, float(metallic))),
        "coat": max(0.0, min(1.0, float(coat))),
        "coat_roughness": max(0.0, min(1.0, float(coat_roughness))),
    }


def resolve_studio_sweep_spec(
    *,
    points: Sequence,
    enabled: bool,
    width_scale: float,
    width_segments: int,
    floor_depth_scale: float,
    wall_height_scale: float,
    radius_scale: float,
    floor_offset_scale: float,
    wall_offset_scale: float,
    segments: int,
    color: tuple[float, float, float, float],
    roughness: float,
    specular: float,
    metallic: float = 0.0,
    coat: float = 0.0,
    coat_roughness: float = 0.08,
    gradient_enabled: bool = False,
    bottom_color: tuple[float, float, float, float] | None = None,
    top_color: tuple[float, float, float, float] | None = None,
    spot_color: tuple[float, float, float, float] | None = None,
    spot_strength: float = 0.0,
    spot_x: float = 0.50,
    spot_y: float = 0.72,
    spot_radius: float = 0.32,
    vignette_strength: float = 0.0,
) -> dict[str, Any]:
    min_x, max_x, min_y, max_y, min_z, max_z = _bounds(points)
    extent = max(max_x - min_x, max_y - min_y, max_z - min_z, 1.0)
    base = {
        "enabled": bool(enabled),
        "extent": float(extent),
        "color": tuple(float(v) for v in color),
        "roughness": max(0.0, min(1.0, float(roughness))),
        "specular": max(0.0, min(1.0, float(specular))),
        "metallic": max(0.0, min(1.0, float(metallic))),
        "coat": max(0.0, min(1.0, float(coat))),
        "coat_roughness": max(0.0, min(1.0, float(coat_roughness))),
        "gradient_enabled": bool(gradient_enabled),
        "bottom_color": tuple(float(v) for v in (bottom_color or color)),
        "top_color": tuple(float(v) for v in (top_color or color)),
        "spot_color": tuple(float(v) for v in (spot_color or (1.0, 1.0, 1.0, 1.0))),
        "spot_strength": max(0.0, float(spot_strength)),
        "spot_x": max(0.0, min(1.0, float(spot_x))),
        "spot_y": max(0.0, min(1.0, float(spot_y))),
        "spot_radius": max(0.01, float(spot_radius)),
        "vignette_strength": max(0.0, float(vignette_strength)),
    }
    if not bool(enabled):
        return {**base, "vertices": [], "faces": [], "reason": "disabled"}
    if bpy is None or Vector is None:
        return {**base, "enabled": False, "vertices": [], "faces": [], "reason": "no_blender"}

    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        return {**base, "enabled": False, "vertices": [], "faces": [], "reason": "no_camera"}

    xyz = []
    for point in points:
        try:
            xyz.append(Vector(_as_xyz(point)))
        except Exception:
            continue
    if not xyz:
        xyz = [Vector((0.0, 0.0, 0.0))]

    center = sum(xyz, Vector((0.0, 0.0, 0.0))) / float(len(xyz))
    quat = cam.matrix_world.to_quaternion()
    right = quat @ Vector((1.0, 0.0, 0.0))
    up = quat @ Vector((0.0, 1.0, 0.0))
    forward = quat @ Vector((0.0, 0.0, -1.0))

    coords = [(float((p - center).dot(right)), float((p - center).dot(up)), float((p - center).dot(forward))) for p in xyz]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    zs = [p[2] for p in coords]
    floor_y = min(ys) - float(floor_offset_scale) * extent
    x_mid = (min(xs) + max(xs)) * 0.5
    half_width = max(1.0, (max(xs) - min(xs)) * 0.5 * float(width_scale), 0.5 * extent * float(width_scale))
    radius = max(0.25, float(radius_scale) * extent)
    z_front = min(zs) - float(floor_depth_scale) * extent
    z_join = max(zs) + float(wall_offset_scale) * extent
    wall_z = z_join + radius
    wall_top_y = max(ys) + float(wall_height_scale) * extent

    cross_section: list[tuple[float, float]] = [(floor_y, z_front), (floor_y, z_join)]
    steps = max(4, int(segments))
    for step in range(1, steps + 1):
        t = (pi * 0.5) * (float(step) / float(steps))
        cross_section.append((floor_y + radius * (1.0 - cos(t)), z_join + radius * sin(t)))
    cross_section.append((wall_top_y, wall_z))

    x_left = x_mid - half_width
    width_steps = max(1, int(width_segments))
    vertices_vec = []
    vertex_colors = []
    color_bottom = base["bottom_color"]
    color_top = base["top_color"]
    color_spot = base["spot_color"]
    for y, z in cross_section:
        denom_y = max(1e-6, float(wall_top_y - floor_y))
        v = max(0.0, min(1.0, float((y - floor_y) / denom_y)))
        for x_step in range(width_steps + 1):
            u = float(x_step) / float(width_steps)
            x = x_left + u * float(half_width * 2.0)
            vertices_vec.append(center + right * float(x) + up * float(y) + forward * float(z))
            vertex_colors.append(
                _sweep_backdrop_color(
                    u=u,
                    v=v,
                    bottom=color_bottom,
                    top=color_top,
                    spot=color_spot,
                    spot_strength=base["spot_strength"],
                    spot_x=base["spot_x"],
                    spot_y=base["spot_y"],
                    spot_radius=base["spot_radius"],
                    vignette_strength=base["vignette_strength"],
                    enabled=bool(gradient_enabled),
                    fallback=color,
                )
            )

    vertices = [(float(v.x), float(v.y), float(v.z)) for v in vertices_vec]
    row_width = width_steps + 1
    faces = []
    for row in range(len(cross_section) - 1):
        for col in range(width_steps):
            idx = row * row_width + col
            faces.append((idx, idx + 1, idx + row_width + 1, idx + row_width))
    sweep_center = sum(vertices_vec, Vector((0.0, 0.0, 0.0))) / float(len(vertices_vec))
    return {
        **base,
        "vertices": vertices,
        "faces": faces,
        "vertex_colors": vertex_colors,
        "center": (float(sweep_center.x), float(sweep_center.y), float(sweep_center.z)),
        "width": float(half_width * 2.0),
        "floor_depth": float(z_join - z_front),
        "wall_height": float(max(0.0, wall_top_y - floor_y - radius)),
        "radius": float(radius),
        "segments": steps,
        "width_segments": width_steps,
    }


def _resolve_effective_mode(*, requested_mode: str, render_engine: str, transparent_bg: bool) -> str:
    can_shadow_catch = str(render_engine).strip().upper() == "CYCLES" and bool(transparent_bg)
    if requested_mode == "auto":
        return "shadow_catcher" if can_shadow_catch else "visible"
    if requested_mode == "shadow_catcher":
        return "shadow_catcher" if can_shadow_catch else "visible"
    return "visible"


def _bounds(points: Sequence) -> tuple[float, float, float, float, float, float]:
    if not points:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    xyz = []
    for point in points:
        try:
            xyz.append(_as_xyz(point))
        except Exception:
            continue
    if not xyz:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    xs = [p[0] for p in xyz]
    ys = [p[1] for p in xyz]
    zs = [p[2] for p in xyz]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def _as_xyz(point: Any) -> tuple[float, float, float]:
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return (float(point.x), float(point.y), float(point.z))
    if isinstance(point, (list, tuple)) and len(point) >= 3:
        return (float(point[0]), float(point[1]), float(point[2]))
    raise ValueError("Point must expose x/y/z or be a 3-length sequence.")


def _assign_ground_material(
    obj: object,
    *,
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
    specular: float,
    metallic: float,
    coat: float,
    coat_roughness: float,
    registry: Any,
) -> None:
    if bpy is None:
        return
    if not hasattr(registry, "get"):
        raise RuntimeError("Ground material assignment requires MaterialRegistry.get().")

    spec = MaterialSpec(
        color=tuple(float(v) for v in color),
        roughness=float(roughness),
        specular=float(specular),
        metallic=float(metallic),
        coat=float(coat),
        coat_roughness=float(coat_roughness),
        alpha=1.0,
    )
    request = MaterialRequest.principled(
        name=name,
        material=spec,
        role="ground",
        style_name=name,
    )
    mat = registry.get(request)

    data = getattr(obj, "data", None)
    if data is None or not hasattr(data, "materials"):
        return
    if len(data.materials) == 0:
        data.materials.append(mat)
    else:
        data.materials[0] = mat


def _assign_sweep_vertex_colors(mesh: object, colors: Any) -> None:
    if bpy is None or not isinstance(colors, list):
        return
    try:
        attributes = getattr(mesh, "color_attributes", None)
        if attributes is not None:
            attr = attributes.new(name="StudioBackdropColor", type="BYTE_COLOR", domain="CORNER")
        else:
            attr = mesh.vertex_colors.new(name="StudioBackdropColor")
        for poly in mesh.polygons:
            for loop_index in poly.loop_indices:
                vertex_index = mesh.loops[loop_index].vertex_index
                if vertex_index < len(colors):
                    attr.data[loop_index].color = colors[vertex_index]
    except Exception:
        return


def _assign_sweep_material(
    obj: object,
    *,
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
    specular: float,
    metallic: float,
    coat: float,
    coat_roughness: float,
    use_vertex_colors: bool,
    registry: Any,
) -> None:
    _assign_ground_material(
        obj,
        name=name,
        color=color,
        roughness=roughness,
        specular=specular,
        metallic=metallic,
        coat=coat,
        coat_roughness=coat_roughness,
        registry=registry,
    )
    if bpy is None or not bool(use_vertex_colors):
        return
    data = getattr(obj, "data", None)
    if data is None or not getattr(data, "materials", None):
        return
    mat = data.materials[0]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "StudioBackdropColor"
    output_name = "Color" if "Color" in attr.outputs else next(iter(attr.outputs.keys()), "")
    if output_name and "Base Color" in bsdf.inputs:
        links.new(attr.outputs[output_name], bsdf.inputs["Base Color"])


def _sweep_backdrop_color(
    *,
    u: float,
    v: float,
    bottom: tuple[float, float, float, float],
    top: tuple[float, float, float, float],
    spot: tuple[float, float, float, float],
    spot_strength: float,
    spot_x: float,
    spot_y: float,
    spot_radius: float,
    vignette_strength: float,
    enabled: bool,
    fallback: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if not bool(enabled):
        return tuple(float(x) for x in fallback)
    base = tuple(_lerp(float(bottom[i]), float(top[i]), v) for i in range(4))
    dx = float(u) - float(spot_x)
    dy = float(v) - float(spot_y)
    radius = max(0.01, float(spot_radius))
    spot_weight = max(0.0, float(spot_strength)) * exp(-((dx * dx + dy * dy) / (2.0 * radius * radius)))
    edge = max(abs(float(u) - 0.5) * 2.0, abs(float(v) - 0.52) * 1.35)
    vignette = max(0.0, min(1.0, 1.0 - max(0.0, float(vignette_strength)) * edge * edge))
    rgb = []
    for i in range(3):
        mixed = _lerp(base[i], float(spot[i]), min(1.0, spot_weight))
        rgb.append(max(0.0, min(1.0, mixed * vignette)))
    return (rgb[0], rgb[1], rgb[2], 1.0)


def _lerp(a: float, b: float, t: float) -> float:
    x = max(0.0, min(1.0, float(t)))
    return (1.0 - x) * float(a) + x * float(b)


def _camera_cover_size(plane_location: tuple[float, float, float]) -> float | None:
    if bpy is None or Vector is None:
        return None
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None or getattr(cam, "data", None) is None:
        return None

    data = cam.data
    forward = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    tilt_scale = 1.0 / max(0.25, abs(float(forward.z)))
    try:
        aspect = float(scene.render.resolution_x) / max(1.0, float(scene.render.resolution_y))
    except Exception:
        aspect = 1.0
    aspect = max(1e-6, float(aspect))

    if str(getattr(data, "type", "")).upper() == "ORTHO":
        ortho = float(getattr(data, "ortho_scale", 0.0))
        return max(1.0, ortho * max(1.0, aspect) * tilt_scale * 40.0)

    cam_loc = cam.matrix_world.translation
    plane_point = Vector((float(plane_location[0]), float(plane_location[1]), float(plane_location[2])))
    distance = abs((plane_point - cam_loc).dot(forward))
    distance = max(0.1, float(distance))

    angle_x = float(getattr(data, "angle_x", 0.0))
    angle_y = float(getattr(data, "angle_y", 0.0))
    if angle_x <= 1e-6 and angle_y <= 1e-6:
        return None
    width = 2.0 * distance * tan(max(1e-6, angle_x) * 0.5) if angle_x > 1e-6 else 0.0
    height = 2.0 * distance * tan(max(1e-6, angle_y) * 0.5) if angle_y > 1e-6 else 0.0
    return max(1.0, max(width, height) * tilt_scale * 40.0)


def _camera_plane_intersection_xy(plane_z: float) -> tuple[float, float] | None:
    if bpy is None or Vector is None:
        return None
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        return None
    forward = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    if abs(float(forward.z)) < 1e-6:
        return None
    cam_loc = cam.matrix_world.translation
    t = (float(plane_z) - float(cam_loc.z)) / float(forward.z)
    hit = cam_loc + forward * float(t)
    return (float(hit.x), float(hit.y))
