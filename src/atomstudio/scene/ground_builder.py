from __future__ import annotations

from math import tan
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
        self.registry = registry

    @classmethod
    def from_cfg(
        cls,
        cfg: RenderJobConfig,
        *,
        registry: Any,
    ) -> "GroundBuilder":
        ground = cfg.lighting.ground
        return cls(
            enabled=ground.enabled,
            mode=ground.mode,
            size_scale=ground.size_scale,
            z_offset_scale=ground.z_offset_scale,
            color=ground.color,
            roughness=ground.roughness,
            specular=ground.specular,
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
        )
        if not bool(spec["enabled"]):
            return None, spec

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
            color=spec["color"],
            roughness=float(spec["roughness"]),
            specular=float(spec["specular"]),
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
    color: tuple[float, float, float, float],
    roughness: float,
    specular: float,
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
        metallic=0.0,
        alpha=1.0,
    )
    request = MaterialRequest.principled(
        name="GroundPlane",
        material=spec,
        role="ground",
        style_name="ground",
    )
    mat = registry.get(request)

    data = getattr(obj, "data", None)
    if data is None or not hasattr(data, "materials"):
        return
    if len(data.materials) == 0:
        data.materials.append(mat)
    else:
        data.materials[0] = mat


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
        return max(1.0, ortho * max(1.0, aspect) * tilt_scale * 8.0)

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
    return max(1.0, max(width, height) * tilt_scale * 8.0)


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
