from __future__ import annotations

from math import exp, tan
from typing import Any, Sequence

from atomstudio.backend.blender.collections import ensure_collection
from atomstudio.config import RenderJobConfig

try:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    Vector = None


class SunbeamBuilder:
    def __init__(
        self,
        *,
        enabled: bool = False,
        color: tuple[float, float, float, float] = (1.0, 0.78, 0.46, 1.0),
        strength: float = 0.18,
        width: float = 0.22,
        softness: float = 0.65,
        start: tuple[float, float] = (0.92, 0.94),
        end: tuple[float, float] = (0.40, 0.44),
    ) -> None:
        self.enabled = bool(enabled)
        self.color = tuple(float(v) for v in color)
        self.strength = max(0.0, float(strength))
        self.width = max(0.01, float(width))
        self.softness = max(0.0, min(1.0, float(softness)))
        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))

    @classmethod
    def from_cfg(cls, cfg: RenderJobConfig) -> "SunbeamBuilder":
        sunbeam = cfg.render.effects.sunbeam
        return cls(
            enabled=sunbeam.enabled,
            color=sunbeam.color,
            strength=sunbeam.strength,
            width=sunbeam.width,
            softness=sunbeam.softness,
            start=sunbeam.start,
            end=sunbeam.end,
        )

    def apply(self, points: Sequence[Any]) -> list[Any]:
        if bpy is None or Vector is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")
        if not self.enabled or self.strength <= 0.0:
            return []

        scene = bpy.context.scene
        cam = scene.camera
        if cam is None:
            return []

        frame = _camera_frame_at_subject(cam, points)
        if frame is None:
            return []
        origin, right, up, width_world, height_world = frame

        collection = ensure_collection("Effects")
        objects: list[Any] = []
        for idx, band in enumerate(self._bands()):
            obj = _make_beam_band(
                name=f"Sunbeam_{idx}",
                origin=origin,
                right=right,
                up=up,
                width_world=width_world,
                height_world=height_world,
                start=self.start,
                end=self.end,
                beam_width=self.width,
                t0=float(band["t0"]),
                t1=float(band["t1"]),
                taper=0.24,
            )
            if obj is None:
                continue
            obj.data.materials.append(_make_beam_material(f"Sunbeam_{idx}", self.color, float(band["alpha"])))
            _camera_only(obj)
            if collection is not None:
                for owner in list(obj.users_collection):
                    if owner != collection:
                        owner.objects.unlink(obj)
                if collection not in obj.users_collection:
                    collection.objects.link(obj)
            objects.append(obj)
        return objects

    def _bands(self) -> tuple[dict[str, float], ...]:
        count = 33
        sigma = 0.22 + self.softness * 0.32
        bands: list[dict[str, float]] = []
        for idx in range(count):
            t0 = -1.0 + 2.0 * idx / count
            t1 = -1.0 + 2.0 * (idx + 1) / count
            mid = (t0 + t1) * 0.5
            profile = exp(-((abs(mid) / max(0.05, sigma)) ** 2.0))
            alpha = self.strength * 0.20 * profile
            if alpha < 0.0025:
                continue
            bands.append({"t0": t0, "t1": t1, "alpha": alpha})
        return tuple(bands)


def _camera_frame_at_subject(cam: Any, points: Sequence[Any]) -> tuple[Any, Any, Any, float, float] | None:
    if bpy is None or Vector is None:
        return None
    quat = cam.matrix_world.to_quaternion()
    right = quat @ Vector((1.0, 0.0, 0.0))
    up = quat @ Vector((0.0, 1.0, 0.0))
    forward = quat @ Vector((0.0, 0.0, -1.0))
    cam_loc = cam.matrix_world.translation
    center = _points_center(points)
    subject_depth = float((center - cam_loc).dot(forward))
    depth = max(float(getattr(cam.data, "clip_start", 0.1)) * 4.0, subject_depth * 0.55)
    origin = cam_loc + forward * depth

    resolution = bpy.context.scene.render
    aspect = max(1e-6, float(resolution.resolution_x) / max(1.0, float(resolution.resolution_y)))
    if str(getattr(cam.data, "type", "")).upper() == "ORTHO":
        height = max(1e-6, float(getattr(cam.data, "ortho_scale", 1.0)))
        width = height * aspect
    else:
        angle_y = float(getattr(cam.data, "angle_y", getattr(cam.data, "angle", 0.785398)))
        height = max(1e-6, 2.0 * depth * tan(max(1e-6, angle_y) * 0.5))
        width = height * aspect
    return origin, right.normalized(), up.normalized(), width, height


def _points_center(points: Sequence[Any]) -> Any:
    if Vector is None:
        raise RuntimeError("mathutils.Vector is not available.")
    coords: list[Any] = []
    for point in points:
        try:
            coords.append(Vector(_as_xyz(point)))
        except Exception:
            continue
    if not coords:
        return Vector((0.0, 0.0, 0.0))
    center = Vector((0.0, 0.0, 0.0))
    for coord in coords:
        center += coord
    return center / float(len(coords))


def _as_xyz(point: Any) -> tuple[float, float, float]:
    if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
        return (float(point.x), float(point.y), float(point.z))
    if hasattr(point, "position"):
        pos = getattr(point, "position")
        return (float(pos[0]), float(pos[1]), float(pos[2]))
    if isinstance(point, (list, tuple)) and len(point) >= 3:
        return (float(point[0]), float(point[1]), float(point[2]))
    raise ValueError("Point must expose x/y/z, position, or be a 3-length sequence.")


def _make_beam_band(
    *,
    name: str,
    origin: Any,
    right: Any,
    up: Any,
    width_world: float,
    height_world: float,
    start: tuple[float, float],
    end: tuple[float, float],
    beam_width: float,
    t0: float,
    t1: float,
    taper: float,
) -> Any | None:
    if bpy is None:
        return None

    start_vec = _screen_point(origin, right, up, width_world, height_world, start)
    end_vec = _screen_point(origin, right, up, width_world, height_world, end)
    direction = end_vec - start_vec
    if float(direction.length) <= 1e-6:
        return None
    normal = (right * float(direction.dot(up)) - up * float(direction.dot(right))).normalized()
    base_width = min(width_world, height_world) * max(0.001, float(beam_width))
    start_width = base_width * max(0.02, float(taper))
    end_width = base_width

    verts = [
        tuple(start_vec + normal * start_width * float(t0)),
        tuple(start_vec + normal * start_width * float(t1)),
        tuple(end_vec + normal * end_width * float(t1)),
        tuple(end_vec + normal * end_width * float(t0)),
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _screen_point(
    origin: Any,
    right: Any,
    up: Any,
    width_world: float,
    height_world: float,
    point: tuple[float, float],
) -> Any:
    x = (float(point[0]) - 0.5) * float(width_world)
    y = (float(point[1]) - 0.5) * float(height_world)
    return origin + right * x + up * y


def _make_beam_material(name: str, color: tuple[float, float, float, float], alpha: float) -> Any:
    if bpy is None:
        return None
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    mat.show_transparent_back = True
    if hasattr(mat, "use_screen_refraction"):
        mat.use_screen_refraction = False
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    emission = nodes.new("ShaderNodeEmission")
    mix = nodes.new("ShaderNodeMixShader")
    emission.inputs["Color"].default_value = tuple(float(v) for v in color)
    emission.inputs["Strength"].default_value = 1.0
    mix.inputs["Fac"].default_value = max(0.0, min(1.0, float(alpha)))
    mat.node_tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    mat.node_tree.links.new(emission.outputs["Emission"], mix.inputs[2])
    mat.node_tree.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    return mat


def _camera_only(obj: Any) -> None:
    visibility = getattr(obj, "cycles_visibility", None)
    if visibility is None:
        return
    for attr in ("shadow", "diffuse", "glossy", "transmission", "scatter"):
        if hasattr(visibility, attr):
            try:
                setattr(visibility, attr, False)
            except Exception:
                pass
