from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import atan, degrees, radians, tan
from typing import Any

from atomstudio.config import RenderJobConfig
from atomstudio.scene.camera_builder import CameraBuilder

try:
    import bpy  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore
except Exception:  # pragma: no cover
    bpy = None
    Matrix = None
    Vector = None


def _value(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        return node.get(key, default)
    return getattr(node, key, default)


def _as_vector3(value: Any, *, default: tuple[float, float, float] | None = None) -> tuple[float, float, float] | None:
    if value is None:
        return default
    if isinstance(value, Sequence) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return default


class BlenderCameraWriter:
    def __init__(self, cfg: RenderJobConfig) -> None:
        self.cfg = cfg

    def write(self, scene: Any, points: Sequence[Any]) -> Any:
        if bpy is None or Vector is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")

        camera = _value(scene, "camera")
        if camera is None:
            return CameraBuilder.from_cfg(self.cfg).build(points)

        bpy.ops.object.camera_add()
        cam = bpy.context.active_object
        cam.name = str(_value(camera, "name", "RenderCamera"))
        cam.data.type = "ORTHO" if str(_value(camera, "projection", "orthographic")).upper().startswith("ORTHO") else "PERSP"
        cam.data.lens = float(_value(camera, "lens_mm", self.cfg.camera.lens_mm))
        cam.data.clip_start = float(_value(camera, "clip_start", self.cfg.camera.clip_start))
        cam.data.clip_end = float(_value(camera, "clip_end", self.cfg.camera.clip_end))

        explicit_position = _as_vector3(_value(camera, "position"))
        explicit_rotation = _as_vector3(_value(camera, "rotation_euler"))
        if explicit_position is not None:
            cam.location = Vector(explicit_position)
            if explicit_rotation is not None:
                cam.rotation_mode = "XYZ"
                cam.rotation_euler = tuple(radians(v) for v in explicit_rotation)
            else:
                center = Vector(_as_vector3(_value(camera, "center"), default=(0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
                cam.rotation_mode = "QUATERNION"
                cam.rotation_quaternion = (center - cam.location).to_track_quat("-Z", "Y")
            if cam.data.type == "ORTHO":
                scale_factor = float(_value(camera, "scale_factor", 1.0))
                cam.data.ortho_scale = max(1e-6, 2.0 * scale_factor)
            bpy.context.scene.camera = cam
            return cam

        center = Vector(_as_vector3(_value(camera, "center"), default=(0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
        right = Vector(_as_vector3(_value(camera, "right"), default=(1.0, 0.0, 0.0)) or (1.0, 0.0, 0.0))
        up = Vector(_as_vector3(_value(camera, "up"), default=(0.0, 1.0, 0.0)) or (0.0, 1.0, 0.0))
        forward = Vector(_as_vector3(_value(camera, "forward"), default=(0.0, 0.0, -1.0)) or (0.0, 0.0, -1.0))
        scale_factor = max(1e-6, float(_value(camera, "scale_factor", 1.0)))
        distance = max(scale_factor * 1.5, 3.0)
        cam.location = center - forward.normalized() * distance

        if cam.data.type == "ORTHO":
            cam.data.ortho_scale = max(1e-6, 2.0 * scale_factor)
        else:
            fov_deg = max(5.0, min(120.0, float(_value(camera, "fov_degrees", 50.0))))
            needed_distance = max(scale_factor / max(1e-6, tan(radians(fov_deg) * 0.5)), 3.0)
            cam.location = center - forward.normalized() * needed_distance
            cam.data.angle = atan(tan(radians(fov_deg) * 0.5) * 2.0)

        rot = Matrix(
            (
                (float(right.x), float(up.x), float(-forward.x)),
                (float(right.y), float(up.y), float(-forward.y)),
                (float(right.z), float(up.z), float(-forward.z)),
            )
        ).to_quaternion()
        cam.rotation_mode = "QUATERNION"
        cam.rotation_quaternion = rot
        bpy.context.scene.camera = cam
        return cam
