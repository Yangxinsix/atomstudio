from __future__ import annotations

from math import cos, radians, sin, sqrt
from typing import Any, Iterable

import numpy as np

from atomstudio.preview.types import PreviewSelection, PreviewRenderScene


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def rotation_basis(
    azimuth: float,
    elevation: float,
    roll: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    az = radians(float(azimuth))
    el = radians(float(elevation))
    right = np.array([cos(az), sin(az), 0.0], dtype=float)
    forward = np.array([-sin(az) * cos(el), cos(az) * cos(el), -sin(el)], dtype=float)
    up = np.cross(right, forward)
    roll_rad = radians(float(roll))
    if abs(roll_rad) > 1e-12:
        right, up = (
            right * cos(roll_rad) + up * sin(roll_rad),
            up * cos(roll_rad) - right * sin(roll_rad),
        )
    return tuple(float(v) for v in right), tuple(float(v) for v in up), tuple(float(v) for v in forward)


def project_point(
    point: tuple[float, float, float],
    camera,
    viewport_size: tuple[int, int],
    *,
    scene_radius: float = 1.0,
    projection: str = "orthographic",
) -> tuple[float, float, float]:
    from atomstudio.preview.camera import camera_matrices

    width = max(1.0, float(viewport_size[0]))
    height = max(1.0, float(viewport_size[1]))
    matrices = camera_matrices(camera, viewport_size, scene_radius=scene_radius, projection=projection)
    clip = (matrices.view_projection @ matrices.model @ _homogeneous(point)).astype(float, copy=False)
    w = float(clip[3])
    if abs(w) <= 1e-12:
        return width * 0.5, height * 0.5, float("inf")
    ndc = clip[:3] / w
    sx = (float(ndc[0]) + 1.0) * 0.5 * width
    sy = (1.0 - float(ndc[1])) * 0.5 * height
    return sx, sy, float(ndc[2])


def point_distance_2d(left: tuple[float, float], right: tuple[float, float]) -> float:
    return sqrt((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2)


def segment_distance_2d(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    start_vec = np.array(start, dtype=float)
    end_vec = np.array(end, dtype=float)
    point_vec = np.array(point, dtype=float)
    segment = end_vec - start_vec
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return point_distance_2d(point, start), 0.0
    weight = float(np.dot(point_vec - start_vec, segment) / denom)
    weight = max(0.0, min(1.0, weight))
    closest = start_vec + segment * weight
    return float(np.linalg.norm(point_vec - closest)), weight


def project_atom_positions(scene: PreviewRenderScene | None, camera, viewport_size: tuple[int, int]) -> dict[int, tuple[float, float, float]]:
    if scene is None:
        return {}
    radius = _scene_radius(scene)
    out: dict[int, tuple[float, float, float]] = {}
    for atom in scene.atoms:
        out.setdefault(int(atom.index), project_point(atom.position, camera, viewport_size, scene_radius=radius))
    return out


def pick_atom_selection(
    scene: PreviewRenderScene | None,
    camera,
    viewport_size: tuple[int, int],
    pos: tuple[float, float],
    *,
    picking_radius_px: float = 8.0,
    projection: str = "orthographic",
) -> PreviewSelection | None:
    hit = _best_atom_hit(
        scene,
        camera,
        viewport_size,
        pos,
        picking_radius_px=picking_radius_px,
        projection=projection,
    )
    return None if hit is None else PreviewSelection(kind="atom", index=hit[0])


def pick_bond_selection(
    scene: PreviewRenderScene | None,
    camera,
    viewport_size: tuple[int, int],
    pos: tuple[float, float],
    *,
    picking_radius_px: float = 8.0,
    bond_scale: float = 18.0,
    projection: str = "orthographic",
) -> PreviewSelection | None:
    hit = _best_bond_hit(
        scene,
        camera,
        viewport_size,
        pos,
        picking_radius_px=picking_radius_px,
        bond_scale=bond_scale,
        projection=projection,
    )
    return None if hit is None else PreviewSelection(kind="bond", index=hit[0])


def pick_selection(
    scene: PreviewRenderScene | None,
    camera,
    viewport_size: tuple[int, int],
    pos: tuple[float, float],
    *,
    picking_radius_px: float = 8.0,
    bond_scale: float = 18.0,
    projection: str = "orthographic",
) -> PreviewSelection | None:
    atom_hit = _best_atom_hit(
        scene,
        camera,
        viewport_size,
        pos,
        picking_radius_px=picking_radius_px,
        projection=projection,
    )
    bond_hit = _best_bond_hit(
        scene,
        camera,
        viewport_size,
        pos,
        picking_radius_px=picking_radius_px,
        bond_scale=bond_scale,
        projection=projection,
    )
    if atom_hit is None and bond_hit is None:
        return None
    if atom_hit is None:
        return PreviewSelection(kind="bond", index=bond_hit[0])
    if bond_hit is None:
        return PreviewSelection(kind="atom", index=atom_hit[0])

    atom_index, atom_ray_t, atom_screen_distance = atom_hit
    bond_index, bond_ray_t, _bond_world_distance, bond_screen_distance = bond_hit
    atom_core_px = max(5.0, min(12.0, float(picking_radius_px) * 0.55))
    if atom_screen_distance <= atom_core_px:
        return PreviewSelection(kind="atom", index=atom_index)
    if bond_screen_distance <= max(8.0, float(picking_radius_px) * 0.9):
        return PreviewSelection(kind="bond", index=bond_index)
    if atom_ray_t <= bond_ray_t:
        return PreviewSelection(kind="atom", index=atom_index)
    return PreviewSelection(kind="bond", index=bond_index)


def _best_atom_hit(
    scene: PreviewRenderScene | None,
    camera: Any,
    viewport_size: tuple[int, int],
    pos: tuple[float, float],
    *,
    picking_radius_px: float,
    projection: str,
) -> tuple[int, float, float] | None:
    atoms = tuple(_iter_atoms(scene))
    if scene is None or not atoms:
        return None
    radius = _scene_radius(scene)
    origin, direction = screen_ray(camera, viewport_size, pos, scene_radius=radius, projection=projection)
    tolerance = _world_pick_tolerance(camera, viewport_size, picking_radius_px)
    best: tuple[int, float, float] | None = None
    for index, position, atom_radius, _size_px in atoms:
        pick_radius = max(float(atom_radius), tolerance)
        hit_t = ray_sphere_intersection(origin, direction, np.asarray(position, dtype=float), pick_radius)
        if hit_t is None:
            continue
        point = project_point(
            tuple(float(v) for v in position),
            camera,
            viewport_size,
            scene_radius=radius,
            projection=projection,
        )
        screen_distance = point_distance_2d((point[0], point[1]), pos)
        if best is None or hit_t < best[1] - 1e-6 or (abs(hit_t - best[1]) <= 1e-6 and screen_distance < best[2]):
            best = (int(index), float(hit_t), float(screen_distance))
    return best


def _best_bond_hit(
    scene: PreviewRenderScene | None,
    camera: Any,
    viewport_size: tuple[int, int],
    pos: tuple[float, float],
    *,
    picking_radius_px: float,
    bond_scale: float,
    projection: str,
) -> tuple[int, float, float, float] | None:
    segments = tuple(_iter_bond_segments(scene, bond_scale=bond_scale))
    if scene is None or not segments:
        return None
    radius = _scene_radius(scene)
    origin, direction = screen_ray(camera, viewport_size, pos, scene_radius=radius, projection=projection)
    tolerance = _world_pick_tolerance(camera, viewport_size, picking_radius_px)
    best: tuple[int, float, float, float] | None = None
    for bond_index, start, end, segment_radius in segments:
        start_screen = project_point(start, camera, viewport_size, scene_radius=radius, projection=projection)
        end_screen = project_point(end, camera, viewport_size, scene_radius=radius, projection=projection)
        screen_distance, _weight = segment_distance_2d(pos, (start_screen[0], start_screen[1]), (end_screen[0], end_screen[1]))
        screen_threshold = max(8.0, float(picking_radius_px) * 0.9, float(segment_radius) * float(bond_scale) * 0.75)
        if screen_distance > screen_threshold:
            continue
        world_distance, ray_t, _segment_t = ray_segment_distance(
            origin,
            direction,
            np.asarray(start, dtype=float),
            np.asarray(end, dtype=float),
        )
        world_threshold = max(float(segment_radius), tolerance * 1.35)
        if ray_t < 0.0 or world_distance > world_threshold:
            continue
        if best is None or ray_t < best[1] - 1e-6 or (
            abs(ray_t - best[1]) <= 1e-6 and screen_distance < best[3]
        ):
            best = (int(bond_index), float(ray_t), float(world_distance), float(screen_distance))
    return best


def screen_ray(
    camera: Any,
    viewport_size: tuple[int, int],
    pos: tuple[float, float],
    *,
    scene_radius: float = 1.0,
    projection: str = "orthographic",
) -> tuple[np.ndarray, np.ndarray]:
    from atomstudio.preview.camera import camera_matrices

    width = max(1.0, float(viewport_size[0]))
    height = max(1.0, float(viewport_size[1]))
    ndc_x = float(pos[0]) / width * 2.0 - 1.0
    ndc_y = 1.0 - float(pos[1]) / height * 2.0
    matrices = camera_matrices(camera, viewport_size, scene_radius=scene_radius, projection=projection)
    inverse_mvp = np.linalg.inv((matrices.view_projection @ matrices.model).astype(float, copy=False))
    near = inverse_mvp @ np.asarray((ndc_x, ndc_y, -1.0, 1.0), dtype=float)
    far = inverse_mvp @ np.asarray((ndc_x, ndc_y, 1.0, 1.0), dtype=float)
    near = near[:3] / max(abs(float(near[3])), 1e-12)
    far = far[:3] / max(abs(float(far[3])), 1e-12)
    direction = normalize(far - near)
    return near.astype(float, copy=False), direction.astype(float, copy=False)


def ray_sphere_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> float | None:
    offset = np.asarray(origin, dtype=float) - np.asarray(center, dtype=float)
    unit = normalize(np.asarray(direction, dtype=float))
    b = float(np.dot(offset, unit))
    c = float(np.dot(offset, offset) - float(radius) * float(radius))
    discriminant = b * b - c
    if discriminant < 0.0:
        return None
    root = sqrt(discriminant)
    near = -b - root
    if near >= 0.0:
        return float(near)
    far = -b + root
    return float(far) if far >= 0.0 else None


def ray_segment_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[float, float, float]:
    ray_dir = normalize(np.asarray(direction, dtype=float))
    segment_dir = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    segment_len_sq = float(np.dot(segment_dir, segment_dir))
    if segment_len_sq <= 1e-16:
        to_point = np.asarray(start, dtype=float) - np.asarray(origin, dtype=float)
        ray_t = max(0.0, float(np.dot(to_point, ray_dir)))
        closest_ray = np.asarray(origin, dtype=float) + ray_dir * ray_t
        return float(np.linalg.norm(closest_ray - start)), ray_t, 0.0

    w0 = np.asarray(origin, dtype=float) - np.asarray(start, dtype=float)
    a = 1.0
    b = float(np.dot(ray_dir, segment_dir))
    c = segment_len_sq
    d = float(np.dot(ray_dir, w0))
    e = float(np.dot(segment_dir, w0))
    denom = a * c - b * b
    if abs(denom) <= 1e-12:
        ray_t = 0.0
        segment_t = max(0.0, min(1.0, e / c))
    else:
        ray_t = (b * e - c * d) / denom
        segment_t = (a * e - b * d) / denom
        if ray_t < 0.0:
            ray_t = 0.0
            segment_t = max(0.0, min(1.0, -e / c))
        else:
            segment_t = max(0.0, min(1.0, segment_t))
            ray_t = max(0.0, float(np.dot(np.asarray(start, dtype=float) + segment_dir * segment_t - np.asarray(origin, dtype=float), ray_dir)))

    closest_ray = np.asarray(origin, dtype=float) + ray_dir * ray_t
    closest_segment = np.asarray(start, dtype=float) + segment_dir * segment_t
    return float(np.linalg.norm(closest_ray - closest_segment)), float(ray_t), float(segment_t)


def atom_indices_in_rect(
    scene: PreviewRenderScene | None,
    camera: Any,
    viewport_size: tuple[int, int],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    picking_radius_px: float = 8.0,
    projection: str = "orthographic",
) -> tuple[int, ...]:
    if scene is None:
        return ()
    x0, x1 = sorted((float(start[0]), float(end[0])))
    y0, y1 = sorted((float(start[1]), float(end[1])))
    radius = _scene_radius(scene)
    selected: list[int] = []
    for index, position, _atom_radius, size_px in _iter_atoms(scene):
        x, y, _z = project_point(tuple(float(v) for v in position), camera, viewport_size, scene_radius=radius, projection=projection)
        pad = max(float(picking_radius_px), float(size_px) * 0.5)
        if x0 - pad <= x <= x1 + pad and y0 - pad <= y <= y1 + pad:
            selected.append(int(index))
    return tuple(sorted(set(selected)))


def bond_indices_in_rect(
    scene: PreviewRenderScene | None,
    camera: Any,
    viewport_size: tuple[int, int],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    picking_radius_px: float = 8.0,
    bond_scale: float = 18.0,
    projection: str = "orthographic",
) -> tuple[int, ...]:
    if scene is None:
        return ()
    x0, x1 = sorted((float(start[0]), float(end[0])))
    y0, y1 = sorted((float(start[1]), float(end[1])))
    radius = _scene_radius(scene)
    selected: set[int] = set()
    for bond_index, segment_start, segment_end, segment_radius in _iter_bond_segments(scene, bond_scale=bond_scale):
        a = project_point(tuple(float(v) for v in segment_start), camera, viewport_size, scene_radius=radius, projection=projection)
        b = project_point(tuple(float(v) for v in segment_end), camera, viewport_size, scene_radius=radius, projection=projection)
        pad = max(float(picking_radius_px), float(segment_radius) * float(bond_scale) * 0.5)
        if _screen_segment_hits_rect((a[0], a[1]), (b[0], b[1]), (x0, y0, x1, y1), pad):
            selected.add(int(bond_index))
    return tuple(sorted(selected))


def _iter_atoms(scene: Any | None) -> Iterable[tuple[int, tuple[float, float, float], float, float]]:
    if scene is None:
        return ()
    atoms = getattr(scene, "atoms", ())
    positions = getattr(atoms, "positions", None)
    if positions is not None:
        positions_arr = np.asarray(positions, dtype=float).reshape((-1, 3))
        radii_arr = np.asarray(getattr(atoms, "radii", np.zeros((positions_arr.shape[0],))), dtype=float).reshape((-1,))
        indices_arr = np.asarray(getattr(atoms, "atom_indices", np.arange(positions_arr.shape[0])), dtype=np.int32).reshape((-1,))
        count = min(positions_arr.shape[0], radii_arr.shape[0], indices_arr.shape[0])
        return tuple((int(indices_arr[idx]), tuple(float(v) for v in positions_arr[idx]), float(radii_arr[idx]), 0.0) for idx in range(count))
    return tuple(
        (
            int(getattr(atom, "index")),
            tuple(float(v) for v in getattr(atom, "position")),
            float(getattr(atom, "radius", 0.0)),
            float(getattr(atom, "size_px", 0.0)),
        )
        for atom in tuple(atoms or ())
    )


def _iter_bond_segments(scene: Any | None, *, bond_scale: float) -> Iterable[tuple[int, tuple[float, float, float], tuple[float, float, float], float]]:
    if scene is None:
        return ()
    bonds = getattr(scene, "bonds", ())
    positions = getattr(bonds, "positions", None)
    if positions is not None:
        positions_arr = np.asarray(positions, dtype=float).reshape((-1, 2, 3))
        radii_arr = np.asarray(getattr(bonds, "radii", np.zeros((positions_arr.shape[0],))), dtype=float).reshape((-1,))
        ids_arr = np.asarray(getattr(bonds, "bond_ids", np.arange(positions_arr.shape[0])), dtype=np.int32).reshape((-1,))
        count = min(positions_arr.shape[0], radii_arr.shape[0], ids_arr.shape[0])
        return tuple(
            (
                int(ids_arr[idx]),
                tuple(float(v) for v in positions_arr[idx, 0]),
                tuple(float(v) for v in positions_arr[idx, 1]),
                float(radii_arr[idx]),
            )
            for idx in range(count)
        )
    segments = []
    for bond in tuple(bonds or ()):
        for segment in tuple(getattr(bond, "segments", ()) or ()):
            segments.append(
                (
                    int(getattr(bond, "index")),
                    tuple(float(v) for v in getattr(segment, "start")),
                    tuple(float(v) for v in getattr(segment, "end")),
                    max(1e-6, float(getattr(segment, "width_px", 1.0)) / max(float(bond_scale), 1e-6)),
                )
            )
    return tuple(segments)


def _scene_radius(scene: Any | None) -> float:
    if scene is None:
        return 1.0
    bounds = getattr(scene, "bounds", None)
    if bounds is not None and getattr(bounds, "radius", None) is not None:
        return max(1.0, float(bounds.radius))
    radius = getattr(scene, "radius", None)
    if radius is not None:
        return max(1.0, float(radius))
    extent = getattr(scene, "extent", None)
    if extent is not None:
        return max(1.0, float(extent))
    return 1.0


def _world_pick_tolerance(camera: Any, viewport_size: tuple[int, int], pixels: float) -> float:
    height = max(1.0, float(viewport_size[1]))
    scale = max(1e-6, float(getattr(camera, "scale_factor", 1.0)))
    return max(0.0, float(pixels)) * scale / height


def _homogeneous(point: tuple[float, float, float]) -> np.ndarray:
    return np.asarray((float(point[0]), float(point[1]), float(point[2]), 1.0), dtype=np.float32)


def _screen_segment_hits_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: tuple[float, float, float, float],
    half_width: float,
) -> bool:
    x0, y0, x1, y1 = rect
    expanded = (x0 - half_width, y0 - half_width, x1 + half_width, y1 + half_width)
    return _screen_segment_intersects_rect(start, end, expanded)


def _screen_segment_intersects_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: tuple[float, float, float, float],
) -> bool:
    x0, y0, x1, y1 = rect
    ax, ay = start
    bx, by = end
    if max(ax, bx) < x0 or min(ax, bx) > x1 or max(ay, by) < y0 or min(ay, by) > y1:
        return False
    if x0 <= ax <= x1 and y0 <= ay <= y1:
        return True
    if x0 <= bx <= x1 and y0 <= by <= y1:
        return True
    edges = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    return any(_segments_intersect(start, end, left, right) for left, right in edges)


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r) -> bool:
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    return (
        abs(o1) <= 1e-9 and on_segment(a, c, b)
        or abs(o2) <= 1e-9 and on_segment(a, d, b)
        or abs(o3) <= 1e-9 and on_segment(c, a, d)
        or abs(o4) <= 1e-9 and on_segment(c, b, d)
    )


__all__ = [
    "normalize",
    "atom_indices_in_rect",
    "bond_indices_in_rect",
    "pick_atom_selection",
    "pick_bond_selection",
    "pick_selection",
    "point_distance_2d",
    "project_atom_positions",
    "project_point",
    "ray_segment_distance",
    "ray_sphere_intersection",
    "rotation_basis",
    "segment_distance_2d",
    "screen_ray",
]
