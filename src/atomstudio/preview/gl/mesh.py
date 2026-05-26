from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MeshData:
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray


def uv_sphere_mesh(*, segments: int = 32, rings: int = 24) -> MeshData:
    segments = max(8, int(segments))
    rings = max(4, int(rings))
    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    indices: list[int] = []

    for ring in range(rings + 1):
        theta = np.pi * float(ring) / float(rings)
        sin_theta = float(np.sin(theta))
        cos_theta = float(np.cos(theta))
        for segment in range(segments):
            phi = 2.0 * np.pi * float(segment) / float(segments)
            x = sin_theta * float(np.cos(phi))
            y = sin_theta * float(np.sin(phi))
            z = cos_theta
            vertices.append((x, y, z))
            normals.append((x, y, z))

    for ring in range(rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = ring * segments + segment
            b = ring * segments + next_segment
            c = (ring + 1) * segments + segment
            d = (ring + 1) * segments + next_segment
            if ring > 0:
                indices.extend((a, c, b))
            if ring < rings - 1:
                indices.extend((b, c, d))

    return MeshData(
        vertices=np.asarray(vertices, dtype=np.float32).reshape((-1, 3)),
        normals=np.asarray(normals, dtype=np.float32).reshape((-1, 3)),
        indices=np.asarray(indices, dtype=np.uint32),
    )


__all__ = ["MeshData", "uv_sphere_mesh"]
