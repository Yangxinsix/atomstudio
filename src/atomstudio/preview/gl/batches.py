from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atomstudio.preview.gl.mesh import MeshData
from atomstudio.preview.gl.shader_styles import OpenGLShaderStyle, resolve_shader_style
from atomstudio.visual_defaults import (
    CELL_LINE_WIDTH_PER_RADIUS,
    HYDROGEN_BOND_LINE_COLOR,
    HYDROGEN_BOND_LINE_WIDTH,
)


@dataclass(frozen=True, slots=True)
class AtomInstancePayload:
    positions: np.ndarray
    radii: np.ndarray
    colors: np.ndarray
    atom_indices: np.ndarray

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def interleaved(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros((0, 8), dtype=np.float32)
        return np.column_stack(
            (
                self.positions.astype(np.float32, copy=False),
                self.radii.astype(np.float32, copy=False).reshape((-1, 1)),
                self.colors.astype(np.float32, copy=False),
            )
        ).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class BondInstancePayload:
    starts: np.ndarray
    ends: np.ndarray
    colors_start: np.ndarray
    colors_end: np.ndarray
    radii: np.ndarray
    bond_ids: np.ndarray

    @property
    def count(self) -> int:
        return int(self.starts.shape[0])

    @property
    def interleaved(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros((0, 15), dtype=np.float32)
        return np.column_stack(
            (
                self.starts.astype(np.float32, copy=False),
                self.ends.astype(np.float32, copy=False),
                self.colors_start.astype(np.float32, copy=False),
                self.colors_end.astype(np.float32, copy=False),
                self.radii.astype(np.float32, copy=False).reshape((-1, 1)),
            )
        ).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class LinePayload:
    positions: np.ndarray
    colors: np.ndarray
    ids: np.ndarray
    width: float = 1.0

    @property
    def count(self) -> int:
        return int(self.positions.shape[0] // 2)

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def interleaved(self) -> np.ndarray:
        if self.vertex_count == 0:
            return np.zeros((0, 7), dtype=np.float32)
        return np.column_stack(
            (
                self.positions.astype(np.float32, copy=False),
                self.colors.astype(np.float32, copy=False),
            )
        ).astype(np.float32, copy=False)


def atom_instance_payload(scene: Any | None) -> AtomInstancePayload:
    if scene is None:
        return _empty_payload()
    atoms = getattr(scene, "atoms", None)
    if atoms is None:
        return _empty_payload()

    positions = getattr(atoms, "positions", None)
    colors = getattr(atoms, "colors", None)
    radii = getattr(atoms, "radii", None)
    atom_indices = getattr(atoms, "atom_indices", None)
    if positions is not None and colors is not None and radii is not None:
        positions_arr = np.asarray(positions, dtype=np.float32).reshape((-1, 3))
        radii_arr = np.asarray(radii, dtype=np.float32).reshape((-1,))
        colors_arr = np.asarray(colors, dtype=np.float32).reshape((-1, 4))
        if atom_indices is None:
            atom_indices_arr = np.arange(positions_arr.shape[0], dtype=np.int32)
        else:
            atom_indices_arr = np.asarray(atom_indices, dtype=np.int32).reshape((-1,))
        count = min(positions_arr.shape[0], radii_arr.shape[0], colors_arr.shape[0], atom_indices_arr.shape[0])
        return AtomInstancePayload(
            positions=positions_arr[:count],
            radii=radii_arr[:count],
            colors=colors_arr[:count],
            atom_indices=atom_indices_arr[:count],
        )

    render_atoms = tuple(atoms or ())
    if not render_atoms:
        return _empty_payload()
    return AtomInstancePayload(
        positions=np.asarray([atom.position for atom in render_atoms], dtype=np.float32).reshape((-1, 3)),
        radii=np.asarray([atom.radius for atom in render_atoms], dtype=np.float32).reshape((-1,)),
        colors=np.asarray([atom.color for atom in render_atoms], dtype=np.float32).reshape((-1, 4)),
        atom_indices=np.asarray([atom.index for atom in render_atoms], dtype=np.int32).reshape((-1,)),
    )


def bond_instance_payload(scene: Any | None) -> BondInstancePayload:
    return _bond_instance_payload(scene, include_hydrogen=True)


def solid_bond_instance_payload(scene: Any | None) -> BondInstancePayload:
    return _bond_instance_payload(scene, include_hydrogen=False)


def _bond_instance_payload(scene: Any | None, *, include_hydrogen: bool) -> BondInstancePayload:
    if scene is None:
        return _empty_bond_payload()
    bonds = getattr(scene, "bonds", None)
    if bonds is None:
        return _empty_bond_payload()

    positions = getattr(bonds, "positions", None)
    colors = getattr(bonds, "colors", None)
    radii = getattr(bonds, "radii", None)
    if positions is None or colors is None or radii is None:
        return _empty_bond_payload()

    positions_arr = np.asarray(positions, dtype=np.float32).reshape((-1, 2, 3))
    colors_arr = np.asarray(colors, dtype=np.float32).reshape((-1, 2, 4))
    radii_arr = np.asarray(radii, dtype=np.float32).reshape((-1,))
    bond_ids = getattr(bonds, "bond_ids", None)
    if bond_ids is None:
        bond_ids_arr = np.arange(positions_arr.shape[0], dtype=np.int32)
    else:
        bond_ids_arr = np.asarray(bond_ids, dtype=np.int32).reshape((-1,))
    count = min(positions_arr.shape[0], colors_arr.shape[0], radii_arr.shape[0], bond_ids_arr.shape[0])
    if count <= 0:
        return _empty_bond_payload()
    if not include_hydrogen:
        bond_types = tuple(getattr(bonds, "bond_types", ()) or ())
        if bond_types:
            mask = np.asarray(
                [str(bond_types[idx]) != "hydrogen" if idx < len(bond_types) else True for idx in range(count)],
                dtype=bool,
            )
            positions_arr = positions_arr[:count][mask]
            colors_arr = colors_arr[:count][mask]
            radii_arr = radii_arr[:count][mask]
            bond_ids_arr = bond_ids_arr[:count][mask]
            count = int(mask.sum())
            if count <= 0:
                return _empty_bond_payload()
    return BondInstancePayload(
        starts=positions_arr[:count, 0],
        ends=positions_arr[:count, 1],
        colors_start=colors_arr[:count, 0],
        colors_end=colors_arr[:count, 1],
        radii=radii_arr[:count],
        bond_ids=bond_ids_arr[:count],
    )


def hydrogen_bond_line_payload(scene: Any | None) -> LinePayload:
    if scene is None:
        return _empty_line_payload()
    bonds = getattr(scene, "bonds", None)
    if bonds is None:
        return _empty_line_payload()

    positions = getattr(bonds, "positions", None)
    if positions is None:
        return _empty_line_payload()

    positions_arr = np.asarray(positions, dtype=np.float32).reshape((-1, 2, 3))
    bond_types = tuple(getattr(bonds, "bond_types", ()) or ())
    if not bond_types:
        return _empty_line_payload()
    bond_ids = getattr(bonds, "bond_ids", None)
    if bond_ids is None:
        bond_ids_arr = np.arange(positions_arr.shape[0], dtype=np.int32)
    else:
        bond_ids_arr = np.asarray(bond_ids, dtype=np.int32).reshape((-1,))
    count = min(positions_arr.shape[0], len(bond_types), bond_ids_arr.shape[0])
    mask = np.asarray([str(bond_types[idx]) == "hydrogen" for idx in range(count)], dtype=bool)
    if not bool(mask.any()):
        return _empty_line_payload()
    selected = positions_arr[:count][mask].reshape((-1, 3))
    selected_ids = bond_ids_arr[:count][mask]
    return LinePayload(
        positions=selected,
        colors=np.tile(np.asarray(HYDROGEN_BOND_LINE_COLOR, dtype=np.float32), (selected.shape[0], 1)),
        ids=np.repeat(selected_ids, 2).astype(np.int32, copy=False),
        width=HYDROGEN_BOND_LINE_WIDTH,
    )


def cell_line_payload(scene: Any | None) -> LinePayload:
    if scene is None:
        return _empty_line_payload()
    cell = getattr(scene, "cell", None)
    if cell is None:
        return _empty_line_payload()

    positions = getattr(cell, "positions", None)
    colors = getattr(cell, "colors", None)
    if positions is None or colors is None:
        return _empty_line_payload()

    positions_arr = np.asarray(positions, dtype=np.float32).reshape((-1, 2, 3))
    colors_arr = np.asarray(colors, dtype=np.float32).reshape((-1, 4))
    edge_indices = getattr(cell, "edge_indices", None)
    if edge_indices is None:
        ids_arr = np.arange(positions_arr.shape[0], dtype=np.int32)
    else:
        ids_arr = np.asarray(edge_indices, dtype=np.int32).reshape((-1, 2))[:, 0]
    count = min(positions_arr.shape[0], colors_arr.shape[0], ids_arr.shape[0])
    if count <= 0:
        return _empty_line_payload()

    radii = getattr(cell, "radii", None)
    width = 1.0
    if radii is not None:
        radii_arr = np.asarray(radii, dtype=np.float32).reshape((-1,))
        if radii_arr.size:
            active_radii = radii_arr[: min(count, radii_arr.shape[0])]
            width = max(1.0, float(np.max(active_radii)) * CELL_LINE_WIDTH_PER_RADIUS)
    return LinePayload(
        positions=positions_arr[:count].reshape((-1, 3)),
        colors=np.repeat(colors_arr[:count], 2, axis=0),
        ids=np.repeat(ids_arr[:count], 2).astype(np.int32, copy=False),
        width=width,
    )


class AtomBatch:
    def __init__(self, *, mesh: MeshData | None = None, shader_style: str | OpenGLShaderStyle | None = None) -> None:
        self.mesh = mesh
        self.shader_style = resolve_shader_style(shader_style)
        self.instance_count = 0
        self.atom_indices = np.zeros((0,), dtype=np.int32)
        self.initialized = False
        self._program: int | None = None
        self._quad_buffer: int | None = None
        self._instance_buffer: int | None = None
        self._locations: dict[str, int] = {}
        self._uniform_locations: dict[str, int] = {}

    def initialize(self) -> None:
        from OpenGL import GL  # type: ignore
        from OpenGL.GL import shaders  # type: ignore

        self._program = int(
            shaders.compileProgram(
                shaders.compileShader(_shader_source("atom.vert"), GL.GL_VERTEX_SHADER),
                shaders.compileShader(_shader_source("atom.frag"), GL.GL_FRAGMENT_SHADER),
            )
        )
        self._locations = {
            name: int(GL.glGetAttribLocation(self._program, name))
            for name in (
                "a_corner",
                "a_instance_position",
                "a_instance_radius",
                "a_instance_color",
            )
        }
        self._uniform_locations = {
            "u_view_projection": int(GL.glGetUniformLocation(self._program, "u_view_projection")),
            "u_view": int(GL.glGetUniformLocation(self._program, "u_view")),
            "u_projection": int(GL.glGetUniformLocation(self._program, "u_projection")),
            "u_model": int(GL.glGetUniformLocation(self._program, "u_model")),
            "u_light_direction": int(GL.glGetUniformLocation(self._program, "u_light_direction")),
            "u_ambient_strength": int(GL.glGetUniformLocation(self._program, "u_ambient_strength")),
            "u_diffuse_strength": int(GL.glGetUniformLocation(self._program, "u_diffuse_strength")),
            "u_wrap_strength": int(GL.glGetUniformLocation(self._program, "u_wrap_strength")),
            "u_specular_strength": int(GL.glGetUniformLocation(self._program, "u_specular_strength")),
            "u_shininess": int(GL.glGetUniformLocation(self._program, "u_shininess")),
            "u_exposure": int(GL.glGetUniformLocation(self._program, "u_exposure")),
            "u_style_mode": int(GL.glGetUniformLocation(self._program, "u_style_mode")),
        }
        quad_vertices = _impostor_quad_vertices()
        self._quad_buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._quad_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL.GL_STATIC_DRAW)

        self._instance_buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._instance_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 0, None, GL.GL_DYNAMIC_DRAW)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.initialized = True

    def update_scene(self, scene: Any | None) -> None:
        payload = atom_instance_payload(scene)
        self.instance_count = payload.count
        self.atom_indices = payload.atom_indices.copy()
        if not self.initialized or self._instance_buffer is None:
            return
        from OpenGL import GL  # type: ignore

        data = payload.interleaved
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._instance_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data if data.size else None, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def set_shader_style(self, shader_style: str | OpenGLShaderStyle | None) -> None:
        self.shader_style = resolve_shader_style(shader_style)

    def draw(
        self,
        view_projection: np.ndarray,
        model: np.ndarray | None = None,
        *,
        view: np.ndarray | None = None,
        projection: np.ndarray | None = None,
    ) -> int:
        if not self.initialized or self._program is None or self._instance_buffer is None or self.instance_count <= 0:
            return 0
        from OpenGL import GL  # type: ignore

        GL.glUseProgram(self._program)
        matrix_location = self._uniform_locations.get("u_view_projection", -1)
        if matrix_location >= 0:
            GL.glUniformMatrix4fv(matrix_location, 1, GL.GL_TRUE, np.asarray(view_projection, dtype=np.float32))
        model_location = self._uniform_locations.get("u_model", -1)
        if model_location >= 0:
            model_matrix = np.eye(4, dtype=np.float32) if model is None else np.asarray(model, dtype=np.float32)
            GL.glUniformMatrix4fv(model_location, 1, GL.GL_TRUE, model_matrix)
        view_location = self._uniform_locations.get("u_view", -1)
        if view_location >= 0:
            view_matrix = np.eye(4, dtype=np.float32) if view is None else np.asarray(view, dtype=np.float32)
            GL.glUniformMatrix4fv(view_location, 1, GL.GL_TRUE, view_matrix)
        projection_location = self._uniform_locations.get("u_projection", -1)
        if projection_location >= 0:
            projection_matrix = np.eye(4, dtype=np.float32) if projection is None else np.asarray(projection, dtype=np.float32)
            GL.glUniformMatrix4fv(projection_location, 1, GL.GL_TRUE, projection_matrix)
        light_location = self._uniform_locations.get("u_light_direction", -1)
        if light_location >= 0:
            GL.glUniform3f(light_location, 0.28, 0.42, 0.86)
        self._apply_shader_style_uniforms(GL)

        self._bind_attribute("a_corner", self._quad_buffer, 2, 0, 0, divisor=0)
        stride = 8 * 4
        self._bind_attribute("a_instance_position", self._instance_buffer, 3, stride, 0, divisor=1)
        self._bind_attribute("a_instance_radius", self._instance_buffer, 1, stride, 12, divisor=1)
        self._bind_attribute("a_instance_color", self._instance_buffer, 4, stride, 16, divisor=1)

        GL.glDrawArraysInstanced(
            GL.GL_TRIANGLES,
            0,
            6,
            int(self.instance_count),
        )
        for name in self._locations:
            location = self._locations.get(name, -1)
            if location >= 0:
                GL.glDisableVertexAttribArray(location)
                _vertex_attrib_divisor(GL, location, 0)
        GL.glUseProgram(0)
        return 1

    def release(self) -> None:
        try:
            from OpenGL import GL  # type: ignore
        except Exception:  # pragma: no cover - optional dependency cleanup
            return
        for buffer_id in (self._quad_buffer, self._instance_buffer):
            if buffer_id is None:
                continue
            try:
                GL.glDeleteBuffers(1, [buffer_id])
            except Exception:
                pass
        if self._program is not None:
            try:
                GL.glDeleteProgram(self._program)
            except Exception:
                pass
        self._quad_buffer = None
        self._instance_buffer = None
        self._program = None
        self._uniform_locations.clear()
        self.initialized = False

    def _bind_attribute(
        self,
        name: str,
        buffer_id: int | None,
        size: int,
        stride: int,
        offset: int,
        *,
        divisor: int,
    ) -> None:
        if buffer_id is None:
            return
        location = self._locations.get(name, -1)
        if location < 0:
            return
        from OpenGL import GL  # type: ignore

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
        GL.glEnableVertexAttribArray(location)
        GL.glVertexAttribPointer(location, int(size), GL.GL_FLOAT, GL.GL_FALSE, int(stride), ctypes.c_void_p(int(offset)))
        _vertex_attrib_divisor(GL, location, int(divisor))

    def _apply_shader_style_uniforms(self, GL: Any) -> None:
        style = self.shader_style
        values = {
            "u_ambient_strength": style.ambient,
            "u_diffuse_strength": style.diffuse,
            "u_wrap_strength": style.wrap,
            "u_specular_strength": style.specular,
            "u_shininess": style.shininess,
            "u_exposure": style.exposure,
        }
        for name, value in values.items():
            location = self._uniform_locations.get(name, -1)
            if location >= 0:
                GL.glUniform1f(location, float(value))
        mode_location = self._uniform_locations.get("u_style_mode", -1)
        if mode_location >= 0:
            GL.glUniform1i(mode_location, int(style.mode))


class BondBatch:
    def __init__(self, *, tube_segments: int = 12, payload_factory: Any = bond_instance_payload) -> None:
        self.tube_segments = max(6, int(tube_segments))
        self.payload_factory = payload_factory
        self.instance_count = 0
        self.bond_ids = np.zeros((0,), dtype=np.int32)
        self.initialized = False
        self._program: int | None = None
        self._vertex_buffer: int | None = None
        self._index_buffer: int | None = None
        self._instance_buffer: int | None = None
        self._index_count = 0
        self._locations: dict[str, int] = {}
        self._uniform_locations: dict[str, int] = {}

    def initialize(self) -> None:
        from OpenGL import GL  # type: ignore
        from OpenGL.GL import shaders  # type: ignore

        self._program = int(
            shaders.compileProgram(
                shaders.compileShader(_shader_source("bond.vert"), GL.GL_VERTEX_SHADER),
                shaders.compileShader(_shader_source("bond.frag"), GL.GL_FRAGMENT_SHADER),
            )
        )
        self._locations = {
            name: int(GL.glGetAttribLocation(self._program, name))
            for name in (
                "a_position",
                "a_instance_start",
                "a_instance_end",
                "a_instance_color_start",
                "a_instance_color_end",
                "a_instance_radius",
            )
        }
        self._uniform_locations = {
            "u_view_projection": int(GL.glGetUniformLocation(self._program, "u_view_projection")),
            "u_model": int(GL.glGetUniformLocation(self._program, "u_model")),
            "u_light_direction": int(GL.glGetUniformLocation(self._program, "u_light_direction")),
        }
        vertices, indices = _open_tube_mesh(self.tube_segments)
        self._index_count = int(indices.shape[0])

        self._vertex_buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vertex_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

        self._index_buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._index_buffer)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)

        self._instance_buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._instance_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 0, None, GL.GL_DYNAMIC_DRAW)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        self.initialized = True

    def update_scene(self, scene: Any | None) -> None:
        payload = self.payload_factory(scene)
        self.instance_count = payload.count
        self.bond_ids = payload.bond_ids.copy()
        if not self.initialized or self._instance_buffer is None:
            return
        from OpenGL import GL  # type: ignore

        data = payload.interleaved
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._instance_buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data if data.size else None, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def draw(self, view_projection: np.ndarray, model: np.ndarray | None = None) -> int:
        if not self.initialized or self._program is None or self._instance_buffer is None or self.instance_count <= 0:
            return 0
        from OpenGL import GL  # type: ignore

        GL.glUseProgram(self._program)
        matrix_location = self._uniform_locations.get("u_view_projection", -1)
        if matrix_location >= 0:
            GL.glUniformMatrix4fv(matrix_location, 1, GL.GL_TRUE, np.asarray(view_projection, dtype=np.float32))
        model_location = self._uniform_locations.get("u_model", -1)
        if model_location >= 0:
            model_matrix = np.eye(4, dtype=np.float32) if model is None else np.asarray(model, dtype=np.float32)
            GL.glUniformMatrix4fv(model_location, 1, GL.GL_TRUE, model_matrix)
        light_location = self._uniform_locations.get("u_light_direction", -1)
        if light_location >= 0:
            GL.glUniform3f(light_location, 0.28, 0.42, 0.86)

        self._bind_attribute("a_position", self._vertex_buffer, 3, 0, 0, divisor=0)
        stride = 15 * 4
        self._bind_attribute("a_instance_start", self._instance_buffer, 3, stride, 0, divisor=1)
        self._bind_attribute("a_instance_end", self._instance_buffer, 3, stride, 12, divisor=1)
        self._bind_attribute("a_instance_color_start", self._instance_buffer, 4, stride, 24, divisor=1)
        self._bind_attribute("a_instance_color_end", self._instance_buffer, 4, stride, 40, divisor=1)
        self._bind_attribute("a_instance_radius", self._instance_buffer, 1, stride, 56, divisor=1)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._index_buffer)
        GL.glDrawElementsInstanced(
            GL.GL_TRIANGLES,
            int(self._index_count),
            GL.GL_UNSIGNED_INT,
            ctypes.c_void_p(0),
            int(self.instance_count),
        )
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
        for name in self._locations:
            location = self._locations.get(name, -1)
            if location >= 0:
                GL.glDisableVertexAttribArray(location)
                _vertex_attrib_divisor(GL, location, 0)
        GL.glUseProgram(0)
        return 1

    def release(self) -> None:
        try:
            from OpenGL import GL  # type: ignore
        except Exception:  # pragma: no cover - optional dependency cleanup
            return
        for buffer_id in (self._vertex_buffer, self._index_buffer, self._instance_buffer):
            if buffer_id is None:
                continue
            try:
                GL.glDeleteBuffers(1, [buffer_id])
            except Exception:
                pass
        if self._program is not None:
            try:
                GL.glDeleteProgram(self._program)
            except Exception:
                pass
        self._vertex_buffer = None
        self._index_buffer = None
        self._instance_buffer = None
        self._program = None
        self._uniform_locations.clear()
        self.initialized = False

    def _bind_attribute(
        self,
        name: str,
        buffer_id: int | None,
        size: int,
        stride: int,
        offset: int,
        *,
        divisor: int,
    ) -> None:
        if buffer_id is None:
            return
        location = self._locations.get(name, -1)
        if location < 0:
            return
        from OpenGL import GL  # type: ignore

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
        GL.glEnableVertexAttribArray(location)
        GL.glVertexAttribPointer(location, int(size), GL.GL_FLOAT, GL.GL_FALSE, int(stride), ctypes.c_void_p(int(offset)))
        _vertex_attrib_divisor(GL, location, int(divisor))


class LineBatch:
    def __init__(self, *, payload_factory: Any, line_width: float = 1.0) -> None:
        self.payload_factory = payload_factory
        self.line_width = float(line_width)
        self._base_line_width = float(line_width)
        self.vertex_count = 0
        self.ids = np.zeros((0,), dtype=np.int32)
        self.initialized = False
        self._program: int | None = None
        self._buffer: int | None = None
        self._locations: dict[str, int] = {}
        self._uniform_locations: dict[str, int] = {}

    def initialize(self) -> None:
        from OpenGL import GL  # type: ignore
        from OpenGL.GL import shaders  # type: ignore

        self._program = int(
            shaders.compileProgram(
                shaders.compileShader(_shader_source("line.vert"), GL.GL_VERTEX_SHADER),
                shaders.compileShader(_shader_source("line.frag"), GL.GL_FRAGMENT_SHADER),
            )
        )
        self._locations = {
            name: int(GL.glGetAttribLocation(self._program, name))
            for name in ("a_position", "a_color")
        }
        self._uniform_locations = {
            "u_view_projection": int(GL.glGetUniformLocation(self._program, "u_view_projection")),
            "u_model": int(GL.glGetUniformLocation(self._program, "u_model")),
        }
        self._buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 0, None, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.initialized = True

    def update_scene(self, scene: Any | None) -> None:
        payload = self.payload_factory(scene)
        self.vertex_count = payload.vertex_count
        self.ids = payload.ids.copy()
        self.line_width = max(float(self._base_line_width), float(getattr(payload, "width", self._base_line_width)))
        if not self.initialized or self._buffer is None:
            return
        from OpenGL import GL  # type: ignore

        data = payload.interleaved
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data if data.size else None, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def draw(self, view_projection: np.ndarray, model: np.ndarray | None = None) -> int:
        if not self.initialized or self._program is None or self._buffer is None or self.vertex_count <= 0:
            return 0
        from OpenGL import GL  # type: ignore

        GL.glUseProgram(self._program)
        matrix_location = self._uniform_locations.get("u_view_projection", -1)
        if matrix_location >= 0:
            GL.glUniformMatrix4fv(matrix_location, 1, GL.GL_TRUE, np.asarray(view_projection, dtype=np.float32))
        model_location = self._uniform_locations.get("u_model", -1)
        if model_location >= 0:
            model_matrix = np.eye(4, dtype=np.float32) if model is None else np.asarray(model, dtype=np.float32)
            GL.glUniformMatrix4fv(model_location, 1, GL.GL_TRUE, model_matrix)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        stride = 7 * 4
        position_location = self._locations.get("a_position", -1)
        if position_location >= 0:
            GL.glEnableVertexAttribArray(position_location)
            GL.glVertexAttribPointer(position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        color_location = self._locations.get("a_color", -1)
        if color_location >= 0:
            GL.glEnableVertexAttribArray(color_location)
            GL.glVertexAttribPointer(color_location, 4, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
        try:
            GL.glLineWidth(max(1.0, float(self.line_width)))
        except Exception:
            pass
        GL.glDrawArrays(GL.GL_LINES, 0, int(self.vertex_count))
        try:
            GL.glLineWidth(1.0)
        except Exception:
            pass
        for location in (position_location, color_location):
            if location >= 0:
                GL.glDisableVertexAttribArray(location)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glUseProgram(0)
        return 1

    def release(self) -> None:
        try:
            from OpenGL import GL  # type: ignore
        except Exception:  # pragma: no cover - optional dependency cleanup
            return
        if self._buffer is not None:
            try:
                GL.glDeleteBuffers(1, [self._buffer])
            except Exception:
                pass
        if self._program is not None:
            try:
                GL.glDeleteProgram(self._program)
            except Exception:
                pass
        self._buffer = None
        self._program = None
        self.initialized = False


def _empty_payload() -> AtomInstancePayload:
    return AtomInstancePayload(
        positions=np.zeros((0, 3), dtype=np.float32),
        radii=np.zeros((0,), dtype=np.float32),
        colors=np.zeros((0, 4), dtype=np.float32),
        atom_indices=np.zeros((0,), dtype=np.int32),
    )


def _empty_bond_payload() -> BondInstancePayload:
    return BondInstancePayload(
        starts=np.zeros((0, 3), dtype=np.float32),
        ends=np.zeros((0, 3), dtype=np.float32),
        colors_start=np.zeros((0, 4), dtype=np.float32),
        colors_end=np.zeros((0, 4), dtype=np.float32),
        radii=np.zeros((0,), dtype=np.float32),
        bond_ids=np.zeros((0,), dtype=np.int32),
    )


def _empty_line_payload() -> LinePayload:
    return LinePayload(
        positions=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 4), dtype=np.float32),
        ids=np.zeros((0,), dtype=np.int32),
        width=1.0,
    )


def _impostor_quad_vertices() -> np.ndarray:
    return np.asarray(
        (
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        ),
        dtype=np.float32,
    )


def _open_tube_mesh(segments: int) -> tuple[np.ndarray, np.ndarray]:
    segments = max(6, int(segments))
    vertices: list[tuple[float, float, float]] = []
    indices: list[int] = []
    for z in (0.0, 1.0):
        for segment in range(segments):
            angle = 2.0 * np.pi * float(segment) / float(segments)
            x = float(np.cos(angle))
            y = float(np.sin(angle))
            vertices.append((x, y, z))
    for segment in range(segments):
        nxt = (segment + 1) % segments
        a = segment
        b = nxt
        c = segments + segment
        d = segments + nxt
        indices.extend((a, c, b, b, c, d))
    return (
        np.asarray(vertices, dtype=np.float32).reshape((-1, 3)),
        np.asarray(indices, dtype=np.uint32),
    )


def _shader_source(name: str) -> str:
    path = Path(__file__).with_name("shaders") / name
    return path.read_text(encoding="utf-8")


def _vertex_attrib_divisor(GL: Any, location: int, divisor: int) -> None:
    function = getattr(GL, "glVertexAttribDivisor", None)
    if function is None:
        function = getattr(GL, "glVertexAttribDivisorARB", None)
    if function is None:
        raise RuntimeError("OpenGL instanced attributes require glVertexAttribDivisor")
    function(int(location), int(divisor))


__all__ = [
    "AtomBatch",
    "AtomInstancePayload",
    "BondBatch",
    "BondInstancePayload",
    "LineBatch",
    "LinePayload",
    "atom_instance_payload",
    "bond_instance_payload",
    "cell_line_payload",
    "hydrogen_bond_line_payload",
    "solid_bond_instance_payload",
]
