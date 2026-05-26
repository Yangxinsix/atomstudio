from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace
from typing import Any

import numpy as np

from atomstudio.preview.camera import camera_matrices
from atomstudio.preview.gl.batches import (
    AtomBatch,
    BondBatch,
    LineBatch,
    cell_line_payload,
    hydrogen_bond_line_payload,
    solid_bond_instance_payload,
)
from atomstudio.preview.gl.shader_styles import resolve_shader_style, shader_style_choices
from atomstudio.preview.gl.state import clear_frame, configure_default_state, context_info, resize_viewport


class OpenGLRenderer:
    """Owns the OpenGL preview draw path."""

    def __init__(self, *, background: tuple[float, float, float, float] = (0.07, 0.08, 0.10, 1.0)) -> None:
        self.background = tuple(float(value) for value in background)
        self.viewport_size = (1, 1)
        self.initialized = False
        self.draw_calls = 0
        self.last_frame_draw_calls = 0
        self.scene_upload_count = 0
        self.last_error: str | None = None
        self.scene: Any | None = None
        self.shader_style_override: str | None = None
        self.shader_style = resolve_shader_style(os.environ.get("ATOMSTUDIO_GL_SHADER_STYLE"))
        self.atom_batch = AtomBatch(shader_style=self.shader_style)
        self.bond_batch = BondBatch(payload_factory=solid_bond_instance_payload)
        self.hydrogen_bond_batch = LineBatch(payload_factory=hydrogen_bond_line_payload, line_width=1.0)
        self.cell_batch = LineBatch(payload_factory=cell_line_payload, line_width=1.0)
        self.selection_atom_batch = AtomBatch(shader_style=self.shader_style)
        self.selection_bond_batch = BondBatch(payload_factory=solid_bond_instance_payload)
        self._program: int | None = None
        self._buffer: int | None = None
        self._position_location = -1
        self._color_location = -1
        self._graphics_info: dict[str, Any] = {
            "preview_renderer": "opengl",
            "opengl_version": "unavailable",
            "renderer": "OpenGL preview",
            "vendor": "unavailable",
            "depth_bits": None,
            "samples": None,
            "shader_style": self.shader_style.name,
            "shader_style_choices": shader_style_choices(),
        }

    def initialize(self, *, enable_msaa: bool = True) -> None:
        configure_default_state(enable_msaa=enable_msaa)
        self._graphics_info.update(context_info())
        self._create_triangle_pipeline()
        self.atom_batch.initialize()
        self.bond_batch.initialize()
        self.hydrogen_bond_batch.initialize()
        self.cell_batch.initialize()
        self.selection_atom_batch.initialize()
        self.selection_bond_batch.initialize()
        self.atom_batch.update_scene(self.scene)
        self.bond_batch.update_scene(self.scene)
        self.hydrogen_bond_batch.update_scene(self.scene)
        self.cell_batch.update_scene(self.scene)
        self.selection_atom_batch.update_scene(None)
        self.selection_bond_batch.update_scene(None)
        self.initialized = True

    def refresh_context_info(self) -> None:
        self._graphics_info.update(context_info())

    def resize(self, width: int, height: int) -> None:
        self.viewport_size = (max(1, int(width)), max(1, int(height)))
        resize_viewport(*self.viewport_size)

    def update_scene(self, scene: Any | None) -> None:
        self.scene = scene
        self.scene_upload_count += 1
        self._update_shader_style_from_scene(scene)
        self.atom_batch.update_scene(scene)
        self.bond_batch.update_scene(scene)
        self.hydrogen_bond_batch.update_scene(scene)
        self.cell_batch.update_scene(scene)

    def update_selection(
        self,
        scene: Any | None,
        selection: Any | None = None,
        *,
        selected_atom_indices: set[int] | frozenset[int] | tuple[int, ...] | list[int] = (),
        selected_bond_indices: set[int] | frozenset[int] | tuple[int, ...] | list[int] = (),
    ) -> None:
        payload_scene = _selection_overlay_scene(
            scene,
            selection,
            selected_atom_indices=selected_atom_indices,
            selected_bond_indices=selected_bond_indices,
        )
        self.selection_atom_batch.update_scene(payload_scene)
        self.selection_bond_batch.update_scene(payload_scene)

    def set_shader_style(self, shader_style: str | None) -> None:
        self.shader_style_override = None if shader_style is None else str(shader_style)
        self._update_shader_style_from_scene(self.scene)

    def draw(self, camera: Any | None = None, *, projection: str = "orthographic") -> None:
        clear_frame(self.background)
        self.last_frame_draw_calls = 0
        if camera is not None and (
            self.atom_batch.instance_count > 0
            or self.bond_batch.instance_count > 0
            or self.hydrogen_bond_batch.vertex_count > 0
            or self.cell_batch.vertex_count > 0
        ):
            radius = self._scene_radius(self.scene)
            matrices = camera_matrices(camera, self.viewport_size, scene_radius=radius, projection=projection)
            bond_calls = self.bond_batch.draw(matrices.view_projection, matrices.model)
            atom_calls = self.atom_batch.draw(
                matrices.view_projection,
                matrices.model,
                view=matrices.view,
                projection=matrices.projection,
            )
            hbond_calls = self.hydrogen_bond_batch.draw(matrices.view_projection, matrices.model)
            cell_calls = self._draw_cell_batch(matrices.view_projection, matrices.model)
            selection_bond_calls = self.selection_bond_batch.draw(matrices.view_projection, matrices.model)
            selection_atom_calls = self.selection_atom_batch.draw(
                matrices.view_projection,
                matrices.model,
                view=matrices.view,
                projection=matrices.projection,
            )
            calls = bond_calls + hbond_calls + cell_calls + atom_calls + selection_bond_calls + selection_atom_calls
            self.draw_calls += calls
            self.last_frame_draw_calls += calls
            return
        if self._program is None or self._buffer is None:
            return

        from OpenGL import GL  # type: ignore

        GL.glUseProgram(self._program)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        stride = 6 * 4
        if self._position_location >= 0:
            GL.glEnableVertexAttribArray(self._position_location)
            GL.glVertexAttribPointer(
                self._position_location,
                3,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                stride,
                ctypes.c_void_p(0),
            )
        if self._color_location >= 0:
            GL.glEnableVertexAttribArray(self._color_location)
            GL.glVertexAttribPointer(
                self._color_location,
                3,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                stride,
                ctypes.c_void_p(12),
            )
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        if self._position_location >= 0:
            GL.glDisableVertexAttribArray(self._position_location)
        if self._color_location >= 0:
            GL.glDisableVertexAttribArray(self._color_location)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glUseProgram(0)
        self.draw_calls += 1
        self.last_frame_draw_calls += 1

    def release(self) -> None:
        try:
            from OpenGL import GL  # type: ignore
        except Exception:  # pragma: no cover - optional dependency cleanup
            return
        self.atom_batch.release()
        self.bond_batch.release()
        self.hydrogen_bond_batch.release()
        self.cell_batch.release()
        self.selection_atom_batch.release()
        self.selection_bond_batch.release()
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

    def graphics_info(self) -> dict[str, Any]:
        info = dict(self._graphics_info)
        info.update(
            {
                "preview_renderer": "opengl",
                "atom_instances": int(self.atom_batch.instance_count),
                "atom_draw_calls": 1 if self.atom_batch.instance_count > 0 else 0,
                "bond_instances": int(self.bond_batch.instance_count),
                "bond_draw_calls": 1 if self.bond_batch.instance_count > 0 else 0,
                "hydrogen_bond_segments": int(self.hydrogen_bond_batch.vertex_count // 2),
                "hydrogen_bond_draw_calls": 1 if self.hydrogen_bond_batch.vertex_count > 0 else 0,
                "cell_segments": int(self.cell_batch.vertex_count // 2),
                "cell_draw_calls": 1 if self.cell_batch.vertex_count > 0 else 0,
                "selection_atom_instances": int(self.selection_atom_batch.instance_count),
                "selection_bond_instances": int(self.selection_bond_batch.instance_count),
                "selection_draw_calls": int(self.selection_atom_batch.instance_count > 0)
                + int(self.selection_bond_batch.instance_count > 0),
                "scene_upload_count": int(self.scene_upload_count),
                "initialized": bool(self.initialized),
                "viewport_size": tuple(int(value) for value in self.viewport_size),
                "draw_calls": int(self.draw_calls),
                "last_frame_draw_calls": int(self.last_frame_draw_calls),
                "last_error": self.last_error,
                "shader_style": self.shader_style.name,
                "shader_style_choices": shader_style_choices(),
            }
        )
        return info

    def _update_shader_style_from_scene(self, scene: Any | None) -> None:
        style_name = None
        if scene is not None:
            style_name = getattr(scene, "style_name", None)
            metadata = getattr(scene, "metadata", None)
            if not style_name and isinstance(metadata, dict):
                style_name = metadata.get("scene_style")
        env_style = os.environ.get("ATOMSTUDIO_GL_SHADER_STYLE")
        resolved = resolve_shader_style(self.shader_style_override or env_style or style_name)
        if resolved == self.shader_style:
            return
        self.shader_style = resolved
        self.atom_batch.set_shader_style(resolved)
        self.selection_atom_batch.set_shader_style(resolved)
        self._graphics_info["shader_style"] = resolved.name

    @staticmethod
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

    def _draw_cell_batch(self, view_projection: np.ndarray, model: np.ndarray | None = None) -> int:
        if self.cell_batch.vertex_count <= 0:
            return 0
        from OpenGL import GL  # type: ignore

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_FALSE)
        try:
            return self.cell_batch.draw(view_projection, model)
        finally:
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)

    def _create_triangle_pipeline(self) -> None:
        from OpenGL import GL  # type: ignore
        from OpenGL.GL import shaders  # type: ignore

        vertex_data = np.asarray(
            [
                -0.62,
                -0.48,
                0.0,
                0.95,
                0.22,
                0.18,
                0.62,
                -0.48,
                0.0,
                0.18,
                0.48,
                0.95,
                0.0,
                0.64,
                0.0,
                0.18,
                0.78,
                0.36,
            ],
            dtype=np.float32,
        )
        self._program = _compile_triangle_program(shaders)
        self._position_location = int(GL.glGetAttribLocation(self._program, "a_position"))
        self._color_location = int(GL.glGetAttribLocation(self._program, "a_color"))
        self._buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)


def _selection_overlay_scene(
    scene: Any | None,
    selection: Any | None,
    *,
    selected_atom_indices: set[int] | frozenset[int] | tuple[int, ...] | list[int],
    selected_bond_indices: set[int] | frozenset[int] | tuple[int, ...] | list[int],
) -> Any:
    atom_ids = {int(index) for index in selected_atom_indices}
    bond_ids = {int(index) for index in selected_bond_indices}
    if selection is not None and getattr(selection, "index", None) is not None:
        if getattr(selection, "kind", None) == "atom":
            atom_ids.add(int(selection.index))
        elif getattr(selection, "kind", None) == "bond":
            bond_ids.add(int(selection.index))
    if scene is None or (not atom_ids and not bond_ids):
        return None
    return SimpleNamespace(
        atoms=_selected_atoms_buffer(scene, atom_ids),
        bonds=_selected_bonds_buffer(scene, bond_ids),
    )


def _selected_atoms_buffer(scene: Any, atom_ids: set[int]) -> Any:
    atoms = getattr(scene, "atoms", None)
    positions = getattr(atoms, "positions", None)
    if atoms is None or positions is None or not atom_ids:
        return SimpleNamespace(
            positions=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 4), dtype=np.float32),
            radii=np.zeros((0,), dtype=np.float32),
            atom_indices=np.zeros((0,), dtype=np.int32),
        )
    positions_arr = np.asarray(positions, dtype=np.float32).reshape((-1, 3))
    radii_arr = np.asarray(getattr(atoms, "radii", np.zeros((positions_arr.shape[0],))), dtype=np.float32).reshape(
        (-1,)
    )
    indices_arr = np.asarray(getattr(atoms, "atom_indices", np.arange(positions_arr.shape[0])), dtype=np.int32).reshape(
        (-1,)
    )
    count = min(positions_arr.shape[0], radii_arr.shape[0], indices_arr.shape[0])
    mask = np.asarray([int(indices_arr[idx]) in atom_ids for idx in range(count)], dtype=bool)
    return SimpleNamespace(
        positions=positions_arr[:count][mask],
        colors=np.tile(np.asarray((1.0, 0.78, 0.20, 1.0), dtype=np.float32), (int(mask.sum()), 1)),
        radii=np.maximum(radii_arr[:count][mask] * 1.12, radii_arr[:count][mask] + 0.018),
        atom_indices=indices_arr[:count][mask],
    )


def _selected_bonds_buffer(scene: Any, bond_ids: set[int]) -> Any:
    bonds = getattr(scene, "bonds", None)
    positions = getattr(bonds, "positions", None)
    if bonds is None or positions is None or not bond_ids:
        return SimpleNamespace(
            positions=np.zeros((0, 2, 3), dtype=np.float32),
            colors=np.zeros((0, 2, 4), dtype=np.float32),
            connect=np.zeros((0, 2), dtype=np.int32),
            bond_ids=np.zeros((0,), dtype=np.int32),
            radii=np.zeros((0,), dtype=np.float32),
        )
    positions_arr = np.asarray(positions, dtype=np.float32).reshape((-1, 2, 3))
    radii_arr = np.asarray(getattr(bonds, "radii", np.zeros((positions_arr.shape[0],))), dtype=np.float32).reshape(
        (-1,)
    )
    ids_arr = np.asarray(getattr(bonds, "bond_ids", np.arange(positions_arr.shape[0])), dtype=np.int32).reshape((-1,))
    bond_types = tuple(getattr(bonds, "bond_types", ()) or ())
    count = min(positions_arr.shape[0], radii_arr.shape[0], ids_arr.shape[0])
    mask = np.asarray([int(ids_arr[idx]) in bond_ids for idx in range(count)], dtype=bool)
    selected_count = int(mask.sum())
    return SimpleNamespace(
        positions=positions_arr[:count][mask],
        colors=np.tile(
            np.asarray(((1.0, 0.78, 0.20, 1.0), (1.0, 0.78, 0.20, 1.0)), dtype=np.float32),
            (selected_count, 1, 1),
        ),
        connect=np.arange(selected_count * 2, dtype=np.int32).reshape((-1, 2)),
        bond_ids=ids_arr[:count][mask],
        radii=np.maximum(radii_arr[:count][mask] * 1.45, radii_arr[:count][mask] + 0.012),
        bond_types=tuple(bond_types[idx] for idx in range(count) if mask[idx]) if bond_types else (),
    )


def _compile_triangle_program(shaders: Any) -> int:
    attempts = (
        (
            """
            #version 120
            attribute vec3 a_position;
            attribute vec3 a_color;
            varying vec3 v_color;
            void main() {
                v_color = a_color;
                gl_Position = vec4(a_position, 1.0);
            }
            """,
            """
            #version 120
            varying vec3 v_color;
            void main() {
                gl_FragColor = vec4(v_color, 1.0);
            }
            """,
        ),
        (
            """
            #version 330 core
            in vec3 a_position;
            in vec3 a_color;
            out vec3 v_color;
            void main() {
                v_color = a_color;
                gl_Position = vec4(a_position, 1.0);
            }
            """,
            """
            #version 330 core
            in vec3 v_color;
            out vec4 frag_color;
            void main() {
                frag_color = vec4(v_color, 1.0);
            }
            """,
        ),
    )
    errors: list[str] = []
    for vertex_source, fragment_source in attempts:
        try:
            return int(
                shaders.compileProgram(
                    shaders.compileShader(vertex_source, 0x8B31),
                    shaders.compileShader(fragment_source, 0x8B30),
                )
            )
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("Failed to compile OpenGL preview triangle shaders: " + " | ".join(errors))


__all__ = ["OpenGLRenderer"]
