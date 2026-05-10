from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from atomstudio.preview.gl.state import clear_frame, configure_default_state, context_info, resize_viewport


class OpenGLRenderer:
    """Owns the first OpenGL preview draw path.

    Phase 1 deliberately renders only a diagnostic triangle. Molecule geometry
    upload and draw passes are added in later phases.
    """

    def __init__(self, *, background: tuple[float, float, float, float] = (0.07, 0.08, 0.10, 1.0)) -> None:
        self.background = tuple(float(value) for value in background)
        self.viewport_size = (1, 1)
        self.initialized = False
        self.draw_calls = 0
        self.last_error: str | None = None
        self.scene: Any | None = None
        self._program: int | None = None
        self._buffer: int | None = None
        self._position_location = -1
        self._color_location = -1
        self._graphics_info: dict[str, Any] = {
            "preview_renderer": "opengl-shell",
            "opengl_version": "unavailable",
            "renderer": "OpenGL preview",
            "vendor": "unavailable",
            "depth_bits": None,
            "samples": None,
        }

    def initialize(self) -> None:
        configure_default_state(enable_msaa=True)
        self._graphics_info.update(context_info())
        self._create_triangle_pipeline()
        self.initialized = True

    def resize(self, width: int, height: int) -> None:
        self.viewport_size = (max(1, int(width)), max(1, int(height)))
        resize_viewport(*self.viewport_size)

    def update_scene(self, scene: Any | None) -> None:
        self.scene = scene

    def draw(self) -> None:
        clear_frame(self.background)
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

    def graphics_info(self) -> dict[str, Any]:
        info = dict(self._graphics_info)
        info.update(
            {
                "preview_renderer": "opengl-shell",
                "initialized": bool(self.initialized),
                "viewport_size": tuple(int(value) for value in self.viewport_size),
                "draw_calls": int(self.draw_calls),
                "last_error": self.last_error,
            }
        )
        return info

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
        self._program = self._compile_triangle_program(shaders)
        self._position_location = int(GL.glGetAttribLocation(self._program, "a_position"))
        self._color_location = int(GL.glGetAttribLocation(self._program, "a_color"))
        self._buffer = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _compile_triangle_program(self, shaders: Any) -> int:
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
