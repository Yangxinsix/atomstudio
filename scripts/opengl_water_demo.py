#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import math
import os
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np


SOFTWARE_RENDERER_MARKERS = (
    "llvmpipe",
    "softpipe",
    "software rasterizer",
    "swrast",
    "mesa offscreen",
    "microsoft basic render",
    "gdi generic",
)


VERTEX_SHADER = """
#version 120
attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec3 a_color;

uniform mat4 u_model;
uniform mat4 u_view_projection;

varying vec3 v_normal;
varying vec3 v_color;

void main() {
    vec4 world_position = u_model * vec4(a_position, 1.0);
    v_normal = normalize((u_model * vec4(a_normal, 0.0)).xyz);
    v_color = a_color;
    gl_Position = u_view_projection * world_position;
}
"""


FRAGMENT_SHADER = """
#version 120
varying vec3 v_normal;
varying vec3 v_color;

uniform vec3 u_light_direction;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(u_light_direction);
    float diffuse = max(dot(normal, light), 0.0);
    float rim = pow(1.0 - max(normal.z, 0.0), 2.0);
    vec3 color = v_color * (0.30 + 0.66 * diffuse) + vec3(0.08) * rim;
    gl_FragColor = vec4(color, 1.0);
}
"""


@dataclass(frozen=True)
class MoleculeMesh:
    vertices: np.ndarray
    indices: np.ndarray


def water_mesh() -> MoleculeMesh:
    bond_length = 0.96
    half_angle = math.radians(104.5 * 0.5)
    oxygen = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    hydrogen_a = np.array(
        [math.sin(half_angle) * bond_length, math.cos(half_angle) * bond_length, 0.0],
        dtype=np.float32,
    )
    hydrogen_b = np.array(
        [-math.sin(half_angle) * bond_length, math.cos(half_angle) * bond_length, 0.0],
        dtype=np.float32,
    )
    center = (oxygen + hydrogen_a + hydrogen_b) / 3.0
    atoms = (
        (oxygen - center, 0.30, (0.93, 0.12, 0.10)),
        (hydrogen_a - center, 0.17, (0.94, 0.94, 0.90)),
        (hydrogen_b - center, 0.17, (0.94, 0.94, 0.90)),
    )

    positions: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    indices: list[int] = []
    for position, radius, color in atoms:
        _append_sphere(positions, normals, colors, indices, position, radius, color)

    bond_color = (0.74, 0.76, 0.78)
    for hydrogen in (hydrogen_a - center, hydrogen_b - center):
        direction = hydrogen - (oxygen - center)
        unit = direction / max(1.0e-6, float(np.linalg.norm(direction)))
        start = oxygen - center + unit * 0.20
        end = hydrogen - unit * 0.13
        _append_cylinder(positions, normals, colors, indices, start, end, 0.055, bond_color)

    vertex_data = np.column_stack(
        (
            np.asarray(positions, dtype=np.float32).reshape((-1, 3)),
            np.asarray(normals, dtype=np.float32).reshape((-1, 3)),
            np.asarray(colors, dtype=np.float32).reshape((-1, 3)),
        )
    ).astype(np.float32, copy=False)
    return MoleculeMesh(vertices=vertex_data, indices=np.asarray(indices, dtype=np.uint32))


def _append_sphere(
    positions: list[np.ndarray],
    normals: list[np.ndarray],
    colors: list[np.ndarray],
    indices: list[int],
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
    *,
    segments: int = 32,
    rings: int = 16,
) -> None:
    offset = len(positions)
    for ring in range(rings + 1):
        theta = math.pi * ring / rings
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for segment in range(segments):
            phi = 2.0 * math.pi * segment / segments
            normal = np.array(
                [sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta],
                dtype=np.float32,
            )
            positions.append(center + normal * float(radius))
            normals.append(normal)
            colors.append(np.asarray(color, dtype=np.float32))

    for ring in range(rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = offset + ring * segments + segment
            b = offset + ring * segments + next_segment
            c = offset + (ring + 1) * segments + segment
            d = offset + (ring + 1) * segments + next_segment
            if ring > 0:
                indices.extend((a, c, b))
            if ring < rings - 1:
                indices.extend((b, c, d))


def _append_cylinder(
    positions: list[np.ndarray],
    normals: list[np.ndarray],
    colors: list[np.ndarray],
    indices: list[int],
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
    *,
    segments: int = 24,
) -> None:
    axis = end - start
    axis_unit = axis / max(1.0e-6, float(np.linalg.norm(axis)))
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(axis_unit, helper))) > 0.92:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent = np.cross(axis_unit, helper)
    tangent /= max(1.0e-6, float(np.linalg.norm(tangent)))
    bitangent = np.cross(axis_unit, tangent)
    bitangent /= max(1.0e-6, float(np.linalg.norm(bitangent)))

    offset = len(positions)
    rgb = np.asarray(color, dtype=np.float32)
    for segment in range(segments):
        angle = 2.0 * math.pi * segment / segments
        normal = tangent * math.cos(angle) + bitangent * math.sin(angle)
        positions.append(start + normal * float(radius))
        positions.append(end + normal * float(radius))
        normals.append(normal.astype(np.float32))
        normals.append(normal.astype(np.float32))
        colors.append(rgb)
        colors.append(rgb)

    for segment in range(segments):
        next_segment = (segment + 1) % segments
        a = offset + segment * 2
        b = offset + segment * 2 + 1
        c = offset + next_segment * 2
        d = offset + next_segment * 2 + 1
        indices.extend((a, b, c, c, b, d))


def rotation_matrix(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(roll), math.sin(roll)
    yaw_matrix = np.array(
        [[cy, 0.0, sy, 0.0], [0.0, 1.0, 0.0, 0.0], [-sy, 0.0, cy, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pitch_matrix = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, cx, -sx, 0.0], [0.0, sx, cx, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    roll_matrix = np.array(
        [[cz, -sz, 0.0, 0.0], [sz, cz, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return (roll_matrix @ pitch_matrix @ yaw_matrix).astype(np.float32)


def view_projection_matrix(width: int, height: int, *, distance: float = 4.2) -> np.ndarray:
    aspect = max(1.0, float(width)) / max(1.0, float(height))
    near_plane = 0.1
    far_plane = 40.0
    f = 1.0 / math.tan(math.radians(34.0) * 0.5)
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = f / aspect
    projection[1, 1] = f
    projection[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
    projection[2, 3] = (2.0 * far_plane * near_plane) / (near_plane - far_plane)
    projection[3, 2] = -1.0
    view = np.eye(4, dtype=np.float32)
    view[2, 3] = -float(distance)
    return (projection @ view).astype(np.float32)


def build_widget_class(QtCore, QtOpenGLWidgets, GL, shaders):
    class OpenGLWaterWidget(QtOpenGLWidgets.QOpenGLWidget):
        def __init__(self, args):
            super().__init__()
            self.args = args
            self.mesh = water_mesh()
            self.program = None
            self.vertex_buffer = None
            self.index_buffer = None
            self.position_location = -1
            self.normal_location = -1
            self.color_location = -1
            self.model_location = -1
            self.view_projection_location = -1
            self.light_location = -1
            self.started_at = time.perf_counter()
            self.last_title_at = self.started_at
            self.frames = 0
            self.title_frames = 0
            self.reported = False
            self.dragging = False
            self.last_mouse = None
            self.yaw = math.radians(args.initial_yaw)
            self.pitch = math.radians(args.initial_pitch)
            self.roll = 0.0
            self.renderer_text = "OpenGL"
            self.setMinimumSize(args.width, args.height)
            self.resize(args.width, args.height)

            self.timer = None
            if args.auto_spin or args.timer_ms >= 0:
                timer = QtCore.QTimer(self)
                timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
                timer.timeout.connect(self.update)
                timer.start(max(0, args.timer_ms))
                self.timer = timer

            if args.seconds > 0:
                QtCore.QTimer.singleShot(int(args.seconds * 1000), self.report_and_quit)

        def initializeGL(self):
            GL.glClearColor(0.055, 0.06, 0.07, 1.0)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
            GL.glDisable(GL.GL_BLEND)
            if self.args.samples > 0 and hasattr(GL, "GL_MULTISAMPLE"):
                GL.glEnable(GL.GL_MULTISAMPLE)

            self.program = shaders.compileProgram(
                shaders.compileShader(VERTEX_SHADER, GL.GL_VERTEX_SHADER),
                shaders.compileShader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER),
            )
            self.position_location = GL.glGetAttribLocation(self.program, "a_position")
            self.normal_location = GL.glGetAttribLocation(self.program, "a_normal")
            self.color_location = GL.glGetAttribLocation(self.program, "a_color")
            self.model_location = GL.glGetUniformLocation(self.program, "u_model")
            self.view_projection_location = GL.glGetUniformLocation(self.program, "u_view_projection")
            self.light_location = GL.glGetUniformLocation(self.program, "u_light_direction")

            self.vertex_buffer = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertex_buffer)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.mesh.vertices.nbytes, self.mesh.vertices, GL.GL_STATIC_DRAW)

            self.index_buffer = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.mesh.indices.nbytes, self.mesh.indices, GL.GL_STATIC_DRAW)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

            version = decode_gl_string(GL.glGetString(GL.GL_VERSION))
            renderer = decode_gl_string(GL.glGetString(GL.GL_RENDERER))
            vendor = decode_gl_string(GL.glGetString(GL.GL_VENDOR))
            depth_bits = GL.GLint()
            GL.glGetIntegerv(GL.GL_DEPTH_BITS, depth_bits)
            self.renderer_text = renderer
            print(f"OpenGL version: {version}", flush=True)
            print(f"OpenGL renderer: {renderer}", flush=True)
            print(f"OpenGL vendor: {vendor}", flush=True)
            print(f"software renderer: {is_software_renderer(renderer, vendor)}", flush=True)
            print(f"OpenGL depth buffer bit: {int(depth_bits.value)}", flush=True)
            print(f"swap interval request: {self.args.swap_interval}, samples request: {self.args.samples}", flush=True)
            print("mode: auto spin" if self.args.auto_spin else "mode: fixed camera/light, left-button model drag", flush=True)

        def resizeGL(self, width, height):
            GL.glViewport(0, 0, max(1, width), max(1, height))

        def paintGL(self):
            now = time.perf_counter()
            elapsed = now - self.started_at
            yaw = self.yaw
            pitch = self.pitch
            roll = self.roll
            if self.args.auto_spin:
                yaw += elapsed * self.args.deg_per_sec * math.pi / 180.0
                pitch += elapsed * self.args.deg_per_sec * 0.38 * math.pi / 180.0
                roll += elapsed * self.args.deg_per_sec * 0.21 * math.pi / 180.0

            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glUseProgram(self.program)
            GL.glUniformMatrix4fv(self.model_location, 1, GL.GL_TRUE, rotation_matrix(yaw, pitch, roll))
            GL.glUniformMatrix4fv(
                self.view_projection_location,
                1,
                GL.GL_TRUE,
                view_projection_matrix(self.width(), self.height(), distance=self.args.camera_distance),
            )
            GL.glUniform3f(self.light_location, 0.34, 0.44, 0.83)

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertex_buffer)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
            stride = 9 * 4
            GL.glEnableVertexAttribArray(self.position_location)
            GL.glVertexAttribPointer(self.position_location, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(self.normal_location)
            GL.glVertexAttribPointer(self.normal_location, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
            GL.glEnableVertexAttribArray(self.color_location)
            GL.glVertexAttribPointer(self.color_location, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(24))
            GL.glDrawElements(GL.GL_TRIANGLES, int(self.mesh.indices.shape[0]), GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
            GL.glDisableVertexAttribArray(self.position_location)
            GL.glDisableVertexAttribArray(self.normal_location)
            GL.glDisableVertexAttribArray(self.color_location)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
            GL.glUseProgram(0)

            self.frames += 1
            self.title_frames += 1
            if now - self.last_title_at >= 0.5:
                fps = self.title_frames / (now - self.last_title_at)
                self.title_frames = 0
                self.last_title_at = now
                self.setWindowTitle(
                    f"OpenGL water demo | {fps:5.1f} FPS | yaw {math.degrees(self.yaw):.0f} | "
                    f"pitch {math.degrees(self.pitch):.0f} | {self.renderer_text}"
                )

        def mousePressEvent(self, event):
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.dragging = True
                self.last_mouse = self._event_xy(event)
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if not self.dragging or self.last_mouse is None:
                super().mouseMoveEvent(event)
                return
            x, y = self._event_xy(event)
            last_x, last_y = self.last_mouse
            self.last_mouse = (x, y)
            scale = math.radians(self.args.drag_deg_per_pixel)
            self.yaw += (x - last_x) * scale
            self.pitch += (y - last_y) * scale
            self.update()
            event.accept()

        def mouseReleaseEvent(self, event):
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.dragging = False
                self.last_mouse = None
                event.accept()
                return
            super().mouseReleaseEvent(event)

        def _event_xy(self, event):
            point = event.position() if hasattr(event, "position") else event.pos()
            return (float(point.x()), float(point.y()))

        def report(self):
            if self.reported:
                return
            self.reported = True
            elapsed = max(0.001, time.perf_counter() - self.started_at)
            print(f"frames: {self.frames}", flush=True)
            print(f"average fps: {self.frames / elapsed:.1f}", flush=True)

        def report_and_quit(self):
            self.report()
            os._exit(0)

        def closeEvent(self, event):
            self.report()
            os._exit(0)

    return OpenGLWaterWidget


def decode_gl_string(value):
    if value is None:
        return "unavailable"
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def is_software_renderer(renderer, vendor):
    text = f"{renderer} {vendor}".lower()
    return any(marker in text for marker in SOFTWARE_RENDERER_MARKERS)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run a standalone PySide6 + PyOpenGL water molecule rotation demo.")
    parser.add_argument(
        "--atomstudio-runtime",
        action="store_true",
        help="Configure AtomStudio's Qt/OpenGL runtime before creating the demo window.",
    )
    parser.add_argument(
        "--atomstudio-runtime-no-vispy",
        action="store_true",
        help="Configure AtomStudio's Qt/OpenGL runtime without loading VisPy.",
    )
    parser.add_argument(
        "--atomstudio-native-runtime",
        action="store_true",
        help="Configure AtomStudio's minimal native OpenGL runtime.",
    )
    parser.add_argument("--auto-spin", action="store_true", help="Rotate the molecule automatically.")
    parser.add_argument("--deg-per-sec", type=float, default=720.0, help="Auto-spin rotation speed in degrees per second.")
    parser.add_argument(
        "--timer-ms",
        type=int,
        default=-1,
        help="QTimer interval. -1 repaints only on mouse drag; 0 repaints as fast as possible.",
    )
    parser.add_argument("--drag-deg-per-pixel", type=float, default=0.45, help="Mouse drag model rotation sensitivity.")
    parser.add_argument("--initial-yaw", type=float, default=25.0, help="Initial model yaw angle in degrees.")
    parser.add_argument("--initial-pitch", type=float, default=18.0, help="Initial model pitch angle in degrees.")
    parser.add_argument("--camera-distance", type=float, default=4.2, help="Fixed camera distance.")
    parser.add_argument("--swap-interval", type=int, default=0, help="Requested OpenGL swap interval. 0 disables vsync.")
    parser.add_argument("--samples", type=int, default=0, help="Requested MSAA sample count.")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--seconds", type=float, default=0.0, help="Auto-close after this many seconds. 0 keeps the window open.")
    return parser.parse_args(argv)


def configure_pyside_plugin_path():
    try:
        import PySide6
    except Exception:
        return
    plugins = pathlib.Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    if not plugins.exists():
        return
    current = os.environ.get("QT_PLUGIN_PATH", "")
    paths = [str(plugins)]
    if current:
        paths.extend(path for path in current.split(os.pathsep) if path and path != str(plugins))
    os.environ["QT_PLUGIN_PATH"] = os.pathsep.join(paths)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.atomstudio_runtime or args.atomstudio_runtime_no_vispy:
        try:
            from atomstudio.app.runtime import configure_qt_runtime

            configure_qt_runtime(enable_vispy=not bool(args.atomstudio_runtime_no_vispy))
        except Exception as exc:
            print(f"AtomStudio runtime setup failed: {exc}", file=sys.stderr)
            return 2
    else:
        try:
            from atomstudio.app.runtime import configure_native_opengl_runtime

            configure_native_opengl_runtime()
        except Exception as exc:
            if args.atomstudio_native_runtime:
                print(f"AtomStudio native runtime setup failed: {exc}", file=sys.stderr)
                return 2
    configure_pyside_plugin_path()
    try:
        from PySide6 import QtCore, QtGui, QtOpenGLWidgets, QtWidgets
        from OpenGL import GL
        from OpenGL.GL import shaders
    except Exception as exc:
        print(f"Missing GUI dependency: {exc}", file=sys.stderr)
        print("Install with: python -m pip install -e '.[gui]'", file=sys.stderr)
        return 2

    surface_format = QtGui.QSurfaceFormat()
    surface_format.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
    surface_format.setDepthBufferSize(24)
    surface_format.setSamples(max(0, args.samples))
    surface_format.setSwapInterval(max(0, args.swap_interval))
    QtGui.QSurfaceFormat.setDefaultFormat(surface_format)
    QtCore.QCoreApplication.setLibraryPaths(
        [QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.PluginsPath)]
    )

    app = QtWidgets.QApplication(sys.argv[:1])
    widget_class = build_widget_class(QtCore, QtOpenGLWidgets, GL, shaders)
    widget = widget_class(args)
    widget.setWindowTitle("OpenGL water demo")
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
