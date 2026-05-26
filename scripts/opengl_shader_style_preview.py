from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from OpenGL import EGL
from OpenGL import GL
from OpenGL.raw.EGL.EXT.platform_base import eglGetPlatformDisplayEXT
from OpenGL.GL import shaders
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ShaderStyle:
    name: str
    ambient: float
    diffuse: float
    wrap: float
    specular: float
    shininess: float
    exposure: float = 1.0
    mode: int = 0


STYLES = (
    ShaderStyle("matcap pearl", ambient=0.56, diffuse=0.34, wrap=0.45, specular=0.055, shininess=48.0, mode=1),
    ShaderStyle("qutemol soft", ambient=0.52, diffuse=0.36, wrap=0.75, specular=0.030, shininess=28.0, mode=2),
    ShaderStyle("studio glossy", ambient=0.44, diffuse=0.46, wrap=0.25, specular=0.080, shininess=64.0, mode=3),
    ShaderStyle("technical gooch", ambient=0.48, diffuse=0.38, wrap=0.35, specular=0.018, shininess=24.0, mode=4),
    ShaderStyle("toon rim", ambient=0.54, diffuse=0.42, wrap=0.15, specular=0.020, shininess=32.0, mode=5),
    ShaderStyle("soft clay", ambient=0.62, diffuse=0.24, wrap=0.85, specular=0.000, shininess=1.0, mode=6),
)

SPHERES = (
    ((-1.62, -0.08, 0.0), 0.58, (1.0, 0.051, 0.051, 1.0)),
    ((-0.58, 0.20, 0.0), 0.42, (1.0, 0.80, 0.80, 1.0)),
    ((0.52, -0.06, 0.0), 0.55, (0.878, 0.40, 0.20, 1.0)),
    ((1.52, 0.18, 0.0), 0.50, (0.941, 0.565, 0.627, 1.0)),
)

VERTEX_SHADER = """
#version 120
attribute vec3 a_position;
attribute vec3 a_normal;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform vec4 u_color;

varying vec3 v_normal;
varying vec4 v_color;

void main() {
    v_normal = normalize((u_model * vec4(a_normal, 0.0)).xyz);
    v_color = u_color;
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 120
varying vec3 v_normal;
varying vec4 v_color;

uniform vec3 u_light_direction;
uniform float u_ambient_strength;
uniform float u_diffuse_strength;
uniform float u_wrap_strength;
uniform float u_specular_strength;
uniform float u_shininess;
uniform float u_exposure;
uniform int u_style_mode;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(u_light_direction);
    vec3 view = vec3(0.0, 0.0, 1.0);
    float ndotl = dot(normal, light);
    float diffuse = max(ndotl, 0.0);
    float wrapped = max((ndotl + u_wrap_strength) / (1.0 + max(u_wrap_strength, 0.0001)), 0.0);
    float rim = pow(1.0 - max(dot(normal, view), 0.0), 2.0);
    float spec = pow(max(dot(reflect(-light, normal), view), 0.0), max(u_shininess, 1.0));
    vec3 base = v_color.rgb;
    vec3 color = base * (u_ambient_strength + u_diffuse_strength * wrapped);

    if (u_style_mode == 1) {
        vec2 uv = normal.xy * 0.5 + 0.5;
        float upper = clamp(1.0 - length(uv - vec2(0.35, 0.70)) * 1.45, 0.0, 1.0);
        float lower = clamp(1.0 - length(uv - vec2(0.68, 0.30)) * 1.05, 0.0, 1.0);
        color = base * (0.52 + 0.26 * max(normal.z, 0.0) + 0.18 * lower);
        color += vec3(0.20) * pow(upper, 5.0);
    } else if (u_style_mode == 2) {
        float face = smoothstep(0.0, 0.75, max(dot(normal, view), 0.0));
        color = base * (0.58 + 0.34 * wrapped) * (0.76 + 0.24 * face);
        color = mix(color * 0.72, color, face);
    } else if (u_style_mode == 3) {
        color = base * (0.42 + 0.48 * wrapped);
        color += vec3(0.20) * pow(spec, 0.85);
    } else if (u_style_mode == 4) {
        vec3 cool = vec3(0.38, 0.48, 0.72);
        vec3 warm = vec3(1.0, 0.84, 0.50);
        float t = clamp(ndotl * 0.5 + 0.5, 0.0, 1.0);
        vec3 gooch = mix(cool, warm, t);
        color = base * 0.64 + gooch * 0.24 + base * 0.12 * diffuse;
    } else if (u_style_mode == 5) {
        float band = diffuse > 0.66 ? 0.98 : (diffuse > 0.28 ? 0.76 : 0.54);
        color = base * band;
        color = mix(color, color * 0.42, smoothstep(0.45, 0.88, rim));
    } else if (u_style_mode == 6) {
        float gray = dot(base, vec3(0.30, 0.59, 0.11));
        color = mix(base, vec3(gray), 0.15);
        color *= 0.66 + 0.24 * wrapped;
        color = mix(color, vec3(0.94), 0.08);
    }

    color += vec3(u_specular_strength) * spec;
    gl_FragColor = vec4(clamp(color * u_exposure, 0.0, 1.0), v_color.a);
}
"""


def main() -> int:
    out = Path("output/opengl_shader_style_preview.png").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    final_width, final_height = 1440, 900
    ssaa = 3
    width, height = final_width * ssaa, final_height * ssaa
    egl_display, egl_surface, egl_context = create_egl_context(width, height)
    fbo, color_tex, depth_rb = create_fbo(width, height)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glViewport(0, 0, width, height)
    GL.glClearColor(0.92, 0.93, 0.94, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glCullFace(GL.GL_BACK)
    try:
        GL.glEnable(GL.GL_MULTISAMPLE)
    except Exception:
        pass

    program = shaders.compileProgram(
        shaders.compileShader(VERTEX_SHADER, GL.GL_VERTEX_SHADER),
        shaders.compileShader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER),
    )
    mesh = sphere_mesh(segments=128, rings=80)
    vao = make_buffers(mesh)

    panel_w = width // 3
    panel_h = height // 2
    for idx, style in enumerate(STYLES):
        col = idx % 3
        row = idx // 3
        x = col * panel_w
        y = height - (row + 1) * panel_h
        render_panel(program, vao, len(mesh[2]), style, x, y, panel_w, panel_h)

    raw = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (width, height), raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    image = image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    draw_labels(image, final_width, final_height)
    image.save(out)

    GL.glDeleteProgram(program)
    GL.glDeleteFramebuffers(1, [fbo])
    GL.glDeleteTextures(1, [color_tex])
    GL.glDeleteRenderbuffers(1, [depth_rb])
    destroy_egl_context(egl_display, egl_surface, egl_context)
    print(out)
    return 0


def create_egl_context(width: int, height: int):
    display = eglGetPlatformDisplayEXT(0x31DD, EGL.EGL_DEFAULT_DISPLAY, None)
    major = EGL.EGLint()
    minor = EGL.EGLint()
    if not EGL.eglInitialize(display, major, minor):
        raise RuntimeError("Failed to initialize EGL surfaceless display")
    if not EGL.eglBindAPI(EGL.EGL_OPENGL_API):
        raise RuntimeError("Failed to bind EGL OpenGL API")

    attributes = [
        EGL.EGL_RED_SIZE,
        8,
        EGL.EGL_GREEN_SIZE,
        8,
        EGL.EGL_BLUE_SIZE,
        8,
        EGL.EGL_ALPHA_SIZE,
        8,
        EGL.EGL_DEPTH_SIZE,
        24,
        EGL.EGL_SURFACE_TYPE,
        EGL.EGL_PBUFFER_BIT,
        EGL.EGL_RENDERABLE_TYPE,
        EGL.EGL_OPENGL_BIT,
        EGL.EGL_NONE,
    ]
    config = (EGL.EGLConfig * 1)()
    count = EGL.EGLint()
    if not EGL.eglChooseConfig(display, attributes, config, 1, count) or count.value < 1:
        raise RuntimeError("Failed to choose EGL framebuffer config")

    surface_attributes = [EGL.EGL_WIDTH, int(width), EGL.EGL_HEIGHT, int(height), EGL.EGL_NONE]
    surface = EGL.eglCreatePbufferSurface(display, config[0], surface_attributes)
    if surface == EGL.EGL_NO_SURFACE:
        raise RuntimeError("Failed to create EGL pbuffer surface")

    context = EGL.eglCreateContext(display, config[0], EGL.EGL_NO_CONTEXT, [EGL.EGL_NONE])
    if context == EGL.EGL_NO_CONTEXT:
        raise RuntimeError("Failed to create EGL OpenGL context")
    if not EGL.eglMakeCurrent(display, surface, surface, context):
        raise RuntimeError("Failed to make EGL context current")
    return display, surface, context


def destroy_egl_context(display, surface, context) -> None:
    EGL.eglMakeCurrent(display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
    EGL.eglDestroyContext(display, context)
    EGL.eglDestroySurface(display, surface)
    EGL.eglTerminate(display)


def create_fbo(width: int, height: int) -> tuple[int, int, int]:
    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    color_tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, color_tex)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, color_tex, 0)
    depth_rb = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth_rb)
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"Incomplete framebuffer: 0x{status:x}")
    return int(fbo), int(color_tex), int(depth_rb)


def sphere_mesh(*, segments: int, rings: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = []
    normals = []
    indices = []
    for ring in range(rings + 1):
        theta = math.pi * ring / rings
        st = math.sin(theta)
        ct = math.cos(theta)
        for segment in range(segments):
            phi = 2.0 * math.pi * segment / segments
            x = st * math.cos(phi)
            y = st * math.sin(phi)
            z = ct
            vertices.append((x, y, z))
            normals.append((x, y, z))
    for ring in range(rings):
        for segment in range(segments):
            nxt = (segment + 1) % segments
            a = ring * segments + segment
            b = ring * segments + nxt
            c = (ring + 1) * segments + segment
            d = (ring + 1) * segments + nxt
            if ring > 0:
                indices.extend((a, c, b))
            if ring < rings - 1:
                indices.extend((b, c, d))
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def make_buffers(mesh: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[int, int, int]:
    vertices, normals, indices = mesh
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
    nbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, nbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, normals.nbytes, normals, GL.GL_STATIC_DRAW)
    ibo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
    return int(vbo), int(nbo), int(ibo)


def render_panel(
    program: int,
    buffers: tuple[int, int, int],
    index_count: int,
    style: ShaderStyle,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    GL.glViewport(x, y, width, height)
    GL.glScissor(x, y, width, height)
    GL.glEnable(GL.GL_SCISSOR_TEST)
    GL.glClearColor(0.92, 0.93, 0.94, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glDisable(GL.GL_SCISSOR_TEST)
    GL.glUseProgram(program)

    projection = panel_projection(width, height)
    view = look_at(np.array((0.0, 0.0, 6.0), dtype=np.float32), np.array((0.0, 0.0, 0.0), dtype=np.float32))
    set_uniform3(program, "u_light_direction", normalize(np.array((0.35, 0.44, 0.82), dtype=np.float32)))
    set_uniform1(program, "u_ambient_strength", style.ambient)
    set_uniform1(program, "u_diffuse_strength", style.diffuse)
    set_uniform1(program, "u_wrap_strength", style.wrap)
    set_uniform1(program, "u_specular_strength", style.specular)
    set_uniform1(program, "u_shininess", style.shininess)
    set_uniform1(program, "u_exposure", style.exposure)
    set_uniform_int(program, "u_style_mode", style.mode)

    vbo, nbo, ibo = buffers
    bind_attribute(program, "a_position", vbo)
    bind_attribute(program, "a_normal", nbo)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibo)
    for position, radius, color in SPHERES:
        model = translate(*position) @ scale(radius)
        mvp = projection @ view @ model
        set_matrix(program, "u_model", model)
        set_matrix(program, "u_mvp", mvp)
        set_uniform4(program, "u_color", color)
        GL.glDrawElements(GL.GL_TRIANGLES, index_count, GL.GL_UNSIGNED_INT, None)
    GL.glUseProgram(0)


def bind_attribute(program: int, name: str, buffer_id: int) -> None:
    loc = GL.glGetAttribLocation(program, name)
    if loc < 0:
        return
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_id)
    GL.glEnableVertexAttribArray(loc)
    GL.glVertexAttribPointer(loc, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)


def set_matrix(program: int, name: str, matrix: np.ndarray) -> None:
    loc = GL.glGetUniformLocation(program, name)
    if loc >= 0:
        GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, np.asarray(matrix, dtype=np.float32))


def set_uniform1(program: int, name: str, value: float) -> None:
    loc = GL.glGetUniformLocation(program, name)
    if loc >= 0:
        GL.glUniform1f(loc, float(value))


def set_uniform_int(program: int, name: str, value: int) -> None:
    loc = GL.glGetUniformLocation(program, name)
    if loc >= 0:
        GL.glUniform1i(loc, int(value))


def set_uniform3(program: int, name: str, value: np.ndarray) -> None:
    loc = GL.glGetUniformLocation(program, name)
    if loc >= 0:
        GL.glUniform3f(loc, float(value[0]), float(value[1]), float(value[2]))


def set_uniform4(program: int, name: str, value: tuple[float, float, float, float]) -> None:
    loc = GL.glGetUniformLocation(program, name)
    if loc >= 0:
        GL.glUniform4f(loc, float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def panel_projection(width: int, height: int) -> np.ndarray:
    world_width = 4.70
    aspect = max(1.0e-6, float(width) / max(1.0, float(height)))
    world_height = world_width / aspect
    return orthographic(-world_width * 0.5, world_width * 0.5, -world_height * 0.5, world_height * 0.5, 1.0, 12.0)


def orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[0, 0] = 2.0 / (right - left)
    mat[1, 1] = 2.0 / (top - bottom)
    mat[2, 2] = -2.0 / (far - near)
    mat[0, 3] = -(right + left) / (right - left)
    mat[1, 3] = -(top + bottom) / (top - bottom)
    mat[2, 3] = -(far + near) / (far - near)
    return mat


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, np.array((0.0, 1.0, 0.0), dtype=np.float32)))
    up = np.cross(right, forward)
    mat = np.eye(4, dtype=np.float32)
    mat[0, :3] = right
    mat[1, :3] = up
    mat[2, :3] = -forward
    mat[0, 3] = -float(np.dot(right, eye))
    mat[1, 3] = -float(np.dot(up, eye))
    mat[2, 3] = float(np.dot(forward, eye))
    return mat


def translate(x: float, y: float, z: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = (x, y, z)
    return mat


def scale(value: float) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[0, 0] = mat[1, 1] = mat[2, 2] = float(value)
    return mat


def normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value if norm <= 1.0e-12 else value / norm


def draw_labels(image: Image.Image, width: int, height: int) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    panel_w = width // 3
    panel_h = height // 2
    for idx, style in enumerate(STYLES):
        col = idx % 3
        row = idx // 3
        x = col * panel_w + 24
        y = row * panel_h + 20
        draw.text((x + 1, y + 1), style.name, fill=(255, 255, 255, 220), font=font)
        draw.text((x, y), style.name, fill=(32, 36, 40, 255), font=font)
        draw.rectangle(
            (col * panel_w, row * panel_h, (col + 1) * panel_w - 1, (row + 1) * panel_h - 1),
            outline=(205, 208, 212, 255),
            width=1,
        )


if __name__ == "__main__":
    raise SystemExit(main())
