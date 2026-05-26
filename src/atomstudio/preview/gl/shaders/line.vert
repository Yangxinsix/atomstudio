#version 120

attribute vec3 a_position;
attribute vec4 a_color;

uniform mat4 u_view_projection;
uniform mat4 u_model;

varying vec4 v_color;

void main() {
    v_color = a_color;
    gl_Position = u_view_projection * u_model * vec4(a_position, 1.0);
}
