#version 120

attribute vec2 a_corner;
attribute vec3 a_instance_position;
attribute float a_instance_radius;
attribute vec4 a_instance_color;

uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_model;

varying vec2 v_corner;
varying vec3 v_center_view;
varying float v_radius;
varying vec4 v_color;

void main() {
    vec4 world_center = u_model * vec4(a_instance_position, 1.0);
    vec4 center_view = u_view * world_center;
    vec4 quad_view = center_view + vec4(a_corner * a_instance_radius, 0.0, 0.0);

    v_corner = a_corner;
    v_center_view = center_view.xyz;
    v_radius = a_instance_radius;
    v_color = a_instance_color;
    gl_Position = u_projection * quad_view;
}
