#version 120

attribute vec3 a_position;
attribute vec3 a_instance_start;
attribute vec3 a_instance_end;
attribute vec4 a_instance_color_start;
attribute vec4 a_instance_color_end;
attribute float a_instance_radius;

uniform mat4 u_view_projection;
uniform mat4 u_model;

varying vec3 v_normal;
varying vec4 v_color;

vec3 safe_normalize(vec3 value, vec3 fallback) {
    float len = length(value);
    if (len <= 0.000001) {
        return fallback;
    }
    return value / len;
}

void main() {
    vec3 start = a_instance_start;
    vec3 end = a_instance_end;
    vec3 axis = safe_normalize(end - start, vec3(0.0, 0.0, 1.0));
    vec3 helper = abs(axis.z) < 0.92 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 side = safe_normalize(cross(helper, axis), vec3(1.0, 0.0, 0.0));
    vec3 up = safe_normalize(cross(axis, side), vec3(0.0, 1.0, 0.0));
    vec3 radial = side * a_position.x + up * a_position.y;
    vec3 world_position = mix(start, end, a_position.z) + radial * a_instance_radius;

    v_normal = normalize((u_model * vec4(radial, 0.0)).xyz);
    v_color = mix(a_instance_color_start, a_instance_color_end, clamp(a_position.z, 0.0, 1.0));
    gl_Position = u_view_projection * u_model * vec4(world_position, 1.0);
}
