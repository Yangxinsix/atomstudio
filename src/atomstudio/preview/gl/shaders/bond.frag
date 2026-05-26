#version 120

varying vec3 v_normal;
varying vec4 v_color;

uniform vec3 u_light_direction;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(u_light_direction);
    float diffuse = max(dot(normal, light), 0.0);
    float shade = 0.50 + 0.42 * diffuse;
    gl_FragColor = vec4(clamp(v_color.rgb * shade, 0.0, 1.0), v_color.a);
}
