#version 120

varying vec2 v_corner;
varying vec3 v_center_view;
varying float v_radius;
varying vec4 v_color;

uniform mat4 u_projection;
uniform vec3 u_light_direction;
uniform float u_ambient_strength;
uniform float u_diffuse_strength;
uniform float u_wrap_strength;
uniform float u_specular_strength;
uniform float u_shininess;
uniform float u_exposure;
uniform int u_style_mode;

void main() {
    float radius2 = dot(v_corner, v_corner);
    if (radius2 > 1.0) {
        discard;
    }

    float z = sqrt(max(1.0 - radius2, 0.0));
    vec3 normal = normalize(vec3(v_corner.xy, z));
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

    vec4 sphere_view = vec4(v_center_view.xy + v_corner * v_radius, v_center_view.z + z * v_radius, 1.0);
    vec4 sphere_clip = u_projection * sphere_view;
    float ndc_depth = sphere_clip.z / sphere_clip.w;
    gl_FragDepth = clamp(ndc_depth * 0.5 + 0.5, 0.0, 1.0);

    color += vec3(u_specular_strength) * spec;
    gl_FragColor = vec4(clamp(color * u_exposure, 0.0, 1.0), v_color.a);
}
