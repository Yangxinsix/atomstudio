import bpy
import argparse
import json
import sys
from pathlib import Path
from mathutils import Vector
from math import tan
from bpy_extras.object_utils import world_to_camera_view

# =========================
# 1) CONFIG
# =========================
NODE_RADIUS = 0.043
EDGE_RADIUS = 0.0012
EDGE_SUBDIV = 12   # 圆柱截面细分，8~16
SPHERE_SEGMENTS = 48
SPHERE_RINGS = 24

DEFAULT_BLUE_RATIO = 0.12

COLORS = {
    "gray":   (0.38, 0.42, 0.47, 1.0),
    "blue":   (0.15, 0.65, 0.90, 1.0),
    "orange": (0.95, 0.56, 0.17, 1.0),
    "purple": (0.44, 0.26, 0.76, 1.0),
    "edge":   (0.12, 0.12, 0.12, 1.0),
}

def parse_script_args():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Render teacher graph preview.")
    parser.add_argument(
        "--json",
        dest="json_path",
        default=str(script_dir / "teacher_graph_3d.json"),
        help="Path to the graph JSON file.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=str(script_dir / "teacher_preview_low.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--engine",
        default="CYCLES",
        choices=["BLENDER_EEVEE", "CYCLES"],
        help="Render engine.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Render samples.",
    )
    parser.add_argument(
        "--res-x",
        type=int,
        default=1024,
        help="Output width in pixels.",
    )
    parser.add_argument(
        "--res-y",
        type=int,
        default=1024,
        help="Output height in pixels.",
    )
    parser.set_defaults(transparent_bg=True)
    parser.add_argument(
        "--transparent-bg",
        dest="transparent_bg",
        action="store_true",
        help="Render with transparent background (default).",
    )
    parser.add_argument(
        "--solid-bg",
        dest="transparent_bg",
        action="store_false",
        help="Render with solid background.",
    )
    parser.add_argument(
        "--use-group-colors",
        dest="use_group_colors",
        action="store_true",
        help="Use node colors from group field in JSON (default).",
    )
    parser.add_argument(
        "--no-group-colors",
        dest="use_group_colors",
        action="store_false",
        help="Disable group colors and use orange+blue style.",
    )
    parser.add_argument(
        "--blue-ratio",
        type=float,
        default=DEFAULT_BLUE_RATIO,
        help="Accent blue node ratio when not using group colors.",
    )
    parser.add_argument(
        "--material-style",
        choices=["glossy", "matte"],
        default="glossy",
        help="Material style for nodes and edges.",
    )
    parser.add_argument(
        "--camera-type",
        choices=["PERSPECTIVE", "ORTHOGRAPHIC"],
        default="PERSPECTIVE",
        help="Camera projection type.",
    )
    parser.add_argument(
        "--camera-lens",
        type=float,
        default=74.0,
        help="Perspective lens in mm.",
    )
    parser.add_argument(
        "--camera-fit-margin",
        type=float,
        default=0.08,
        help="Normalized frame margin used for camera auto-fit (0~0.4).",
    )

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser.set_defaults(use_group_colors=True)
    return parser.parse_args(argv)

ARGS = parse_script_args()

# =========================
# 2) HELPERS
# =========================
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def get_or_create_material(
    name,
    rgba,
    roughness=0.35,
    specular=0.4,
    metallic=0.0,
    coat=0.0,
    coat_roughness=0.08,
):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True

    bsdf = next(
        (node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        # Blender 版本兼容
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = specular
        elif "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = specular
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = coat
        elif "Coat" in bsdf.inputs:
            bsdf.inputs["Coat"].default_value = coat
        if "Coat Roughness" in bsdf.inputs:
            bsdf.inputs["Coat Roughness"].default_value = coat_roughness
    return mat

def assign_mat(obj, mat):
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

def add_sphere(location, radius, mat=None, name="Node"):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=SPHERE_SEGMENTS, ring_count=SPHERE_RINGS, radius=radius, location=location
    )
    obj = bpy.context.active_object
    obj.name = name
    for poly in obj.data.polygons:
        poly.use_smooth = True
    if mat:
        assign_mat(obj, mat)
    return obj

def add_cylinder_between(p1, p2, radius, mat=None, name="Edge"):
    p1 = Vector(p1)
    p2 = Vector(p2)
    vec = p2 - p1
    length = vec.length
    if length < 1e-8:
        return None

    mid = (p1 + p2) / 2.0
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=EDGE_SUBDIV,
        radius=radius,
        depth=length,
        location=mid
    )
    obj = bpy.context.active_object
    obj.name = name

    # 将圆柱本地 z 轴旋转到边方向
    z_axis = Vector((0, 0, 1))
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = z_axis.rotation_difference(vec.normalized())
    for poly in obj.data.polygons:
        poly.use_smooth = True

    if mat:
        assign_mat(obj, mat)
    return obj

def setup_world_and_render(args):
    scene = bpy.context.scene
    scene.render.engine = args.engine

    if hasattr(scene, "cycles"):
        scene.cycles.samples = max(1, args.samples)
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = max(1, args.samples)

    scene.render.resolution_x = max(1, args.res_x)
    scene.render.resolution_y = max(1, args.res_y)
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = args.transparent_bg
    scene.render.image_settings.color_mode = "RGBA" if args.transparent_bg else "RGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = -0.8

    # 背景颜色（可改）
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = next(
        (node for node in world.node_tree.nodes if node.type == "BACKGROUND"),
        None,
    )
    if bg:
        bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0) # 类似你之前背景
        bg.inputs[1].default_value = 1.0

def setup_camera_and_lights(args):
    # Camera
    bpy.ops.object.camera_add(location=(0, -3.45, 1.72), rotation=(1.10, 0.0, 0.0))
    cam = bpy.context.active_object
    cam.data.type = 'ORTHO' if args.camera_type == "ORTHOGRAPHIC" else 'PERSP'
    cam.data.lens = max(18.0, args.camera_lens)
    cam.data.clip_start = 0.01
    cam.data.clip_end = 1000.0
    bpy.context.scene.camera = cam

    # Key light
    bpy.ops.object.light_add(type='AREA', location=(2.0, -2.0, 2.3))
    key = bpy.context.active_object
    key.data.energy = 260
    key.data.size = 1.5

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-2.2, -1.5, 1.2))
    fill = bpy.context.active_object
    fill.data.energy = 95
    fill.data.size = 2.3

    # Rim light
    bpy.ops.object.light_add(type='POINT', location=(0.0, 2.5, 1.8))
    rim = bpy.context.active_object
    rim.data.energy = 26
    return cam

def fit_camera_to_objects(camera, objs, margin=0.08):
    if not objs:
        return

    margin = min(max(margin, 0.01), 0.40)
    usable = max(0.05, 1.0 - 2.0 * margin)

    points = []
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for obj in objs:
        for corner in obj.bound_box:
            p = obj.matrix_world @ Vector(corner)
            points.append(p)
            mins.x = min(mins.x, p.x)
            mins.y = min(mins.y, p.y)
            mins.z = min(mins.z, p.z)
            maxs.x = max(maxs.x, p.x)
            maxs.y = max(maxs.y, p.y)
            maxs.z = max(maxs.z, p.z)

    center = (mins + maxs) / 2.0
    forward = (center - camera.location)
    if forward.length < 1e-8:
        forward = Vector((0.0, 1.0, -0.25))
    forward.normalize()
    quat = forward.to_track_quat('-Z', 'Y')
    rot_inv = quat.to_matrix().inverted()

    local = [rot_inv @ (p - center) for p in points]
    max_abs_x = max(abs(v.x) for v in local)
    max_abs_y = max(abs(v.y) for v in local)
    max_z = max(v.z for v in local)
    min_z = min(v.z for v in local)

    scene = bpy.context.scene
    aspect = scene.render.resolution_x / max(1.0, scene.render.resolution_y)

    if camera.data.type == 'ORTHO':
        half_w = max_abs_x / usable
        half_h = max_abs_y / usable
        camera.data.ortho_scale = max(2.0 * half_w, 2.0 * half_h * aspect)
        distance = max(max_z - min_z + 1.0, 3.0)
    else:
        tan_x = tan(camera.data.angle_x * 0.5) * usable
        tan_y = tan(camera.data.angle_y * 0.5) * usable
        need_d = 0.0
        for v in local:
            need_d = max(
                need_d,
                v.z + abs(v.x) / max(1e-6, tan_x),
                v.z + abs(v.y) / max(1e-6, tan_y),
            )
        distance = max(need_d + NODE_RADIUS * 4.0, 2.5)

    camera.location = center - forward * distance
    camera.rotation_euler = quat.to_euler()
    bpy.context.view_layer.update()

    # Perspective case: final iterative expand to guarantee full framing
    if camera.data.type == 'PERSP':
        for _ in range(48):
            xs, ys, zs = [], [], []
            for p in points:
                c = world_to_camera_view(bpy.context.scene, camera, p)
                xs.append(c.x)
                ys.append(c.y)
                zs.append(c.z)

            if min(zs) > 0 and min(xs) >= margin and max(xs) <= (1.0 - margin) and min(ys) >= margin and max(ys) <= (1.0 - margin):
                break

            distance *= 1.06
            camera.location = center - forward * distance
            bpy.context.view_layer.update()

def choose_blue_nodes_by_degree(nodes, edges, ratio):
    node_ids = [int(n["id"]) for n in nodes]
    if not node_ids:
        return set()

    ratio = max(0.0, min(1.0, ratio))
    target = max(1, int(round(len(node_ids) * ratio)))

    degree = {nid: 0 for nid in node_ids}
    for e in edges:
        u = int(e["source"])
        v = int(e["target"])
        if u in degree:
            degree[u] += 1
        if v in degree:
            degree[v] += 1

    ranked = sorted(node_ids, key=lambda nid: (degree[nid], -nid), reverse=True)
    return set(ranked[:target])

def frame_all_objects(objs, target_extent=2.0):
    # 简单居中缩放，方便相机看到整体
    if not objs:
        return

    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))

    for obj in objs:
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, wc.x); mins.y = min(mins.y, wc.y); mins.z = min(mins.z, wc.z)
            maxs.x = max(maxs.x, wc.x); maxs.y = max(maxs.y, wc.y); maxs.z = max(maxs.z, wc.z)

    center = (mins + maxs) / 2.0
    extent = max(maxs.x - mins.x, maxs.y - mins.y, maxs.z - mins.z)
    scale = target_extent / extent if extent > 1e-8 else 1.0

    for obj in objs:
        obj.location = (obj.location - center) * scale
        obj.scale *= scale

# =========================
# 3) LOAD JSON
# =========================
json_path = Path(ARGS.json_path).expanduser()
if not json_path.is_absolute():
    json_path = (Path.cwd() / json_path).resolve()

out_path = Path(ARGS.out_path).expanduser()
if not out_path.is_absolute():
    out_path = (Path.cwd() / out_path).resolve()
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data["nodes"]
edges = data["edges"]

# 建一个 id -> xyz 映射
node_pos = {}
node_group = {}
for n in nodes:
    nid = int(n["id"])
    node_pos[nid] = (float(n["x"]), float(n["y"]), float(n.get("z", 0.0)))
    node_group[nid] = n.get("group", "gray")

# =========================
# 4) BUILD SCENE
# =========================
clear_scene()
setup_world_and_render(ARGS)

# 材质
if ARGS.material_style == "glossy":
    mat_gray = get_or_create_material("MatGray", COLORS["gray"], roughness=0.24, specular=0.62, metallic=0.0, coat=0.30, coat_roughness=0.03)
    mat_blue = get_or_create_material("MatBlue", COLORS["blue"], roughness=0.16, specular=0.72, metallic=0.0, coat=0.35, coat_roughness=0.03)
    mat_orange = get_or_create_material("MatOrange", COLORS["orange"], roughness=0.15, specular=0.70, metallic=0.0, coat=0.35, coat_roughness=0.03)
    mat_purple = get_or_create_material("MatPurple", COLORS["purple"], roughness=0.18, specular=0.68, metallic=0.0, coat=0.32, coat_roughness=0.03)
    mat_edge = get_or_create_material("MatEdge", COLORS["edge"], roughness=0.30, specular=0.30, metallic=0.0, coat=0.08, coat_roughness=0.04)
else:
    mat_gray = get_or_create_material("MatGray", COLORS["gray"], roughness=0.58, specular=0.10)
    mat_blue = get_or_create_material("MatBlue", COLORS["blue"], roughness=0.46, specular=0.18)
    mat_orange = get_or_create_material("MatOrange", COLORS["orange"], roughness=0.44, specular=0.16)
    mat_purple = get_or_create_material("MatPurple", COLORS["purple"], roughness=0.48, specular=0.18)
    mat_edge = get_or_create_material("MatEdge", COLORS["edge"], roughness=0.55, specular=0.04)

group_to_mat = {
    "gray": mat_gray,
    "blue": mat_blue,
    "orange": mat_orange,
    "purple": mat_purple,
    "center": mat_purple,  # 兼容有些导出写 center
}

created_objs = []
blue_node_ids = choose_blue_nodes_by_degree(nodes, edges, ARGS.blue_ratio)

# 先画 edges（在后面球体更显眼）
for e in edges:
    u = int(e["source"])
    v = int(e["target"])
    if u not in node_pos or v not in node_pos:
        continue
    obj = add_cylinder_between(node_pos[u], node_pos[v], EDGE_RADIUS, mat=mat_edge, name=f"E_{u}_{v}")
    if obj:
        created_objs.append(obj)

# 再画 nodes
for n in nodes:
    nid = int(n["id"])
    xyz = node_pos[nid]
    if ARGS.use_group_colors:
        g = node_group.get(nid, "gray")
        mat = group_to_mat.get(g, mat_gray)
    else:
        mat = mat_blue if nid in blue_node_ids else mat_orange
    obj = add_sphere(xyz, NODE_RADIUS, mat=mat, name=f"N_{nid}")
    created_objs.append(obj)

# 整体居中缩放，方便构图
frame_all_objects(created_objs, target_extent=1.85)

# 灯光和相机
camera = setup_camera_and_lights(ARGS)
fit_camera_to_objects(camera, created_objs, margin=ARGS.camera_fit_margin)

# 打印拟合范围，方便排查是否完整入镜
xs, ys = [], []
for obj in created_objs:
    for corner in obj.bound_box:
        p = obj.matrix_world @ Vector(corner)
        c = world_to_camera_view(bpy.context.scene, camera, p)
        xs.append(c.x)
        ys.append(c.y)
print(f"Camera fit range x:[{min(xs):.3f}, {max(xs):.3f}] y:[{min(ys):.3f}, {max(ys):.3f}]")

print(f"Imported graph: {len(nodes)} nodes, {len(edges)} edges")

scene = bpy.context.scene
scene.render.filepath = str(out_path)
bpy.ops.render.render(write_still=True)
print(f"Saved render to: {out_path}")
