__all__ = [
    "RenderScene",
    "SceneAtom",
    "SceneBond",
    "SceneBondSegment",
    "SceneBounds",
    "SceneBuilder",
    "SceneBuildResult",
    "SceneCamera",
    "SceneCellEdge",
    "SceneLight",
    "ScenePolyhedron",
    "build_render_scene",
    "resolve_scene_camera",
    "resolve_scene_lights",
]


def __getattr__(name: str):
    if name in {
        "RenderScene",
        "SceneAtom",
        "SceneBond",
        "SceneBondSegment",
        "SceneBounds",
        "SceneCamera",
        "SceneCellEdge",
        "SceneLight",
        "ScenePolyhedron",
    }:
        from atomstudio.scene import model as _model

        return getattr(_model, name)
    if name in {"SceneBuilder", "SceneBuildResult", "build_render_scene"}:
        from atomstudio.scene import builder as _builder

        return getattr(_builder, name)
    if name == "resolve_scene_camera":
        from atomstudio.scene.camera_resolver import resolve_scene_camera

        return resolve_scene_camera
    if name == "resolve_scene_lights":
        from atomstudio.scene.light_resolver import resolve_scene_lights

        return resolve_scene_lights
    raise AttributeError(name)
