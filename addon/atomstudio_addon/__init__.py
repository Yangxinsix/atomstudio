bl_info = {
    "name": "atomstudio",
    "author": "atomstudio contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > AtomStudio",
    "description": "Render ASE structures with Blender",
    "category": "Import-Export",
}

import bpy

from .operators import (
    ATOMSTUDIO_OT_apply_style,
    ATOMSTUDIO_OT_import_structure,
    ATOMSTUDIO_OT_render_batch_from_yaml,
    ATOMSTUDIO_OT_render_current,
)
from .panel import ATOMSTUDIO_PT_main_panel


classes = (
    ATOMSTUDIO_OT_import_structure,
    ATOMSTUDIO_OT_apply_style,
    ATOMSTUDIO_OT_render_current,
    ATOMSTUDIO_OT_render_batch_from_yaml,
    ATOMSTUDIO_PT_main_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.atomstudio_input = bpy.props.StringProperty(name="Input Path", subtype="FILE_PATH")
    bpy.types.Scene.atomstudio_output = bpy.props.StringProperty(name="Output Path", subtype="FILE_PATH")
    bpy.types.Scene.atomstudio_config = bpy.props.StringProperty(name="Config Path", subtype="FILE_PATH")
    bpy.types.Scene.atomstudio_preset = bpy.props.EnumProperty(
        name="Preset",
        items=[
            ("default", "default", "Default glossy style"),
            ("darklab", "darklab", "Dark lab style"),
            ("monochrome", "monochrome", "Monochrome style"),
            ("ceramic_studio", "ceramic_studio", "Matte ceramic studio style"),
            ("glass_lab", "glass_lab", "Glass material on dark lab lighting"),
            ("clean_glossy", "clean_glossy", "Clean glossy material candidate"),
            ("porcelain", "porcelain", "Porcelain material candidate"),
            ("solid_glass", "solid_glass", "Solid glass material candidate"),
            ("studio_darklab", "studio_darklab", "Dark studio lighting with solid glass material"),
            ("jade", "jade", "Jade material candidate"),
            ("pearl", "pearl", "Pearl material candidate"),
            ("holographic", "holographic", "Holographic material candidate"),
            ("warm_clay", "warm_clay", "Warm clay material candidate"),
            ("handdrawn", "handdrawn", "Handdrawn cartoon style"),
            ("handdrawn_v2", "handdrawn_v2", "Handdrawn cartoon style v2"),
        ],
        default="default",
    )


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.atomstudio_input
    del bpy.types.Scene.atomstudio_output
    del bpy.types.Scene.atomstudio_config
    del bpy.types.Scene.atomstudio_preset
