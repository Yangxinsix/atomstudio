import bpy


class ATOMSTUDIO_PT_main_panel(bpy.types.Panel):
    bl_label = "AtomStudio"
    bl_idname = "VIEW3D_PT_atomstudio"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AtomStudio"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        box = layout.box()
        box.label(text="Single Structure")
        box.prop(scene, "atomstudio_input")
        box.prop(scene, "atomstudio_output")
        box.prop(scene, "atomstudio_preset")
        box.operator("atomstudio.import_structure", text="Import Structure")
        box.operator("atomstudio.apply_style", text="Apply Style")
        box.operator("atomstudio.render_current", text="Render Current")

        box = layout.box()
        box.label(text="Batch")
        box.prop(scene, "atomstudio_config")
        box.operator("atomstudio.render_batch_from_yaml", text="Render Batch from YAML")
