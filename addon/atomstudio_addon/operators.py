from __future__ import annotations

import traceback

import bpy


class ATOMSTUDIO_OT_import_structure(bpy.types.Operator):
    bl_idname = "atomstudio.import_structure"
    bl_label = "Import ASE Structure"

    def execute(self, context):
        try:
            from atomstudio.io.ase_loader import load_structure
            from atomstudio.backend.blender.scene_writer import BlenderSceneWriter
            from atomstudio.config import RenderJobConfig
            from atomstudio.scene.builder import build_render_scene
            from atomstudio.scene.materials.registry import MaterialRegistry
            from atomstudio.style.registry import get_scene_style
            from atomstudio.style.resolver import resolve_style_bundle

            scene = context.scene
            if not scene.atomstudio_input:
                self.report({"ERROR"}, "Input path is empty")
                return {"CANCELLED"}

            structure = load_structure(scene.atomstudio_input, "last")
            payload = {
                "id": "addon_import",
                "input": {"path": scene.atomstudio_input, "frames": "last"},
                "output": {"path": scene.atomstudio_output or "render.png"},
                "style": {"scene_style": scene.atomstudio_preset},
                "render": {"output_path": scene.atomstudio_output or "render.png"},
            }
            cfg = RenderJobConfig.from_dict(payload)
            style_bundle = resolve_style_bundle(cfg.style, get_scene_style(cfg.style.scene_style))
            render_scene = build_render_scene(structure, cfg)
            _, stats, _ = BlenderSceneWriter(
                cfg,
                registry=MaterialRegistry(),
                default_material_pipeline=str(style_bundle.material_style.pipeline),
                default_material_style_name=str(style_bundle.material_style_name),
            ).write(render_scene)
            self.report({"INFO"}, f"Imported atoms={stats['atoms']} bonds={stats['bonds']}")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            traceback.print_exc()
            return {"CANCELLED"}


class ATOMSTUDIO_OT_apply_style(bpy.types.Operator):
    bl_idname = "atomstudio.apply_style"
    bl_label = "Apply Style"

    def execute(self, context):
        try:
            from atomstudio.style.registry import get_scene_style

            style_name = context.scene.atomstudio_preset
            scene_style = get_scene_style(style_name)
            for mat in bpy.data.materials:
                if not mat.use_nodes:
                    continue
                bsdf = next((node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
                if bsdf:
                    bsdf.inputs["Roughness"].default_value = scene_style.material_style.atom_default.roughness
            self.report({"INFO"}, f"Applied style: {style_name}")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            traceback.print_exc()
            return {"CANCELLED"}


class ATOMSTUDIO_OT_render_current(bpy.types.Operator):
    bl_idname = "atomstudio.render_current"
    bl_label = "Render Current Scene"

    def execute(self, context):
        try:
            from atomstudio.config import RenderJobConfig

            output = context.scene.atomstudio_output or "render.png"
            payload = {
                "id": "addon_render",
                "input": {"path": context.scene.atomstudio_input or "_", "frames": "last"},
                "output": {"path": output},
                "style": {"scene_style": context.scene.atomstudio_preset},
                "render": {"output_path": output},
            }
            cfg = RenderJobConfig.from_dict(payload)
            bpy.context.scene.render.filepath = str(cfg.output.path or output)
            bpy.ops.render.render(write_still=True)
            self.report({"INFO"}, f"Rendered: {bpy.context.scene.render.filepath}")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            traceback.print_exc()
            return {"CANCELLED"}


class ATOMSTUDIO_OT_render_batch_from_yaml(bpy.types.Operator):
    bl_idname = "atomstudio.render_batch_from_yaml"
    bl_label = "Render Batch from YAML"

    def execute(self, context):
        try:
            from atomstudio.render.pipeline import render_batch

            config = context.scene.atomstudio_config
            if not config:
                self.report({"ERROR"}, "Config path is empty")
                return {"CANCELLED"}
            result = render_batch(config)
            self.report({"INFO"}, f"Batch success={result.success}, jobs={len(result.reports)}")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            traceback.print_exc()
            return {"CANCELLED"}
