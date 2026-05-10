from __future__ import annotations

from pathlib import Path
from typing import Any

from atomstudio.backend.blender.collections import ensure_collection
from atomstudio.backend.blender.geometry import collect_bbox_points, set_cylinder_between, tuple3
from atomstudio.backend.blender.material_adapter import BlenderMaterialAdapter, scene_value
from atomstudio.backend.blender.scene_setup import apply_camera_lights_effects, apply_render_environment, write_scene_metadata
from atomstudio.config import RenderJobConfig
from atomstudio.scene.materials.registry import MaterialRegistry
from atomstudio.structure.atom import Atom
from atomstudio.structure.bond import add_cylinder_between
from atomstudio.style.registry import get_scene_style
from atomstudio.style.resolver import resolve_style_bundle

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


class BlenderAnimationRenderer:
    """Render frame sequences by updating existing Blender objects."""

    def __init__(self, cfg: RenderJobConfig) -> None:
        self.cfg = cfg
        self.scene_style = get_scene_style(cfg.style.scene_style)
        self.style_bundle = resolve_style_bundle(cfg.style, self.scene_style)
        self.registry = MaterialRegistry()
        self.materials = BlenderMaterialAdapter(self.registry)
        self.collections: dict[str, Any] = {}
        self.atom_objects: dict[int, Any] = {}
        self.bond_objects: dict[int, list[Any]] = {}
        self.cell_objects: dict[int, Any] = {}
        self.objects: list[Any] = []

    def render_frames(self, frames: list[dict[str, Any]], *, output_dir: str) -> dict[str, Any]:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")
        if not frames:
            return {
                "success": False,
                "output_dir": str(output_dir),
                "outputs": [],
                "failed_frames": [],
                "frame_results": [],
                "message": "no animation frames were provided",
            }

        self._create_scene(frames[0]["render_scene"], RenderJobConfig.from_dict(frames[0]["config"]))
        outputs: list[str] = []
        failed_frames: list[int] = []
        frame_results: list[dict[str, Any]] = []
        for sequence_index, frame_payload in enumerate(frames):
            cfg = RenderJobConfig.from_dict(frame_payload["config"])
            scene = frame_payload["render_scene"]
            frame_index = int(scene_value(scene, "frame_index", sequence_index))
            try:
                self._update_scene(scene)
                bpy.context.scene.frame_set(sequence_index + 1)
                output_path = self._render_frame(cfg)
                result = {"success": True, "output_path": output_path, "frame_index": frame_index, "message": "ok"}
                outputs.append(output_path)
            except Exception as exc:  # pragma: no cover - executed inside Blender
                failed_frames.append(frame_index)
                result = {
                    "success": False,
                    "output_path": cfg.output.path or "",
                    "frame_index": frame_index,
                    "message": str(exc),
                }
            frame_results.append(result)

        failed_frames = sorted(set(failed_frames))
        return {
            "success": not failed_frames,
            "output_dir": str(output_dir),
            "outputs": outputs,
            "failed_frames": failed_frames,
            "frame_results": frame_results,
            "message": "ok" if not failed_frames else "some frames failed",
        }

    def _create_scene(self, scene: Any, cfg: RenderJobConfig) -> None:
        self._validate_stable_scene(scene)
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        self.collections = {
            "Atoms": ensure_collection("Atoms"),
            "Bonds": ensure_collection("Bonds"),
            "Cell": ensure_collection("Cell"),
        }

        background = scene_value(scene, "background", self.style_bundle.background)
        apply_render_environment(cfg, background=background)
        self._create_atoms(list(scene_value(scene, "atoms", []) or []))
        self._create_bonds(list(scene_value(scene, "bonds", []) or []))
        self._create_cell_edges(list(scene_value(scene, "cell_edges", []) or []))

        points = collect_bbox_points(self.objects)
        ground_spec = apply_camera_lights_effects(
            cfg,
            scene,
            points,
            registry=self.registry,
            style_bundle=self.style_bundle,
        )
        write_scene_metadata(scene, style_bundle=self.style_bundle, ground_spec=ground_spec)

    @staticmethod
    def _validate_stable_scene(scene: Any) -> None:
        if scene_value(scene, "polyhedra", []):
            raise ValueError("Blender animation rendering does not support polyhedra yet. Disable polyhedra for animation.")

    def _create_atoms(self, atoms: list[Any]) -> None:
        for raw_atom in atoms:
            index = int(scene_value(raw_atom, "index", 0))
            symbol = str(scene_value(raw_atom, "symbol", "X"))
            material = self.materials.resolve(
                scene_value(raw_atom, "material"),
                name=f"Atom_{index}_{symbol}",
                role="atom",
                pipeline=self.style_bundle.material_style.pipeline,
                style_name=self.style_bundle.material_style_name,
            )
            atom = Atom(
                index=index,
                atomic_number=int(scene_value(raw_atom, "atomic_number", 0)),
                symbol=symbol,
                position=tuple3(scene_value(raw_atom, "position")),
                radius=float(scene_value(raw_atom, "radius", 0.5)),
                segments=int(scene_value(raw_atom, "segments", self.cfg.structure.sphere_segments)),
                rings=int(scene_value(raw_atom, "rings", self.cfg.structure.sphere_rings)),
                style=scene_value(raw_atom, "style"),
                representation=scene_value(raw_atom, "representation"),
                tag=str(scene_value(raw_atom, "tag", "")),
                metadata=dict(scene_value(raw_atom, "metadata", {}) or {}),
            )
            obj = Atom.build(atom, material=material, collection=self.collections["Atoms"], name=f"Atom_{index}_{symbol}")
            self.atom_objects[index] = obj
            self.objects.append(obj)

    def _create_bonds(self, bonds: list[Any]) -> None:
        for raw_bond in bonds:
            bond_id = int(scene_value(raw_bond, "id", 0))
            objects: list[Any] = []
            for segment_index, segment in enumerate(scene_value(raw_bond, "segments", []) or []):
                material = self.materials.resolve(
                    scene_value(segment, "material"),
                    name=f"Bond_{bond_id}_segment_{segment_index}",
                    role="bond",
                    pipeline=self.style_bundle.material_style.pipeline,
                    style_name=self.style_bundle.material_style_name,
                )
                obj = add_cylinder_between(
                    tuple3(scene_value(segment, "start")),
                    tuple3(scene_value(segment, "end")),
                    float(scene_value(segment, "radius", scene_value(raw_bond, "radius", self.cfg.structure.bond_radius))),
                    material,
                    f"Bond_{bond_id}_segment_{segment_index}",
                    int(self.cfg.structure.bond_vertices),
                    collection=self.collections["Bonds"],
                )
                if obj is not None:
                    objects.append(obj)
                    self.objects.append(obj)
            self.bond_objects[bond_id] = objects

    def _create_cell_edges(self, edges: list[Any]) -> None:
        for raw_edge in edges:
            edge_index = int(scene_value(raw_edge, "index", len(self.cell_objects)))
            material = self.materials.resolve(
                scene_value(raw_edge, "material"),
                name=f"CellEdge_{edge_index}",
                role="cell",
                pipeline=self.style_bundle.material_style.pipeline,
                style_name=self.style_bundle.material_style_name,
            )
            obj = add_cylinder_between(
                tuple3(scene_value(raw_edge, "start")),
                tuple3(scene_value(raw_edge, "end")),
                float(scene_value(raw_edge, "radius", self.cfg.structure.cell_style.radius)),
                material,
                f"CellEdge_{edge_index}",
                12,
                collection=self.collections["Cell"],
            )
            if obj is not None:
                self.cell_objects[edge_index] = obj
                self.objects.append(obj)

    def _update_scene(self, scene: Any) -> None:
        self._validate_stable_scene(scene)
        atoms = {int(scene_value(atom, "index", -1)): atom for atom in scene_value(scene, "atoms", [])}
        if set(atoms) != set(self.atom_objects):
            raise ValueError("Animation atom topology changed; BlenderAnimationRenderer requires stable atom indices.")
        for index, atom in atoms.items():
            obj = self.atom_objects[index]
            radius = float(scene_value(atom, "radius", 0.5))
            obj.location = tuple3(scene_value(atom, "position"))
            obj.scale = (radius, radius, radius)

        bonds = {int(scene_value(bond, "id", -1)): bond for bond in scene_value(scene, "bonds", [])}
        if set(bonds) != set(self.bond_objects):
            raise ValueError("Animation bond topology changed; BlenderAnimationRenderer requires stable bond ids.")
        for bond_id, bond in bonds.items():
            segments = list(scene_value(bond, "segments", []) or [])
            objects = self.bond_objects[bond_id]
            if len(segments) != len(objects):
                raise ValueError(f"Animation bond {bond_id} segment count changed; BlenderAnimationRenderer requires stable bond segments.")
            for obj, segment in zip(objects, segments):
                set_cylinder_between(
                    obj,
                    tuple3(scene_value(segment, "start")),
                    tuple3(scene_value(segment, "end")),
                    float(scene_value(segment, "radius", scene_value(bond, "radius", self.cfg.structure.bond_radius))),
                )

        edges = {int(scene_value(edge, "index", -1)): edge for edge in scene_value(scene, "cell_edges", [])}
        if set(edges) != set(self.cell_objects):
            raise ValueError("Animation cell topology changed; BlenderAnimationRenderer requires stable cell edge indices.")
        for edge_index, edge in edges.items():
            set_cylinder_between(
                self.cell_objects[edge_index],
                tuple3(scene_value(edge, "start")),
                tuple3(scene_value(edge, "end")),
                float(scene_value(edge, "radius", self.cfg.structure.cell_style.radius)),
            )

        write_scene_metadata(scene, style_bundle=self.style_bundle)

    def _render_frame(self, cfg: RenderJobConfig) -> str:
        output_path = cfg.output.path
        if not output_path:
            raise ValueError("output.path must be set before rendering an animation frame")
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        return str(path)
