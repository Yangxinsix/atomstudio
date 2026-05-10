from __future__ import annotations

from atomstudio.config import HanddrawnStyleConfig, RenderJobConfig
from atomstudio.scene.materials.specs import HandDrawnMaterialSpec, handdrawn_spec_from_any
from atomstudio.style.outline_style import OutlineRoleStyle, OutlineStyle

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


class OutlineBuilder:
    def __init__(
        self,
        *,
        style_name: str = "default",
        material_pipeline: str = "principled",
        structure_representation: str = "auto",
        outline: OutlineStyle | None = None,
        handdrawn_spec: HandDrawnMaterialSpec | None = None,
    ) -> None:
        self.style_name = str(style_name)
        self.material_pipeline = str(material_pipeline)
        self.structure_representation = str(structure_representation)
        self.outline = OutlineStyle.from_any(outline)
        self.handdrawn_spec = handdrawn_spec

    @classmethod
    def from_cfg(
        cls,
        cfg: RenderJobConfig,
        *,
        style_name: str,
        material_pipeline: str,
        outline: OutlineStyle | None = None,
        handdrawn: HanddrawnStyleConfig | None = None,
    ) -> "OutlineBuilder":
        return cls(
            style_name=style_name,
            material_pipeline=material_pipeline,
            structure_representation=cfg.structure.representation,
            outline=outline,
            handdrawn_spec=handdrawn_spec_from_any(handdrawn if handdrawn is not None else cfg.style.handdrawn),
        )

    def apply(self) -> None:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")

        scene = bpy.context.scene
        scene.render.use_freestyle = bool(self.outline.enabled)
        if not scene.render.use_freestyle:
            return

        view_layer = bpy.context.view_layer
        fs = view_layer.freestyle_settings
        self._clear_linesets(fs)

        if self.material_pipeline == "handdrawn":
            hd = self.handdrawn_spec or HandDrawnMaterialSpec()
            representation = _effective_representation(self.style_name, self.structure_representation)
            factor = 0.82 if representation == "space_filling" else 1.0
            atom_groups = []
            for coll in bpy.data.collections:
                if str(coll.name).startswith("AtomOutlineGroup_") and len(coll.objects) > 0:
                    atom_groups.append(coll)
            atom_groups.sort(key=lambda c: str(c.name))
            if atom_groups:
                for i, coll in enumerate(atom_groups):
                    color_raw = coll.get("atomstudio_outline_color", list(_shade(self.outline.color, 0.96)))
                    secondary_color_raw = coll.get(
                        "atomstudio_outline_secondary_color",
                        list(hd.outline_secondary_color),
                    )
                    color = tuple(float(v) for v in color_raw[:4])
                    secondary_color = tuple(float(v) for v in secondary_color_raw[:4])
                    thickness = max(
                        0.4,
                        float(
                            coll.get(
                                "atomstudio_outline_thickness",
                                max(0.4, float(hd.outline_molecule) * factor),
                            )
                        ),
                    )
                    secondary_thickness = max(
                        0.0,
                        float(
                            coll.get(
                                "atomstudio_outline_secondary_thickness",
                                max(0.0, float(hd.outline_secondary_thickness) * factor),
                            )
                        ),
                    )
                    ignore_occlusion = bool(
                        coll.get("atomstudio_outline_ignore_occlusion", self.outline.atoms.ignore_occlusion)
                    )
                    self._add_lineset(
                        fs,
                        name=f"AtomOutline_{i}",
                        color=color,
                        thickness=thickness,
                        collection_name=str(coll.name),
                        select_by_visibility=not ignore_occlusion,
                    )
                    if secondary_thickness > 0.0:
                        self._add_lineset(
                            fs,
                            name=f"AtomOutline_{i}_Secondary",
                            color=secondary_color,
                            thickness=max(0.15, secondary_thickness),
                            collection_name=str(coll.name),
                            select_by_visibility=not ignore_occlusion,
                        )
            elif self.outline.atoms.enabled:
                atom_color = self.outline.atoms.color or _shade(self.outline.color, 0.96)
                atom_thickness = (
                    float(self.outline.atoms.thickness)
                    if self.outline.atoms.thickness is not None
                    else float(hd.outline_molecule)
                )
                self._add_lineset(
                    fs,
                    name="AtomOutline",
                    color=atom_color,
                    thickness=max(0.4, atom_thickness * factor),
                    collection_name="Atoms",
                    select_by_visibility=not bool(self.outline.atoms.ignore_occlusion),
                )
                self._add_secondary_outline(fs, "AtomOutline", hd, factor, "Atoms", role_style=self.outline.atoms)
            if self.outline.bonds.enabled:
                bond_color = self.outline.bonds.color or _shade(self.outline.color, 0.85)
                bond_thickness = (
                    float(self.outline.bonds.thickness)
                    if self.outline.bonds.thickness is not None
                    else float(hd.outline_bond)
                )
                self._add_lineset(
                    fs,
                    name="BondOutline",
                    color=bond_color,
                    thickness=max(0.25, bond_thickness * factor),
                    collection_name="Bonds",
                    select_by_visibility=not bool(self.outline.bonds.ignore_occlusion),
                )
                self._add_secondary_outline(fs, "BondOutline", hd, factor, "Bonds", role_style=self.outline.bonds)
            if self.outline.cell.enabled:
                cell_color = self.outline.cell.color or _shade(self.outline.color, 0.82)
                cell_thickness = (
                    float(self.outline.cell.thickness)
                    if self.outline.cell.thickness is not None
                    else float(hd.outline_bond)
                )
                self._add_lineset(
                    fs,
                    name="CellOutline",
                    color=cell_color,
                    thickness=max(0.25, cell_thickness * factor),
                    collection_name="Cell",
                    select_by_visibility=not bool(self.outline.cell.ignore_occlusion),
                )
                self._add_secondary_outline(fs, "CellOutline", hd, factor, "Cell", role_style=self.outline.cell)
            return

        if self._has_role_outline_overrides():
            self._add_role_linesets(fs)
            return
        self._add_lineset(
            fs,
            name="DefaultOutline",
            color=self.outline.color,
            thickness=max(0.5, float(self.outline.thickness)),
            collection_name=None,
            select_by_visibility=True,
        )

    def _clear_linesets(self, fs) -> None:
        while len(fs.linesets) > 0:
            fs.linesets.remove(fs.linesets[0])

    def _add_lineset(
        self,
        fs,
        name: str,
        color: tuple[float, float, float, float],
        thickness: float,
        collection_name: str | None,
        select_by_visibility: bool = True,
    ) -> None:
        bpy.ops.scene.freestyle_lineset_add()
        line_set = fs.linesets[-1]
        line_set.name = name
        line_set.select_silhouette = True
        line_set.select_border = True
        line_set.select_crease = False
        line_set.select_ridge_valley = False
        line_set.select_external_contour = True
        line_set.select_by_visibility = bool(select_by_visibility)
        if bool(select_by_visibility) and hasattr(line_set, "visibility"):
            line_set.visibility = "VISIBLE"

        if collection_name:
            collection = bpy.data.collections.get(collection_name)
            if collection is None:
                collection = bpy.data.collections.new(collection_name)
                root = bpy.context.scene.collection
                if root.children.get(collection.name) is None:
                    root.children.link(collection)
            line_set.select_by_collection = True
            line_set.collection = collection

        line_style = line_set.linestyle
        line_style.color = color[:3]
        line_style.thickness = float(thickness)

    def _add_secondary_outline(
        self,
        fs,
        base_name: str,
        hd: HandDrawnMaterialSpec,
        factor: float,
        collection_name: str,
        role_style: OutlineRoleStyle | None = None,
    ) -> None:
        t = float(hd.outline_secondary_thickness)
        c = tuple(float(v) for v in hd.outline_secondary_color)
        if role_style is not None and role_style.secondary_thickness is not None:
            t = float(role_style.secondary_thickness)
        if role_style is not None and role_style.secondary_color is not None:
            c = tuple(float(v) for v in role_style.secondary_color)
        t = max(0.0, t)
        if t <= 0.0:
            return
        self._add_lineset(
            fs,
            name=f"{base_name}Secondary",
            color=c,
            thickness=max(0.15, t * factor),
            collection_name=collection_name,
            select_by_visibility=not bool(role_style and role_style.ignore_occlusion),
        )

    def _has_role_outline_overrides(self) -> bool:
        default_role = OutlineRoleStyle()
        return any(
            role != default_role
            for role in (self.outline.atoms, self.outline.bonds, self.outline.cell)
        )

    def _add_role_linesets(self, fs) -> None:
        role_specs = (
            ("AtomOutline", "Atoms", self.outline.atoms),
            ("BondOutline", "Bonds", self.outline.bonds),
            ("CellOutline", "Cell", self.outline.cell),
        )
        for name, collection_name, role in role_specs:
            if not role.enabled:
                continue
            color = self.outline.color if role.color is None else role.color
            thickness = self.outline.thickness if role.thickness is None else role.thickness
            self._add_lineset(
                fs,
                name=name,
                color=tuple(float(v) for v in color),
                thickness=max(0.5, float(thickness)),
                collection_name=collection_name,
                select_by_visibility=not bool(role.ignore_occlusion),
            )


def _shade(color: tuple[float, float, float, float], scale: float) -> tuple[float, float, float, float]:
    s = float(scale)
    return (
        max(0.0, min(1.0, float(color[0]) * s)),
        max(0.0, min(1.0, float(color[1]) * s)),
        max(0.0, min(1.0, float(color[2]) * s)),
        float(color[3]),
    )


def _effective_representation(style_name: str, representation: str | None) -> str:
    rep = str(representation or "auto").strip().lower()
    if rep not in {"auto", "space_filling", "ball_stick"}:
        rep = "auto"
    if rep == "auto":
        return "space_filling" if str(style_name).lower() == "handdrawn" else "ball_stick"
    return rep
