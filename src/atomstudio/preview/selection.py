from __future__ import annotations

from typing import Any

from atomstudio.preview.types import PreviewAtomRecord, PreviewBondRecord, PreviewScene, PreviewSelection, RenderAtom, RenderBond, PreviewRenderScene


def lookup_atom(scene: PreviewRenderScene | None, atom_index: int) -> RenderAtom | None:
    if scene is None:
        return None
    for atom in scene.atoms:
        if int(atom.index) == int(atom_index):
            return atom
    return None


def lookup_bond(scene: PreviewRenderScene | None, bond_index: int) -> RenderBond | None:
    if scene is None:
        return None
    for bond in scene.bonds:
        if int(bond.index) == int(bond_index):
            return bond
    return None


def selection_payload(
    scene: PreviewRenderScene | None,
    selection: PreviewSelection | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if scene is None or selection is None:
        return None, None
    if selection.kind == "atom":
        atom = lookup_atom(scene, selection.index)
        if atom is None:
            return None, None
        selected_object = {
            "kind": "atom",
            "index": int(atom.index),
            "symbol": atom.symbol,
            "position": list(atom.position),
            "radius": float(atom.radius),
        }
        payload = atom.material.to_dict() if atom.material is not None else {}
        return selected_object, payload
    bond = lookup_bond(scene, selection.index)
    if bond is None:
        return None, None
    selected_object = {
        "kind": "bond",
        "index": int(bond.index),
        "a": int(bond.a),
        "b": int(bond.b),
        "order": int(bond.order),
        "bond_type": bond.bond_type,
    }
    material = bond.material_uniform or bond.material_left or bond.material_right
    payload = material.to_dict() if material is not None else {}
    return selected_object, payload


def build_selection_payload(
    scene: PreviewScene | PreviewRenderScene | None,
    selection: PreviewSelection | None,
) -> dict[str, Any] | None:
    if scene is None or selection is None or selection.index is None:
        return None
    if isinstance(scene, PreviewRenderScene):
        return _build_render_scene_selection_payload(scene, selection)
    return _build_preview_scene_selection_payload(scene, selection)


def _build_render_scene_selection_payload(
    scene: PreviewRenderScene,
    selection: PreviewSelection,
) -> dict[str, Any] | None:
    if selection.kind == "atom":
        atom = lookup_atom(scene, selection.index)
        if atom is None:
            return None
        if atom.record is not None:
            return _payload_from_atom_record(atom.record)
        return {
            "object": {
                "index": int(atom.index),
                "symbol": atom.symbol,
                "atomic_number": None,
                "position": list(atom.position),
                "radius": float(atom.radius),
                "representation": atom.representation,
                "style": None,
                "tag": "",
            },
            "material": {} if atom.material is None else atom.material.to_dict(),
            "metadata": {"selection_kind": "atom", "selection_index": int(atom.index)},
        }
    bond = lookup_bond(scene, selection.index)
    if bond is None:
        return None
    if bond.record is not None:
        return _payload_from_bond_record(bond.record)
    material = bond.material_uniform or bond.material_left or bond.material_right
    return {
        "object": {
            "id": int(bond.index),
            "a": int(bond.a),
            "b": int(bond.b),
            "bond_type": bond.bond_type,
            "order": int(bond.order),
            "distance": bond.distance,
            "split_ratio": float(bond.split_ratio),
        },
        "material": {} if material is None else material.to_dict(),
        "metadata": {"selection_kind": "bond", "selection_index": int(bond.index)},
    }


def _build_preview_scene_selection_payload(
    scene: PreviewScene,
    selection: PreviewSelection,
) -> dict[str, Any] | None:
    if selection.kind == "atom":
        record = next((item for item in scene.atom_records if int(item.index) == int(selection.index)), None)
        return None if record is None else _payload_from_atom_record(record)
    record = next((item for item in scene.bond_records if int(item.id) == int(selection.index)), None)
    return None if record is None else _payload_from_bond_record(record)


def _payload_from_atom_record(record: PreviewAtomRecord) -> dict[str, Any]:
    metadata = dict(record.metadata)
    metadata.setdefault("selection_kind", "atom")
    metadata.setdefault("selection_index", int(record.index))
    return {
        "object": {
            "index": int(record.index),
            "symbol": str(record.symbol),
            "atomic_number": int(record.atomic_number),
            "position": list(record.position),
            "radius": float(record.radius),
            "representation": str(record.representation),
            "style": record.style,
            "tag": str(record.tag),
        },
        "material": record.material.to_dict(),
        "metadata": metadata,
    }


def _payload_from_bond_record(record: PreviewBondRecord) -> dict[str, Any]:
    metadata = dict(record.metadata)
    metadata.setdefault("selection_kind", "bond")
    metadata.setdefault("selection_index", int(record.id))
    return {
        "object": {
            "id": int(record.id),
            "a": int(record.a),
            "b": int(record.b),
            "bond_type": str(record.bond_type),
            "order": int(record.order),
            "distance": float(record.distance),
            "split_ratio": float(record.split_ratio),
        },
        "material": {
            "uniform": record.material_uniform.to_dict(),
            "left": record.material_left.to_dict(),
            "right": record.material_right.to_dict(),
        },
        "metadata": metadata,
    }


def cycle_atom_selection(scene: PreviewRenderScene | None, current: PreviewSelection | None, step: int) -> PreviewSelection | None:
    if scene is None or not scene.atoms:
        return None
    indices = [int(atom.index) for atom in scene.atoms]
    current_index = current.index if current is not None and current.kind == "atom" else None
    if current_index not in indices:
        resolved = indices[0] if step >= 0 else indices[-1]
        return PreviewSelection(kind="atom", index=resolved)
    offset = (indices.index(current_index) + step) % len(indices)
    return PreviewSelection(kind="atom", index=indices[offset])


def cycle_bond_selection(scene: PreviewRenderScene | None, current: PreviewSelection | None, step: int) -> PreviewSelection | None:
    if scene is None or not scene.bonds:
        return None
    indices = [int(bond.index) for bond in scene.bonds]
    current_index = current.index if current is not None and current.kind == "bond" else None
    if current_index not in indices:
        resolved = indices[0] if step >= 0 else indices[-1]
        return PreviewSelection(kind="bond", index=resolved)
    offset = (indices.index(current_index) + step) % len(indices)
    return PreviewSelection(kind="bond", index=indices[offset])


__all__ = [
    "build_selection_payload",
    "cycle_atom_selection",
    "cycle_bond_selection",
    "lookup_atom",
    "lookup_bond",
    "selection_payload",
]
