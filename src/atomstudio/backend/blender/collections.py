from __future__ import annotations

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


def ensure_collection(name: str):
    if bpy is None:
        return None
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    root = bpy.context.scene.collection
    if root.children.get(coll.name) is None:
        root.children.link(coll)
    return coll


def prepare_collections() -> dict[str, object]:
    if bpy is None:
        return {}
    return {
        "Atoms": ensure_collection("Atoms"),
        "Bonds": ensure_collection("Bonds"),
        "Polyhedra": ensure_collection("Polyhedra"),
        "PolyhedraEdges": ensure_collection("PolyhedraEdges"),
        "Cell": ensure_collection("Cell"),
    }
