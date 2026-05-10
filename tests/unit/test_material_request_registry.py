from __future__ import annotations

from atomstudio.scene.materials.registry import MaterialRegistry
from atomstudio.scene.materials.request import MaterialRequest
from atomstudio.scene.materials.specs import HandDrawnMaterialSpec, MaterialSpec, material_from_dict


def test_material_request_factories_set_pipeline():
    p = MaterialRequest.principled(name="A", material=MaterialSpec(), role="atom", style_name="default")
    h = MaterialRequest.handdrawn(name="B", material=HandDrawnMaterialSpec(), role="atom", style_name="handdrawn")
    assert p.pipeline == "principled"
    assert h.pipeline == "handdrawn"


def test_registry_cache_key_changes_with_handdrawn_fields():
    req_a = MaterialRequest.handdrawn(
        name="Atom_C",
        material=HandDrawnMaterialSpec(shadow_area=0.2),
        role="atom",
        style_name="handdrawn",
    )
    req_b = MaterialRequest.handdrawn(
        name="Atom_C",
        material=HandDrawnMaterialSpec(shadow_area=0.5),
        role="atom",
        style_name="handdrawn",
    )
    assert MaterialRegistry._cache_key(req_a) != MaterialRegistry._cache_key(req_b)


def test_material_from_dict_parses_handdrawn_payload():
    parsed = material_from_dict({"color": [0.2, 0.3, 0.4, 1.0], "shadow_strength": 0.66})
    assert isinstance(parsed, HandDrawnMaterialSpec)
    assert parsed.shadow_strength == 0.66
