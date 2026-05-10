from atomstudio.config import MaterialPolicy
from atomstudio.scene.materials.specs import HandDrawnMaterialSpec, MaterialSpec


def test_atom_material_priority_index_rule_element():
    policy = MaterialPolicy.from_dict(
        {
            "atom_defaults": {
                "O": {"color": [1.0, 0.0, 0.0, 1.0]},
            },
            "atom_rules": [
                {
                    "selector": {"symbol": "O", "z_range": [0.0, 5.0]},
                    "material": {"color": [0.0, 0.0, 1.0, 1.0]},
                }
            ],
            "atom_overrides": {
                "0": {"color": [0.0, 1.0, 0.0, 1.0]},
            },
        }
    )

    fallback = MaterialSpec(color=(0.5, 0.5, 0.5, 1.0))

    m0 = policy.resolve_atom_material(0, "O", (0.0, 0.0, 10.0), "", fallback, fallback)
    assert m0.color[:3] == (0.0, 1.0, 0.0)

    m1 = policy.resolve_atom_material(1, "O", (0.0, 0.0, 1.0), "", fallback, fallback)
    assert m1.color[:3] == (0.0, 0.0, 1.0)

    m2 = policy.resolve_atom_material(2, "O", (0.0, 0.0, 10.0), "", fallback, fallback)
    assert m2.color[:3] == (1.0, 0.0, 0.0)


def test_bond_material_normalized_pair_and_index_override():
    policy = MaterialPolicy.from_dict(
        {
            "bond_defaults": {
                "O-C": {"color": [1.0, 1.0, 0.0, 1.0]},
            },
            "bond_overrides": {
                "by_index": {
                    "0": {"color": [0.0, 1.0, 1.0, 1.0]},
                },
                "by_pair": {
                    "3-7": {"color": [1.0, 0.0, 1.0, 1.0]},
                },
            },
        }
    )

    fallback = MaterialSpec(color=(0.5, 0.5, 0.5, 1.0))

    m0 = policy.resolve_bond_material(0, 1, 2, "C", "O", 1.2, fallback, fallback)
    assert m0.color[:3] == (0.0, 1.0, 1.0)

    m1 = policy.resolve_bond_material(2, 7, 3, "Mg", "O", 2.0, fallback, fallback)
    assert m1.color[:3] == (1.0, 0.0, 1.0)

    m2 = policy.resolve_bond_material(3, 9, 11, "C", "O", 1.2, fallback, fallback)
    assert m2.color[:3] == (1.0, 1.0, 0.0)


def test_material_policy_accepts_handdrawn_fields():
    policy = MaterialPolicy.from_dict(
        {
            "atom_defaults": {
                "O": {"color": [0.3, 0.4, 0.5, 1.0], "shadow_area": 0.61},
            }
        }
    )
    fallback = MaterialSpec(color=(0.5, 0.5, 0.5, 1.0))
    mat = policy.resolve_atom_material(0, "O", (0.0, 0.0, 0.0), "", fallback, fallback)
    assert isinstance(mat, HandDrawnMaterialSpec)
    assert mat.shadow_area == 0.61


def test_material_policy_partially_overrides_principled_fallback():
    policy = MaterialPolicy.from_dict(
        {
            "atom_defaults": {
                "Cu": {"color": [0.14, 0.52, 0.96, 1.0]},
            }
        }
    )
    base_material = MaterialSpec(
        color=(0.6, 0.6, 0.6, 1.0),
        roughness=0.10,
        specular=0.65,
        metallic=0.0,
        ior=1.52,
        coat=0.65,
        coat_roughness=0.05,
        specular_tint=0.10,
        alpha=0.88,
    )
    fallback = MaterialSpec(color=(0.9, 0.9, 0.9, 1.0), roughness=0.95, specular=0.05, alpha=0.33)
    mat = policy.resolve_atom_material(0, "Cu", (0.0, 0.0, 0.0), "", base_material, fallback)
    assert isinstance(mat, MaterialSpec)
    assert mat.color == (0.14, 0.52, 0.96, 1.0)
    assert mat.roughness == 0.10
    assert mat.specular == 0.65
    assert mat.ior == 1.52
    assert mat.coat == 0.65
    assert mat.alpha == 0.88


def test_material_policy_partially_overrides_handdrawn_fallback():
    policy = MaterialPolicy.from_dict(
        {
            "atom_defaults": {
                "O": {"color": [0.2, 0.3, 0.4, 1.0]},
            }
        }
    )
    base_material = HandDrawnMaterialSpec(
        color=(0.7, 0.7, 0.7, 1.0),
        roughness=0.82,
        specular=0.04,
        shadow_area=0.61,
        highlight_strength=0.33,
    )
    fallback = HandDrawnMaterialSpec(
        color=(0.9, 0.9, 0.9, 1.0),
        roughness=0.2,
        specular=0.2,
        shadow_area=0.1,
        highlight_strength=0.1,
    )
    mat = policy.resolve_atom_material(0, "O", (0.0, 0.0, 0.0), "", base_material, fallback)
    assert isinstance(mat, HandDrawnMaterialSpec)
    assert mat.color == (0.2, 0.3, 0.4, 1.0)
    assert mat.roughness == 0.82
    assert mat.specular == 0.04
    assert mat.shadow_area == 0.61
    assert mat.highlight_strength == 0.33
