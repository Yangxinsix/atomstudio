from __future__ import annotations

from atomstudio.structure.selectors import norm_symbol_pair


def normalize_pair_key(value: str) -> str:
    raw = str(value).strip()
    if "-" not in raw:
        raise ValueError("Pair key must be an element pair like 'O-Ti'.")
    left, right = raw.split("-", 1)
    return norm_symbol_pair(left.strip(), right.strip())


def normalize_pair_cutoffs(values: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, raw_cutoff in values.items():
        norm_key = normalize_pair_key(key)
        cutoff = float(raw_cutoff)
        if cutoff <= 0.0:
            raise ValueError(f"Pair cutoff for {norm_key} must be > 0.")
        out[norm_key] = cutoff
    return dict(sorted(out.items()))


# Pairwise bond distance cutoffs in Angstrom.
# Used as default bonding table when bonding.pair_cutoffs is not provided.
PRESET_PAIR_BOND_CUTOFFS: dict[str, float] = normalize_pair_cutoffs(
    {
        "H-H": 0.90,
        "C-H": 1.25,
        "N-H": 1.20,
        "O-H": 1.15,
        "C-C": 1.80,
        "C-N": 1.75,
        "C-O": 1.75,
        "N-N": 1.70,
        "N-O": 1.70,
        "O-O": 1.65,
        "C-S": 2.05,
        "C-P": 2.05,
        "Si-O": 2.05,
        "S-O": 1.95,
        "P-O": 1.90,
        "Na-O": 2.75,
        "K-O": 3.10,
        "Mg-O": 2.35,
        "Ca-O": 2.90,
        "Ti-O": 2.35,
        "Sr-O": 3.20,
        "Fe-O": 2.30,
        "Cu-O": 2.20,
        "Zn-O": 2.20,
        "Cl-H": 1.45,
        "Cl-C": 2.00,
    }
)
