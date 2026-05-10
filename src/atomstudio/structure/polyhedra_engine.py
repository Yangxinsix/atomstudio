from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

from atomstudio.structure.polyhedron import Polyhedron

if TYPE_CHECKING:
    from atomstudio.config import BondingConfig, PolyhedraConfig, PolyhedraRuleConfig
    from atomstudio.structure.structure import Structure


class PolyhedraEngine:
    def compute(
        self,
        structure: "Structure",
        polyhedra_config: "PolyhedraConfig",
        bonding_config: "BondingConfig | None" = None,
    ) -> list[Polyhedron]:
        _ = bonding_config
        if not polyhedra_config.rules:
            return []
        return self._compute_pairwise(structure, polyhedra_config)

    def _compute_pairwise(
        self,
        structure: "Structure",
        polyhedra_config: "PolyhedraConfig",
    ) -> list[Polyhedron]:
        out: list[Polyhedron] = []
        poly_id = 0
        for rule_index, rule in enumerate(polyhedra_config.rules):
            center_symbols = set(rule.center_symbols)
            neighbor_symbols = set(rule.neighbor_symbols)
            for center_idx, center_atom in enumerate(structure.atoms):
                if center_atom.symbol not in center_symbols:
                    continue
                data = self._neighbors_for_center(center_idx, structure, rule, neighbor_symbols)
                if data is None:
                    continue
                vertices, neighbor_indices = data
                out.append(
                    Polyhedron(
                        id=int(poly_id),
                        center=int(center_idx),
                        center_symbol=str(center_atom.symbol),
                        vertex_positions=vertices,
                        neighbor_indices=neighbor_indices,
                        neighbor_offsets=[(0, 0, 0)] * len(neighbor_indices),
                        metadata={"rule_index": int(rule_index)},
                        style=rule.style,
                        material=rule.material,
                        color=rule.color,
                        show_edges=bool(rule.show_edges),
                        edge_radius=rule.edge_radius,
                        edge_color=rule.edge_color,
                    )
                )
                poly_id += 1
        return out

    def _neighbors_for_center(
        self,
        center_idx: int,
        structure: "Structure",
        rule: "PolyhedraRuleConfig",
        neighbor_symbols: set[str],
    ) -> tuple[list[tuple[float, float, float]], list[int]] | None:
        center = structure.atoms[int(center_idx)]
        p1 = center.position
        candidates: list[tuple[float, int, tuple[float, float, float]]] = []
        for atom in structure.atoms:
            if int(atom.index) == int(center_idx):
                continue
            if atom.symbol not in neighbor_symbols:
                continue
            p2 = atom.position
            d = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
            if rule.max_distance is not None and d > float(rule.max_distance):
                continue
            candidates.append((float(d), int(atom.index), (float(p2[0]), float(p2[1]), float(p2[2]))))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        if rule.max_neighbors is not None:
            candidates = candidates[: max(0, int(rule.max_neighbors))]
        min_neighbors = max(4, int(rule.min_neighbors))
        if len(candidates) < min_neighbors:
            return None

        return [item[2] for item in candidates], [item[1] for item in candidates]
