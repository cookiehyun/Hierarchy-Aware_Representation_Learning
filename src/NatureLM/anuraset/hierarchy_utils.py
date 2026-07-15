"""
AnuraSet Hierarchy Utilities for Tree-Triplet Loss

Manages the taxonomic hierarchy: Order → Family → Genus → Species
Provides tree distance (ψ) calculation for hierarchy-aware triplet sampling.

✏️ 수정: get_tree_distance가 "분기 레벨"(0,1,2,3) 대신 논문 정의 그대로
   "tree에서의 진짜 edge 거리"(0,2,4,6)를 반환하도록 수정함.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List


class AnuraSetHierarchy:
    """
    Hierarchy levels (top-to-bottom):
    0: Order (e.g., "Anura")
    1: Family (e.g., "Hylidae")
    2: Genus (e.g., "Boana")
    3: Species (e.g., "Boana albopunctata")
    """

    def __init__(self, metadata_path: str):
        self.meta = pd.read_csv(metadata_path)

        self.hierarchy = self._build_hierarchy()

        self.order = "Anura"
        self.order_id = 0

        self.families = sorted(self.meta['family'].unique())
        self.genera = sorted(self.meta['genus'].unique())
        self.species = sorted(self.meta['species'].unique())

        print(f"[Hierarchy] Order: {self.order}")
        print(f"[Hierarchy] Families: {len(self.families)}")
        print(f"[Hierarchy] Genera: {len(self.genera)}")
        print(f"[Hierarchy] Species: {len(self.species)}")

    def _build_hierarchy(self) -> Dict:
        hierarchy = {}
        for _, row in self.meta.iterrows():
            order = "Anura"
            family = row['family']
            genus = row['genus']
            species = row['species']
            key = (order, family, genus, species)
            if key not in hierarchy:
                hierarchy[key] = {
                    'order': order, 'family': family,
                    'genus': genus, 'species': species
                }
        return hierarchy

    def get_tree_distance(self, species_i: str, species_j: str,
                         family_i: str, family_j: str,
                         genus_i: str, genus_j: str) -> int:
        """
        진짜 edge-distance (Order -> Family -> Genus -> Species, depth=3):
        0: same species
        2: same genus, different species   (genus까지 올라갔다 내려옴: 1+1)
        4: same family, different genus    (family까지 올라갔다 내려옴: 2+2)
        6: different family                (order까지 올라갔다 내려옴: 3+3)
        """
        if species_i == species_j:
            return 0
        elif genus_i == genus_j:
            return 2
        elif family_i == family_j:
            return 4
        else:
            return 6

    def batch_tree_distance(self, species_ids: torch.Tensor,
                           family_ids: torch.Tensor,
                           genus_ids: torch.Tensor,
                           species_enc, family_enc, genus_enc) -> torch.Tensor:
        batch_size = species_ids.shape[0]
        distances = torch.zeros(batch_size, batch_size, dtype=torch.long)

        species_names = species_enc.inverse_transform(species_ids.cpu().numpy())
        family_names = family_enc.inverse_transform(family_ids.cpu().numpy())
        genus_names = genus_enc.inverse_transform(genus_ids.cpu().numpy())

        for i in range(batch_size):
            for j in range(batch_size):
                dist = self.get_tree_distance(
                    species_names[i], species_names[j],
                    family_names[i], family_names[j],
                    genus_names[i], genus_names[j]
                )
                distances[i, j] = dist

        return distances


def compute_hierarchy_aware_margin(tree_dist_pos: int, tree_dist_neg: int,
                                   max_depth: int = 3) -> float:
    """
    From HSSN paper (Eq. 8): m = m_ε + 0.5 * m_τ
    m_τ = (ψ(neg) - ψ(pos)) / (2*D)
    """
    m_epsilon = 0.1
    m_tau = (tree_dist_neg - tree_dist_pos) / (2.0 * max_depth)
    margin = m_epsilon + 0.5 * m_tau
    return margin