"""
ESC-50 Hierarchy Utilities for Tree-Triplet Loss

Hierarchy: BGA -> Midcat -> Category

✏️ 수정: get_tree_distance가 "분기 레벨"(0,1,2,3) 대신 논문 정의 그대로
   "tree에서의 진짜 edge 거리"(0,2,4,6)를 반환하도록 수정함.
   (AnuraSet의 hierarchy_utils.py에서 한 수정과 동일한 이유 — Eq.8의
   margin 계산이 literal edge-distance를 전제로 하기 때문)
"""

import torch
import pandas as pd


class ESC50Hierarchy:
    """
    Hierarchy levels (top-to-bottom):
    0: BGA      (e.g., "biophony")
    1: Midcat   (e.g., "bird")
    2: Category (e.g., "chirping_birds")
    """

    def __init__(self, metadata_path: str):
        self.meta = pd.read_csv(metadata_path)

        self.bgas       = sorted(self.meta['bga_label'].unique())
        self.midcats    = sorted(self.meta['midcat'].unique())
        self.categories = sorted(self.meta['category'].unique())

        print(f"[Hierarchy] BGA: {len(self.bgas)}")
        print(f"[Hierarchy] Midcat: {len(self.midcats)}")
        print(f"[Hierarchy] Category: {len(self.categories)}")

    def get_tree_distance(self, category_i: str, category_j: str,
                           bga_i: str, bga_j: str,
                           midcat_i: str, midcat_j: str) -> int:
        """
        진짜 edge-distance (BGA -> Midcat -> Category, 2-level depth):
        0: same category
        2: same midcat, different category   (category 올라갔다 내려옴: 1+1)
        4: same bga, different midcat        (midcat까지 올라갔다 내려옴: 2+2)
        6: different bga                     (bga까지 올라갔다 내려옴: 3+3)
        """
        if category_i == category_j:
            return 0
        elif midcat_i == midcat_j:
            return 2
        elif bga_i == bga_j:
            return 4
        else:
            return 6

    def batch_tree_distance(self, category_ids: torch.Tensor,
                             bga_ids: torch.Tensor,
                             midcat_ids: torch.Tensor,
                             category_enc, bga_enc, midcat_enc) -> torch.Tensor:
        batch_size = category_ids.shape[0]
        distances = torch.zeros(batch_size, batch_size, dtype=torch.long)

        category_names = category_enc.inverse_transform(category_ids.cpu().numpy())
        bga_names      = bga_enc.inverse_transform(bga_ids.cpu().numpy())
        midcat_names   = midcat_enc.inverse_transform(midcat_ids.cpu().numpy())

        for i in range(batch_size):
            for j in range(batch_size):
                dist = self.get_tree_distance(
                    category_names[i], category_names[j],
                    bga_names[i], bga_names[j],
                    midcat_names[i], midcat_names[j]
                )
                distances[i, j] = dist

        return distances