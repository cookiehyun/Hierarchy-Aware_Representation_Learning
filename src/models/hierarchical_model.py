"""
Shared-backbone Hierarchical Model

논문의 f_ENC(공유 인코더) -> f_PROJ(triplet 임베딩) / f_SEG(hierarchy score) 구조를
오디오 임베딩(4096-d)에 맞게 적용.

- backbone   : 4096 -> 1024 (공유)
- triplet_head    : 1024 -> 128, L2-normalized   (Tree-Triplet Loss, Eq.7 용)
- hierarchy_head  : 1024 -> n_nodes, sigmoid      (Focal Tree-Min Loss, Eq.6 용)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalModel(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024, triplet_dim=128, n_hierarchy_nodes=None):
        super().__init__()
        assert n_hierarchy_nodes is not None, "n_hierarchy_nodes (HierarchyTree.n_nodes)를 지정해야 함"

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.triplet_head   = nn.Linear(hidden_dim, triplet_dim)
        self.hierarchy_head = nn.Linear(hidden_dim, n_hierarchy_nodes)

    def forward(self, x):
        """
        Args:
            x: [bsz, 4096] NatureLM embedding
        Returns:
            z: [bsz, triplet_dim]      L2-normalized, Tree-Triplet Loss 입력용
            s: [bsz, n_hierarchy_nodes] sigmoid 확률, Focal Tree-Min Loss 입력용
        """
        h = self.backbone(x)
        z = F.normalize(self.triplet_head(h), dim=1)
        s = torch.sigmoid(self.hierarchy_head(h))
        return z, s


if __name__ == "__main__":
    model = HierarchicalModel(n_hierarchy_nodes=20)
    x = torch.randn(8, 4096)
    z, s = model(x)
    print(f"z: {z.shape} (norm~1: {z.norm(dim=1).mean():.4f})")
    print(f"s: {s.shape} (range: [{s.min():.3f}, {s.max():.3f}])")