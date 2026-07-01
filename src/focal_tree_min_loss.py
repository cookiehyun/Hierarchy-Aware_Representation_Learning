"""
Focal Tree-Min Loss (HSSN, Li et al. CVPR 2022, Eq. 4 & Eq. 6)

원래 픽셀 단위(semantic segmentation) multi-label hierarchy scoring을
샘플 단위(오디오 임베딩, AnuraSet) multi-label hierarchy scoring으로 변환.

Eq.4 (tree-min consistency):
    p_v = min_{u in A_v}(s_u)              if l_v = 1   (positive node)
    p_v = max_{u in C_v}(s_u)              if l_v = 0   (negative node)
    [원문: 1-p_v = min_{u in C_v}(1-s_u) = 1-max_{u in C_v}(s_u)  =>  p_v = max_{u in C_v}(s_u)]

Eq.6 (focal tree-min loss, γ=2 기본값):
    L^FTM(p) = Σ_v [ -l_v (1-p_v)^γ log(p_v) - (1-l_v)(p_v)^γ log(1-p_v) ]
"""

import torch
import torch.nn as nn


class FocalTreeMinLoss(nn.Module):
    def __init__(self, ancestor_matrix: torch.Tensor, descendant_matrix: torch.Tensor,
                 gamma: float = 2.0, eps: float = 1e-6):
        """
        Args:
            ancestor_matrix:   [n_nodes, n_nodes] bool, A[v, u] = True if u ∈ A_v
            descendant_matrix: [n_nodes, n_nodes] bool, C[v, u] = True if u ∈ C_v
            gamma: focusing parameter (논문 기본값 = 2, §3.2.1 Table 7에서 확인됨)
        """
        super().__init__()
        self.register_buffer("ancestor_mask", ancestor_matrix)
        self.register_buffer("descendant_mask", descendant_matrix)
        self.gamma = gamma
        self.eps = eps
        self.n_nodes = ancestor_matrix.shape[0]

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [bsz, n_nodes] sigmoid 확률값 s_v ∈ [0,1] (모델이 이미 sigmoid 적용해서 줘야 함)
            labels: [bsz, n_nodes] binary ground-truth multi-hot (HierarchyTree.build_ground_truth 결과)
        Returns:
            loss: scalar
        """
        bsz, n = scores.shape
        device = scores.device

        anc_mask  = self.ancestor_mask.to(device)    # [n, n]
        desc_mask = self.descendant_mask.to(device)  # [n, n]

        # s_expand[i, v, u] = s[i, u]  (모든 v에 대해 같은 score row를 broadcast)
        s_expand = scores.unsqueeze(1).expand(bsz, n, n)

        # p_pos[i, v] = min_{u in A_v} s[i, u]   (Eq.4, l_v=1 분기)
        masked_min = s_expand.masked_fill(~anc_mask.unsqueeze(0), float("inf"))
        p_pos = masked_min.min(dim=2).values

        # p_neg[i, v] = max_{u in C_v} s[i, u]   (Eq.4, l_v=0 분기 — 주의: 1-p가 아니라 p 자체가 max)
        masked_max = s_expand.masked_fill(~desc_mask.unsqueeze(0), float("-inf"))
        p_neg = masked_max.max(dim=2).values

        p = torch.where(labels.bool(), p_pos, p_neg)         # [bsz, n]
        p = p.clamp(self.eps, 1.0 - self.eps)

        # ── Eq.6 ──
        pos_term = -labels * (1 - p).pow(self.gamma) * torch.log(p)
        neg_term = -(1 - labels) * p.pow(self.gamma) * torch.log(1 - p)

        loss = (pos_term + neg_term).sum(dim=1).mean()
        return loss