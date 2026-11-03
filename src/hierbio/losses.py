"""Hierarchical objectives adapted to frozen audio embeddings (paper Sec. 2).

    HMCE   Zhang et al. 2022 (HiMulConE), + explicit finest (species) level    Eq. 1-3
    FTM    Li et al. 2022 (HSSN) Focal Tree-Min                                Eq. 4-5
    TT     Li et al. 2022 (HSSN) Tree-Triplet, + same-class positives           Eq. 6

Single- and multi-label versions are kept separate because their positive sets are
defined differently (class ids vs. multi-hot vectors).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchy import pairwise_tree_distance


# ── Supervised contrastive base (Khosla et al. 2020, code from Zhang et al. 2022) ──
class SupConLoss(nn.Module):
    """features [bsz, n_views, dim]; mask [bsz, bsz] with mask[i, j] = 1 if j is a positive of i."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask):
        device = features.device
        bsz, n_views = features.shape[:2]
        mask = mask.float().to(device)

        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
        logits = contrast @ contrast.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        mask = mask.repeat(n_views, n_views)
        not_self = torch.scatter(torch.ones_like(mask), 1,
                                 torch.arange(bsz * n_views, device=device).view(-1, 1), 0)
        mask = mask * not_self

        exp_logits = torch.exp(logits) * not_self
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.view(n_views, bsz).mean()


def _unique_rows(x):
    """Index of the last occurrence of every unique row of x (as in the HiMulCon code)."""
    _, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=0)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return inverse.new_empty(int(inverse.max()) + 1).scatter_(0, inverse, perm)


class HMCELoss(nn.Module):
    """Single-label HiMulConE with the species level added (paper Sec. 2.1, 'Implementation gap').

    labels: [bsz, 3] (coarse, mid, fine) ids. Step 1 = exact match on all levels (new),
    step 2 = coarse+mid, step 3 = coarse only. Level weight exp(1/step), and every level's
    loss is lower-bounded by the maximum loss of the finer levels (HiConE).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.sup_con = SupConLoss(temperature)

    def forward(self, features, labels):
        labels = labels.float()
        n_levels = labels.shape[1]
        mask = torch.ones_like(labels)
        total = torch.tensor(0.0, device=features.device)
        max_finer = torch.tensor(float('-inf'), device=features.device)
        for step in range(1, n_levels + 1):
            if step > 1:
                mask[:, n_levels - (step - 1):] = 0
            layer_labels = labels * mask
            pos = torch.all(layer_labels.unsqueeze(1) == layer_labels.unsqueeze(0), dim=2)
            layer_loss = torch.max(max_finer, self.sup_con(features, pos))
            total = total + math.exp(1 / step) * layer_loss
            max_finer = torch.max(max_finer, layer_loss)
            keep = _unique_rows(layer_labels)
            labels, mask, features = labels[keep], mask[keep], features[keep]
        return total / n_levels


class MultiLabelHMCELoss(nn.Module):
    """Multi-label HiMulConE: two samples are positives at a level if their multi-hot vectors
    at that level are identical. level_labels are ordered fine to coarse."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.sup_con = SupConLoss(temperature)

    def forward(self, features, level_labels):
        total = torch.tensor(0.0, device=features.device)
        max_finer = torch.tensor(float('-inf'), device=features.device)
        for step, labels in enumerate(level_labels, start=1):
            pos = torch.all(labels.unsqueeze(1) == labels.unsqueeze(0), dim=2)
            layer_loss = torch.max(max_finer, self.sup_con(features, pos))
            total = total + math.exp(1 / step) * layer_loss
            max_finer = torch.max(max_finer, layer_loss)
        return total / len(level_labels)


# ── Focal Tree-Min (Eq. 4-5) ─────────────────────────────────────────────
class FocalTreeMinLoss(nn.Module):
    """scores [bsz, n_nodes] sigmoid outputs, labels [bsz, n_nodes] node multi-hot.

    Positive node v: p_v = min over ancestors of s_u.  Negative node v: p_v = max over
    descendants of s_u (so the loss pushes the largest descendant score down).
    Passing identity matrices removes the tree-min step (plain focal BCE).
    """

    def __init__(self, ancestor_matrix, descendant_matrix, gamma=2.0, eps=1e-6):
        super().__init__()
        self.register_buffer('ancestor_mask', ancestor_matrix.bool())
        self.register_buffer('descendant_mask', descendant_matrix.bool())
        self.gamma = gamma
        self.eps = eps

    def forward(self, scores, labels):
        s = scores.unsqueeze(1).expand(-1, scores.shape[1], -1)
        p_pos = s.masked_fill(~self.ancestor_mask.unsqueeze(0), float('inf')).min(dim=2).values
        p_neg = s.masked_fill(~self.descendant_mask.unsqueeze(0), float('-inf')).max(dim=2).values
        p = torch.where(labels.bool(), p_pos, p_neg).clamp(self.eps, 1.0 - self.eps)
        pos_term = -labels * (1 - p).pow(self.gamma) * torch.log(p)
        neg_term = -(1 - labels) * p.pow(self.gamma) * torch.log(1 - p)
        return (pos_term + neg_term).sum(dim=1).mean()


# ── Tree-Triplet (Eq. 6) ────────────────────────────────────────────────
def tree_margin(pos_dist, neg_dist, m_epsilon=0.1, depth=3):
    """m = m_eps + 0.5 * m_tau, m_tau = (psi(anchor, neg) - psi(anchor, pos)) / 2D."""
    return m_epsilon + 0.5 * (neg_dist - pos_dist) / (2.0 * depth)


class TreeTripletLoss(nn.Module):
    """Single-label Tree-Triplet.

    For every fine class c in the batch: anchors = samples of c, positives = samples sharing
    c's mid class (same-class samples included, paper Sec. 2.2 'Implementation gap'),
    negatives = samples from a different coarse class.
    """

    def __init__(self, m_epsilon=0.1, depth=3, max_triplets_per_class=200):
        super().__init__()
        self.m_epsilon = m_epsilon
        self.depth = depth
        self.max_triplets = max_triplets_per_class

    def forward(self, features, ids):
        device = features.device
        f = F.normalize(features, dim=1)
        coarse, mid, fine = ids[:, 0], ids[:, 1], ids[:, 2]
        dist = pairwise_tree_distance(ids)
        total, n_classes = 0.0, 0
        for c in torch.unique(fine):
            anchor_idx = torch.where(fine == c)[0]
            first = anchor_idx[0]
            pos_idx = torch.where(mid == mid[first])[0]
            neg_idx = torch.where(coarse != coarse[first])[0]
            if len(neg_idx) == 0:
                continue
            k = min(len(anchor_idx), len(pos_idx), len(neg_idx), self.max_triplets)
            pos_idx = pos_idx[torch.randperm(len(pos_idx), device=device)]
            a, p, n = anchor_idx[:k], pos_idx[:k], neg_idx[:k]
            d_pos = 1.0 - (f[a] * f[p]).sum(dim=1)
            d_neg = 1.0 - (f[a] * f[n]).sum(dim=1)
            margin = tree_margin(dist[a, p].float(), dist[a, n].float(), self.m_epsilon, self.depth)
            total = total + F.relu(d_pos - d_neg + margin).mean()
            n_classes += 1
        if n_classes == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total / n_classes


class MultiLabelTreeTripletLoss(nn.Module):
    """Multi-label Tree-Triplet. For every fine class v present in the batch: anchors contain
    v; positives are the closest available tier (same class 0 -> same mid 2 -> same coarse 4,
    measured as the minimum tree distance between v and any label of the sample); negatives
    share no coarse class with v (distance 6)."""

    def __init__(self, fine_distance_matrix, m_epsilon=0.1, depth=3, max_triplets_per_class=200):
        super().__init__()
        self.register_buffer('dist_matrix', fine_distance_matrix)
        self.m_epsilon = m_epsilon
        self.depth = depth
        self.max_triplets = max_triplets_per_class

    def forward(self, features, fine_multihot):
        device = features.device
        f = F.normalize(features, dim=1)
        present = fine_multihot.bool().unsqueeze(1)
        # min_dist[i, v] = min over labels u of sample i of psi(u, v); 999 for all-zero rows
        min_dist = self.dist_matrix.unsqueeze(0).expand(len(f), -1, -1).masked_fill(~present, 999).min(dim=2).values
        total, n_classes = torch.tensor(0.0, device=device), 0
        for v in torch.where(fine_multihot.sum(dim=0) > 0)[0].tolist():
            anchor_idx = torch.where(fine_multihot[:, v] == 1)[0]
            col = min_dist[:, v]
            for pos_dist in (0, 2, 4):
                pos_idx = torch.where(col == pos_dist)[0]
                if len(pos_idx):
                    break
            neg_idx = torch.where(col == 6)[0]
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            pos_idx = pos_idx[torch.randperm(len(pos_idx), device=device)]
            k = min(len(anchor_idx), len(pos_idx), len(neg_idx), self.max_triplets)
            a, p, n = anchor_idx[:k], pos_idx[:k], neg_idx[:k]
            d_pos = 1.0 - (f[a] * f[p]).sum(dim=1)
            d_neg = 1.0 - (f[a] * f[n]).sum(dim=1)
            margin = tree_margin(pos_dist, 6, self.m_epsilon, self.depth)
            total = total + F.relu(d_pos - d_neg + margin).mean()
            n_classes += 1
        if n_classes == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total / n_classes


def beta_schedule(epoch, total_epochs):
    """FTM+TT weight on L_TT (Eq. 7): cosine ramp 0 -> 0.5."""
    return 0.25 * (1 - math.cos(math.pi * epoch / total_epochs))
