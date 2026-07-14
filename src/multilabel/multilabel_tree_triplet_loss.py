"""
Multi-label Tree-Triplet Loss -- same-species variant.

Same idea as tree_triplet_loss_self.py (single-label): the original
multi-label version excluded distance-0 (same species) pairs from the
positive pool, matching HSSN's original design where anchor and positive
are never from the same class. This version adds that tier back in,
falling back through 0 -> 2 -> 4 depending on what's available in the
batch, exactly mirroring the single-label same-genus -> same-family
fallback.

For a given species v that occurs in at least one sample of the batch:
  - anchor samples: contain species v
  - positive samples: prefer other samples that ALSO contain species v
    (distance 0); if none, same genus (distance 2); if still none, same
    family (distance 4)
  - negative samples: no species sharing v's family at all (distance 6)

Margin formula is unchanged (HSSN Eq.8); m_tau=(6-pos_dist)/(2D) already
gives distance-0 positives the largest margin, same as the single-label fix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelTreeTripletLoss(nn.Module):
    def __init__(self, species_distance_matrix, m_epsilon=0.1,
                 max_hierarchy_depth=3, max_triplets_per_species=200):
        super().__init__()
        self.register_buffer('dist_matrix', species_distance_matrix)
        self.m_epsilon = m_epsilon
        self.max_hierarchy_depth = max_hierarchy_depth
        self.max_triplets_per_species = max_triplets_per_species

    def _margin(self, pos_dist):
        m_tau = (6 - pos_dist) / (2.0 * self.max_hierarchy_depth)
        return self.m_epsilon + 0.5 * m_tau

    def forward(self, features, species_multihot):
        device = features.device
        bsz, n_species = species_multihot.shape
        dist_matrix = self.dist_matrix.to(device)

        features_norm = F.normalize(features, p=2, dim=1)

        mask = species_multihot.bool().unsqueeze(1)  # [bsz, 1, n_species]
        expanded = dist_matrix.unsqueeze(0).expand(bsz, -1, -1)
        masked = expanded.masked_fill(~mask, 999)
        min_dist = masked.min(dim=2).values  # [bsz, n_species]

        triplet_loss = torch.tensor(0.0, device=device)
        species_count = 0

        present_species = torch.where(species_multihot.sum(dim=0) > 0)[0]
        for v in present_species:
            v = v.item()
            anchor_idx = torch.where(species_multihot[:, v] == 1)[0]
            if len(anchor_idx) == 0:
                continue

            dist_col = min_dist[:, v]

            # fallback tiers: same species first, then same genus, then same family
            # (only difference from the original: distance 0 is tried first)
            pos_idx = torch.where(dist_col == 0)[0]
            pos_dist = 0
            if len(pos_idx) == 0:
                pos_idx = torch.where(dist_col == 2)[0]
                pos_dist = 2
            if len(pos_idx) == 0:
                pos_idx = torch.where(dist_col == 4)[0]
                pos_dist = 4
            neg_idx = torch.where(dist_col == 6)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            # shuffle positives so anchor[k] and pos[k] don't coincide on the
            # same sample when pos_dist == 0 (anchor pool and positive pool
            # overlap completely in that case)
            perm = torch.randperm(len(pos_idx), device=device)
            pos_idx = pos_idx[perm]

            min_size = min(len(anchor_idx), len(pos_idx), len(neg_idx), self.max_triplets_per_species)
            a_idx = anchor_idx[:min_size]
            p_idx = pos_idx[:min_size]
            n_idx = neg_idx[:min_size]

            feats_a = features_norm[a_idx]
            feats_p = features_norm[p_idx]
            feats_n = features_norm[n_idx]

            dist_to_pos = 1.0 - torch.sum(feats_a * feats_p, dim=1)
            dist_to_neg = 1.0 - torch.sum(feats_a * feats_n, dim=1)

            margin = self._margin(pos_dist)
            tl = F.relu(dist_to_pos - dist_to_neg + margin)

            if tl.numel() > 0:
                triplet_loss = triplet_loss + tl.mean()
                species_count += 1

        if species_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return triplet_loss / species_count