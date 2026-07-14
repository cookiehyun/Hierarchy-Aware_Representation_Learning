"""
Patched HierarchicalModel: adds a separate `encode_hidden()` method that
exposes h, the shared backbone's hidden representation -- WITHOUT changing
forward()'s existing (z, s) return signature, so every existing training
script (`z, s = model(emb)`) keeps working unmodified, and existing
checkpoints still load fine (state_dict is unchanged).

Rida's point (correct): for FTM-only, triplet_head is never trained, but
hierarchy_head's gradient DOES flow back through the shared backbone
(z = triplet_head(h), s = hierarchy_head(h) -- both come off the same h).
So h is a genuinely learned representation for FTM-only, unlike s (a
downstream probability vector, not an embedding). Gap should be computed
on h for FTM-only, not s, to be comparable to the embedding-space gap used
for UALB/TT-only/FTM+TT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalModel(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=1024, triplet_dim=128, n_hierarchy_nodes=None):
        super().__init__()
        assert n_hierarchy_nodes is not None

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.triplet_head   = nn.Linear(hidden_dim, triplet_dim)
        self.hierarchy_head = nn.Linear(hidden_dim, n_hierarchy_nodes)

    def forward(self, x):
        """UNCHANGED from the original -- every existing training script
        that does `z, s = model(emb)` keeps working exactly as before."""
        h = self.backbone(x)
        z = F.normalize(self.triplet_head(h), dim=1)
        s = torch.sigmoid(self.hierarchy_head(h))
        return z, s

    def encode_hidden(self, x):
        """🆕 Eval-only helper: returns h, the shared backbone's hidden
        representation (pre-triplet_head, pre-hierarchy_head). Use this for
        FTM-only's gap computation instead of s -- h is trained (gradient
        flows back through backbone from L_FTM via hierarchy_head), so it's
        a genuine learned embedding, unlike s (raw sigmoid probabilities)."""
        return self.backbone(x)