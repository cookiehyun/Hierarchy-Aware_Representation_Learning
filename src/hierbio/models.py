"""Trainable heads on top of frozen embeddings (paper Table 1).

Parameter names match the original scripts, so checkpoints from results/<old dirs>
still load with load_state_dict.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM = 1024
EMBED_DIM = 128


class ProjectionHead(nn.Module):
    """HMCE: Linear(d->1024) -> ReLU -> Linear(1024->128), L2-normalised."""

    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, proj_dim=EMBED_DIM):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.head(x), dim=1)

    def embed(self, x):
        return self(x)


class HierarchicalModel(nn.Module):
    """FTM / TT / FTM+TT: shared Linear(d->1024)+ReLU trunk, branching into a triplet head
    (z, 128-d, L2-normalised) and a hierarchy head (s, one sigmoid score per node)."""

    def __init__(self, input_dim, n_hierarchy_nodes, hidden_dim=HIDDEN_DIM, triplet_dim=EMBED_DIM):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True))
        self.triplet_head = nn.Linear(hidden_dim, triplet_dim)
        self.hierarchy_head = nn.Linear(hidden_dim, n_hierarchy_nodes)

    def forward(self, x):
        h = self.backbone(x)
        return F.normalize(self.triplet_head(h), dim=1), torch.sigmoid(self.hierarchy_head(h))

    def encode_hidden(self, x):
        """Shared trunk output h. For FTM-only the triplet head is never trained, so Gap/NMI
        are computed on h instead of z."""
        return self.backbone(x)


class FlatHead(nn.Module):
    """Flat controls: Linear(d->out) or the same MLP as above, Linear(d->1024)->ReLU->Linear(1024->out).
    Returns logits."""

    def __init__(self, input_dim, n_out, mlp=True, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.mlp = mlp
        if mlp:
            self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True))
            self.out = nn.Linear(hidden_dim, n_out)
        else:
            self.out = nn.Linear(input_dim, n_out)

    def forward(self, x):
        return self.out(self.backbone(x) if self.mlp else x)

    def encode_hidden(self, x):
        return self.backbone(x) if self.mlp else x
