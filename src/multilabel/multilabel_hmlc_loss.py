"""
Multi-label HMLC (UALB training objective), adapted for AnuraSet's
multi-species-per-recording setting.

The original HMLC assumes labels = [bsz, n_levels] of integer class ids and
builds positive masks by progressively dropping the finest level. That trick
doesn't work here because each hierarchy level is itself a multi-hot vector
(family: 5-dim, genus: 12-dim, species: 42-dim), not a single integer.

Instead, each level gets its own positive mask: two samples are positive at
a given level if they have the exact same multi-hot vector at that level.
This mirrors what compute_gap already does for the gap metric, and we
confirmed on the test set that exact-combination classes lose almost no
samples (<0.3% at species level), so this is a safe way to define "same
class" here. Levels must be passed fine to coarse so the finer levels keep
getting the stronger penalty, same as the original HMLC.

This only implements the 'hmce' variant (the one train_ualb.py used), which
also enforces that a coarser layer's loss never drops below a finer layer's
loss already seen -- the 'hmc' and 'hce' variants from the original code
aren't reproduced here since they weren't used.
"""

import torch
import torch.nn as nn

from hmlc_loss import SupConLoss


class MultiLabelHMLC(nn.Module):
    def __init__(self, temperature=0.07, layer_penalty=None):
        super().__init__()
        self.temperature = temperature
        self.layer_penalty = layer_penalty if layer_penalty is not None else torch.exp
        self.sup_con_loss = SupConLoss(temperature)

    def forward(self, features, level_labels):
        """
        Args:
            features: [bsz, n_views, feat_dim]
            level_labels: list of label tensors ordered fine to coarse,
                e.g. [species_labels, genus_labels, family_labels], each
                [bsz, n_classes_at_level] multi-hot. Order matters: the
                first entry gets the strongest penalty, same as the
                original HMLC weighting finer levels more.
        """
        device = features.device
        n_levels = len(level_labels)
        cumulative_loss = torch.tensor(0.0, device=device)
        max_loss_lower_layer = torch.tensor(float('-inf'), device=device)

        for l, labels in enumerate(level_labels, start=1):
            labels = labels.to(device)
            mask = torch.all(labels.unsqueeze(1) == labels.unsqueeze(0), dim=2).to(torch.uint8)
            layer_loss = self.sup_con_loss(features, mask=mask)

            # hmce: a coarser layer's loss can't be lower than any finer
            # layer already seen, same constraint as the single-label HMLC
            layer_loss = torch.max(max_loss_lower_layer, layer_loss)
            cumulative_loss += self.layer_penalty(torch.tensor(1 / l, dtype=torch.float)) * layer_loss
            max_loss_lower_layer = torch.max(max_loss_lower_layer, layer_loss)

        return cumulative_loss / n_levels