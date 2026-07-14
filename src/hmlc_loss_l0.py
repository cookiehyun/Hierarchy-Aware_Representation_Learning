"""
Modified HMLC: adds the finest-level (exact full-label match) layer that the
original loop skips.

The original HMLC loop is `for l in range(1, labels.shape[1])`, which never
includes l=0 (no columns zeroed, i.e. "all levels match exactly"). That's
fine for DeepFashion, where the finest column is "variation" (different
photos of the exact same product) -- a level that doesn't need its own
exact-match layer, since two augmented views of the same image already
share it implicitly. For AnuraSet, the finest column is species, which is
the thing we actually want to discriminate, so skipping it is a real gap.

This version iterates one more step, adding a layer where no columns are
zeroed (family+genus+species must all match to be positive), and shifts
the layer_penalty indexing so this new layer gets the strongest penalty
(matching the original's convention of weighting finer levels more).
"""

import torch
import torch.nn as nn

from hmlc_loss import SupConLoss, unique


class HMLCWithSpeciesLevel(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07,
                 layer_penalty=None, loss_type='hmce'):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.layer_penalty = layer_penalty if layer_penalty is not None else self.pow_2
        self.sup_con_loss = SupConLoss(temperature)
        self.loss_type = loss_type

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        n_levels = labels.shape[1]
        mask = torch.ones(labels.shape).to(device)
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))

        # step=1 -> no columns zeroed (species+genus+family exact match, new)
        # step=2 -> zero last 1 column   (family+genus match, same as original l=1)
        # step=3 -> zero last 2 columns  (family only, same as original l=2)
        for step in range(1, n_levels + 1):
            n_zero = step - 1
            if n_zero > 0:
                mask[:, n_levels - n_zero:] = 0
            layer_labels = labels * mask
            mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                       for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)
            layer_loss = self.sup_con_loss(features, mask=mask_labels)

            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(torch.tensor(1 / step).type(torch.float)) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += self.layer_penalty(torch.tensor(1 / step).type(torch.float)) * layer_loss
            else:
                raise NotImplementedError('Unknown loss')

            _, unique_indices = unique(layer_labels, dim=0)
            max_loss_lower_layer = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
            labels = labels[unique_indices]
            mask = mask[unique_indices]
            features = features[unique_indices]

        return cumulative_loss / n_levels