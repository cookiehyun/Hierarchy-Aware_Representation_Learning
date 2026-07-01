"""
Tree-Triplet Loss for Hierarchical Embedding Space Learning

Based on HSSN (Deep Hierarchical Semantic Segmentation Networks)
paper: Li et al., CVPR 2022
Adapted for AnuraSet audio embedding space instead of semantic segmentation.

Loss formulation (Eq. 7 in paper):
L^TT(i, i^+, i^-) = max{⟨i, i^+⟩ - ⟨i, i^-⟩ + m, 0}

where:
- i, i^+, i^- are anchor, positive, negative embeddings
- ⟨·,·⟩ is cosine distance
- m is hierarchy-induced margin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TreeTripletLoss(nn.Module):
    """
    Hierarchical triplet loss with tree-induced margin.
    
    Triplet sampling strategy:
    - anchor: all samples of class c
    - positive: samples from same subtree as anchor (closer in hierarchy)
    - negative: samples from different subtree (farther in hierarchy)
    
    Margin is dynamically adjusted based on tree distances.
    """
    
    def __init__(self, temperature: float = 0.07, m_epsilon: float = 0.1,
                 max_hierarchy_depth: int = 3, use_hierarchy_margin: bool = True):
        """
        Args:
            temperature: temperature for triplet loss scaling
            m_epsilon: constant margin tolerance (from HSSN paper)
            max_hierarchy_depth: maximum depth of hierarchy (for margin calculation)
            use_hierarchy_margin: if True, use hierarchy-aware margin; else constant margin
        """
        super(TreeTripletLoss, self).__init__()
        self.temperature = temperature
        self.m_epsilon = m_epsilon
        self.max_hierarchy_depth = max_hierarchy_depth
        self.use_hierarchy_margin = use_hierarchy_margin
    
    def forward(self, features: torch.Tensor, 
                species_labels: torch.Tensor,
                family_labels: torch.Tensor,
                genus_labels: torch.Tensor,
                tree_distances: Optional[torch.Tensor] = None,
                max_triplets_per_class: int = 200) -> torch.Tensor:
        """
        Compute tree-triplet loss.
        
        Args:
            features: [bsz, feat_dim] embeddings
            species_labels: [bsz] species class indices
            family_labels: [bsz] family class indices
            genus_labels: [bsz] genus class indices
            tree_distances: [bsz, bsz] pairwise tree distances (optional)
            max_triplets_per_class: max triplets to sample per class
            
        Returns:
            loss: scalar tensor
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)  # [bsz, feat_dim]
        
        triplet_loss = 0.0
        class_count = 0
        
        # Iterate over each unique species in batch
        unique_species = torch.unique(species_labels)
        
        for species_id in unique_species:
            # Anchor samples: all samples of current species
            anchor_mask = species_labels == species_id
            anchor_indices = torch.where(anchor_mask)[0]
            
            if len(anchor_indices) == 0:
                continue
            
            # Positive samples: same genus but different species
            # (or same family if genus too small)
            pos_mask = (genus_labels == genus_labels[anchor_indices[0]]) & (~anchor_mask)
            
            # If no positive samples from same genus, use same family
            if pos_mask.sum() == 0:
                pos_mask = (family_labels == family_labels[anchor_indices[0]]) & (~anchor_mask)
            
            # Negative samples: different family
            family_id = family_labels[anchor_indices[0]]
            neg_mask = family_labels != family_id
            
            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]
            
            # Skip if not enough positive or negative samples
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            # Sample equal number of triplets
            min_size = min(len(anchor_indices), len(pos_indices), len(neg_indices),
                          max_triplets_per_class)
            
            anchor_idx = anchor_indices[:min_size]
            pos_idx = pos_indices[:min_size]
            neg_idx = neg_indices[:min_size]
            
            # Get feature vectors
            feats_anchor = features_norm[anchor_idx]  # [min_size, feat_dim]
            feats_pos = features_norm[pos_idx]        # [min_size, feat_dim]
            feats_neg = features_norm[neg_idx]        # [min_size, feat_dim]
            
            # Compute cosine distances
            # distance = 1 - cosine_similarity
            dist_to_pos = 1.0 - torch.sum(feats_anchor * feats_pos, dim=1)  # [min_size]
            dist_to_neg = 1.0 - torch.sum(feats_anchor * feats_neg, dim=1)  # [min_size]
            
            # Compute margin
            if self.use_hierarchy_margin and tree_distances is not None:
                # Dynamic hierarchy-aware margin
                margins = []
                for i in range(min_size):
                    anchor_pos_dist = tree_distances[anchor_idx[i], pos_idx[i]].float()
                    anchor_neg_dist = tree_distances[anchor_idx[i], neg_idx[i]].float()
                    
                    m_tau = (anchor_neg_dist - anchor_pos_dist) / (2.0 * self.max_hierarchy_depth)
                    margin = self.m_epsilon + 0.5 * m_tau
                    margins.append(margin)
                
                margin = torch.stack(margins).to(device)  # [min_size]
            else:
                # Constant margin (HSSN uses 0.6 in experiments)
                margin = (0.1 + 0.5 * (3 - 1) / (2.0 * self.max_hierarchy_depth)) * torch.ones(min_size, device=device)
            
            # Triplet loss: max(d_pos - d_neg + margin, 0)
            tl = dist_to_pos - dist_to_neg + margin
            tl = F.relu(tl)
            
            if tl.numel() > 0:
                triplet_loss += tl.mean()
                class_count += 1
        
        # Average over classes
        if class_count == 0:
            # Return zero loss if no valid triplets found
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        triplet_loss = triplet_loss / class_count
        
        return triplet_loss


class CombinedHierarchicalLoss(nn.Module):
    """
    Combined loss for hierarchical embedding learning.
    Combines Focal Tree-Min Loss (from UALB method) with Tree-Triplet Loss.
    
    Total Loss = α * L_FTM + β * L_TT
    """
    
    def __init__(self, ftm_loss_fn, tree_triplet_loss_fn, alpha: float = 1.0, beta: float = 0.1):
        """
        Args:
            ftm_loss_fn: Focal Tree-Min Loss function (from Method 1)
            tree_triplet_loss_fn: Tree-Triplet Loss function (from Method 2)
            alpha: weight for FTM loss
            beta: weight for Tree-Triplet loss (can be scheduled)
        """
        super(CombinedHierarchicalLoss, self).__init__()
        self.ftm_loss = ftm_loss_fn
        self.triplet_loss = tree_triplet_loss_fn
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, features: torch.Tensor, hier_labels: torch.Tensor,
                tree_distances: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: [bsz, 2, feat_dim] (2 views from projection head)
            hier_labels: [bsz, 3] hierarchical labels [family, genus, species]
            tree_distances: [bsz, bsz] pairwise tree distances (optional)
            
        Returns:
            combined_loss: scalar
        """
        # FTM loss (requires 3D features with views)
        ftm_loss = self.ftm_loss(features, hier_labels.float())
        
        # Tree-Triplet loss (use single view)
        features_single = features[:, 0, :]  # Take first view
        triplet_loss = self.triplet_loss(
            features_single,
            hier_labels[:, 2],  # species
            hier_labels[:, 0],  # family
            hier_labels[:, 1],  # genus
            tree_distances=tree_distances
        )
        
        combined_loss = self.alpha * ftm_loss + self.beta * triplet_loss
        
        return combined_loss