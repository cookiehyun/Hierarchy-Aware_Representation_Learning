"""
Hierarchical F1 (hF1), Kiritchenko et al. 2006.

Unlike the per-level (family/genus/species) macro F1 probes, hF1 needs a
single joint prediction over the whole node space V = family U genus U
species, so it can catch cases where a prediction is hierarchically
inconsistent (e.g. predicting a species positive without its genus/family
also being positive).

hP = |Yhat_aug ∩ Y| / |Yhat_aug|
hR = |Yhat_aug ∩ Y| / |Y|
hF1 = 2*hP*hR / (hP+hR)

where Yhat_aug is the predicted label set with all ancestors added in
(using the same ancestor_matrix built for FocalTreeMinLoss).
"""

import numpy as np
import torch
import torch.nn as nn


def joint_probe(train_emb, train_gt, test_emb, test_gt, n_nodes,
                epochs=300, lr=0.01, weight_decay=1e-4, seed=42, device=None):
    """Single linear probe over all n_nodes (family+genus+species
    concatenated), sigmoid + BCE with pos_weight, same setup as
    multilabel_probe but predicting the whole node space at once."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)

    clf = nn.Linear(train_emb.shape[1], n_nodes).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)

    pos_count = train_gt.sum(axis=0)
    neg_count = train_gt.shape[0] - pos_count
    pos_weight = torch.FloatTensor(neg_count / np.clip(pos_count, 1, None)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train = torch.FloatTensor(train_emb).to(device)
    Y_train = torch.FloatTensor(train_gt).to(device)
    X_test = torch.FloatTensor(test_emb).to(device)

    clf.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(clf(X_train), Y_train)
        loss.backward()
        optimizer.step()

    clf.eval()
    with torch.no_grad():
        probs = torch.sigmoid(clf(X_test))
        preds = (probs > 0.5).float().cpu().numpy()

    return preds


def hierarchical_f1(preds, ground_truth, ancestor_matrix, eps=1e-8):
    """
    Args:
        preds: [n_samples, n_nodes] binary predictions
        ground_truth: [n_samples, n_nodes] binary ground truth (already
            ancestor-consistent since it comes straight from family/genus/
            species multi-hot)
        ancestor_matrix: [n_nodes, n_nodes] bool, ancestor_matrix[v, u] = True
            if u is an ancestor of v (v included)
    Returns:
        hP, hR, hF1 (micro-averaged over the whole test set)
    """
    anc = ancestor_matrix.numpy().astype(np.float32)

    # preds_aug[i, u] = 1 if u is an ancestor of any node predicted positive in sample i
    preds_aug = (preds @ anc > 0).astype(np.float32)

    intersection = (preds_aug * ground_truth).sum()
    hP = intersection / (preds_aug.sum() + eps)
    hR = intersection / (ground_truth.sum() + eps)
    hF1 = 2 * hP * hR / (hP + hR + eps)

    return hP, hR, hF1