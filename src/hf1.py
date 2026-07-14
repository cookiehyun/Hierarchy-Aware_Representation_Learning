"""
Joint probe + hierarchical F1 (Kiritchenko et al. 2006) for SINGLE-LABEL
AnuraSet evaluation.

Rida's point: in the existing single-label eval, family/genus/species are
each predicted by an INDEPENDENTLY trained linear probe. Nothing forces
those three probes to agree with each other, so a model can predict
species correctly while its separately-trained genus probe is wrong --
prediction-level inconsistency is real and worth measuring, even though
the GROUND-TRUTH labels are trivially consistent by construction. Ground-
truth consistency was my earlier (wrong) justification for skipping hF1
here; it doesn't say anything about prediction consistency.

Fix: train ONE joint linear probe over the full family+genus+species node
space (mirrors the multi-label hF1's joint_probe), convert its per-level
argmax predictions into one multi-hot vector per sample, and score with
the same Kiritchenko et al. (2006) ancestor-augmented hP/hR/hF1 used for
multi-label. Only applicable to the closed-set (all species known)
scenario -- NOT used for the unseen-species open-set tier, since a held-out
species has no corresponding node in this space at all (see literature
note: Digital Taxonomist / HComP-Net report flat per-level accuracy for
unseen classes instead of a Kiritchenko-style hF1, for exactly this
reason).
"""

import numpy as np
import torch
import torch.nn as nn


def joint_probe_singlelabel(train_emb, train_hier_labels, test_emb, n_family, n_genus, n_species,
                            epochs=300, lr=0.01, weight_decay=1e-4, seed=42, device=None):
    """
    Single linear probe over n_family+n_genus+n_species nodes, trained with
    per-level cross-entropy (each level is still a single-label/argmax
    classification, just sharing one backbone layer instead of three
    separate probes). Returns a [n_test, n_nodes] one-hot-per-level binary
    prediction matrix (family block one-hot, genus block one-hot, species
    block one-hot), in the same node order as the ancestor_matrix
    (family, then genus, then species).

    Args:
        train_hier_labels: [n_train, 3] int array/tensor, columns =
            [family_id, genus_id, species_id]
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    n_nodes = n_family + n_genus + n_species

    clf = nn.Linear(train_emb.shape[1], n_nodes).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)

    fam = train_hier_labels[:, 0]
    gen = train_hier_labels[:, 1]
    spe = train_hier_labels[:, 2]

    X_train = torch.FloatTensor(train_emb).to(device)
    y_fam = torch.LongTensor(fam).to(device)
    y_gen = torch.LongTensor(gen).to(device)
    y_spe = torch.LongTensor(spe).to(device)

    clf.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = clf(X_train)
        logits_fam = logits[:, 0:n_family]
        logits_gen = logits[:, n_family:n_family + n_genus]
        logits_spe = logits[:, n_family + n_genus:n_nodes]
        loss = (nn.functional.cross_entropy(logits_fam, y_fam)
               + nn.functional.cross_entropy(logits_gen, y_gen)
               + nn.functional.cross_entropy(logits_spe, y_spe))
        loss.backward()
        optimizer.step()

    clf.eval()
    with torch.no_grad():
        X_test = torch.FloatTensor(test_emb).to(device)
        logits = clf(X_test)
        pred_fam = logits[:, 0:n_family].argmax(dim=1).cpu().numpy()
        pred_gen = logits[:, n_family:n_family + n_genus].argmax(dim=1).cpu().numpy()
        pred_spe = logits[:, n_family + n_genus:n_nodes].argmax(dim=1).cpu().numpy()

    n_test = test_emb.shape[0]
    preds = np.zeros((n_test, n_nodes), dtype=np.float32)
    preds[np.arange(n_test), pred_fam] = 1.0
    preds[np.arange(n_test), n_family + pred_gen] = 1.0
    preds[np.arange(n_test), n_family + n_genus + pred_spe] = 1.0
    return preds


def ftm_only_preds_singlelabel(scores, n_family, n_genus, n_species):
    """For FTM-only: argmax each level's node range directly from its
    already-trained sigmoid scores s, same one-hot-per-level format as
    joint_probe_singlelabel's output."""
    n_nodes = n_family + n_genus + n_species
    n_test = scores.shape[0]
    pred_fam = scores[:, 0:n_family].argmax(axis=1)
    pred_gen = scores[:, n_family:n_family + n_genus].argmax(axis=1)
    pred_spe = scores[:, n_family + n_genus:n_nodes].argmax(axis=1)

    preds = np.zeros((n_test, n_nodes), dtype=np.float32)
    preds[np.arange(n_test), pred_fam] = 1.0
    preds[np.arange(n_test), n_family + pred_gen] = 1.0
    preds[np.arange(n_test), n_family + n_genus + pred_spe] = 1.0
    return preds


def ground_truth_singlelabel(hier_labels, n_family, n_genus, n_species):
    """[n_samples, 3] int labels -> [n_samples, n_nodes] one-hot-per-level
    ground truth, same node order as above. Already hierarchically
    consistent by construction (that part of my original reasoning was
    correct -- it's only the PREDICTIONS that need checking)."""
    n_nodes = n_family + n_genus + n_species
    n_samples = hier_labels.shape[0]
    gt = np.zeros((n_samples, n_nodes), dtype=np.float32)
    gt[np.arange(n_samples), hier_labels[:, 0]] = 1.0
    gt[np.arange(n_samples), n_family + hier_labels[:, 1]] = 1.0
    gt[np.arange(n_samples), n_family + n_genus + hier_labels[:, 2]] = 1.0
    return gt


def hierarchical_f1(preds, ground_truth, ancestor_matrix, eps=1e-8):
    """Same Kiritchenko et al. (2006) micro-averaged hP/hR/hF1 used for
    multi-label (identical formula, just fed single-label-derived one-hot
    vectors here)."""
    anc = ancestor_matrix.numpy().astype(np.float32) if hasattr(ancestor_matrix, 'numpy') else ancestor_matrix.astype(np.float32)
    preds_aug = (preds @ anc > 0).astype(np.float32)
    intersection = (preds_aug * ground_truth).sum()
    hP = intersection / (preds_aug.sum() + eps)
    hR = intersection / (ground_truth.sum() + eps)
    hF1 = 2 * hP * hR / (hP + hR + eps)
    return hP, hR, hF1