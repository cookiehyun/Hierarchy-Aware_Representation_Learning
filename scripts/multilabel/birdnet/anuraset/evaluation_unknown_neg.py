"""
Negative-Clustering Diagnostic (eval-only, no retraining)

Question this answers: do negative ("unknown"/no-call, all-zero label)
test samples get pulled into their own implicit cluster in embedding
space, separate from the real species/genus/family clusters -- for each
of Baseline / UALB / FTM-only / TT-only / FTM+TT?

This matters specifically for UALB: MultiLabelHMLC's positive mask is
built by comparing whether two samples' label vectors are exactly equal
(torch.all(labels_i == labels_j)). All-zero negative samples are always
mutually "equal" under this comparison, so in principle they could get
treated as one shared implicit class and attracted together. FTM (sample-
independent per-node penalty) and TT (negatives excluded from
present_species entirely, since min_dist stays at the 999 sentinel) have
no such mechanism, so they're included here as a built-in point of
comparison -- if UALB's numbers look like an outlier relative to the
other four, that's evidence for the effect actually manifesting in the
learned embeddings on this dataset (not just as a theoretical concern
from reading the loss code).

Two complementary metrics, both computed on frozen test-set embeddings
from already-trained checkpoints:

1. Negative Gap: same intra/inter cosine-similarity construction as
   compute_gap() in the eval scripts, but with "negative" as the target
   class instead of a species/genus/family label.
     intra = mean cosine similarity among negative samples
     inter = mean cosine similarity between negative and known samples
     gap   = intra - inter
   A large positive gap means negatives sit closer to each other than to
   known samples -- consistent with (though not sole proof of) clustering.

2. k-NN Purity: for each negative sample, look at its k nearest neighbors
   (cosine distance) among ALL test samples (negative + known) and compute
   what fraction are also negative. Compare against the base rate (overall
   fraction of test samples that are negative). If negatives were
   distributed the same as everything else, purity ~= base rate. Purity
   noticeably above base rate is a more direct, less assumption-laden
   signal of clustering than the gap metric.

This script only extracts embeddings and computes these two metrics; it
does not retrain anything and does not modify any checkpoint.

Usage: adjust EMBED_SOURCE below ('naturelm' or 'birdnet') and the
MODEL_PATHS dict to point at your trained checkpoints, then run.
"""

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from multilabel.NatureLM.anuraset.anuraset_dataset_un import AnuraSetMultiLabelDataset
from models.projection_head import ProjectionHead
from models.hierarchical_model import HierarchicalModel

# ── Config: pick which embedding source's checkpoints to check ────────────
EMBED_SOURCE = 'naturelm'   # 'naturelm' or 'birdnet'

if EMBED_SOURCE == 'naturelm':
    INPUT_DIM = 4096
    EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
    META_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
    RESULTS_ROOT = '/user/gaad2403/Bachelor-Thesis/results/multilabel/NatureLM/anuraset'
    MODEL_PATHS = {
        'UALB':         ('projectionhead', f'{RESULTS_ROOT}/wp4_ualb_multilabel_unknown/best_model.pt'),
        'FTM-only':     ('hierarchical',   f'{RESULTS_ROOT}/wp4_ftm_only_multilabel_unknown/best_model.pt'),
        'TT-only_self': ('hierarchical',   f'{RESULTS_ROOT}/wp4_tt_only_multilabel_self_unknown/best_model.pt'),
        'FTM+TT_self':  ('hierarchical',   f'{RESULTS_ROOT}/wp4_hssn_multilabel_self_unknown/best_model.pt'),
    }
else:  # birdnet
    INPUT_DIM = 1024
    EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings.npy'
    META_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings_metadata.csv'
    RESULTS_ROOT = '/user/gaad2403/Bachelor-Thesis/results/multilabel/birdnet/anuraset'
    MODEL_PATHS = {
        'UALB':         ('projectionhead', f'{RESULTS_ROOT}/wp4_ualb_multilabel_unknown/best_model.pt'),
        'FTM-only':     ('hierarchical',   f'{RESULTS_ROOT}/wp4_ftm_only_multilabel_unknown/best_model.pt'),
        'TT-only_self': ('hierarchical',   f'{RESULTS_ROOT}/wp4_tt_only_multilabel_self_unknown/best_model.pt'),
        'FTM+TT_self':  ('hierarchical',   f'{RESULTS_ROOT}/wp4_hssn_multilabel_self_unknown/best_model.pt'),
    }

TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_INTER_SAMPLE = 20000   # cap on inter-group pairs sampled, for speed on large test sets
K_NEIGHBORS = [5, 20, 50]
SEED = 42


# ── Dataset ────────────────────────────────────────────────────────────────
def load_test_set():
    train_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='train',
                                        include_negative=True)
    test_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='test',
                                        species_cols=train_ds.species_cols,
                                        genus_cols=train_ds.genus_cols,
                                        family_cols=train_ds.family_cols,
                                        include_negative=True)
    return train_ds, test_ds


# ── Embedding extractors (mirrors the eval scripts' extractors) ───────────
def extract_baseline(ds):
    emb = ds.embeddings.numpy()
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)


def extract_projectionhead(ds, model_path):
    model = ProjectionHead(input_dim=INPUT_DIM, proj_dim=128).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            z = model(ds.embeddings[i:i + 512].to(DEVICE))
            out.append(z.cpu().numpy())
    return np.vstack(out)


def extract_hierarchical(ds, model_path, n_hierarchy_nodes):
    """For TT-only / FTM+TT (triplet_head is trained)."""
    model = HierarchicalModel(input_dim=INPUT_DIM, hidden_dim=1024, triplet_dim=128,
                               n_hierarchy_nodes=n_hierarchy_nodes).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            z, _ = model(ds.embeddings[i:i + 512].to(DEVICE))
            out.append(z.cpu().numpy())
    return np.vstack(out)


def extract_hidden(ds, model_path, n_hierarchy_nodes):
    """For FTM-only: triplet_head is never trained, so use the shared
    backbone representation h instead (same choice as the main eval
    scripts make for FTM-only's Gap)."""
    model = HierarchicalModel(input_dim=INPUT_DIM, hidden_dim=1024, triplet_dim=128,
                               n_hierarchy_nodes=n_hierarchy_nodes).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            h = model.encode_hidden(ds.embeddings[i:i + 512].to(DEVICE))
            out.append(h.cpu().numpy())
    emb = np.vstack(out)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)


# ── Metric 1: Negative Gap ─────────────────────────────────────────────────
def negative_gap(embeddings, neg_mask, seed=SEED):
    """intra = mean cosine sim among negative samples.
    inter  = mean cosine sim between negative and known samples.
    gap    = intra - inter (same construction as compute_gap, target = negative)."""
    np.random.seed(seed)
    neg_idx = np.where(neg_mask)[0]
    known_idx = np.where(~neg_mask)[0]

    if len(neg_idx) < 2 or len(known_idx) == 0:
        return float('nan'), float('nan'), float('nan')

    neg_emb = embeddings[neg_idx]
    sim = cosine_similarity(neg_emb)
    mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
    intra = sim[mask].mean()

    n_sample = min(N_INTER_SAMPLE, len(neg_idx) * len(known_idx))
    # sample pairs rather than the full neg x known matrix, for speed
    sampled_known = known_idx[np.random.choice(len(known_idx),
                                               min(len(known_idx), max(1, n_sample // max(len(neg_idx), 1))),
                                               replace=False)]
    inter = cosine_similarity(neg_emb, embeddings[sampled_known]).flatten().mean()

    return float(intra), float(inter), float(intra - inter)


# ── Metric 2: k-NN Purity ──────────────────────────────────────────────────
def knn_purity(embeddings, neg_mask, k_values=K_NEIGHBORS):
    """For each negative sample, fraction of its k nearest neighbors (over
    ALL test samples, cosine distance) that are also negative. Compared
    against the base rate (overall negative fraction in the test set)."""
    n = len(embeddings)
    base_rate = neg_mask.mean()

    # cosine distance via normalized L2 (embeddings assumed L2-normalizable;
    # NearestNeighbors with metric='cosine' handles un-normalized inputs too)
    max_k = max(k_values)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric='cosine', n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings[neg_mask])  # neighbors of each negative sample
    indices = indices[:, 1:]  # drop self (index 0 is always the point itself)

    neg_mask_arr = neg_mask.astype(np.float32)
    purities = {}
    for k in k_values:
        neighbor_labels = neg_mask_arr[indices[:, :k]]  # [n_neg, k]
        purity = neighbor_labels.mean()
        purities[k] = float(purity)

    return purities, float(base_rate)


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print(f"Embedding source: {EMBED_SOURCE}")
    print("Loading test set...")
    train_ds, test_ds = load_test_set()
    n_hierarchy_nodes = train_ds.n_families + train_ds.n_genera + train_ds.n_species

    # negative = species vector is all-zero (equivalently genus/family are too)
    neg_mask = test_ds.species_labels.numpy().sum(axis=1) == 0
    n_neg = int(neg_mask.sum())
    n_total = len(neg_mask)
    print(f"Test set: {n_total} samples | negative: {n_neg} ({100 * n_neg / n_total:.1f}%)")

    embeddings = {'Baseline': extract_baseline(test_ds)}

    for name, (kind, path) in MODEL_PATHS.items():
        try:
            if name == 'FTM-only':
                embeddings[name] = extract_hidden(test_ds, path, n_hierarchy_nodes)
            elif kind == 'hierarchical':
                embeddings[name] = extract_hierarchical(test_ds, path, n_hierarchy_nodes)
            else:
                embeddings[name] = extract_projectionhead(test_ds, path)
            print(f"  {name}: loaded")
        except FileNotFoundError:
            print(f"  {name}: checkpoint not found ({path}), skipping")

    print("\nComputing negative-clustering diagnostics...")
    print(f"(base rate for k-NN purity: {100 * n_neg / n_total:.1f}% negative in test set)\n")

    results = {}
    header = f"{'Model':<14} | {'Gap(intra-inter)':>16} | " + " | ".join(f"kNN-{k} purity".rjust(14) for k in K_NEIGHBORS)
    print(header)
    print("-" * len(header))

    for name, emb in embeddings.items():
        intra, inter, gap = negative_gap(emb, neg_mask)
        purities, base_rate = knn_purity(emb, neg_mask)
        results[name] = {'intra': intra, 'inter': inter, 'gap': gap, 'purity': purities, 'base_rate': base_rate}

        row = f"{name:<14} | {gap:>16.4f} | " + " | ".join(f"{purities[k]*100:>13.2f}%" for k in K_NEIGHBORS)
        print(row)

    print(f"\n(For reference, if negatives were NOT clustered, kNN purity should be close to the "
          f"base rate of {100 * n_neg / n_total:.1f}% for all methods.)")

    print("\nInterpretation:")
    print("- Gap noticeably higher for a method than for Baseline/FTM-only suggests that")
    print("  method is pulling negatives together relative to what the raw embedding shows.")
    print("- kNN purity well above the base rate means a negative sample's neighbors are")
    print("  disproportionately also negative -- i.e., negatives form their own neighborhood.")

    return results


if __name__ == '__main__':
    main()