"""
WP4 full evaluation on AnuraSet, multi-label (NatureLM embeddings).

Computes, for Baseline / UALB / FTM-only / TT-only / FTM+TT, at Family /
Genus / Species level:
  - Cosine similarity gap (intra-class - inter-class), computed PER-LABEL
    as a one-vs-rest problem (see compute_gap docstring below -- changed
    from the original exact-label-combination version per Rida's feedback).
  - Multi-label F1: macro/micro F1 at threshold 0.5.
  - Hierarchical F1 (Kiritchenko et al. 2006).

FTM-only is evaluated differently from TT-only/FTM+TT: its triplet_head is
never trained (only hierarchy_head receives gradients from L_FTM), so
extracting z would evaluate a random projection. Instead FTM-only uses s
(the hierarchy_head's sigmoid scores) directly and thresholds it at 0.5 --
no separate probe is trained on top of it. This matches how the official
HieraSeg code computes accuracy straight from seg_logit, never from the
projection head's embedding, and mirrors the single-label fix in
wp3_full_eval_fixed.py (which used direct_node_accuracy the same way).

For Baseline / UALB / TT-only / FTM+TT, the multilabel_probe (trained
linear classifier) is still used, since those embeddings are not already
predictions -- only FTM-only's s is.

Merges: anuraset_baseline_multilabel_eval.py, eval_ualb_multilabel.py, hf1.py.
"""

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')
sys.path.append('/user/gaad2403/Bachelor-Thesis/scripts')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from multilabel.NatureLM.anuraset.anuraset_dataset import AnuraSetMultiLabelDataset
from multilabel.multilabel_hierarchy_tree import MultiLabelHierarchyTree
from models.projection_head import ProjectionHead
from models.hierarchical_model import HierarchicalModel
from multilabel.NatureLM.anuraset.hf1 import joint_probe, hierarchical_f1

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings.npy'
META_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
# 🆕 separate output dir -- gap values are no longer comparable to the old
# exact-label-combination results, so don't overwrite them
OUTPUT_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/multilabel/birdnet/anuraset/wp4_full_eval_onevsrest')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    'UALB': ('projectionhead', '/user/gaad2403/Bachelor-Thesis/results/multilabel/birdnet/anuraset/wp4_ualb_multilabel/best_model.pt'),
    'FTM-only': ('hierarchical', '/user/gaad2403/Bachelor-Thesis/results/multilabel/birdnet/anuraset/wp4_ftm_only_multilabel/best_model.pt'),
    'TT-only_self':   ('hierarchical', '/user/gaad2403/Bachelor-Thesis/results/multilabel/birdnet/anuraset/wp4_tt_only_multilabel_self/best_model.pt'),
    'FTM+TT_self':  ('hierarchical', '/user/gaad2403/Bachelor-Thesis/results/multilabel/birdnet/anuraset/wp4_hssn_multilabel_self/best_model.pt'),
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROBE_EPOCHS = 300
PROBE_LR = 0.01
N_INTER_MULT = 5
THRESHOLD = 0.5
SEED = 42

LEVELS = ['Family', 'Genus', 'Species']


# ── Dataset ────────────────────────────────────────────────────────────────
def load_datasets():
    train_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='train')
    test_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='test',
                                        species_cols=train_ds.species_cols,
                                        genus_cols=train_ds.genus_cols,
                                        family_cols=train_ds.family_cols)
    return train_ds, test_ds


# ── Embedding extractors ────────────────────────────────────────────────────
def extract_baseline(ds):
    emb = ds.embeddings.numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)


def extract_projectionhead(ds, model_path):
    model = ProjectionHead(input_dim=1024, proj_dim=128).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            z = model(ds.embeddings[i:i + 512].to(DEVICE))
            out.append(z.cpu().numpy())
    return np.vstack(out)  # already L2-normalized by the model's forward()


def extract_hierarchical(ds, model_path, n_hierarchy_nodes):
    """For TT-only / FTM+TT: triplet_head IS trained (L_TT backprops into it),
    so z is a valid learned embedding here."""
    model = HierarchicalModel(input_dim=1024, hidden_dim=1024, triplet_dim=128,
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


def extract_hierarchical_scores(ds, model_path, n_hierarchy_nodes):
    """For FTM-only: triplet_head is never trained, so extract s (sigmoid
    scores) instead of z. s is already family+genus+species concatenated,
    in the same node order used at training time (tree.ground_truth)."""
    model = HierarchicalModel(input_dim=1024, hidden_dim=1024, triplet_dim=128,
                               n_hierarchy_nodes=n_hierarchy_nodes).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            _, s = model(ds.embeddings[i:i + 512].to(DEVICE))
            out.append(s.cpu().numpy())
    return np.vstack(out)  # sigmoid scores in [0,1], no re-normalization


def extract_hidden(ds, model_path, n_hierarchy_nodes):
    """🆕 FTM-only's shared backbone representation h -- see naturelm_wp4_full_eval_onevsrest.py."""
    model = HierarchicalModel(input_dim=1024, hidden_dim=1024, triplet_dim=128,
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


EXTRACTORS = {'projectionhead': extract_projectionhead, 'hierarchical': extract_hierarchical}


# ── Metrics ──────────────────────────────────────────────────────────────
def compute_gap(embeddings, multihot_labels, n_inter_mult=N_INTER_MULT, seed=SEED):
    """
    🆕 Per-label one-vs-rest gap (replaces the exact-label-combination
    version). Rationale (Rida's feedback): grouping samples by their exact
    multi-hot combination treats "species A alone" and "species A +
    species B" as two unrelated classes, even though both genuinely
    contain species A. Since combinations are numerous relative to sample
    count, most "classes" ended up with only 1-2 samples, diluting what
    the gap is meant to measure.

    Fix: for each label/species column independently,
      intra = mean cosine similarity among all samples where that label is
              present (regardless of what else co-occurs),
      inter = mean cosine similarity between those samples and samples
              where the label is absent,
    then average (intra - inter) across all labels that have at least 2
    positive samples and at least 1 negative sample.
    """
    np.random.seed(seed)
    n_samples, n_labels = multihot_labels.shape

    per_label_intra, per_label_inter, per_label_gap = [], [], []

    for j in range(n_labels):
        pos_idx = np.where(multihot_labels[:, j] == 1)[0]
        neg_idx = np.where(multihot_labels[:, j] == 0)[0]
        if len(pos_idx) < 2 or len(neg_idx) == 0:
            continue

        pos_emb = embeddings[pos_idx]
        sim = cosine_similarity(pos_emb)
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        intra_sims = sim[mask]

        n_sample = min(len(pos_idx) * n_inter_mult, len(neg_idx))
        sample_neg = neg_idx[np.random.choice(len(neg_idx), n_sample, replace=False)]
        inter_sims = cosine_similarity(pos_emb, embeddings[sample_neg]).flatten()

        li, le = intra_sims.mean(), inter_sims.mean()
        per_label_intra.append(li)
        per_label_inter.append(le)
        per_label_gap.append(li - le)

    if not per_label_gap:
        return float('nan'), float('nan'), float('nan')

    return float(np.mean(per_label_intra)), float(np.mean(per_label_inter)), float(np.mean(per_label_gap))


def multilabel_probe(train_emb, train_labels, test_emb, test_labels, n_classes):
    """Linear probe, sigmoid + BCE, one independent binary classifier per
    class. Used for Baseline/UALB/TT-only/FTM+TT, where the extracted
    features are not already predictions."""
    torch.manual_seed(SEED)
    clf = nn.Linear(train_emb.shape[1], n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=PROBE_LR, weight_decay=1e-4)

    pos_count = train_labels.sum(axis=0)
    neg_count = train_labels.shape[0] - pos_count
    pos_weight = torch.FloatTensor(neg_count / np.clip(pos_count, 1, None)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train = torch.FloatTensor(train_emb).to(DEVICE)
    y_train = torch.FloatTensor(train_labels).to(DEVICE)
    X_test = torch.FloatTensor(test_emb).to(DEVICE)
    y_test = torch.FloatTensor(test_labels).to(DEVICE)

    clf.train()
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(clf(X_train), y_train)
        loss.backward()
        optimizer.step()

    clf.eval()
    with torch.no_grad():
        probs = torch.sigmoid(clf(X_test))
        preds = (probs > THRESHOLD).float().cpu().numpy()

    y_true = y_test.cpu().numpy()
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, preds, average='micro', zero_division=0)
    return macro_f1, micro_f1


def direct_node_f1(scores, node_start, node_end, true_labels):
    """Official-style prediction for FTM-only: no separate probe, just
    threshold the already-trained per-node scores for this level's node
    range. Mirrors direct_node_accuracy in the single-label eval script."""
    preds = (scores[:, node_start:node_end] > THRESHOLD).astype(np.float32)
    macro_f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(true_labels, preds, average='micro', zero_division=0)
    return macro_f1, micro_f1


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    print("Loading datasets...")
    train_ds, test_ds = load_datasets()
    n_hierarchy_nodes = train_ds.n_families + train_ds.n_genera + train_ds.n_species
    print(f"n_hierarchy_nodes (family+genus+species) = {n_hierarchy_nodes}")

    level_labels_train = {
        'Family': train_ds.family_labels.numpy(),
        'Genus': train_ds.genus_labels.numpy(),
        'Species': train_ds.species_labels.numpy(),
    }
    level_labels_test = {
        'Family': test_ds.family_labels.numpy(),
        'Genus': test_ds.genus_labels.numpy(),
        'Species': test_ds.species_labels.numpy(),
    }
    level_node_range = {
        'Family': (0, train_ds.n_families),
        'Genus': (train_ds.n_families, train_ds.n_families + train_ds.n_genera),
        'Species': (train_ds.n_families + train_ds.n_genera, n_hierarchy_nodes),
    }

    # joint family+genus+species ground truth, same node order as the
    # ancestor matrix (family, then genus, then species)
    train_gt_all = np.concatenate(
        [level_labels_train['Family'], level_labels_train['Genus'], level_labels_train['Species']], axis=1)
    test_gt_all = np.concatenate(
        [level_labels_test['Family'], level_labels_test['Genus'], level_labels_test['Species']], axis=1)

    tree = MultiLabelHierarchyTree(TAXONOMY_PATH, train_ds.species_cols,
                                   train_ds.genus_cols, train_ds.family_cols)
    ancestor_matrix = tree.ancestor_matrix()

    print("\nExtracting embeddings...")
    train_emb = {'Baseline': extract_baseline(train_ds)}
    test_emb = {'Baseline': extract_baseline(test_ds)}

    for name, (kind, path) in MODEL_PATHS.items():
        try:
            if name == 'FTM-only':
                train_emb[name] = extract_hierarchical_scores(train_ds, path, n_hierarchy_nodes)
                test_emb[name] = extract_hierarchical_scores(test_ds, path, n_hierarchy_nodes)
            elif kind == 'hierarchical':
                train_emb[name] = extract_hierarchical(train_ds, path, n_hierarchy_nodes)
                test_emb[name] = extract_hierarchical(test_ds, path, n_hierarchy_nodes)
            else:
                train_emb[name] = extract_projectionhead(train_ds, path)
                test_emb[name] = extract_projectionhead(test_ds, path)
            print(f"  {name}: loaded")
        except FileNotFoundError:
            print(f"  {name}: checkpoint not found ({path}), skipping")

    # 🆕 h for FTM-only's Gap (s stays in test_emb['FTM-only'] for Macro/Micro F1/hF1)
    test_hidden = {}
    if 'FTM-only' in test_emb:
        try:
            test_hidden['FTM-only'] = extract_hidden(test_ds, MODEL_PATHS['FTM-only'][1], n_hierarchy_nodes)
        except FileNotFoundError:
            pass

    print("\nComputing gap (one-vs-rest), per-level macro/micro F1, and hF1...")
    results = {}
    for name in train_emb:
        results[name] = {'levels': {}}
        for level in LEVELS:
            train_labels = level_labels_train[level]
            test_labels = level_labels_test[level]
            n_classes = train_labels.shape[1]

            gap_emb = test_hidden['FTM-only'] if name == 'FTM-only' else test_emb[name]  # 🆕
            intra, inter, gap = compute_gap(gap_emb, test_labels)

            if name == 'FTM-only':
                node_start, node_end = level_node_range[level]
                macro_f1, micro_f1 = direct_node_f1(test_emb[name], node_start, node_end, test_labels)
            else:
                macro_f1, micro_f1 = multilabel_probe(
                    train_emb[name], train_labels, test_emb[name], test_labels, n_classes)

            results[name]['levels'][level] = {'gap': gap, 'macro_f1': macro_f1, 'micro_f1': micro_f1}
            print(f"  [{name:<10}] {level:<8} | Gap: {gap:.4f} | "
                  f"Macro F1: {macro_f1 * 100:.2f}% | Micro F1: {micro_f1 * 100:.2f}%")

        # hF1: FTM-only uses s directly (already the prediction, node order
        # matches train_gt_all); everyone else trains a joint probe
        if name == 'FTM-only':
            preds = (test_emb[name] > THRESHOLD).astype(np.float32)
        else:
            preds = joint_probe(train_emb[name], train_gt_all, test_emb[name], test_gt_all,
                                n_nodes=n_hierarchy_nodes)
        hP, hR, hF1 = hierarchical_f1(preds, test_gt_all, ancestor_matrix)
        results[name]['hf1'] = {'hP': hP, 'hR': hR, 'hF1': hF1}
        print(f"  [{name:<10}] hierarchical  | hP: {hP*100:.2f}% | hR: {hR*100:.2f}% | hF1: {hF1*100:.2f}%")

    print("\nSummary")
    header = (f"{'Model':<22} | " + " | ".join(
        f"{lv} Gap".rjust(10) + " " + f"{lv} MacroF1".rjust(11) for lv in LEVELS)
        + " | hF1".rjust(8))
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        row = f"{name:<22} | "
        row += " | ".join(
            f"{res['levels'][lv]['gap']:>10.4f} {res['levels'][lv]['macro_f1'] * 100:>10.2f}%" for lv in LEVELS)
        row += f" | {res['hf1']['hF1'] * 100:>6.2f}%"
        print(row)

    np.save(OUTPUT_DIR / 'wp4_full_eval_results.npy', results)
    print(f"\nSaved to: {OUTPUT_DIR}/wp4_full_eval_results.npy")


if __name__ == '__main__':
    main()