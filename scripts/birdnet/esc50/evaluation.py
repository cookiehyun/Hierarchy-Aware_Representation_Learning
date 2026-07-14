"""
WP3 evaluation on ESC-50 (BirdNET embeddings), NORMAL (all 50 categories) + hF1.

Note: ESC-50 has no negative/"unknown" analogue (every clip already belongs
to exactly one of 50 categories), so there is no separate "unknown" variant
of this script -- just normal and unseen (see the unseen-category script).

Adds hF1 (Kiritchenko et al. 2006) via a joint linear probe over
bga+midcat+category, reusing singlelabel_hf1_helper (family/genus/species
naming there maps positionally to bga/midcat/category here).
"""

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from birdnet.esc50.esc50_dataset import ESC50EmbeddingDataset
from birdnet.esc50.esc50_hierarchy_tree import ESC50HierarchyTree
from models.projection_head import ProjectionHead
from models.hierarchical_model import HierarchicalModel
from hf1 import (joint_probe_singlelabel, ftm_only_preds_singlelabel,
                                    ground_truth_singlelabel, hierarchical_f1)

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_esc50_birdnet/embeddings.npy'
META_PATH  = '/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv'
OUTPUT_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_eval_total_hf1')  # 🆕 fixed naming
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = [1, 2, 3, 4]
TEST_FOLDS  = [5]

MODEL_PATHS = {
    'UALB':          ('projectionhead', '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_ualb_l0/best_model.pt'),
    'FTM-only':      ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_ftm_only/best_model.pt'),
    'TT-only_self':  ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_tt_only_self/best_model.pt'),
    'FTM+TT_self':   ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_hssn_self/best_model.pt'),
}

INPUT_DIM    = 1024
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROBE_EPOCHS = 300
PROBE_LR     = 0.01
SEED         = 42
LEVELS       = [('BGA', 'bga_label', 0), ('Midcat', 'midcat', 1), ('Category', 'category', 2)]


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


def extract_hierarchical_scores(ds, model_path, n_hierarchy_nodes):
    model = HierarchicalModel(input_dim=INPUT_DIM, hidden_dim=1024, triplet_dim=128,
                              n_hierarchy_nodes=n_hierarchy_nodes).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            _, s = model(ds.embeddings[i:i + 512].to(DEVICE))
            out.append(s.cpu().numpy())
    return np.vstack(out)


def direct_node_accuracy(scores, node_start, node_end, true_labels):
    preds = scores[:, node_start:node_end].argmax(axis=1)
    return (preds == true_labels).mean() * 100


def compute_gap(embeddings, labels):
    np.random.seed(SEED)
    classes = np.unique(labels)
    intra_sims, inter_sims = [], []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        if len(idx) < 2:
            continue
        sub = embeddings[idx]
        sim = cosine_similarity(sub)
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        intra_sims.extend(sim[mask].tolist())
        other_idx = np.where(labels != cls)[0]
        if len(other_idx) == 0:
            continue
        sample_other = other_idx[np.random.choice(len(other_idx),
                                  min(len(idx) * 5, len(other_idx)), replace=False)]
        inter_sims.extend(cosine_similarity(sub, embeddings[sample_other]).flatten().tolist())
    intra, inter = np.mean(intra_sims), np.mean(inter_sims)
    return intra, inter, intra - inter


def linear_probe_accuracy(train_emb, train_labels, test_emb, test_labels, n_classes):
    torch.manual_seed(SEED)
    clf = nn.Linear(train_emb.shape[1], n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    class_counts = np.bincount(train_labels, minlength=n_classes).astype(np.float32)
    class_weights = torch.FloatTensor(1.0 / np.clip(class_counts, 1, None)).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    X_train = torch.FloatTensor(train_emb).to(DEVICE)
    y_train = torch.LongTensor(train_labels).to(DEVICE)
    X_test = torch.FloatTensor(test_emb).to(DEVICE)
    y_test = torch.LongTensor(test_labels).to(DEVICE)
    clf.train()
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(clf(X_train), y_train)
        loss.backward()
        optimizer.step()
    clf.eval()
    with torch.no_grad():
        test_acc = (clf(X_test).argmax(dim=1) == y_test).float().mean().item() * 100
    majority_class = np.bincount(train_labels, minlength=n_classes).argmax()
    majority_acc = (test_labels == majority_class).mean() * 100
    return test_acc, majority_acc


def clustering_nmi(test_emb, test_labels, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    cluster_ids = km.fit_predict(test_emb)
    return normalized_mutual_info_score(test_labels, cluster_ids) * 100


def main():
    print("Loading datasets...")
    train_ds = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TRAIN_FOLDS)
    test_ds = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TEST_FOLDS,
                                    bga_enc=train_ds.bga_enc,
                                    midcat_enc=train_ds.midcat_enc,
                                    category_enc=train_ds.category_enc)
    meta = test_ds.meta.reset_index(drop=True)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    tree = ESC50HierarchyTree(META_PATH, train_ds.bga_enc, train_ds.midcat_enc, train_ds.category_enc)
    ancestor_matrix = tree.ancestor_matrix()
    n_bga, n_midcat, n_category = train_ds.n_bga, train_ds.n_midcat, train_ds.n_category
    print(f"n_hierarchy_nodes = {tree.n_nodes}")

    LEVEL_INFO = {'BGA': (0, n_bga), 'Midcat': (1, n_midcat), 'Category': (2, n_category)}
    LEVEL_NODE_RANGE = {
        'BGA': (0, n_bga), 'Midcat': (n_bga, n_bga + n_midcat),
        'Category': (n_bga + n_midcat, tree.n_nodes),
    }

    print("\nExtracting embeddings...")
    train_emb = {'Baseline': extract_baseline(train_ds)}
    test_emb = {'Baseline': extract_baseline(test_ds)}
    for name, (kind, path) in MODEL_PATHS.items():
        try:
            if name == 'FTM-only':
                train_emb[name] = extract_hierarchical_scores(train_ds, path, tree.n_nodes)
                test_emb[name] = extract_hierarchical_scores(test_ds, path, tree.n_nodes)
            else:
                extractor = extract_hierarchical if kind == 'hierarchical' else extract_projectionhead
                train_emb[name] = extractor(train_ds, path, tree.n_nodes) if kind == 'hierarchical' else extractor(train_ds, path)
                test_emb[name] = extractor(test_ds, path, tree.n_nodes) if kind == 'hierarchical' else extractor(test_ds, path)
            print(f"  {name}: loaded")
        except FileNotFoundError:
            print(f"  {name}: checkpoint not found ({path}), skipping")

    train_hier = train_ds.hier_labels.numpy()
    test_hier = test_ds.hier_labels.numpy()
    test_gt = ground_truth_singlelabel(test_hier, n_bga, n_midcat, n_category)

    print("\nComputing gap / probe accuracy / NMI / hF1...")
    results = {}
    for name in train_emb:
        results[name] = {}
        for level, col, col_idx in LEVELS:
            n_classes = LEVEL_INFO[level][1]
            intra, inter, gap = compute_gap(test_emb[name], meta[col].values)
            train_labels = train_hier[:, col_idx]
            test_labels = test_hier[:, col_idx]

            if name == 'FTM-only':
                node_start, node_end = LEVEL_NODE_RANGE[level]
                acc = direct_node_accuracy(test_emb[name], node_start, node_end, test_labels)
            else:
                acc, majority_acc = linear_probe_accuracy(
                    train_emb[name], train_labels, test_emb[name], test_labels, n_classes)

            nmi = clustering_nmi(test_emb[name], test_labels, n_classes)
            results[name][level] = {'gap': gap, 'acc': acc, 'nmi': nmi}
            print(f"  [{name:<12}] {level:<9} | Gap: {gap:.4f} | Acc: {acc:.2f}% | NMI: {nmi:.2f}%")

        # 🆕 hF1
        if name == 'FTM-only':
            preds = ftm_only_preds_singlelabel(test_emb[name], n_bga, n_midcat, n_category)
        else:
            preds = joint_probe_singlelabel(train_emb[name], train_hier, test_emb[name],
                                            n_bga, n_midcat, n_category)
        hP, hR, hF1 = hierarchical_f1(preds, test_gt, ancestor_matrix)
        results[name]['hf1'] = {'hP': hP, 'hR': hR, 'hF1': hF1}
        print(f"  [{name:<12}] hierarchical  | hP: {hP*100:.2f}% | hR: {hR*100:.2f}% | hF1: {hF1*100:.2f}%")

    print("\nSummary")
    for name, res in results.items():
        row = f"{name:<14} | " + " | ".join(
            f"{res[lv]['gap']:.4f} {res[lv]['acc']:.2f}%" for lv in ['BGA', 'Midcat', 'Category'])
        row += f" | hF1={res['hf1']['hF1']*100:.2f}%"
        print(row)

    np.save(OUTPUT_DIR / 'wp3_eval_results.npy', results)
    print(f"\nSaved to: {OUTPUT_DIR}/wp3_eval_results.npy")


if __name__ == '__main__':
    main()