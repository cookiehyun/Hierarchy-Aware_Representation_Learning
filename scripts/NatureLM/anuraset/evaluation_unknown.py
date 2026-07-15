"""
WP3 full evaluation on AnuraSet (NatureLM embeddings, single-label subset, NORMAL/42-species) + hF1.
"""

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from NatureLM.anuraset.anuraset_dataset_unknown import AnuraSetEmbeddingDataset  # negative/unknown-inclusive
from NatureLM.anuraset.hierarchy_tree import HierarchyTree
from models.projection_head import ProjectionHead
from models.hierarchical_model import HierarchicalModel
from hf1 import (joint_probe_singlelabel, ftm_only_preds_singlelabel,
                                    ground_truth_singlelabel, hierarchical_f1)

EMBED_PATH    = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH     = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
OUTPUT_DIR    = Path('/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp3_eval_total_unknown_hf1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    'UALB':     ('projectionhead', '/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_ualb_singlelabel_l0_unknown/best_model.pt'),
    'FTM-only': ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_ftm_only_singlelabel_unknown/best_model.pt'),
    'TT-only':  ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_tt_only_singlelabel_self_unknown/best_model.pt'),
    'FTM+TT':   ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_hssn_singlelabel_self_unknown/best_model.pt'),
}

INPUT_DIM    = 4096
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROBE_EPOCHS = 300
PROBE_LR     = 0.01
N_INTER_MULT = 5
SEED         = 42

LEVELS = ['Family', 'Genus', 'Species']
LEVEL_COL = {'Family': 'family', 'Genus': 'genus', 'Species': 'species'}


def load_datasets():
    train_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='train', single_label_only=True)
    test_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='test', single_label_only=True,
                                       family_enc=train_ds.family_enc,
                                       genus_enc=train_ds.genus_enc,
                                       species_enc=train_ds.species_enc)
    return train_ds, test_ds


def extract_baseline(ds):
    emb = ds.embeddings.numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)


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
    emb = np.vstack(out)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)


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
    emb = np.vstack(out)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)


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


def extract_hidden(ds, model_path, n_hierarchy_nodes):
    """🆕 FTM-only's shared backbone representation h -- see naturelm_wp3_full_eval_normal_hf1.py."""
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
                                  min(len(idx) * N_INTER_MULT, len(other_idx)), replace=False)]
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
    train_ds, test_ds = load_datasets()
    tree = HierarchyTree(TAXONOMY_PATH, train_ds.family_enc, train_ds.genus_enc, train_ds.species_enc)
    ancestor_matrix = tree.ancestor_matrix()
    n_family, n_genus, n_species = train_ds.n_families, train_ds.n_genera, train_ds.n_species
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | n_hierarchy_nodes={tree.n_nodes}")

    level_n_classes = {'Family': n_family, 'Genus': n_genus, 'Species': n_species}
    level_col_idx = {'Family': 0, 'Genus': 1, 'Species': 2}
    level_node_range = {
        'Family': (0, n_family), 'Genus': (n_family, n_family + n_genus),
        'Species': (n_family + n_genus, tree.n_nodes),
    }

    print("\nExtracting embeddings...")
    train_emb = {'Baseline': extract_baseline(train_ds)}
    test_emb = {'Baseline': extract_baseline(test_ds)}
    for name, (kind, path) in MODEL_PATHS.items():
        try:
            if name == 'FTM-only':
                train_emb[name] = extract_hierarchical_scores(train_ds, path, tree.n_nodes)
                test_emb[name] = extract_hierarchical_scores(test_ds, path, tree.n_nodes)
            elif kind == 'hierarchical':
                train_emb[name] = extract_hierarchical(train_ds, path, tree.n_nodes)
                test_emb[name] = extract_hierarchical(test_ds, path, tree.n_nodes)
            else:
                train_emb[name] = extract_projectionhead(train_ds, path)
                test_emb[name] = extract_projectionhead(test_ds, path)
            print(f"  {name}: loaded")
        except FileNotFoundError:
            print(f"  {name}: checkpoint not found ({path}), skipping")

    train_hier = train_ds.hier_labels.numpy()
    test_hier = test_ds.hier_labels.numpy()
    test_gt = ground_truth_singlelabel(test_hier, n_family, n_genus, n_species)

    # 🆕 h for FTM-only's Gap/NMI (s stays in test_emb['FTM-only'] for acc/hF1)
    test_hidden = {}
    if 'FTM-only' in test_emb:
        try:
            test_hidden['FTM-only'] = extract_hidden(test_ds, MODEL_PATHS['FTM-only'][1], tree.n_nodes)
        except FileNotFoundError:
            pass

    print("\nComputing gap / probe accuracy / NMI / hF1...")
    results = {}
    for name in train_emb:
        results[name] = {}
        for level in LEVELS:
            col_idx = level_col_idx[level]
            n_classes = level_n_classes[level]
            gap_emb = test_hidden['FTM-only'] if name == 'FTM-only' else test_emb[name]  # 🆕
            intra, inter, gap = compute_gap(gap_emb, test_ds.meta[LEVEL_COL[level]].values)
            train_labels = train_hier[:, col_idx]
            test_labels = test_hier[:, col_idx]

            if name == 'FTM-only':
                node_start, node_end = level_node_range[level]
                acc = direct_node_accuracy(test_emb[name], node_start, node_end, test_labels)
            else:
                acc, majority_acc = linear_probe_accuracy(
                    train_emb[name], train_labels, test_emb[name], test_labels, n_classes)

            nmi = clustering_nmi(gap_emb, test_labels, n_classes)  # 🆕 h instead of s
            results[name][level] = {'gap': gap, 'test_acc': acc, 'nmi': nmi}
            print(f"  [{name:<10}] {level:<8} | Gap: {gap:.4f} | Acc: {acc:.2f}% | NMI: {nmi:.2f}%")

        if name == 'FTM-only':
            preds = ftm_only_preds_singlelabel(test_emb[name], n_family, n_genus, n_species)
        else:
            preds = joint_probe_singlelabel(train_emb[name], train_hier, test_emb[name],
                                            n_family, n_genus, n_species)
        hP, hR, hF1 = hierarchical_f1(preds, test_gt, ancestor_matrix)
        results[name]['hf1'] = {'hP': hP, 'hR': hR, 'hF1': hF1}
        print(f"  [{name:<10}] hierarchical  | hP: {hP*100:.2f}% | hR: {hR*100:.2f}% | hF1: {hF1*100:.2f}%")

    print("\nSummary")
    header = f"{'Model':<12} | " + " | ".join(
        f"{lv} Gap".rjust(10) + " " + f"{lv} Acc".rjust(9) for lv in LEVELS) + " | hF1".rjust(8)
    print(header)
    for name, res in results.items():
        row = f"{name:<12} | " + " | ".join(
            f"{res[lv]['gap']:>10.4f} {res[lv]['test_acc']:>8.2f}%" for lv in LEVELS)
        row += f" | {res['hf1']['hF1'] * 100:>6.2f}%"
        print(row)

    np.save(OUTPUT_DIR / 'wp3_full_eval_results.npy', results)
    print(f"\nSaved to: {OUTPUT_DIR}/wp3_full_eval_results.npy")


if __name__ == '__main__':
    main()