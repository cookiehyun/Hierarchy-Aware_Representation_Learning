"""
Unseen-category evaluation (ESC-50, BirdNET), with hF1 added to the
known-category (closed-set) metrics block only -- same rationale as the
AnuraSet unseen-species version: a held-out category has no corresponding
node in the trained model's output space, so Kiritchenko-style hF1 isn't
computable for the open-set tier itself (literature reports flat per-level
accuracy there instead, which bga_acc_given_flagged_unseen already does).
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

from birdnet.esc50.esc50_dataset_unseen import ESC50EmbeddingDataset
from birdnet.esc50.esc50_hierarchy_tree import ESC50HierarchyTree
from models.projection_head import ProjectionHead
from models.hierarchical_model import HierarchicalModel
from hf1 import (joint_probe_singlelabel, ftm_only_preds_singlelabel,
                                    ground_truth_singlelabel, hierarchical_f1)

SPLIT_PATH = '/user/gaad2403/Bachelor-Thesis/data/unseen_category_split.json'
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_esc50_birdnet/embeddings.npy'
META_PATH  = '/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv'
OUTPUT_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/unseen_category_eval_hf1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = [1, 2, 3, 4]
TEST_FOLDS  = [5]

MODEL_PATHS = {
    'UALB':     ('projectionhead', '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_ualb_l0_unseen/best_model.pt'),
    'FTM-only': ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_ftm_only_unseen/best_model.pt'),
    'TT-only':  ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_tt_only_self_unseen/best_model.pt'),
    'FTM+TT':   ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_hssn_self_unseen/best_model.pt'),
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_PERCENTILE = 5
SEED_METRICS = 42


def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)


def extract_baseline(ds):
    emb = ds.embeddings.numpy()
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)


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
    emb = np.vstack(out)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)


def extract_hierarchical(ds, model_path, n_hierarchy_nodes):
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
    emb = np.vstack(out)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)


def extract_hierarchical_scores(ds, model_path, n_hierarchy_nodes):
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
    return np.vstack(out)


def extract_hidden(ds, model_path, n_hierarchy_nodes):
    """🆕 FTM-only's shared backbone representation h -- see birdnet_wp3_full_eval_normal_hf1.py."""
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


def anomaly_score_from_ftm(scores, cat_node_start, cat_node_end):
    cat_scores = scores[:, cat_node_start:cat_node_end]
    return 1.0 - cat_scores.max(axis=1)


def anomaly_score_from_embedding(test_emb, train_emb, train_category_labels, n_known_categories):
    centroids = np.stack([
        train_emb[train_category_labels == c].mean(axis=0) for c in range(n_known_categories)
    ])
    centroids = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8, None)
    sims = cosine_similarity(test_emb, centroids)
    return 1.0 - sims.max(axis=1)


def bga_pred_from_embedding(test_emb, train_emb, train_bga_labels, n_known_bga):
    centroids = np.stack([
        train_emb[train_bga_labels == c].mean(axis=0) for c in range(n_known_bga)
    ])
    centroids = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8, None)
    sims = cosine_similarity(test_emb, centroids)
    return sims.argmax(axis=1)


def compute_gap(embeddings, labels):
    np.random.seed(SEED_METRICS)
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
    if not intra_sims or not inter_sims:
        return float('nan')
    return np.mean(intra_sims) - np.mean(inter_sims)


def linear_probe_accuracy(train_emb, train_labels, test_emb, test_labels, n_classes):
    torch.manual_seed(SEED_METRICS)
    clf = nn.Linear(train_emb.shape[1], n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.01, weight_decay=1e-4)
    class_counts = np.bincount(train_labels, minlength=n_classes).astype(np.float32)
    class_weights = torch.FloatTensor(1.0 / np.clip(class_counts, 1, None)).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    X_train = torch.FloatTensor(train_emb).to(DEVICE)
    y_train = torch.LongTensor(train_labels).to(DEVICE)
    X_test = torch.FloatTensor(test_emb).to(DEVICE)
    y_test = torch.LongTensor(test_labels).to(DEVICE)
    clf.train()
    for _ in range(300):
        optimizer.zero_grad()
        loss = criterion(clf(X_train), y_train)
        loss.backward()
        optimizer.step()
    clf.eval()
    with torch.no_grad():
        return (clf(X_test).argmax(dim=1) == y_test).float().mean().item() * 100


def clustering_nmi(test_emb, test_labels, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=SEED_METRICS, n_init=10)
    cluster_ids = km.fit_predict(test_emb)
    return normalized_mutual_info_score(test_labels, cluster_ids) * 100


def evaluate(anomaly_scores, is_unseen, val_anomaly_scores_id_only, bga_correct=None):
    auroc = roc_auc_score(is_unseen, anomaly_scores)
    threshold = np.percentile(val_anomaly_scores_id_only, 100 - VAL_PERCENTILE)
    flagged_unseen = anomaly_scores > threshold
    recall_unseen = flagged_unseen[is_unseen].mean() if is_unseen.sum() > 0 else float('nan')
    false_reject_rate = flagged_unseen[~is_unseen].mean() if (~is_unseen).sum() > 0 else float('nan')
    result = {'auroc': auroc, 'threshold': threshold,
              'recall_unseen': recall_unseen, 'false_reject_rate': false_reject_rate}
    correctly_flagged_unseen = flagged_unseen & is_unseen
    if bga_correct is not None and correctly_flagged_unseen.sum() > 0:
        result['bga_acc_given_flagged_unseen'] = bga_correct[correctly_flagged_unseen].mean()
    return result


def main():
    split = load_split()
    near_ood_cats = set(split['near_ood_categories'])
    far_ood_cats = set(split['far_ood_categories'])
    exclude_categories = split['all_held_out_categories']

    print("Loading datasets (train excludes held-out categories; test keeps everything)...")
    train_ds = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TRAIN_FOLDS,
                                     exclude_categories=exclude_categories)

    full_meta = pd.read_csv(META_PATH)
    from sklearn.preprocessing import LabelEncoder
    full_bga_enc = LabelEncoder().fit(full_meta['bga_label'])
    full_midcat_enc = LabelEncoder().fit(full_meta['midcat'])
    full_category_enc = LabelEncoder().fit(full_meta['category'])

    test_ds = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TEST_FOLDS,
                                    bga_enc=full_bga_enc,
                                    midcat_enc=full_midcat_enc,
                                    category_enc=full_category_enc)

    print(f"Train (held-in only): {len(train_ds)} | Test: {len(test_ds)}")

    test_meta = test_ds.meta.reset_index(drop=True)
    is_unseen = test_meta['category'].isin(exclude_categories).values
    is_near_ood = test_meta['category'].isin(near_ood_cats).values
    is_far_ood = test_meta['category'].isin(far_ood_cats).values
    print(f"Test set: {is_unseen.sum()} held-out samples ({is_near_ood.sum()} near-OOD, "
          f"{is_far_ood.sum()} far-OOD) / {len(test_ds)} total")

    n_known_categories = train_ds.n_category
    model_tree = ESC50HierarchyTree(META_PATH, train_ds.bga_enc, train_ds.midcat_enc, train_ds.category_enc)
    n_bga, n_midcat = model_tree.n_bga, model_tree.n_midcat
    cat_node_start = n_bga + n_midcat
    cat_node_end = model_tree.n_nodes
    ancestor_matrix = model_tree.ancestor_matrix()  # 🆕 for hF1, built on train_ds's reduced node space
    print(f"[Model dims] n_nodes={model_tree.n_nodes} "
          f"(bga={n_bga}, midcat={n_midcat}, category={n_known_categories}) "
          f"-- must match the checkpoint's training-time shape")

    bga_ids_test = full_bga_enc.transform(test_meta['bga_label'])

    all_results = {}
    model_list = [('Baseline', None, None)] + [(name, kind, path) for name, (kind, path) in MODEL_PATHS.items()]

    for name, kind, path in model_list:
        print(f"\n--- {name} ---")
        try:
            if name == 'Baseline':
                train_emb = extract_baseline(train_ds)
                test_emb = extract_baseline(test_ds)
                train_cat_labels = train_ds.hier_labels[:, 2].numpy()
                train_bga_labels = train_ds.hier_labels[:, 0].numpy()
                anomaly = anomaly_score_from_embedding(test_emb, train_emb, train_cat_labels, n_known_categories)
                bga_pred = bga_pred_from_embedding(test_emb, train_emb, train_bga_labels, train_ds.n_bga)
            elif name == 'FTM-only':
                test_scores = extract_hierarchical_scores(test_ds, path, model_tree.n_nodes)
                anomaly = anomaly_score_from_ftm(test_scores, cat_node_start, cat_node_end)
                bga_pred = test_scores[:, 0:n_bga].argmax(axis=1)
            else:
                extractor = extract_hierarchical if kind == 'hierarchical' else extract_projectionhead
                train_emb = extractor(train_ds, path, model_tree.n_nodes) if kind == 'hierarchical' else extractor(train_ds, path)
                test_emb = extractor(test_ds, path, model_tree.n_nodes) if kind == 'hierarchical' else extractor(test_ds, path)
                train_cat_labels = train_ds.hier_labels[:, 2].numpy()
                train_bga_labels = train_ds.hier_labels[:, 0].numpy()
                anomaly = anomaly_score_from_embedding(test_emb, train_emb, train_cat_labels, n_known_categories)
                bga_pred = bga_pred_from_embedding(test_emb, train_emb, train_bga_labels, train_ds.n_bga)

            val_anomaly = anomaly[~is_unseen]
            for tier_name, tier_mask in [('near_ood', is_near_ood | ~is_unseen),
                                         ('far_ood', is_far_ood | ~is_unseen)]:
                tier_is_unseen = is_unseen[tier_mask] if tier_name == 'near_ood' else is_far_ood[tier_mask]
                bga_correct = (bga_pred[tier_mask] == bga_ids_test[tier_mask])
                res = evaluate(anomaly[tier_mask], tier_is_unseen, val_anomaly, bga_correct=bga_correct)
                all_results[f"{name}_{tier_name}"] = res
                print(f"  [{tier_name}] AUROC={res['auroc']:.3f} | "
                      f"recall_unseen={res['recall_unseen']:.3f} | "
                      f"false_reject_rate={res['false_reject_rate']:.3f}"
                      + (f" | bga_acc_given_flagged={res.get('bga_acc_given_flagged_unseen', float('nan')):.3f}"
                         if 'bga_acc_given_flagged_unseen' in res else ""))

            # ── Known-category Gap/Acc/NMI + 🆕 hF1, held-in samples only ──
            held_in_mask = ~is_unseen
            test_cat_labels_known = train_ds.category_enc.transform(test_meta['category'][held_in_mask])
            test_bga_labels_known = train_ds.bga_enc.transform(test_meta['bga_label'][held_in_mask])
            test_midcat_labels_known = train_ds.midcat_enc.transform(test_meta['midcat'][held_in_mask])
            known_hier = np.stack([test_bga_labels_known, test_midcat_labels_known,
                                   test_cat_labels_known], axis=1)
            known_gt = ground_truth_singlelabel(known_hier, n_bga, n_midcat, n_known_categories)

            if name == 'FTM-only':
                cat_scores = test_scores[held_in_mask, cat_node_start:cat_node_end]
                acc = (cat_scores.argmax(axis=1) == test_cat_labels_known).mean() * 100
                # 🆕 h instead of s for gap/NMI
                test_hidden_ftm = extract_hidden(test_ds, path, model_tree.n_nodes)
                gap_hidden = test_hidden_ftm[held_in_mask]
                gap = compute_gap(gap_hidden, test_cat_labels_known)
                nmi = clustering_nmi(gap_hidden, test_cat_labels_known, n_known_categories)
                preds = ftm_only_preds_singlelabel(test_scores[held_in_mask], n_bga, n_midcat, n_known_categories)
            else:
                gap_emb = test_emb[held_in_mask]
                gap = compute_gap(gap_emb, test_cat_labels_known)
                acc = linear_probe_accuracy(train_emb, train_cat_labels, gap_emb,
                                            test_cat_labels_known, n_known_categories)
                nmi = clustering_nmi(gap_emb, test_cat_labels_known, n_known_categories)
                train_hier = train_ds.hier_labels.numpy()
                preds = joint_probe_singlelabel(train_emb, train_hier, gap_emb, n_bga, n_midcat, n_known_categories)

            hP, hR, hF1 = hierarchical_f1(preds, known_gt, ancestor_matrix)

            all_results[f"{name}_known_category_metrics"] = {
                'gap': gap, 'acc': acc, 'nmi': nmi, 'hP': hP, 'hR': hR, 'hF1': hF1}
            print(f"  [known-category] Gap={gap:.4f} | Acc={acc:.2f}% | NMI={nmi:.2f}% | hF1={hF1*100:.2f}%")

        except FileNotFoundError:
            print(f"  checkpoint not found ({path}), skipping -- train it first")

    np.save(OUTPUT_DIR / 'unseen_category_eval_results.npy', all_results)
    print(f"\nSaved to: {OUTPUT_DIR}/unseen_category_eval_results.npy")


if __name__ == '__main__':
    main()