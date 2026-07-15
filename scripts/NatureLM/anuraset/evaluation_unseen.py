"""
Unseen-species evaluation (BirdNET), with hF1 added to the known-species
(closed-set) metrics block.

Protocol (Bjerge et al. 2023, Ecological Informatics -- same anomaly-tagging
idea, adapted to our FTM sigmoid scores / embedding-distance setting):

  1. Train each model on the AnuraSet single-label subset with the
     held-out species (from unseen_species_split.json) completely removed
     from train/val.
  2. On a test set containing BOTH held-in and held-out species:
       - Anomaly score, AUROC, recall_unseen, false_reject_rate, family
         accuracy given flagged unseen -- unchanged from before.
  3. 🆕 Known-species (held-in) metrics now also include hF1 (Kiritchenko
     et al. 2006), via a joint linear probe over the reduced
     family+genus+species node space.

     NOT added to the open-set/unseen tier itself: per literature (Digital
     Taxonomist, HComP-Net -- see chat), a held-out species has no
     corresponding node in this model's output space at all (it was
     excluded from training), so predicted and ground-truth label sets
     can't be compared in the same space the way Kiritchenko's hF1
     requires. Those papers report flat per-level accuracy for unseen
     classes instead, which is exactly what family_acc_given_flagged_unseen
     / genus_acc_given_flagged_unseen already do here.
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

from NatureLM.anuraset.anuraset_dataset_unseen import AnuraSetEmbeddingDataset
from NatureLM.anuraset.hierarchy_tree import HierarchyTree
from models.projection_head import ProjectionHead
from models.hierarchical_model import HierarchicalModel
from hf1 import (joint_probe_singlelabel, ftm_only_preds_singlelabel,
                                    ground_truth_singlelabel, hierarchical_f1)

SPLIT_PATH = '/user/gaad2403/Bachelor-Thesis/data/unseen_species_split.json'
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
OUTPUT_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/unseen_species_eval_hf1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_PERCENTILE = 5
SEED_METRICS = 42


def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)


MODEL_PATHS = {
    'UALB':     ('projectionhead', '/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_ualb_singlelabel_l0_unseen/best_model.pt'),
    'FTM-only': ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_ftm_only_singlelabel_unseen/best_model.pt'),
    'TT-only':  ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_tt_only_singlelabel_self_unseen/best_model.pt'),
    'FTM+TT':   ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_hssn_singlelabel_self_unseen/best_model.pt'),
}


def extract_projectionhead(ds, model_path):
    model = ProjectionHead(input_dim=4096, proj_dim=128).to(DEVICE)
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
    """TT-only / FTM+TT: triplet_head IS trained, so z is a valid learned embedding."""
    model = HierarchicalModel(input_dim=4096, hidden_dim=1024, triplet_dim=128,
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
    """FTM-only: triplet_head is never trained, so use sigmoid scores s directly."""
    model = HierarchicalModel(input_dim=4096, hidden_dim=1024, triplet_dim=128,
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
    """🆕 FTM-only's shared backbone representation h -- genuinely trained
    (gradient flows hierarchy_head -> h -> backbone), unlike s. Use for
    known-species Gap/NMI; keep s for accuracy/hF1/anomaly-score."""
    model = HierarchicalModel(input_dim=4096, hidden_dim=1024, triplet_dim=128,
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


def anomaly_score_from_ftm(scores, species_node_start, species_node_end):
    species_scores = scores[:, species_node_start:species_node_end]
    return 1.0 - species_scores.max(axis=1)


def anomaly_score_from_embedding(test_emb, train_emb, train_species_labels, n_known_species):
    centroids = np.stack([
        train_emb[train_species_labels == c].mean(axis=0) for c in range(n_known_species)
    ])
    centroids = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8, None)
    sims = cosine_similarity(test_emb, centroids)
    return 1.0 - sims.max(axis=1)


def family_pred_from_embedding(test_emb, train_emb, train_family_labels, n_known_families):
    centroids = np.stack([
        train_emb[train_family_labels == c].mean(axis=0) for c in range(n_known_families)
    ])
    centroids = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8, None)
    sims = cosine_similarity(test_emb, centroids)
    return sims.argmax(axis=1)


def extract_baseline(ds):
    emb = ds.embeddings.numpy()
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)


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
        return float('nan'), float('nan'), float('nan')
    intra, inter = np.mean(intra_sims), np.mean(inter_sims)
    return intra, inter, intra - inter


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
        test_acc = (clf(X_test).argmax(dim=1) == y_test).float().mean().item() * 100
    return test_acc


def clustering_nmi(test_emb, test_labels, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=SEED_METRICS, n_init=10)
    cluster_ids = km.fit_predict(test_emb)
    return normalized_mutual_info_score(test_labels, cluster_ids) * 100


def evaluate(anomaly_scores, is_unseen, val_anomaly_scores_id_only, family_correct=None, genus_correct=None):
    auroc = roc_auc_score(is_unseen, anomaly_scores)
    threshold = np.percentile(val_anomaly_scores_id_only, 100 - VAL_PERCENTILE)
    flagged_unseen = anomaly_scores > threshold

    recall_unseen = flagged_unseen[is_unseen].mean() if is_unseen.sum() > 0 else float('nan')
    false_reject_rate = flagged_unseen[~is_unseen].mean() if (~is_unseen).sum() > 0 else float('nan')

    result = {'auroc': auroc, 'threshold': threshold,
              'recall_unseen': recall_unseen, 'false_reject_rate': false_reject_rate}

    correctly_flagged_unseen = flagged_unseen & is_unseen
    if family_correct is not None and correctly_flagged_unseen.sum() > 0:
        result['family_acc_given_flagged_unseen'] = family_correct[correctly_flagged_unseen].mean()
    if genus_correct is not None and correctly_flagged_unseen.sum() > 0:
        result['genus_acc_given_flagged_unseen'] = genus_correct[correctly_flagged_unseen].mean()
    return result


def main():
    split = load_split()
    exclude_pairs = split['all_held_out_pairs']
    near_ood_codes = {p['code'] for p in split['near_ood_pairs']}
    far_ood_codes = {p['code'] for p in split['far_ood_pairs']}

    print("Loading datasets (train/val exclude held-out species; test keeps everything)...")
    train_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='train',
                                        single_label_only=True, exclude_pairs=exclude_pairs)

    full_meta = pd.read_csv(META_PATH)
    full_species_enc = LabelEncoder().fit(full_meta['species'])
    full_genus_enc = LabelEncoder().fit(full_meta['genus'])

    test_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='test',
                                       single_label_only=True,
                                       family_enc=train_ds.family_enc,
                                       genus_enc=full_genus_enc,
                                       species_enc=full_species_enc)

    print(f"Train (held-in only): {len(train_ds)} | Test (held-in + held-out): {len(test_ds)}")

    test_meta = test_ds.meta.reset_index(drop=True)
    held_out_code_set = {p['code'] for p in exclude_pairs}
    tax = pd.read_csv(TAXONOMY_PATH)
    pair_to_code = {(row['genus'], row['species']): row['code'] for _, row in tax.iterrows()}
    test_codes = np.array([pair_to_code.get((g, s), None)
                           for g, s in zip(test_meta['genus'], test_meta['species'])])
    is_unseen = np.isin(test_codes, list(held_out_code_set))
    is_near_ood = np.isin(test_codes, list(near_ood_codes))
    is_far_ood = np.isin(test_codes, list(far_ood_codes))
    print(f"Test set: {is_unseen.sum()} held-out samples ({is_near_ood.sum()} near-OOD, "
          f"{is_far_ood.sum()} far-OOD) / {len(test_ds)} total")

    n_known_species = train_ds.n_species
    model_tree = HierarchyTree(TAXONOMY_PATH, train_ds.family_enc,
                               train_ds.genus_enc, train_ds.species_enc)
    n_family, n_genus = model_tree.n_family, model_tree.n_genus
    species_node_start = n_family + n_genus
    species_node_end = model_tree.n_nodes
    ancestor_matrix = model_tree.ancestor_matrix()  # 🆕 for hF1, built on train_ds's reduced node space
    print(f"[Model dims] n_nodes={model_tree.n_nodes} "
          f"(family={n_family}, genus={n_genus}, species={model_tree.n_species}) "
          f"-- must match the checkpoint's training-time shape")

    family_ids_test = train_ds.family_enc.transform(test_meta['family'])
    train_genus_set = set(train_ds.genus_enc.classes_)
    genus_checkable = test_meta['genus'].isin(train_genus_set).values

    all_results = {}
    model_list = [('Baseline', None, None)] + [(name, kind, path) for name, (kind, path) in MODEL_PATHS.items()]

    for name, kind, path in model_list:
        print(f"\n--- {name} ---")
        try:
            if name == 'Baseline':
                train_emb = extract_baseline(train_ds)
                test_emb = extract_baseline(test_ds)
                train_species_labels = train_ds.hier_labels[:, 2].numpy()
                train_family_labels = train_ds.hier_labels[:, 0].numpy()
                anomaly = anomaly_score_from_embedding(test_emb, train_emb, train_species_labels, n_known_species)
                family_pred = family_pred_from_embedding(test_emb, train_emb, train_family_labels, train_ds.n_families)
            elif name == 'FTM-only':
                test_scores = extract_hierarchical_scores(test_ds, path, model_tree.n_nodes)
                anomaly = anomaly_score_from_ftm(test_scores, species_node_start, species_node_end)
                family_pred = test_scores[:, 0:n_family].argmax(axis=1)
            else:
                extractor = extract_hierarchical if kind == 'hierarchical' else extract_projectionhead
                train_emb = extractor(train_ds, path, model_tree.n_nodes) if kind == 'hierarchical' else extractor(train_ds, path)
                test_emb = extractor(test_ds, path, model_tree.n_nodes) if kind == 'hierarchical' else extractor(test_ds, path)
                train_species_labels = train_ds.hier_labels[:, 2].numpy()
                train_family_labels = train_ds.hier_labels[:, 0].numpy()
                anomaly = anomaly_score_from_embedding(test_emb, train_emb, train_species_labels, n_known_species)
                family_pred = family_pred_from_embedding(test_emb, train_emb, train_family_labels, train_ds.n_families)

            # ── Open-set (unseen-detection) evaluation, per tier -- unchanged ──
            val_anomaly = anomaly[~is_unseen]
            for tier_name, tier_mask in [('near_ood', is_near_ood | ~is_unseen),
                                         ('far_ood', is_far_ood | ~is_unseen)]:
                tier_is_unseen = is_unseen[tier_mask] if tier_name == 'near_ood' else is_far_ood[tier_mask]
                family_correct = None
                if family_pred is not None:
                    family_correct = (family_pred[tier_mask] == family_ids_test[tier_mask])
                res = evaluate(anomaly[tier_mask], tier_is_unseen, val_anomaly, family_correct=family_correct)
                all_results[f"{name}_{tier_name}"] = res
                print(f"  [{tier_name}] AUROC={res['auroc']:.3f} | "
                      f"recall_unseen={res['recall_unseen']:.3f} | "
                      f"false_reject_rate={res['false_reject_rate']:.3f}"
                      + (f" | family_acc_given_flagged={res.get('family_acc_given_flagged_unseen', float('nan')):.3f}"
                         if 'family_acc_given_flagged_unseen' in res else ""))

            # ── Known-species Gap/Acc/NMI + 🆕 hF1, held-in samples only ──
            held_in_mask = ~is_unseen
            test_species_labels_known = train_ds.species_enc.transform(test_meta['species'][held_in_mask])
            test_family_labels_known = train_ds.family_enc.transform(test_meta['family'][held_in_mask])
            test_genus_labels_known = train_ds.genus_enc.transform(test_meta['genus'][held_in_mask])
            known_hier = np.stack([test_family_labels_known, test_genus_labels_known,
                                   test_species_labels_known], axis=1)
            known_gt = ground_truth_singlelabel(known_hier, n_family, n_genus, n_known_species)

            if name != 'FTM-only':
                gap_emb = test_emb[held_in_mask]
                intra, inter, gap = compute_gap(gap_emb, test_species_labels_known)
                acc = linear_probe_accuracy(train_emb, train_species_labels, gap_emb,
                                            test_species_labels_known, n_known_species)
                nmi = clustering_nmi(gap_emb, test_species_labels_known, n_known_species)
                train_hier = train_ds.hier_labels.numpy()
                preds = joint_probe_singlelabel(train_emb, train_hier, gap_emb, n_family, n_genus, n_known_species)
            else:
                species_scores = test_scores[held_in_mask, species_node_start:species_node_end]
                acc = (species_scores.argmax(axis=1) == test_species_labels_known).mean() * 100
                # 🆕 h instead of s for gap/NMI -- s (species_scores) stays for acc/preds only
                test_hidden_ftm = extract_hidden(test_ds, path, model_tree.n_nodes)
                gap_hidden = test_hidden_ftm[held_in_mask]
                intra, inter, gap = compute_gap(gap_hidden, test_species_labels_known)
                nmi = clustering_nmi(gap_hidden, test_species_labels_known, n_known_species)
                preds = ftm_only_preds_singlelabel(test_scores[held_in_mask], n_family, n_genus, n_known_species)

            hP, hR, hF1 = hierarchical_f1(preds, known_gt, ancestor_matrix)

            all_results[f"{name}_known_species_metrics"] = {
                'gap': gap, 'acc': acc, 'nmi': nmi, 'hP': hP, 'hR': hR, 'hF1': hF1}
            print(f"  [known-species] Gap={gap:.4f} | Acc={acc:.2f}% | NMI={nmi:.2f}% | hF1={hF1*100:.2f}%")

        except FileNotFoundError:
            print(f"  checkpoint not found ({path}), skipping -- train it first (see train_*_unseen_holdout.py)")

    np.save(OUTPUT_DIR / 'unseen_species_eval_results.npy', all_results)
    print(f"\nSaved to: {OUTPUT_DIR}/unseen_species_eval_results.npy")


if __name__ == '__main__':
    main()