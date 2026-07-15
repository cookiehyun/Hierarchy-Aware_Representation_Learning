"""
Unseen-species evaluation.

Protocol (Bjerge et al. 2023, Ecological Informatics -- same anomaly-tagging
idea, adapted to our FTM sigmoid scores / embedding-distance setting):

  1. Train each model on the AnuraSet single-label subset with the
     held-out species (from unseen_species_split.json) completely removed
     from train/val.
  2. On a test set containing BOTH held-in and held-out species:
       - Anomaly score:
           * FTM-only / FTM+TT (have hierarchy_head): 1 - max(species-level
             sigmoid scores). High score = model isn't confident about any
             known species = likely unseen.
           * UALB / TT-only / Baseline (embedding only): distance to the
             nearest of the 42 (now: n_known) training species centroids in
             embedding space. Large distance = likely unseen.
       - Threshold: taken from the held-in validation distribution only
         (5th/10th percentile of "confidence", i.e. threshold-free choice
         deferred to AUROC; a concrete operating threshold is also reported
         for convenience).
  3. Metrics:
       - AUROC (held-out vs held-in, using the anomaly score) -- primary,
         threshold-free metric per standard OSR evaluation.
       - At the chosen threshold: recall on held-out (they should score as
         "unseen") and false-rejection rate on held-in (they should NOT).
       - For samples correctly flagged unseen: accuracy of the family
         prediction (and genus prediction, for near-OOD only, since
         far-OOD species have no correct genus to predict against).
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

from NatureLM.anuraset.anuraset_dataset_unseen import AnuraSetEmbeddingDataset
from NatureLM.anuraset.hierarchy_tree import HierarchyTree
from models.projection_head import ProjectionHead
from models.hierarchical_model import HierarchicalModel

SPLIT_PATH = '/user/gaad2403/Bachelor-Thesis/data/unseen_species_split.json'
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
OUTPUT_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/unseen_species_eval')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAL_PERCENTILE = 5  # operating threshold = 5th percentile of held-in val confidence


def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)


def anomaly_score_from_ftm(scores, species_node_start, species_node_end):
    """scores: [n, n_nodes] sigmoid. Returns anomaly score = 1 - max species score."""
    species_scores = scores[:, species_node_start:species_node_end]
    return 1.0 - species_scores.max(axis=1)


def anomaly_score_from_embedding(test_emb, train_emb, train_species_labels, n_known_species):
    """Distance to nearest known-species centroid (cosine distance)."""
    centroids = np.stack([
        train_emb[train_species_labels == c].mean(axis=0) for c in range(n_known_species)
    ])
    centroids = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8, None)
    sims = cosine_similarity(test_emb, centroids)  # [n_test, n_known_species]
    return 1.0 - sims.max(axis=1)  # anomaly score: higher = farther from all known centroids


def evaluate(anomaly_scores, is_unseen, val_anomaly_scores_id_only, family_correct=None, genus_correct=None):
    auroc = roc_auc_score(is_unseen, anomaly_scores)

    # operating threshold from held-in validation distribution
    threshold = np.percentile(val_anomaly_scores_id_only, 100 - VAL_PERCENTILE)
    flagged_unseen = anomaly_scores > threshold

    recall_unseen = flagged_unseen[is_unseen].mean()          # correctly flagged as unseen
    false_reject_rate = flagged_unseen[~is_unseen].mean()      # wrongly flagged known samples

    result = {
        'auroc': auroc,
        'threshold': threshold,
        'recall_unseen': recall_unseen,
        'false_reject_rate': false_reject_rate,
    }

    correctly_flagged_unseen = flagged_unseen & is_unseen
    if family_correct is not None and correctly_flagged_unseen.sum() > 0:
        result['family_acc_given_flagged_unseen'] = family_correct[correctly_flagged_unseen].mean()
    if genus_correct is not None and correctly_flagged_unseen.sum() > 0:
        result['genus_acc_given_flagged_unseen'] = genus_correct[correctly_flagged_unseen].mean()

    return result


def main():
    split = load_split()
    exclude_pairs = split['all_held_out_pairs']  # list of {'genus', 'species', 'code'}
    near_ood_codes = {p['code'] for p in split['near_ood_pairs']}
    far_ood_codes = {p['code'] for p in split['far_ood_pairs']}

    # NOTE: this assumes a variant of AnuraSetEmbeddingDataset that accepts
    # an `exclude_species` argument to drop held-out species from train/val
    # while keeping them in test. See unseen_species_split.py for the split
    # itself; wiring `exclude_species` into the dataset loader is the one
    # piece of glue code still needed on top of the existing class.
    print("Loading datasets (train/val exclude held-out species; test keeps everything)...")
    train_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='train',
                                        single_label_only=True, exclude_pairs=exclude_pairs)
    print(sorted(train_ds.family_enc.classes_))   # <-- 이 줄 추가


    # IMPORTANT: train_ds.species_enc was fit AFTER excluding held-out
    # species, so it doesn't know their labels. Building test with that
    # encoder would crash (LabelEncoder.transform on unseen labels) as soon
    # as a held-out species shows up in test. Instead, fit a separate
    # species encoder on the FULL (unfiltered) taxonomy so every species --
    # held-in or held-out -- gets a valid, stable id. family_enc/genus_enc
    # are safe to reuse from train_ds since no whole family/genus is fully
    # removed for near-OOD (far-OOD removes a whole genus, so genus_enc
    # from train_ds would ALSO be missing that genus -- same fix needed).
    from sklearn.preprocessing import LabelEncoder
    full_meta = pd.read_csv(META_PATH)
    full_species_enc = LabelEncoder().fit(full_meta['species'])
    full_genus_enc = LabelEncoder().fit(full_meta['genus'])

    test_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='test',
                                       single_label_only=True,
                                       family_enc=train_ds.family_enc,
                                       genus_enc=full_genus_enc,
                                       species_enc=full_species_enc)
    # test_ds intentionally NOT filtered by exclude_species -- we need both
    # held-in and held-out samples in test to compute AUROC.

    print(f"Train (held-in only): {len(train_ds)} | Test (held-in + held-out): {len(test_ds)}")
    print("Wire in model-specific embedding extraction + anomaly_score_from_* here,")
    print("then call evaluate(...) per model, per tier (near-OOD / far-OOD separately).")



if __name__ == '__main__':
    main()