"""
Checks how many samples compute_gap's "exact label combination = one class"
approach would actually use, at each hierarchy level. Combinations with only
1 sample get skipped (no intra-class pair possible), so this tells us how
much data that costs us.
"""

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import numpy as np
from multilabel.NatureLM.anuraset.anuraset_dataset import AnuraSetMultiLabelDataset

EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'


def check_level(name, multihot_labels):
    label_strings = np.array([''.join(map(str, row.astype(int))) for row in multihot_labels])
    unique, counts = np.unique(label_strings, return_counts=True)

    n_total = len(label_strings)
    n_singleton_combos = (counts == 1).sum()
    n_covered = counts[counts >= 2].sum()

    print(f"[{name}]")
    print(f"  total samples: {n_total}")
    print(f"  unique label combinations: {len(unique)}")
    print(f"  combinations with exactly 1 sample (skipped by compute_gap): {n_singleton_combos}")
    print(f"  samples usable for intra-class pairs: {n_covered} ({100 * n_covered / n_total:.1f}%)")
    print()


print("Loading test set...")
train_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='train')
test_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='test',
                                    species_cols=train_ds.species_cols,
                                    genus_cols=train_ds.genus_cols,
                                    family_cols=train_ds.family_cols)

print()
check_level('Family', test_ds.family_labels.numpy())
check_level('Genus', test_ds.genus_labels.numpy())
check_level('Species', test_ds.species_labels.numpy())