"""
AnuraSet multi-label embedding dataset.

Uses the full dataset (no single-label filtering) with the official train/test
split from the metadata. Species labels are multi-hot as provided by AnuraSet.
Genus and family labels are derived by aggregating species multi-hot vectors
through the taxonomy table (a sample is positive for a genus/family if it
contains at least one species from that genus/family).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AnuraSetMultiLabelDataset(Dataset):
    def __init__(self, embeddings_path, metadata_path, taxonomy_path, subset=None,
                 species_cols=None, genus_cols=None, family_cols=None):
        embeddings_all = np.load(embeddings_path)
        meta_all = pd.read_csv(metadata_path)
        taxonomy = pd.read_csv(taxonomy_path)

        if species_cols is None:
            # 'family'/'genus'/'species' here are derived columns (first active
            # species only) and unreliable for multi-label samples, so they're
            # excluded rather than used.
            non_species = ['sample_name', 'family', 'genus', 'species', 'subset']
            species_cols = [c for c in meta_all.columns if c not in non_species]
        self.species_cols = species_cols

        code_to_genus = dict(zip(taxonomy['code'], taxonomy['genus']))
        code_to_family = dict(zip(taxonomy['code'], taxonomy['family']))

        missing = [c for c in species_cols if c not in code_to_genus]
        if missing:
            print(f"[Dataset] warning: {len(missing)} species columns not found in taxonomy: {missing}")

        if genus_cols is None:
            genus_cols = sorted(set(code_to_genus[c] for c in species_cols if c in code_to_genus))
        if family_cols is None:
            family_cols = sorted(set(code_to_family[c] for c in species_cols if c in code_to_family))
        self.genus_cols = genus_cols
        self.family_cols = family_cols

        # drop unlabeled background samples (species_number == 0)
        species_sum = meta_all[species_cols].sum(axis=1)
        labeled_mask = species_sum > 0
        embeddings_all = embeddings_all[labeled_mask.values]
        meta_all = meta_all[labeled_mask].reset_index(drop=True)

        if subset is not None:
            split_mask = meta_all['subset'] == subset
            embeddings_all = embeddings_all[split_mask.values]
            meta_all = meta_all[split_mask].reset_index(drop=True)

        species_multihot = meta_all[species_cols].values.astype(np.float32)

        # species -> genus/family mapping matrices, built once from the taxonomy
        species_to_genus = np.zeros((len(species_cols), len(genus_cols)), dtype=np.float32)
        species_to_family = np.zeros((len(species_cols), len(family_cols)), dtype=np.float32)
        for i, code in enumerate(species_cols):
            if code in code_to_genus:
                species_to_genus[i, genus_cols.index(code_to_genus[code])] = 1.0
            if code in code_to_family:
                species_to_family[i, family_cols.index(code_to_family[code])] = 1.0

        genus_multihot = (species_multihot @ species_to_genus > 0).astype(np.float32)
        family_multihot = (species_multihot @ species_to_family > 0).astype(np.float32)

        self.embeddings = torch.FloatTensor(embeddings_all)
        self.species_labels = torch.FloatTensor(species_multihot)
        self.genus_labels = torch.FloatTensor(genus_multihot)
        self.family_labels = torch.FloatTensor(family_multihot)
        self.meta = meta_all

        n_multi = int((species_multihot.sum(axis=1) > 1).sum())
        n_total = len(self)
        print(f"[Dataset] subset={subset} | samples={n_total} | "
              f"multi-label samples={n_multi} ({100 * n_multi / n_total:.1f}%) | "
              f"species={len(species_cols)} genus={len(genus_cols)} family={len(family_cols)}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (self.embeddings[idx], self.family_labels[idx],
                self.genus_labels[idx], self.species_labels[idx])

    @property
    def n_families(self):
        return len(self.family_cols)

    @property
    def n_genera(self):
        return len(self.genus_cols)

    @property
    def n_species(self):
        return len(self.species_cols)


if __name__ == '__main__':
    # sanity check: species_sum == 0 should match the 33,913 rows labeled
    # family == 'unknown' in the raw metadata
    meta_check = pd.read_csv('/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv')
    non_species_check = ['sample_name', 'family', 'genus', 'species', 'subset']
    species_cols_check = [c for c in meta_check.columns if c not in non_species_check]
    n_zero = (meta_check[species_cols_check].sum(axis=1) == 0).sum()
    n_unknown = (meta_check['family'] == 'unknown').sum()
    print(f"[sanity check] species_sum==0: {n_zero} | family=='unknown': {n_unknown} | match: {n_zero == n_unknown}")

    train_ds = AnuraSetMultiLabelDataset(
        embeddings_path='/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy',
        metadata_path='/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv',
        taxonomy_path='/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv',
        subset='train',
    )
    test_ds = AnuraSetMultiLabelDataset(
        embeddings_path='/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy',
        metadata_path='/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv',
        taxonomy_path='/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv',
        subset='test',
        species_cols=train_ds.species_cols,
        genus_cols=train_ds.genus_cols,
        family_cols=train_ds.family_cols,
    )

    emb, family, genus, species = train_ds[0]
    print(f"embedding shape: {emb.shape}")
    print(f"family: {family.shape}, genus: {genus.shape}, species: {species.shape}")