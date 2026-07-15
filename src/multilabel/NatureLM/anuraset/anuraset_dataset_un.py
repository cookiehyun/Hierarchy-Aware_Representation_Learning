"""
AnuraSet Multi-Label Dataset -- with exclude_species_codes (unseen-species
hold-out) and include_negative (negative/"unknown" sample inclusion).

Two independent 🆕 additions to the original class:

1. exclude_species_codes: drops specific species (by code, e.g. 'ADEDIP')
   from species_cols entirely -- so the resulting species/genus/family
   multi-hot vectors never have a column for them. Pass this for TRAIN in
   the unseen-species experiment; leave None for TEST so held-out species
   remain (as full-width vectors, built with the same taxonomy) for eval.

   IMPORTANT: after removing those columns, any training sample whose only
   positive label(s) were among the excluded species would incorrectly
   look like an all-zero ("negative") sample. Those samples are dropped
   entirely rather than kept as false negatives.

2. include_negative: if True, keeps species_sum==0 ("no call present")
   samples instead of filtering them out. Unlike the single-label case,
   multi-label doesn't need an extra explicit class for this -- the
   all-zero multi-hot vector IS the correct representation of "no known
   species present," so no other code changes are needed.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AnuraSetMultiLabelDataset(Dataset):
    def __init__(self, embeddings_path, metadata_path, taxonomy_path, subset=None,
                 species_cols=None, genus_cols=None, family_cols=None,
                 exclude_species_codes=None, include_negative=False):
        embeddings_all = np.load(embeddings_path)
        meta_all = pd.read_csv(metadata_path)
        taxonomy = pd.read_csv(taxonomy_path)

        non_species = ['sample_name', 'family', 'genus', 'species', 'subset']
        full_species_cols = [c for c in meta_all.columns if c not in non_species]

        if species_cols is None:
            # 🆕 drop held-out species columns entirely (unseen-species experiment)
            if exclude_species_codes:
                exclude_set = set(exclude_species_codes)
                species_cols = [c for c in full_species_cols if c not in exclude_set]
                print(f"[Dataset] exclude_species_codes -> dropped {len(exclude_set)} species columns: "
                      f"{sorted(exclude_set)}")
            else:
                species_cols = full_species_cols
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

        # 🆕 include_negative=True skips this filter -- species_sum==0 rows
        # are kept as legitimate all-zero multi-hot vectors (no extra class
        # needed, unlike single-label, since multi-label already supports
        # "no positive label" natively).
        original_species_sum = meta_all[full_species_cols].sum(axis=1)
        if not include_negative:
            labeled_mask = original_species_sum > 0
            embeddings_all = embeddings_all[labeled_mask.values]
            meta_all = meta_all[labeled_mask].reset_index(drop=True)
            original_species_sum = original_species_sum[labeled_mask].reset_index(drop=True)

        # 🆕 drop samples whose only positive label(s) fell entirely within
        # the excluded species -- otherwise they'd falsely become all-zero
        # ("negative-looking") rows after column removal, contaminating
        # training with mislabeled negatives.
        if exclude_species_codes:
            new_species_sum = meta_all[species_cols].sum(axis=1)
            falsely_emptied = (original_species_sum > 0) & (new_species_sum == 0)
            if falsely_emptied.any():
                n_drop = int(falsely_emptied.sum())
                keep_mask = ~falsely_emptied
                embeddings_all = embeddings_all[keep_mask.values]
                meta_all = meta_all[keep_mask].reset_index(drop=True)
                print(f"[Dataset] exclude_species_codes -> also dropped {n_drop} samples "
                      f"whose only positive label(s) were held-out species")

        if subset is not None:
            split_mask = meta_all['subset'] == subset
            embeddings_all = embeddings_all[split_mask.values]
            meta_all = meta_all[split_mask].reset_index(drop=True)

        species_multihot = meta_all[species_cols].values.astype(np.float32)

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
        n_negative = int((species_multihot.sum(axis=1) == 0).sum())
        n_total = len(self)
        print(f"[Dataset] subset={subset} | samples={n_total} | "
              f"multi-label samples={n_multi} ({100 * n_multi / max(n_total,1):.1f}%) | "
              f"negative(all-zero) samples={n_negative} | "
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