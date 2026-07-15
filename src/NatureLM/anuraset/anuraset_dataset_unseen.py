"""
AnuraSet Embedding Dataset (single_label_only + exclude_species)

Adds one thing on top of the single_label_only version: an `exclude_species`
argument that drops specific species (by their code, e.g. 'ADEDIP') from the
dataset entirely. Used for the unseen-species experiment -- pass the
held-out set when building the TRAIN dataset (so those species never appear
during training/val), and leave it as None when building the TEST dataset
(so held-out species remain in test, alongside held-in ones, for AUROC/
anomaly-detection evaluation).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class AnuraSetEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, metadata_path, subset=None,
                 family_enc=None, genus_enc=None, species_enc=None,
                 single_label_only=False, exclude_pairs=None):
        """
        Args:
            single_label_only: True -> keep only species_number==1 samples.
            exclude_pairs: optional list of (genus, species) tuples -- or
                dicts with 'genus'/'species' keys, as saved by
                unseen_species_split.py -- to drop entirely. IMPORTANT: the
                metadata's 'species' column is the epithet only (e.g.
                'surdus'), which is not guaranteed unique across genera, so
                matching is done on the (genus, species) pair together, not
                species alone. Pass this when building the TRAIN dataset for
                the unseen-species experiment; leave None for TEST so
                held-out species remain for evaluation.
        """
        embeddings_all = np.load(embeddings_path)
        meta_all       = pd.read_csv(metadata_path)

        species_cols = [c for c in meta_all.columns
                        if c not in ['sample_name', 'family', 'genus', 'species', 'subset']]
        species_number = meta_all[species_cols].sum(axis=1)

        # unlabeled(배경음, species_number==0) 제거
        labeled_mask   = species_number > 0
        embeddings_all = embeddings_all[labeled_mask.values]
        meta_all       = meta_all[labeled_mask].reset_index(drop=True)
        species_number = species_number[labeled_mask].reset_index(drop=True)

        # multi-label(2종 이상 동시발생) 제거
        if single_label_only:
            single_mask    = species_number == 1
            embeddings_all = embeddings_all[single_mask.values]
            meta_all       = meta_all[single_mask].reset_index(drop=True)
            print(f"[Dataset] single_label_only=True -> "
                  f"{single_mask.sum()}/{len(single_mask)} 샘플 유지 "
                  f"({single_mask.mean()*100:.1f}%)")

        # 🆕 unseen-species hold-out: drop specific (genus, species) pairs entirely
        if exclude_pairs:
            pair_set = set()
            for p in exclude_pairs:
                if isinstance(p, dict):
                    pair_set.add((p['genus'], p['species']))
                else:
                    pair_set.add(tuple(p))
            pair_keys = list(zip(meta_all['genus'], meta_all['species']))
            keep_mask = pd.Series([pk not in pair_set for pk in pair_keys], index=meta_all.index)
            n_before = len(meta_all)
            embeddings_all = embeddings_all[keep_mask.values]
            meta_all       = meta_all[keep_mask].reset_index(drop=True)
            print(f"[Dataset] exclude_pairs={sorted(pair_set)} -> "
                  f"removed {n_before - len(meta_all)}/{n_before} samples "
                  f"({100*(n_before - len(meta_all))/n_before:.1f}%)")

        if subset is not None:
            split_mask     = meta_all['subset'] == subset
            embeddings_all = embeddings_all[split_mask.values]
            meta_all       = meta_all[split_mask].reset_index(drop=True)

        if family_enc is None:
            self.family_enc  = LabelEncoder().fit(meta_all['family'])
            self.genus_enc   = LabelEncoder().fit(meta_all['genus'])
            self.species_enc = LabelEncoder().fit(meta_all['species'])
        else:
            self.family_enc  = family_enc
            self.genus_enc   = genus_enc
            self.species_enc = species_enc

        # NOTE: when exclude_species is used for TRAIN and family_enc/etc.
        # are then fit only on the reduced train set, the resulting
        # species_enc will NOT contain the held-out species' classes at
        # all. When later building TEST with family_enc=train_ds.family_enc
        # etc. (no exclude_species), any held-out species in meta_all will
        # fail LabelEncoder.transform (unseen label error) for species_enc
        # specifically (family_enc/genus_enc are fine since those classes
        # still exist). For the unseen-species eval script, build TEST's
        # species labels separately (e.g. via a fixed global species_enc
        # fit on the FULL taxonomy) rather than reusing train_ds.species_enc
        # for the species column.
        family_ids  = self.family_enc.transform(meta_all['family'])
        genus_ids   = self.genus_enc.transform(meta_all['genus'])
        species_ids = self.species_enc.transform(meta_all['species'])

        hier_labels = np.stack([family_ids, genus_ids, species_ids], axis=1)

        self.embeddings  = torch.FloatTensor(embeddings_all)
        self.hier_labels = torch.LongTensor(hier_labels)
        self.meta        = meta_all

        print(f"[Dataset] subset={subset} | samples={len(self)} | "
              f"families={len(self.family_enc.classes_)} | "
              f"genera={len(self.genus_enc.classes_)} | "
              f"species={len(self.species_enc.classes_)}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.hier_labels[idx]

    @property
    def n_families(self):
        return len(self.family_enc.classes_)

    @property
    def n_genera(self):
        return len(self.genus_enc.classes_)

    @property
    def n_species(self):
        return len(self.species_enc.classes_)