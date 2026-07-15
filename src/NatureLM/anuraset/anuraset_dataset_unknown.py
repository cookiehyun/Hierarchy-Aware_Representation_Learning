"""
AnuraSet Embedding Dataset (single_label_only + negative-sample inclusion)

CHANGE from the previous version: the `labeled_mask = species_number > 0`
filter has been REMOVED. Previously, species_number==0 ("no call present")
samples were dropped entirely, treating them as unlabeled. Per the AnuraSet
paper and Rida's feedback, these are NOT unannotated -- they are properly
annotated negatives (background/silence), and the extraction pipeline
already stores them with family/genus/species == "unknown" (see
extract_embeddings_birdnet.py's `all_families.append("unknown")` etc.).

Since single-label classification requires mutually exclusive categories
(there's no "all inactive" option the way there is in multi-label), the
correct fix here is simply to STOP filtering these samples out --
LabelEncoder.fit will then automatically include "unknown" as one more
valid class at each level (family/genus/species), the same way it already
handles every other class name. No other code changes needed.

Everything else (single_label_only multi-call filtering) is unchanged.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class AnuraSetEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, metadata_path, subset=None,
                 family_enc=None, genus_enc=None, species_enc=None,
                 single_label_only=False):
        """
        Args:
            single_label_only: True면 species_number==1(2종 이상 동시발생 제외)
                샘플만 사용. species_number==0(negative/"unknown") 샘플은
                이제 제외하지 않고 "unknown" 클래스로 유지됨 -- 아래 참고.
        """
        embeddings_all = np.load(embeddings_path)
        meta_all       = pd.read_csv(metadata_path)

        species_cols = [c for c in meta_all.columns
                        if c not in ['sample_name', 'family', 'genus', 'species', 'subset']]
        species_number = meta_all[species_cols].sum(axis=1)

        # 🆕 negative samples (species_number==0, family/genus/species=="unknown")
        # are NO LONGER filtered out here. They are properly annotated
        # negatives, not unlabeled data, and single-label classification
        # needs them as an explicit "unknown"/background class rather than
        # silently dropping ~34% of the dataset. LabelEncoder.fit picks up
        # "unknown" automatically as one more class value, same as any
        # other family/genus/species name.
        n_negative = (species_number == 0).sum()
        print(f"[Dataset] negative (species_number==0) samples kept: "
              f"{n_negative}/{len(meta_all)} ({100*n_negative/len(meta_all):.1f}%) "
              f"-- stored as family/genus/species=='unknown'")

        # 🆕 multi-label(2종 이상 동시발생) 제거
        if single_label_only:
            single_mask    = species_number == 1
            # negative(species_number==0) 샘플은 "multi-species 동시발생"이
            # 아니므로 single_label_only 필터에서도 항상 유지되어야 함
            keep_mask      = single_mask | (species_number == 0)
            embeddings_all = embeddings_all[keep_mask.values]
            meta_all       = meta_all[keep_mask].reset_index(drop=True)
            species_number = species_number[keep_mask].reset_index(drop=True)
            print(f"[Dataset] single_label_only=True -> "
                  f"{keep_mask.sum()}/{len(keep_mask)} 샘플 유지 "
                  f"({keep_mask.mean()*100:.1f}%) "
                  f"[single-species: {single_mask.sum()}, negative: {n_negative}]")

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
              f"species={len(self.species_enc.classes_)} "
              f"(includes 'unknown' negative class at each level)")

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