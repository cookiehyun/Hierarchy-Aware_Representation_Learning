"""
AnuraSet Embedding Dataset (single_label_only 옵션 추가됨)

기존 AnuraSetEmbeddingDataset 그대로 + 'multi-label(2종 이상 동시발생)
샘플 제외' 옵션. baseline_probing_singlelabel.py와 동일한 필터링 기준을
학습 코드에도 똑같이 적용하기 위함 (둘이 다른 subset을 쓰면 비교가 깨짐).
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
                샘플만 사용. baseline_probing_singlelabel.py와 동일 기준.
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

        # 🆕 multi-label(2종 이상 동시발생) 제거
        if single_label_only:
            single_mask    = species_number == 1
            embeddings_all = embeddings_all[single_mask.values]
            meta_all       = meta_all[single_mask].reset_index(drop=True)
            print(f"[Dataset] single_label_only=True -> "
                  f"{single_mask.sum()}/{len(single_mask)} 샘플 유지 "
                  f"({single_mask.mean()*100:.1f}%)")

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