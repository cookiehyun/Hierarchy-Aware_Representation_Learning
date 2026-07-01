"""
ESC-50 Embedding Dataset for HMLC / Tree-Triplet Training (BirdNET)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class ESC50EmbeddingDataset(Dataset):
    """
    Returns:
        embedding  : torch.FloatTensor [1024]  ← BirdNET embedding dim
        hier_label : torch.LongTensor  [3]  -> [bga_id, midcat_id, category_id]
    """

    def __init__(self, embeddings_path, metadata_path, folds=None,
                 bga_enc=None, midcat_enc=None, category_enc=None):
        """
        Args:
            folds : list[int] (e.g. [1,2,3,4] for train, [5] for test), None이면 전체
        """
        embeddings_all = np.load(embeddings_path)
        meta_all       = pd.read_csv(metadata_path)

        if folds is not None:
            split_mask     = meta_all['fold'].isin(folds)
            embeddings_all = embeddings_all[split_mask.values]
            meta_all       = meta_all[split_mask].reset_index(drop=True)

        if bga_enc is None:
            self.bga_enc      = LabelEncoder().fit(meta_all['bga_label'])
            self.midcat_enc   = LabelEncoder().fit(meta_all['midcat'])
            self.category_enc = LabelEncoder().fit(meta_all['category'])
        else:
            self.bga_enc      = bga_enc
            self.midcat_enc   = midcat_enc
            self.category_enc = category_enc

        bga_ids      = self.bga_enc.transform(meta_all['bga_label'])
        midcat_ids   = self.midcat_enc.transform(meta_all['midcat'])
        category_ids = self.category_enc.transform(meta_all['category'])

        hier_labels = np.stack([bga_ids, midcat_ids, category_ids], axis=1)

        self.embeddings  = torch.FloatTensor(embeddings_all)
        self.hier_labels = torch.LongTensor(hier_labels)
        self.meta        = meta_all

        print(f"[Dataset] folds={folds} | samples={len(self)} | "
              f"bga={len(self.bga_enc.classes_)} | "
              f"midcat={len(self.midcat_enc.classes_)} | "
              f"category={len(self.category_enc.classes_)}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.hier_labels[idx]

    @property
    def n_bga(self):      return len(self.bga_enc.classes_)
    @property
    def n_midcat(self):   return len(self.midcat_enc.classes_)
    @property
    def n_category(self): return len(self.category_enc.classes_)