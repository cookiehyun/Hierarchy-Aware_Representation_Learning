"""
Hierarchy Tree Builder for Focal Tree-Min Loss — ESC-50 (BGA -> Midcat -> Category)

AnuraSet의 hierarchy_tree.py와 동일한 역할 (Eq.4용 ancestor/descendant 집합 계산).
ESC-50은 별도 taxonomy.csv가 없으므로, esc50_bga_labels.csv 자체에서
category -> midcat -> bga 매핑을 직접 추출 (각 row가 이미 셋을 다 갖고 있음).
"""

import pandas as pd
import torch


class ESC50HierarchyTree:
    def __init__(self, metadata_path, bga_enc, midcat_enc, category_enc):
        """
        Args:
            metadata_path: esc50_bga_labels.csv 경로
            bga_enc, midcat_enc, category_enc: 학습에 쓰는 (train) Dataset에서 fit된 LabelEncoder
        """
        meta = pd.read_csv(metadata_path)

        self.bga_classes      = list(bga_enc.classes_)
        self.midcat_classes   = list(midcat_enc.classes_)
        self.category_classes = list(category_enc.classes_)

        self.n_bga      = len(self.bga_classes)
        self.n_midcat   = len(self.midcat_classes)
        self.n_category = len(self.category_classes)
        self.n_nodes     = self.n_bga + self.n_midcat + self.n_category

        self.bga_offset      = 0
        self.midcat_offset   = self.n_bga
        self.category_offset = self.n_bga + self.n_midcat

        # metadata 자체에서 직접 매핑 추출 (taxonomy 파일 불필요)
        self.category_to_midcat = dict(zip(meta["category"], meta["midcat"]))
        self.midcat_to_bga      = dict(zip(meta["midcat"], meta["bga_label"]))

        self.ancestors   = [set() for _ in range(self.n_nodes)]
        self.descendants = [set() for _ in range(self.n_nodes)]
        self._build()

    def bga_idx(self, name):      return self.bga_offset + self.bga_classes.index(name)
    def midcat_idx(self, name):   return self.midcat_offset + self.midcat_classes.index(name)
    def category_idx(self, name): return self.category_offset + self.category_classes.index(name)

    def _build(self):
        n = self.n_nodes
        for v in range(n):
            self.ancestors[v].add(v)
            self.descendants[v].add(v)

        # midcat -> bga
        for m_name in self.midcat_classes:
            b_name = self.midcat_to_bga.get(m_name)
            if b_name is None or b_name not in self.bga_classes:
                continue
            m_idx, b_idx = self.midcat_idx(m_name), self.bga_idx(b_name)
            self.ancestors[m_idx].add(b_idx)
            self.descendants[b_idx].add(m_idx)

        # category -> midcat, category -> bga (직접 추가, depth=3이라 closure 불필요)
        for c_name in self.category_classes:
            c_idx = self.category_idx(c_name)
            m_name = self.category_to_midcat.get(c_name)
            b_name = self.midcat_to_bga.get(m_name) if m_name else None

            if m_name in self.midcat_classes:
                m_idx = self.midcat_idx(m_name)
                self.ancestors[c_idx].add(m_idx)
                self.descendants[m_idx].add(c_idx)

            if b_name in self.bga_classes:
                b_idx = self.bga_idx(b_name)
                self.ancestors[c_idx].add(b_idx)
                self.descendants[b_idx].add(c_idx)

    def positive_path(self, bga_name, midcat_name, category_name):
        idxs = set()
        if bga_name in self.bga_classes:
            idxs.add(self.bga_idx(bga_name))
        if midcat_name in self.midcat_classes:
            idxs.add(self.midcat_idx(midcat_name))
        if category_name in self.category_classes:
            idxs.add(self.category_idx(category_name))
        return idxs

    def build_ground_truth(self, bga_ids, midcat_ids, category_ids,
                            bga_enc, midcat_enc, category_enc):
        bsz = bga_ids.shape[0]
        gt = torch.zeros(bsz, self.n_nodes)
        bga_names      = bga_enc.inverse_transform(bga_ids.cpu().numpy())
        midcat_names   = midcat_enc.inverse_transform(midcat_ids.cpu().numpy())
        category_names = category_enc.inverse_transform(category_ids.cpu().numpy())
        for i in range(bsz):
            for idx in self.positive_path(bga_names[i], midcat_names[i], category_names[i]):
                gt[i, idx] = 1.0
        return gt

    def ancestor_matrix(self) -> torch.Tensor:
        n = self.n_nodes
        M = torch.zeros(n, n, dtype=torch.bool)
        for v in range(n):
            for u in self.ancestors[v]:
                M[v, u] = True
        return M

    def descendant_matrix(self) -> torch.Tensor:
        n = self.n_nodes
        M = torch.zeros(n, n, dtype=torch.bool)
        for v in range(n):
            for u in self.descendants[v]:
                M[v, u] = True
        return M