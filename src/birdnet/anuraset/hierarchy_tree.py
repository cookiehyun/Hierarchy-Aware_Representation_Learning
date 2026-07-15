"""
Hierarchy Tree Builder for Focal Tree-Min Loss (HSSN, Li et al. CVPR 2022, Eq. 4 & 6)

V = Family ∪ Genus ∪ Species (Order 제외 — 전체 샘플이 다 "Anura"라서 변별력 없음)

각 노드 v ∈ V에 대해:
  - ancestors[v]   : v의 조상 집합 (v 자신 포함) -> A_v  (Eq.4의 min 계산에 사용)
  - descendants[v] : v의 자손 집합 (v 자신 포함) -> C_v  (Eq.4의 max 계산에 사용)

단일 라벨 설정에서 ground truth positive path:
  species s를 가진 샘플 -> {family(s), genus(s), species(s)} 가 positive,
  나머지 전부 negative.
"""

import pandas as pd
import torch


class HierarchyTree:
    def __init__(self, taxonomy_path, family_enc, genus_enc, species_enc):
        """
        Args:
            taxonomy_path: anuraset_taxonomy.csv 경로 (code,genus,species,full_name,family,order)
            family_enc, genus_enc, species_enc: (단일 라벨 필터링된) AnuraSetEmbeddingDataset에서
                fit된 LabelEncoder들. 실제 데이터에 존재하는 클래스만 V에 포함됨.
        """
        tax = pd.read_csv(taxonomy_path)

        self.family_classes  = list(family_enc.classes_)
        self.genus_classes   = list(genus_enc.classes_)
        self.species_classes = list(species_enc.classes_)

        self.n_family  = len(self.family_classes)
        self.n_genus   = len(self.genus_classes)
        self.n_species = len(self.species_classes)
        self.n_nodes   = self.n_family + self.n_genus + self.n_species

        # 전역 노드 인덱스 오프셋: V = [family...] + [genus...] + [species...]
        self.family_offset  = 0
        self.genus_offset   = self.n_family
        self.species_offset = self.n_family + self.n_genus

        # taxonomy.csv 기준 직접 매핑 (species 컬럼은 epithet만, AnuraSetEmbeddingDataset의
        # 'species' 컬럼 표기와 동일한 convention)
        self.species_to_genus  = dict(zip(tax["species"], tax["genus"]))
        self.species_to_family = dict(zip(tax["species"], tax["family"]))
        self.genus_to_family   = dict(zip(tax["genus"], tax["family"]))

        self.ancestors   = [set() for _ in range(self.n_nodes)]
        self.descendants = [set() for _ in range(self.n_nodes)]
        self._build()

    # ── 노드 이름 -> 전역 인덱스 ──
    def family_idx(self, name):
        return self.family_offset + self.family_classes.index(name)

    def genus_idx(self, name):
        return self.genus_offset + self.genus_classes.index(name)

    def species_idx(self, name):
        return self.species_offset + self.species_classes.index(name)

    def _build(self):
        n = self.n_nodes
        # (v, v) ∈ E  — 자기 자신은 항상 조상이자 자손
        for v in range(n):
            self.ancestors[v].add(v)
            self.descendants[v].add(v)

        # genus -> family
        for g_name in self.genus_classes:
            f_name = self.genus_to_family.get(g_name)
            if f_name is None or f_name not in self.family_classes:
                continue  # 이 genus의 family가 (단일 라벨 필터링 후) 데이터에 없음
            g_idx, f_idx = self.genus_idx(g_name), self.family_idx(f_name)
            self.ancestors[g_idx].add(f_idx)
            self.descendants[f_idx].add(g_idx)

        # species -> genus, species -> family (직접 추가, depth=3이라 transitive closure 불필요)
        for s_name in self.species_classes:
            s_idx = self.species_idx(s_name)
            g_name = self.species_to_genus.get(s_name)
            f_name = self.species_to_family.get(s_name)

            if g_name in self.genus_classes:
                g_idx = self.genus_idx(g_name)
                self.ancestors[s_idx].add(g_idx)
                self.descendants[g_idx].add(s_idx)

            if f_name in self.family_classes:
                f_idx = self.family_idx(f_name)
                self.ancestors[s_idx].add(f_idx)
                self.descendants[f_idx].add(s_idx)  # family의 자손에 species 직접 추가

    def positive_path(self, family_name, genus_name, species_name):
        """단일 라벨 샘플 하나의 positive 노드 인덱스 집합 {family, genus, species}."""
        idxs = set()
        if family_name in self.family_classes:
            idxs.add(self.family_idx(family_name))
        if genus_name in self.genus_classes:
            idxs.add(self.genus_idx(genus_name))
        if species_name in self.species_classes:
            idxs.add(self.species_idx(species_name))
        return idxs

    def build_ground_truth(self, family_ids, genus_ids, species_ids,
                            family_enc, genus_enc, species_enc):
        """
        Args:
            family_ids, genus_ids, species_ids: [bsz] LongTensor (encoded label id)
        Returns:
            gt: [bsz, n_nodes] FloatTensor, V 전체에 대한 multi-hot ground truth
        """
        bsz = family_ids.shape[0]
        gt = torch.zeros(bsz, self.n_nodes)

        family_names  = family_enc.inverse_transform(family_ids.cpu().numpy())
        genus_names   = genus_enc.inverse_transform(genus_ids.cpu().numpy())
        species_names = species_enc.inverse_transform(species_ids.cpu().numpy())

        for i in range(bsz):
            for idx in self.positive_path(family_names[i], genus_names[i], species_names[i]):
                gt[i, idx] = 1.0
        return gt

    def ancestor_matrix(self) -> torch.Tensor:
        """[n_nodes, n_nodes] bool, M[v, u] = True  <=>  u ∈ A_v"""
        n = self.n_nodes
        M = torch.zeros(n, n, dtype=torch.bool)
        for v in range(n):
            for u in self.ancestors[v]:
                M[v, u] = True
        return M

    def descendant_matrix(self) -> torch.Tensor:
        """[n_nodes, n_nodes] bool, M[v, u] = True  <=>  u ∈ C_v"""
        n = self.n_nodes
        M = torch.zeros(n, n, dtype=torch.bool)
        for v in range(n):
            for u in self.descendants[v]:
                M[v, u] = True
        return M