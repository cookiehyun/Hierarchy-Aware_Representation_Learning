"""
독립 실행형 검증 스크립트: HierarchyTree가 taxonomy.csv에 추가된 "unknown" 행을
제대로 인식해서 family/genus/species 3-level을 서로 연결하는지 확인.

HierarchyTree 클래스는 기존 코드를 그대로 복사해왔음 (수정 없음).

사용법:
    python verify_hierarchy_tree_unknown.py /path/to/anuraset_taxonomy_with_unknown.csv
"""

import sys
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────
# HierarchyTree: 기존 코드 그대로 복사 (수정 없음)
# ─────────────────────────────────────────────────────────────
class HierarchyTree:
    def __init__(self, taxonomy_path, family_enc, genus_enc, species_enc):
        tax = pd.read_csv(taxonomy_path)

        self.family_classes  = list(family_enc.classes_)
        self.genus_classes   = list(genus_enc.classes_)
        self.species_classes = list(species_enc.classes_)

        self.n_family  = len(self.family_classes)
        self.n_genus   = len(self.genus_classes)
        self.n_species = len(self.species_classes)
        self.n_nodes   = self.n_family + self.n_genus + self.n_species

        self.family_offset  = 0
        self.genus_offset   = self.n_family
        self.species_offset = self.n_family + self.n_genus

        self.species_to_genus  = dict(zip(tax["species"], tax["genus"]))
        self.species_to_family = dict(zip(tax["species"], tax["family"]))
        self.genus_to_family   = dict(zip(tax["genus"], tax["family"]))

        self.ancestors   = [set() for _ in range(self.n_nodes)]
        self.descendants = [set() for _ in range(self.n_nodes)]
        self._build()

    def family_idx(self, name):
        return self.family_offset + self.family_classes.index(name)

    def genus_idx(self, name):
        return self.genus_offset + self.genus_classes.index(name)

    def species_idx(self, name):
        return self.species_offset + self.species_classes.index(name)

    def _build(self):
        n = self.n_nodes
        for v in range(n):
            self.ancestors[v].add(v)
            self.descendants[v].add(v)

        for g_name in self.genus_classes:
            f_name = self.genus_to_family.get(g_name)
            if f_name is None or f_name not in self.family_classes:
                continue
            g_idx, f_idx = self.genus_idx(g_name), self.family_idx(f_name)
            self.ancestors[g_idx].add(f_idx)
            self.descendants[f_idx].add(g_idx)

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
                self.descendants[f_idx].add(s_idx)

    def positive_path(self, family_name, genus_name, species_name):
        idxs = set()
        if family_name in self.family_classes:
            idxs.add(self.family_idx(family_name))
        if genus_name in self.genus_classes:
            idxs.add(self.genus_idx(genus_name))
        if species_name in self.species_classes:
            idxs.add(self.species_idx(species_name))
        return idxs

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


def build_label_encoders(taxonomy_path):
    """taxonomy.csv (unknown 포함)의 family/genus/species 클래스 목록으로
    LabelEncoder를 fit. 실제 파이프라인에서는 AnuraSetEmbeddingDataset이
    실제 metadata로 fit하지만, family/genus/species 클래스 '집합' 자체는
    taxonomy.csv에 있는 것과 동일해야 하므로 이 방식으로 재현 가능."""
    tax = pd.read_csv(taxonomy_path)
    family_enc  = LabelEncoder().fit(tax["family"])
    genus_enc   = LabelEncoder().fit(tax["genus"])
    species_enc = LabelEncoder().fit(tax["species"])
    return family_enc, genus_enc, species_enc


def check(label, condition):
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  [{status}] {label}")
    return condition


def main(taxonomy_path):
    print(f"[INFO] taxonomy 로드: {taxonomy_path}")
    family_enc, genus_enc, species_enc = build_label_encoders(taxonomy_path)

    print(f"[INFO] family classes ({len(family_enc.classes_)}): {list(family_enc.classes_)}")
    print(f"[INFO] genus classes  ({len(genus_enc.classes_)}): {list(genus_enc.classes_)}")
    print(f"[INFO] species classes ({len(species_enc.classes_)}): {len(species_enc.classes_)}개 (unknown 포함 여부만 확인)")
    print()

    tree = HierarchyTree(taxonomy_path, family_enc, genus_enc, species_enc)
    print(f"[INFO] n_nodes = {tree.n_nodes} "
          f"(family={tree.n_family}, genus={tree.n_genus}, species={tree.n_species})")
    print()

    all_pass = True

    # ── 1. unknown 3-level이 서로 연결되는지 ──
    print("[검증 1] unknown 3-level이 ancestor/descendant로 연결되는지")
    s_unk = tree.species_idx("unknown")
    g_unk = tree.genus_idx("unknown")
    f_unk = tree.family_idx("unknown")

    all_pass &= check(
        "species='unknown'의 ancestors에 genus='unknown' 포함",
        g_unk in tree.ancestors[s_unk]
    )
    all_pass &= check(
        "species='unknown'의 ancestors에 family='unknown' 포함",
        f_unk in tree.ancestors[s_unk]
    )
    all_pass &= check(
        "genus='unknown'의 ancestors에 family='unknown' 포함",
        f_unk in tree.ancestors[g_unk]
    )
    all_pass &= check(
        "family='unknown'의 descendants에 genus='unknown', species='unknown' 포함",
        (g_unk in tree.descendants[f_unk]) and (s_unk in tree.descendants[f_unk])
    )
    print()

    # ── 2. unknown이 실제 종/속/과와는 연결 안 되는지 (오염 방지 확인) ──
    print("[검증 2] unknown이 실제(42종) 노드와는 섞이지 않는지")
    real_family_names = [f for f in tree.family_classes if f != "unknown"]
    contaminated = False
    for f_name in real_family_names:
        f_idx = tree.family_idx(f_name)
        if s_unk in tree.descendants[f_idx] or g_unk in tree.descendants[f_idx]:
            contaminated = True
            print(f"    ⚠️ family='{f_name}'의 descendants에 unknown 노드가 섞여 있음!")
    all_pass &= check("실제 family들의 descendants에 unknown이 섞이지 않음", not contaminated)
    print()

    # ── 3. 기존 42종 관계가 정상적으로 유지되는지 (회귀 확인) ──
    print("[검증 3] 기존 실제 종 하나(RHIICT: Rhinella icterica, Bufonidae)로 회귀 테스트")
    s_idx = tree.species_idx("icterica")
    g_idx = tree.genus_idx("Rhinella")
    f_idx = tree.family_idx("Bufonidae")
    all_pass &= check(
        "species='icterica'의 ancestors에 genus='Rhinella' 포함",
        g_idx in tree.ancestors[s_idx]
    )
    all_pass &= check(
        "species='icterica'의 ancestors에 family='Bufonidae' 포함",
        f_idx in tree.ancestors[s_idx]
    )
    all_pass &= check(
        "family='Bufonidae'의 descendants에 species='icterica' 포함",
        s_idx in tree.descendants[f_idx]
    )
    print()

    # ── 4. ancestor_matrix / descendant_matrix가 서로 transpose 관계인지 ──
    print("[검증 4] ancestor_matrix와 descendant_matrix가 서로 일관되는지 (M_anc[v,u] == M_desc[u,v])")
    M_anc = tree.ancestor_matrix()
    M_desc = tree.descendant_matrix()
    consistent = torch.equal(M_anc, M_desc.T)
    all_pass &= check("ancestor_matrix == descendant_matrix.T", consistent)
    print()

    print("=" * 60)
    if all_pass:
        print("✅ 모든 검증 통과 — unknown이 taxonomy에 정상적으로 통합되었습니다.")
    else:
        print("❌ 일부 검증 실패 — 위 로그를 확인하세요.")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python verify_hierarchy_tree_unknown.py /path/to/anuraset_taxonomy_with_unknown.csv")
        sys.exit(1)
    main(sys.argv[1])