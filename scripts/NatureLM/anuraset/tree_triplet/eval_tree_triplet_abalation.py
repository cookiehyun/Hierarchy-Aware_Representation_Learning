"""
WP2 Ablation Evaluation - WP1 vs FTM-only vs TT-only vs FTM+TT

HSSN 논문 Table 5 ablation을 그대로 재현해서, 6000% 개선이
"FTM 단독 효과"인지 "Tree-Triplet의 추가 기여"인지 분리해서 본다.

모든 모델은 동일한 단일라벨(species_number==1) test set으로 평가.
"""

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from anuraset.anuraset_dataset import AnuraSetEmbeddingDataset
from anuraset.hierarchy_tree import HierarchyTree
from models.hierarchical_model import HierarchicalModel

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH    = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH     = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
OUTPUT_DIR    = Path('/user/gaad2403/Bachelor-Thesis/results/wp2_ablation_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    'FTM-only': '/user/gaad2403/Bachelor-Thesis/results/wp2_ftm_only_singlelabel/best_model.pt',
    'TT-only':  '/user/gaad2403/Bachelor-Thesis/results/wp2_tt_only_singlelabel/best_model.pt',
    'FTM+TT':   '/user/gaad2403/Bachelor-Thesis/results/wp2_hssn_singlelabel/best_model.pt',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TSNE = 5000

print("=" * 80)
print("WP2 Ablation Evaluation: WP1 vs FTM-only vs TT-only vs FTM+TT")
print("=" * 80)

# ── Datasets (전부 동일 단일라벨 기준, 동일 encoder) ───────────────────────
print("\n[1/4] Loading datasets...")
train_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='train', single_label_only=True)
test_ds  = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='test', single_label_only=True,
                                    family_enc=train_ds.family_enc,
                                    genus_enc=train_ds.genus_enc,
                                    species_enc=train_ds.species_enc)
meta = test_ds.meta.reset_index(drop=True)
print(f"✓ Test set: {len(test_ds)} samples")

tree = HierarchyTree(TAXONOMY_PATH, train_ds.family_enc, train_ds.genus_enc, train_ds.species_enc)
print(f"✓ n_hierarchy_nodes = {tree.n_nodes} (모델 구조 통일에 사용)")


def extract_embeddings(model_path):
    model = HierarchicalModel(input_dim=4096, hidden_dim=1024, triplet_dim=128,
                              n_hierarchy_nodes=tree.n_nodes).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_z = []
    with torch.no_grad():
        for i in range(0, len(test_ds), 512):
            batch = test_ds.embeddings[i:i + 512].to(DEVICE)
            z, _ = model(batch)
            all_z.append(z.cpu().numpy())
    return np.vstack(all_z), ckpt.get('epoch')


def compute_gap(embeddings, level_col):
    np.random.seed(42)
    labels  = meta[level_col].values
    classes = np.unique(labels)
    intra_sims, inter_sims = [], []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        if len(idx) < 2:
            continue
        sub = embeddings[idx]
        sim = cosine_similarity(sub)
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        intra_sims.extend(sim[mask].tolist())

        other_idx = np.where(labels != cls)[0]
        if len(other_idx) == 0:
            continue
        sample_other = other_idx[np.random.choice(len(other_idx),
                                  min(len(idx) * 5, len(other_idx)), replace=False)]
        inter_sims.extend(cosine_similarity(sub, embeddings[sample_other]).flatten().tolist())
    intra, inter = np.mean(intra_sims), np.mean(inter_sims)
    return intra, inter, intra - inter


# ── WP1 baseline (single-label, 이미 계산된 값 — baseline_probing_singlelabel.py) ──
wp1 = {
    'Family':  (0.5294, 0.5194, 0.0100),
    'Genus':   (0.5338, 0.5215, 0.0124),
    'Species': (0.6026, 0.5174, 0.0853),
}

# ── 각 ablation 모델 임베딩 추출 + gap 계산 ────────────────────────────────
print("\n[2/4] Extracting embeddings & computing gaps for each ablation...")
results = {'Baseline': wp1}
embeddings_cache = {}

for name, path in MODEL_PATHS.items():
    try:
        emb, epoch = extract_embeddings(path)
        embeddings_cache[name] = emb
        res = {}
        for level, col in [('Family', 'family'), ('Genus', 'genus'), ('Species', 'species')]:
            res[level] = compute_gap(emb, col)
        results[name] = res
        print(f"✓ {name} loaded (best epoch={epoch})")
    except FileNotFoundError:
        print(f"⚠ {name}: 체크포인트 없음 ({path}) — 아직 학습 안 했으면 건너뜀")

# ── 표 출력 ─────────────────────────────────────────────────────────────────
print("\n[3/4] Results")
print("=" * 80)
for name, res in results.items():
    print(f"\n{name}")
    print(f"{'Level':<10} {'Intra':>10} {'Inter':>10} {'Gap':>10}")
    print("-" * 42)
    for level in ['Family', 'Genus', 'Species']:
        intra, inter, gap = res[level]
        print(f"{level:<10} {intra:>10.4f} {inter:>10.4f} {gap:>10.4f}")

# FTM 단독 효과 vs TT 추가 기여 분리
if 'FTM-only' in results and 'FTM+TT' in results:
    print("\n" + "=" * 80)
    print("핵심 비교: FTM 단독 효과 vs Tree-Triplet의 '추가' 기여")
    print("=" * 80)
    print(f"{'Level':<10} {'WP1':>10} {'FTM-only':>10} {'FTM+TT':>10} {'TT의 추가기여':>14}")
    print("-" * 60)
    for level in ['Family', 'Genus', 'Species']:
        wp1_gap     = results['Baseline'][level][2]
        ftm_gap     = results['FTM-only'][level][2]
        full_gap    = results['FTM+TT'][level][2]
        tt_marginal = full_gap - ftm_gap
        print(f"{level:<10} {wp1_gap:>10.4f} {ftm_gap:>10.4f} {full_gap:>10.4f} {tt_marginal:>+14.4f}")

# ── Bar chart (4-way) ──────────────────────────────────────────────────────
print("\n[4/4] Creating visualizations...")
levels = ['Family', 'Genus', 'Species']
colors = {'Baseline': '#4C72B0', 'FTM-only': '#DD8452',
         'TT-only': '#937860', 'FTM+TT': '#55A868'}

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(levels))
n_bars = len(results)
w = 0.8 / n_bars

for i, (name, res) in enumerate(results.items()):
    gaps = [res[l][2] for l in levels]
    offset = (i - (n_bars - 1) / 2) * w
    ax.bar(x + offset, gaps, w, label=name, color=colors.get(name, '#9E9E9E'),
           alpha=0.85, edgecolor='black', linewidth=1.2)

ax.set_xlabel('Hierarchy Level', fontsize=12, fontweight='bold')
ax.set_ylabel('Intra - Inter Similarity Gap', fontsize=12, fontweight='bold')
ax.set_title('Baseline vs FTM-only vs TT-only vs FTM+TT\n(Single-label)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(levels, fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/ablation_comparison.png")

print("\n✓ Done!")