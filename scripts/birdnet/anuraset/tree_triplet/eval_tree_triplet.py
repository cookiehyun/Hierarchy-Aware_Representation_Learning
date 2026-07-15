"""
WP2 Evaluation - Method 2: HSSN (faithful reproduction, single-label only)

학습된 HierarchicalModel의 triplet head(z, 128-d)로 test set 임베딩을 변환하고
WP1(single-label baseline)과 동일한 방식으로 cosine similarity gap + t-SNE 비교.

⚠️ UALB(Method 1) 비교는 여기 없음 — UALB는 아직 multi-label 버그가 있는
   데이터로 학습된 상태라, 지금 단일라벨로 정제된 HSSN과 나란히 비교하면
   공정하지 않음. UALB도 single_label_only로 재학습한 뒤에 추가할 것.
"""

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import torch
import numpy as np
import pandas as pd
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
MODEL_PATH    = '/user/gaad2403/Bachelor-Thesis/results/wp2_hssn_singlelabel/best_model.pt'
OUTPUT_DIR    = Path('/user/gaad2403/Bachelor-Thesis/results/wp2_hssn_singlelabel')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TSNE = 5000

print("=" * 80)
print("WP2 Evaluation - Method 2: HSSN (single-label only)")
print("=" * 80)

# ── Load datasets (single_label_only=True, train으로 encoder fit -> test에 재사용) ──
print("\n[1/5] Loading datasets...")
train_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='train', single_label_only=True)
test_ds  = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='test', single_label_only=True,
                                    family_enc=train_ds.family_enc,
                                    genus_enc=train_ds.genus_enc,
                                    species_enc=train_ds.species_enc)
print(f"✓ Test set loaded: {len(test_ds)} samples")

# n_hierarchy_nodes는 학습 때와 정확히 같은 값이어야 state_dict가 로드됨
# (하드코딩 대신, 학습 때와 동일하게 HierarchyTree로 재계산해서 일치 보장)
tree = HierarchyTree(TAXONOMY_PATH, train_ds.family_enc, train_ds.genus_enc, train_ds.species_enc)
print(f"✓ n_hierarchy_nodes = {tree.n_nodes} (model 구조 확인용)")

# ── Load model ─────────────────────────────────────────────────────────────
print("\n[2/5] Loading model...")
model = HierarchicalModel(input_dim=4096, hidden_dim=1024, triplet_dim=128,
                          n_hierarchy_nodes=tree.n_nodes).to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"✓ Loaded checkpoint (epoch={ckpt['epoch']}, val_ftm_loss={ckpt['val_ftm_loss']:.4f})")

# ── Extract triplet embeddings (z) ─────────────────────────────────────────
print("\n[3/5] Extracting projected embeddings (z)...")
all_embeddings = []
with torch.no_grad():
    batch_size = 512
    for i in range(0, len(test_ds), batch_size):
        batch_emb = test_ds.embeddings[i:i+batch_size].to(DEVICE)
        z, _ = model(batch_emb)          # s(hierarchy score)는 평가에서 안 씀
        all_embeddings.append(z.cpu().numpy())

embeddings = np.vstack(all_embeddings)   # [N, 128], 이미 L2-normalized (모델 내부에서 처리됨)
meta       = test_ds.meta.reset_index(drop=True)
print(f"✓ Projected embeddings shape: {embeddings.shape}")

# ── Cosine similarity gap ──────────────────────────────────────────────────
print("\n[4/5] Computing cosine similarity gap...")

# WP1 (single-label baseline_probing_singlelabel.py 결과 — 직접 입력)
wp1 = {
    'Family':  (0.5294, 0.5194, 0.0100),
    'Genus':   (0.5338, 0.5215, 0.0124),
    'Species': (0.6026, 0.5174, 0.0853),
}

hssn_results = {}
np.random.seed(42)
for level, col in [('Family', 'family'), ('Genus', 'genus'), ('Species', 'species')]:
    labels  = meta[col].values
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
        sim_inter = cosine_similarity(sub, embeddings[sample_other])
        inter_sims.extend(sim_inter.flatten().tolist())

    intra = np.mean(intra_sims)
    inter = np.mean(inter_sims)
    gap   = intra - inter
    hssn_results[level] = (intra, inter, gap)

print(f"\n{'WP1 (Baseline, single-label)':<28} {'Intra':>10} {'Inter':>10} {'Gap':>10}")
print("-" * 60)
for level in ['Family', 'Genus', 'Species']:
    intra, inter, gap = wp1[level]
    print(f"{level:<28} {intra:>10.4f} {inter:>10.4f} {gap:>10.4f}")

print(f"\n{'WP2-M2 (HSSN, single-label)':<28} {'Intra':>10} {'Inter':>10} {'Gap':>10}")
print("-" * 60)
for level in ['Family', 'Genus', 'Species']:
    intra, inter, gap = hssn_results[level]
    print(f"{level:<28} {intra:>10.4f} {inter:>10.4f} {gap:>10.4f}")

# ── Bar chart ──────────────────────────────────────────────────────────────
print("\n[5/5] Creating visualizations...")

levels = ['Family', 'Genus', 'Species']
wp1_gaps  = [wp1[l][2] for l in levels]
hssn_gaps = [hssn_results[l][2] for l in levels]

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(levels))
w = 0.35

ax.bar(x - w/2, wp1_gaps, w, label='NatureLM Baseline(single-label)',
       color='#4C72B0', alpha=0.8, edgecolor='black', linewidth=1.5)
ax.bar(x + w/2, hssn_gaps, w, label='HSSN(single-label)',
       color='#55A868', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Hierarchy Level', fontsize=12, fontweight='bold')
ax.set_ylabel('Intra - Inter Similarity Gap', fontsize=12, fontweight='bold')
ax.set_title('Cosine Similarity Gap: Baseline vs HSSN',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(levels, fontsize=11)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gap_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: gap_comparison.png")

# ── t-SNE ──────────────────────────────────────────────────────────────────
print(f"\nRunning t-SNE (n={min(N_TSNE, len(embeddings))})...")
idx_sample  = np.random.choice(len(embeddings), min(N_TSNE, len(embeddings)), replace=False)
emb_sample  = embeddings[idx_sample]
meta_sample = meta.iloc[idx_sample].reset_index(drop=True)

tsne   = TSNE(n_components=2, random_state=42, max_iter=1000, perplexity=30)
emb_2d = tsne.fit_transform(emb_sample)

FAMILY_COLORS = {
    'Hylidae':         '#4C72B0',
    'Leptodactylidae': '#DD8452',
    'Bufonidae':       '#C44E52',
    'Microhylidae':    '#8172B2',
}

genera = sorted(meta_sample['genus'].unique())
cmap   = plt.colormaps['tab20'].resampled(len(genera))   # ✏️ matplotlib 신버전 호환
GENUS_COLORS = {g: cmap(i) for i, g in enumerate(genera)}

# Family level t-SNE
fig, ax = plt.subplots(figsize=(12, 9))
labels_plot = meta_sample['family'].values
for label in sorted(set(labels_plot)):
    idx_l = [i for i, l in enumerate(labels_plot) if l == label]
    color = FAMILY_COLORS.get(label, '#9E9E9E')
    ax.scatter(emb_2d[idx_l, 0], emb_2d[idx_l, 1],
               c=[color], label=label, alpha=0.6, s=10, linewidths=0.5, edgecolors='black')
ax.legend(markerscale=2.5, fontsize=10, bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_title(f'HSSN(single-label) t-SNE — Family Level (n={len(emb_sample)})',
             fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE 1', fontsize=11)
ax.set_ylabel('t-SNE 2', fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tsne_family.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: tsne_family.png")

# Genus level t-SNE
fig, ax = plt.subplots(figsize=(14, 10))
labels_plot = meta_sample['genus'].values
for label in sorted(set(labels_plot)):
    idx_l = [i for i, l in enumerate(labels_plot) if l == label]
    color = GENUS_COLORS.get(label, '#9E9E9E')
    ax.scatter(emb_2d[idx_l, 0], emb_2d[idx_l, 1],
               c=[color], label=label, alpha=0.6, s=10, linewidths=0.5, edgecolors='black')
ax.legend(markerscale=2.5, fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
ax.set_title(f'HSSN(single-label) t-SNE — Genus Level (n={len(emb_sample)})',
             fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE 1', fontsize=11)
ax.set_ylabel('t-SNE 2', fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'tsne_genus.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: tsne_genus.png")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print(f"\n📊 Results saved to: {OUTPUT_DIR}")

print("\n📈 Gap Improvements (vs WP1 single-label baseline):")
for level in levels:
    hssn_gap = hssn_results[level][2]
    wp1_gap  = wp1[level][2]
    improvement = ((hssn_gap - wp1_gap) / wp1_gap) * 100
    direction = "↑" if improvement > 0 else "↓"
    print(f"   {level:<10}: {direction} {abs(improvement):>6.2f}% "
          f"(WP1={wp1_gap:.4f} → HSSN={hssn_gap:.4f})")

print("\n✓ Done!")