"""
WP2 Evaluation - Method 1: UALB (single-label only)

기존 eval_ualb.py와 동일 구조. 바뀐 것:
  1. single_label_only=True (FTM-only/TT-only/FTM+TT와 동일 기준으로 통일)
  2. MODEL_PATH/OUTPUT_DIR -> wp2_ualb_singlelabel
  3. WP1 baseline 수치 -> 단일라벨 기준(baseline_probing_singlelabel.py 결과)
  4. matplotlib get_cmap 신버전 호환
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
from models.projection_head import ProjectionHead

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH  = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH   = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
MODEL_PATH  = '/user/gaad2403/Bachelor-Thesis/results/wp2_ualb_singlelabel/best_model.pt'
OUTPUT_DIR  = Path('/user/gaad2403/Bachelor-Thesis/results/wp2_ualb_singlelabel')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TSNE = 5000

# ── Load model ─────────────────────────────────────────────────────────────
print("Loading model...")
model = ProjectionHead(input_dim=4096, proj_dim=128).to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded checkpoint (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

# ── Load test set (single_label_only=True) ─────────────────────────────────
print("Loading test set...")
train_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='train', single_label_only=True)
test_ds  = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='test', single_label_only=True,
                                    family_enc=train_ds.family_enc,
                                    genus_enc=train_ds.genus_enc,
                                    species_enc=train_ds.species_enc)

# ── Extract projected embeddings ───────────────────────────────────────────
print("Extracting projected embeddings...")
all_embeddings = []
with torch.no_grad():
    batch_size = 512
    for i in range(0, len(test_ds), batch_size):
        batch_emb = test_ds.embeddings[i:i+batch_size].to(DEVICE)
        z = model(batch_emb)
        all_embeddings.append(z.cpu().numpy())

embeddings = np.vstack(all_embeddings)
meta       = test_ds.meta.reset_index(drop=True)
print(f"Projected embeddings shape: {embeddings.shape}")

norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.clip(norms, 1e-8, None)

# ── Cosine similarity gap ──────────────────────────────────────────────────
print("\n=== Cosine Similarity Gap (single-label) ===")
print(f"{'Level':<10} {'Intra':>8} {'Inter':>8} {'Gap':>8}")
print("-" * 38)

wp1 = {
    'Family':  (0.5294, 0.5194, 0.0100),
    'Genus':   (0.5338, 0.5215, 0.0124),
    'Species': (0.6026, 0.5174, 0.0853),
}

results = {}
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
                                  min(len(idx)*5, len(other_idx)), replace=False)]
        sim_inter = cosine_similarity(sub, embeddings[sample_other])
        inter_sims.extend(sim_inter.flatten().tolist())

    intra, inter = np.mean(intra_sims), np.mean(inter_sims)
    gap = intra - inter
    results[level] = (intra, inter, gap)

    print(f"{level:<10} {intra:>8.4f} {inter:>8.4f} {gap:>8.4f}  (WP1 gap={wp1[level][2]:.4f})")

# ── Bar chart ──────────────────────────────────────────────────────────────
levels   = ['Family', 'Genus', 'Species']
wp1_gaps = [wp1[l][2] for l in levels]
wp2_gaps = [results[l][2] for l in levels]

x = np.arange(len(levels))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, wp1_gaps, w, label='WP1 (NatureLM, single-label)', color='steelblue', alpha=0.8)
ax.bar(x + w/2, wp2_gaps, w, label='WP2 (UALB, single-label)',     color='coral',     alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel('Intra - Inter Similarity Gap')
ax.set_title('Cosine Similarity Gap: WP1 vs WP2 (UALB), single-label')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gap_comparison.png', dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'gap_comparison.png'}")

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

for level, col, color_map in [('family', 'family', FAMILY_COLORS), ('genus', 'genus', GENUS_COLORS)]:
    fig, ax = plt.subplots(figsize=(12, 9))
    labels_plot = meta_sample[col].values
    for label in sorted(set(labels_plot)):
        idx_l = [i for i, l in enumerate(labels_plot) if l == label]
        color = color_map.get(label, '#9E9E9E')
        ax.scatter(emb_2d[idx_l, 0], emb_2d[idx_l, 1],
                   c=[color], label=label, alpha=0.5, s=8, linewidths=0)
    ax.legend(markerscale=3, fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_title(f'WP2 (UALB, single-label) t-SNE — {level.capitalize()} Level (n={len(emb_sample)})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    save_path = OUTPUT_DIR / f'tsne_{level}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

print("\n=== Evaluation Complete ===")
print(f"All results saved to: {OUTPUT_DIR}")