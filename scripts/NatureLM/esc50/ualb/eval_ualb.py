"""
WP3 Evaluation - Method 1: UALB on ESC-50
학습된 projection head로 test fold(5) 임베딩을 변환하고
WP3-A baseline과 동일한 방식으로 cosine similarity gap + t-SNE 비교
"""

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from esc50.esc50_dataset import ESC50EmbeddingDataset
from models.projection_head import ProjectionHead

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_esc50/embeddings.npy'
META_PATH  = '/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv'
MODEL_PATH = '/user/gaad2403/Bachelor-Thesis/results/esc50_wp3_ualb/best_model.pt'
OUTPUT_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/esc50_wp3_ualb')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = [1, 2, 3, 4]
TEST_FOLDS  = [5]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load model ─────────────────────────────────────────────────────────────
print("Loading model...")
model = ProjectionHead(input_dim=4096, proj_dim=128).to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Loaded checkpoint (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

# ── Load train(fit encoder) / test set ────────────────────────────────────
print("Loading datasets...")
train_ds = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TRAIN_FOLDS)
test_ds  = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TEST_FOLDS,
                                  bga_enc=train_ds.bga_enc,
                                  midcat_enc=train_ds.midcat_enc,
                                  category_enc=train_ds.category_enc)

# ── Extract projected embeddings ───────────────────────────────────────────
print("Extracting projected embeddings...")
all_embeddings = []
with torch.no_grad():
    batch_size = 512
    for i in range(0, len(test_ds), batch_size):
        batch_emb = test_ds.embeddings[i:i+batch_size].to(DEVICE)
        z = model(batch_emb)
        all_embeddings.append(z.cpu().numpy())

embeddings = np.vstack(all_embeddings)   # [N, 128]
meta       = test_ds.meta.reset_index(drop=True)
print(f"Projected embeddings shape: {embeddings.shape}")

# L2 normalize
norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.clip(norms, 1e-8, None)

# ── Cosine similarity gap (WP3-A baseline과 비교) ──────────────────────────
print("\n=== Cosine Similarity Gap ===")
print(f"{'Level':<10} {'Intra':>8} {'Inter':>8} {'Gap':>8}")
print("-" * 38)

# WP3-A baseline_probing.py 결과 (전체 2000개 기준, 참고용)
baseline = {
    'BGA':      (0.5595, 0.5025, 0.0570),
    'Midcat':   (0.5912, 0.5195, 0.0717),
    'Category': (0.7022, 0.5253, 0.1769),
}

np.random.seed(42)
results = {}
for level, col in [('BGA', 'bga_label'), ('Midcat', 'midcat'), ('Category', 'category')]:
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

    intra = np.mean(intra_sims)
    inter = np.mean(inter_sims)
    gap   = intra - inter
    results[level] = (intra, inter, gap)

    base_gap = baseline[level][2]
    print(f"{level:<10} {intra:>8.4f} {inter:>8.4f} {gap:>8.4f}  (Baseline gap={base_gap:.4f})")

# ── Bar chart: Baseline vs UALB gap 비교 ───────────────────────────────────
levels       = ['BGA', 'Midcat', 'Category']
baseline_gaps = [baseline[l][2] for l in levels]
ualb_gaps     = [results[l][2] for l in levels]

x = np.arange(len(levels))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, baseline_gaps, w, label='Baseline (NatureLM)', color='steelblue', alpha=0.8)
ax.bar(x + w/2, ualb_gaps,     w, label='WP3 (UALB)',          color='coral',     alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.set_ylabel('Intra - Inter Similarity Gap')
ax.set_title('ESC-50: Cosine Similarity Gap — Baseline vs UALB')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'gap_comparison.png', dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'gap_comparison.png'}")

# ── t-SNE ─────────────────────────────────────────────────────────────────
N_TSNE = len(embeddings)  # test fold가 400개뿐이라 전체 사용
print(f"\nRunning t-SNE (n={N_TSNE})...")

tsne   = TSNE(n_components=2, random_state=42, max_iter=1000, perplexity=30)
emb_2d = tsne.fit_transform(embeddings)

BGA_COLORS = {
    "biophony":    "#4CAF50",
    "geophony":    "#2196F3",
    "anthrophony": "#F44336",
}

midcats = sorted(meta['midcat'].unique())
base_colors = mpl.colormaps["tab20"].colors   # get_cmap 대신 안전한 방식
MIDCAT_COLORS = {m: base_colors[i % 20] for i, m in enumerate(midcats)}

for level, col, color_map in [('bga', 'bga_label', BGA_COLORS), ('midcat', 'midcat', MIDCAT_COLORS)]:
    fig, ax = plt.subplots(figsize=(12, 9))
    labels_plot = meta[col].values
    for label in sorted(set(labels_plot)):
        idx_l = [i for i, l in enumerate(labels_plot) if l == label]
        color = color_map.get(label, '#9E9E9E')
        ax.scatter(emb_2d[idx_l, 0], emb_2d[idx_l, 1],
                   c=[color], label=label, alpha=0.5, s=8, linewidths=0)
    ax.legend(markerscale=3, fontsize=9, bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.set_title(f'WP3 (UALB) t-SNE — {level.upper()} Level (n={N_TSNE})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    save_path = OUTPUT_DIR / f'tsne_{level}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

print("\n=== Evaluation Complete ===")
print(f"All results saved to: {OUTPUT_DIR}")