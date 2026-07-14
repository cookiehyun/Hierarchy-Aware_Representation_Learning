import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR   = Path("/user/gaad2403/bachelor_thesis")
EMBED_DIR  = BASE_DIR / "embeddings_esc50"
META_PATH  = Path("/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv")
OUTPUT_DIR = Path("/user/gaad2403/Bachelor-Thesis/results/esc50_baseline_probing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
print("Loading embeddings...")
embeddings = np.load(EMBED_DIR / "embeddings.npy")   # [2000, 4096]
metadata   = pd.read_csv(META_PATH)                  # filename, fold, category, midcat, bga_label, ...

print(f"Embeddings shape: {embeddings.shape}")
print(f"Metadata shape:   {metadata.shape}")
print(f"\nBGA distribution:\n{metadata['bga_label'].value_counts()}\n")

# L2 정규화 (cosine similarity를 위해)
norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.clip(norms, 1e-8, None)

# ─────────────────────────────────────────────
# 1. Per-class mean embedding 계산
# ─────────────────────────────────────────────
def compute_class_means(embeddings, labels):
    classes = sorted(set(labels))
    means   = []
    for c in classes:
        idx = [i for i, l in enumerate(labels) if l == c]
        means.append(embeddings[idx].mean(axis=0))
    return np.array(means), classes

# ─────────────────────────────────────────────
# 2. Cosine Similarity Matrix 시각화
# ─────────────────────────────────────────────
def plot_similarity_matrix(sim_matrix, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

bga_means, bga_classes = compute_class_means(embeddings, metadata["bga_label"].tolist())
bga_sim = cosine_similarity(bga_means)
plot_similarity_matrix(
    bga_sim, bga_classes,
    "NatureLM Embeddings (ESC-50): BGA-level Cosine Similarity",
    OUTPUT_DIR / "similarity_bga.png"
)

midcat_means, midcat_classes = compute_class_means(embeddings, metadata["midcat"].tolist())
midcat_sim = cosine_similarity(midcat_means)
plot_similarity_matrix(
    midcat_sim, midcat_classes,
    "NatureLM Embeddings (ESC-50): Mid-category-level Cosine Similarity",
    OUTPUT_DIR / "similarity_midcat.png"
)

# ─────────────────────────────────────────────
# 3. Intra- vs Inter-cluster similarity 분석
# ─────────────────────────────────────────────
def compute_intra_inter(embeddings, labels, level_name, n_sample=2000):
    n_sample = min(n_sample, len(embeddings))
    idx   = np.random.choice(len(embeddings), n_sample, replace=False)
    emb_s = embeddings[idx]
    lab_s = [labels[i] for i in idx]

    sim_matrix = cosine_similarity(emb_s)
    intra, inter = [], []

    for i in range(len(lab_s)):
        for j in range(i+1, len(lab_s)):
            if lab_s[i] == lab_s[j]:
                intra.append(sim_matrix[i, j])
            else:
                inter.append(sim_matrix[i, j])

    intra = np.array(intra)
    inter = np.array(inter)

    print(f"\n[{level_name}]")
    print(f"  Intra-class similarity: mean={intra.mean():.4f}, std={intra.std():.4f}")
    print(f"  Inter-class similarity: mean={inter.mean():.4f}, std={inter.std():.4f}")
    print(f"  Gap (intra - inter):    {intra.mean() - inter.mean():.4f}")

    return intra, inter

np.random.seed(42)
# 전체가 2000개뿐이라 n_sample은 전부 사용 (AnuraSet은 93k라 5000개 샘플링했었음)
intra_bga,    inter_bga    = compute_intra_inter(embeddings, metadata["bga_label"].tolist(), "BGA")
intra_midcat, inter_midcat = compute_intra_inter(embeddings, metadata["midcat"].tolist(),    "Midcat")
intra_cat,    inter_cat    = compute_intra_inter(embeddings, metadata["category"].tolist(),  "Category")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
levels = [
    ("BGA",      intra_bga,    inter_bga),
    ("Midcat",   intra_midcat, inter_midcat),
    ("Category", intra_cat,    inter_cat),
]
for ax, (name, intra, inter) in zip(axes, levels):
    ax.boxplot([intra, inter], labels=["Intra", "Inter"], patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.set_title(f"{name} Level")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)

fig.suptitle("NatureLM Embeddings (ESC-50): Intra- vs Inter-class Cosine Similarity", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "intra_inter_similarity.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/intra_inter_similarity.png")

# ─────────────────────────────────────────────
# 4. t-SNE 시각화
# ─────────────────────────────────────────────
print("\nRunning t-SNE...")

# AnuraSet은 93k 중 5000개만 샘플링했지만, ESC-50은 전체가 2000개라 다 사용
N_TSNE = len(embeddings)
idx    = np.arange(N_TSNE)
emb_s  = embeddings[idx]
meta_s = metadata.iloc[idx].reset_index(drop=True)

tsne   = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
emb_2d = tsne.fit_transform(emb_s)

BGA_COLORS = {
    "biophony":    "#4CAF50",
    "geophony":    "#2196F3",
    "anthrophony": "#F44336",
}

def plot_tsne(emb_2d, labels, color_map, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 9))
    for label in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == label]
        color = color_map.get(label, "#9E9E9E")
        ax.scatter(emb_2d[idx, 0], emb_2d[idx, 1],
                   c=color, label=label, alpha=0.5, s=8, linewidths=0)
    ax.legend(markerscale=3, fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

plot_tsne(
    emb_2d, meta_s["bga_label"].tolist(), BGA_COLORS,
    f"NatureLM Embeddings t-SNE (ESC-50, n={N_TSNE}) — BGA Level",
    OUTPUT_DIR / "tsne_bga.png"
)

midcats = sorted(meta_s["midcat"].unique())
import matplotlib as mpl
base_colors = mpl.colormaps["tab20"].colors  # 20개 고정 색상 리스트, 버전 무관
MIDCAT_COLORS = {m: base_colors[i % 20] for i, m in enumerate(midcats)}
plot_tsne(
    emb_2d, meta_s["midcat"].tolist(), MIDCAT_COLORS,
    f"NatureLM Embeddings t-SNE (ESC-50, n={N_TSNE}) — Mid-category Level",
    OUTPUT_DIR / "tsne_midcat.png"
)

print("\n=== ESC-50 Baseline Probing Complete ===")
print(f"All results saved to: {OUTPUT_DIR}")