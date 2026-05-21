import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR     = Path("/user/gaad2403/bachelor_thesis")
EMBED_DIR    = BASE_DIR / "embeddings"
OUTPUT_DIR   = Path("/user/gaad2403/Bachelor-Thesis/results/baseline_probing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
print("Loading embeddings...")
embeddings = np.load(EMBED_DIR / "embeddings.npy")        # [93378, 4096]
metadata   = pd.read_csv(EMBED_DIR / "embeddings_metadata.csv")

print(f"Embeddings shape (raw): {embeddings.shape}")
print(f"Metadata shape (raw):   {metadata.shape}")

# unknown (종 라벨 없는 배경 소음) 제거
metadata   = metadata[metadata["family"] != "unknown"].reset_index(drop=True)
embeddings = embeddings[metadata.index]

print(f"Embeddings shape (filtered): {embeddings.shape}")
print(f"Metadata shape (filtered):   {metadata.shape}")
print(f"\nFamily distribution:\n{metadata['family'].value_counts()}\n")

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

family_means, family_classes = compute_class_means(embeddings, metadata["family"].tolist())
family_sim = cosine_similarity(family_means)
plot_similarity_matrix(
    family_sim, family_classes,
    "NatureLM Embeddings: Family-level Cosine Similarity",
    OUTPUT_DIR / "similarity_family.png"
)

genus_means, genus_classes = compute_class_means(embeddings, metadata["genus"].tolist())
genus_sim = cosine_similarity(genus_means)
plot_similarity_matrix(
    genus_sim, genus_classes,
    "NatureLM Embeddings: Genus-level Cosine Similarity",
    OUTPUT_DIR / "similarity_genus.png"
)

# ─────────────────────────────────────────────
# 3. Intra- vs Inter-cluster similarity 분석
# ─────────────────────────────────────────────
def compute_intra_inter(embeddings, labels, level_name, n_sample=5000):
    idx = np.random.choice(len(embeddings), min(n_sample, len(embeddings)), replace=False)
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
intra_family, inter_family = compute_intra_inter(embeddings, metadata["family"].tolist(), "Family")
intra_genus,  inter_genus  = compute_intra_inter(embeddings, metadata["genus"].tolist(),  "Genus")
intra_species,inter_species= compute_intra_inter(embeddings, metadata["species"].tolist(),"Species")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
levels = [
    ("Family",  intra_family,  inter_family),
    ("Genus",   intra_genus,   inter_genus),
    ("Species", intra_species, inter_species),
]
for ax, (name, intra, inter) in zip(axes, levels):
    ax.boxplot([intra, inter], labels=["Intra", "Inter"], patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.set_title(f"{name} Level")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8)

fig.suptitle("NatureLM Embeddings: Intra- vs Inter-class Cosine Similarity", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "intra_inter_similarity.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/intra_inter_similarity.png")

# ─────────────────────────────────────────────
# 4. t-SNE 시각화
# ─────────────────────────────────────────────
print("\nRunning t-SNE (this may take a few minutes)...")

N_TSNE = 5000
idx    = np.random.choice(len(embeddings), N_TSNE, replace=False)
emb_s  = embeddings[idx]
meta_s = metadata.iloc[idx].reset_index(drop=True)

tsne   = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
emb_2d = tsne.fit_transform(emb_s)

FAMILY_COLORS = {
    "Hylidae":         "#2196F3",
    "Leptodactylidae": "#F44336",
    "Bufonidae":       "#4CAF50",
    "Phyllomedusidae": "#FF9800",
    "Microhylidae":    "#9C27B0",
    "Dendrobatidae":   "#00BCD4",
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
    emb_2d, meta_s["family"].tolist(), FAMILY_COLORS,
    f"NatureLM Embeddings t-SNE (n={N_TSNE}) — Family Level",
    OUTPUT_DIR / "tsne_family.png"
)

genera = sorted(meta_s["genus"].unique())
cmap   = plt.cm.get_cmap("tab20", len(genera))
GENUS_COLORS = {g: cmap(i) for i, g in enumerate(genera)}
plot_tsne(
    emb_2d, meta_s["genus"].tolist(), GENUS_COLORS,
    f"NatureLM Embeddings t-SNE (n={N_TSNE}) — Genus Level",
    OUTPUT_DIR / "tsne_genus.png"
)

print("\n=== Baseline Probing Complete ===")
print(f"All results saved to: {OUTPUT_DIR}")