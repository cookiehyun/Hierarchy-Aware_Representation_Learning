import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from NatureLM.anuraset.anuraset_dataset import AnuraSetEmbeddingDataset
from NatureLM.anuraset.hierarchy_tree import HierarchyTree
from models.hierarchical_model import HierarchicalModel

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
EMBED_PATH    = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH     = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
OUTPUT_DIR    = Path('/user/gaad2403/Bachelor-Thesis/results/tsne_before_after')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FTM_ONLY_PATH = '/user/gaad2403/Bachelor-Thesis/results/NatureLM/anuraset/wp2_ftm_only_singlelabel/best_model.pt'

INPUT_DIM = 4096
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED      = 42

# ─────────────────────────────────────────────
# 1) 데이터 로드
# ─────────────────────────────────────────────
def load_datasets():
    train_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='train', single_label_only=True)
    test_ds  = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='test', single_label_only=True,
                                        family_enc=train_ds.family_enc,
                                        genus_enc=train_ds.genus_enc,
                                        species_enc=train_ds.species_enc)
    return train_ds, test_ds

print("Loading datasets...")
train_ds, test_ds = load_datasets()
tree = HierarchyTree(TAXONOMY_PATH, train_ds.family_enc, train_ds.genus_enc, train_ds.species_enc)
print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | n_hierarchy_nodes={tree.n_nodes}")

# ─────────────────────────────────────────────
# 2) 임베딩 추출 함수
# ─────────────────────────────────────────────
def extract_baseline(ds):
    emb = ds.embeddings.numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)

def extract_hidden(ds, model_path, n_hierarchy_nodes):
    model = HierarchicalModel(input_dim=INPUT_DIM, hidden_dim=1024, triplet_dim=128,
                               n_hierarchy_nodes=n_hierarchy_nodes).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            h = model.encode_hidden(ds.embeddings[i:i + 512].to(DEVICE))
            out.append(h.cpu().numpy())
    emb = np.vstack(out)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)

# ─────────────────────────────────────────────
# 3) Gap 계산
# ─────────────────────────────────────────────
N_INTER_MULT = 5

def compute_gap(embeddings, labels):
    np.random.seed(SEED)
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
                                  min(len(idx) * N_INTER_MULT, len(other_idx)), replace=False)]
        inter_sims.extend(cosine_similarity(sub, embeddings[sample_other]).flatten().tolist())
    intra, inter = np.mean(intra_sims), np.mean(inter_sims)
    return intra, inter, intra - inter

# ─────────────────────────────────────────────
# 4) 임베딩 추출
# ─────────────────────────────────────────────
print("\nExtracting embeddings (test set)...")
test_emb_baseline = extract_baseline(test_ds)
test_emb_ftmonly  = extract_hidden(test_ds, FTM_ONLY_PATH, tree.n_nodes)

test_family = test_ds.meta['family'].values
test_genus  = test_ds.meta['genus'].values

# ─────────────────────────────────────────────
# 5) Gap 계산
# ─────────────────────────────────────────────
print("\nComputing Gap...")
fam_gap_b = compute_gap(test_emb_baseline, test_family)[2]
gen_gap_b = compute_gap(test_emb_baseline, test_genus)[2]
fam_gap_f = compute_gap(test_emb_ftmonly, test_family)[2]
gen_gap_f = compute_gap(test_emb_ftmonly, test_genus)[2]
print(f"  Baseline: Family Gap={fam_gap_b:.4f} Genus Gap={gen_gap_b:.4f}")
print(f"  FTM-only: Family Gap={fam_gap_f:.4f} Genus Gap={gen_gap_f:.4f}")

# ─────────────────────────────────────────────
# 6) t-SNE
# ─────────────────────────────────────────────
print(f"\nRunning t-SNE on test set (n={len(test_ds)})...")
tsne_b = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
emb2d_baseline = tsne_b.fit_transform(test_emb_baseline)

tsne_f = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
emb2d_ftmonly = tsne_f.fit_transform(test_emb_ftmonly)

# ─────────────────────────────────────────────
# 7) 시각화 -- Genus level, 2-panel, 폰트 확대 + 배경색 steelblue!12
# ─────────────────────────────────────────────
genera = sorted(set(test_genus))
cmap = plt.colormaps["tab20"].resampled(len(genera))
GENUS_COLORS = {g: cmap(i) for i, g in enumerate(genera)}

POSTER_BG = '#E9EDF3'   # steelblue!12 (RGB 68,102,158 @ 12% on white)

TITLE_FS    = 26
SUPTITLE_FS = 28
LABEL_FS    = 20
TICK_FS     = 16
LEGEND_FS   = 18
POINT_SIZE  = 22

def plot_tsne_ax(ax, emb_2d, labels, color_map, title):
    ax.set_facecolor(POSTER_BG)
    for label in sorted(set(labels)):
        mask = labels == label
        color = color_map.get(label, "#9E9E9E")
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=color, label=label, alpha=0.7, s=POINT_SIZE, linewidths=0)
    ax.set_title(title, fontsize=TITLE_FS, pad=14)
    ax.set_xlabel("t-SNE 1", fontsize=LABEL_FS)
    ax.set_ylabel("t-SNE 2", fontsize=LABEL_FS)
    ax.tick_params(axis='both', labelsize=TICK_FS)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.patch.set_facecolor(POSTER_BG)

plot_tsne_ax(axes[0], emb2d_baseline, test_genus, GENUS_COLORS, "Before (Baseline)")
plot_tsne_ax(axes[1], emb2d_ftmonly, test_genus, GENUS_COLORS, "After (FTM-only)")

handles, labels_ = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels_, loc='upper center', bbox_to_anchor=(0.5, 1.14),
           ncol=6, markerscale=3, fontsize=LEGEND_FS, frameon=True)
legend.get_frame().set_facecolor(POSTER_BG)
legend.get_frame().set_edgecolor('#C8CDD4')  # rulegray, 테두리도 포스터 톤에 맞춤

fig.suptitle("NatureLM Embeddings t-SNE — Genus Level, Single-label, Test Set",
             fontsize=SUPTITLE_FS, y=1.24)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tsne_baseline_ftmonly_genus_singlelabel_testset.png",
            dpi=300, bbox_inches="tight", facecolor=POSTER_BG)
plt.close()
print(f"\nSaved: {OUTPUT_DIR / 'tsne_baseline_ftmonly_genus_singlelabel_testset.png'}")