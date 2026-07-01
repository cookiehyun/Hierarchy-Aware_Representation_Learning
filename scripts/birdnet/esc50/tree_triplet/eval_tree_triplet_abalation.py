"""
WP3 Evaluation - ESC-50 통합 평가
Baseline / UALB / FTM-only / TT-only / FTM+TT 비교

AnuraSet에서 배운 교훈 반영: cosine similarity gap만으로는 collapse를
못 잡아내므로(예: 클래스가 과격하게 떨어져서 gap은 커지지만 실제 분류는
못하는 경우), 처음부터 아래 3가지를 모두 계산함:
  1. Cosine Similarity Gap (intra-inter)
  2. Linear Probe Top-1 Accuracy (frozen feature + Linear classifier, Adam,
     class-balanced loss — collapse 방지)
  3. Clustering NMI (K-means)
"""

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from pathlib import Path

from esc50.esc50_dataset import ESC50EmbeddingDataset
from esc50.esc50_hierarchy_tree import ESC50HierarchyTree
from models.projection_head import ProjectionHead
from models.hierarchical_model import HierarchicalModel

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_esc50_birdnet/embeddings.npy'
META_PATH  = '/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv'
OUTPUT_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_ablation_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = [1, 2, 3, 4]
TEST_FOLDS  = [5]

MODEL_PATHS = {
    'UALB':     ('projectionhead', '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_ualb/best_model.pt'),
    'FTM-only': ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_ftm_only/best_model.pt'),
    'TT-only':  ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_tt_only/best_model.pt'),
    'FTM+TT':   ('hierarchical',   '/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_hssn/best_model.pt'),
}

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROBE_EPOCHS = 300
PROBE_LR     = 0.01
SEED         = 42
LEVELS       = [('BGA', 'bga_label', 0), ('Midcat', 'midcat', 1), ('Category', 'category', 2)]

print("=" * 80)
print("WP3 Evaluation (ESC-50): Baseline vs UALB vs FTM-only vs TT-only vs FTM+TT")
print("=" * 80)

# ── Datasets ───────────────────────────────────────────────────────────────
print("\n[1/5] Loading datasets...")
train_ds = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TRAIN_FOLDS)
test_ds  = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TEST_FOLDS,
                                 bga_enc=train_ds.bga_enc,
                                 midcat_enc=train_ds.midcat_enc,
                                 category_enc=train_ds.category_enc)
meta = test_ds.meta.reset_index(drop=True)
print(f"✓ Train: {len(train_ds)} | Test: {len(test_ds)}")

tree = ESC50HierarchyTree(META_PATH, train_ds.bga_enc, train_ds.midcat_enc, train_ds.category_enc)
print(f"✓ n_hierarchy_nodes = {tree.n_nodes}")

LEVEL_INFO = {
    'BGA':      (0, train_ds.n_bga),
    'Midcat':   (1, train_ds.n_midcat),
    'Category': (2, train_ds.n_category),
}


# ── 임베딩 추출 ─────────────────────────────────────────────────────────────
def extract_baseline(ds):
    emb = ds.embeddings.numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)


def extract_projectionhead(ds, model_path):
    model = ProjectionHead(input_dim=1024, proj_dim=128).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    all_z = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            z = model(ds.embeddings[i:i+512].to(DEVICE))
            all_z.append(z.cpu().numpy())
    return np.vstack(all_z)


def extract_hierarchical(ds, model_path):
    model = HierarchicalModel(input_dim=1024, hidden_dim=1024, triplet_dim=128,
                              n_hierarchy_nodes=tree.n_nodes).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    all_z = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            z, _ = model(ds.embeddings[i:i+512].to(DEVICE))
            all_z.append(z.cpu().numpy())
    return np.vstack(all_z)


EXTRACTORS = {'baseline': extract_baseline,
             'projectionhead': extract_projectionhead,
             'hierarchical': extract_hierarchical}

print("\n[2/5] Extracting embeddings for all models...")
embeddings_train = {'Baseline': extract_baseline(train_ds)}
embeddings_test  = {'Baseline': extract_baseline(test_ds)}

for name, (kind, path) in MODEL_PATHS.items():
    try:
        embeddings_train[name] = EXTRACTORS[kind](train_ds, path)
        embeddings_test[name]  = EXTRACTORS[kind](test_ds, path)
        print(f"✓ {name} loaded")
    except FileNotFoundError:
        print(f"⚠ {name}: 체크포인트 없음 ({path}) — 건너뜀")


# ── 1) Cosine Similarity Gap ────────────────────────────────────────────────
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
                                  min(len(idx)*5, len(other_idx)), replace=False)]
        inter_sims.extend(cosine_similarity(sub, embeddings[sample_other]).flatten().tolist())
    intra, inter = np.mean(intra_sims), np.mean(inter_sims)
    return intra, inter, intra - inter


# ── 2) Linear Probe (frozen feature + Linear + class-balanced CE) ──────────
def linear_probe_accuracy(train_emb, train_labels, test_emb, test_labels, n_classes):
    torch.manual_seed(SEED)
    clf = nn.Linear(train_emb.shape[1], n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=PROBE_LR, weight_decay=1e-4)

    class_counts = np.bincount(train_labels, minlength=n_classes).astype(np.float32)
    class_weights = torch.FloatTensor(1.0 / np.clip(class_counts, 1, None)).to(DEVICE)
    class_weights = class_weights / class_weights.sum() * n_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    X_train = torch.FloatTensor(train_emb).to(DEVICE)
    y_train = torch.LongTensor(train_labels).to(DEVICE)
    X_test  = torch.FloatTensor(test_emb).to(DEVICE)
    y_test  = torch.LongTensor(test_labels).to(DEVICE)

    clf.train()
    for epoch in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(clf(X_train), y_train)
        loss.backward()
        optimizer.step()

    clf.eval()
    with torch.no_grad():
        test_acc = (clf(X_test).argmax(dim=1) == y_test).float().mean().item() * 100

    majority_class = np.bincount(train_labels, minlength=n_classes).argmax()
    majority_acc = (test_labels == majority_class).mean() * 100
    return test_acc, majority_acc


# ── 3) Clustering NMI ───────────────────────────────────────────────────────
def clustering_nmi(test_emb, test_labels, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    cluster_ids = km.fit_predict(test_emb)
    return normalized_mutual_info_score(test_labels, cluster_ids) * 100


# ── 전체 실행 ───────────────────────────────────────────────────────────────
print("\n[3/5] Computing Gap / Linear Probe / NMI for each (model, level)...")
gap_results, probe_results, nmi_results = {}, {}, {}

for name, train_emb in embeddings_train.items():
    test_emb = embeddings_test[name]
    gap_results[name], probe_results[name], nmi_results[name] = {}, {}, {}

    for level, col, col_idx in LEVELS:
        test_labels_str = meta[col].values
        gap_results[name][level] = compute_gap(test_emb, test_labels_str)

        n_classes = LEVEL_INFO[level][1]
        train_labels = train_ds.hier_labels[:, col_idx].numpy()
        test_labels  = test_ds.hier_labels[:, col_idx].numpy()
        acc, majority_acc = linear_probe_accuracy(train_emb, train_labels, test_emb, test_labels, n_classes)
        nmi = clustering_nmi(test_emb, test_labels, n_classes)

        probe_results[name][level] = acc
        nmi_results[name][level]   = nmi

        flag = "  ⚠ COLLAPSE?" if acc <= majority_acc + 1.0 else ""
        gap = gap_results[name][level][2]
        print(f"  [{name:<10}] {level:<9} | Gap: {gap:7.4f} | Probe Acc: {acc:6.2f}% "
              f"(다수클래스={majority_acc:.2f}%) | NMI: {nmi:6.2f}%{flag}")

# ── 표 출력 ─────────────────────────────────────────────────────────────────
print("\n[4/5] Summary")
print("=" * 100)
print(f"{'Model':<12} | {'BGA Gap':>9} {'BGA Acc':>9} {'BGA NMI':>9} | "
      f"{'Mid Gap':>9} {'Mid Acc':>9} {'Mid NMI':>9} | "
      f"{'Cat Gap':>9} {'Cat Acc':>9} {'Cat NMI':>9}")
print("-" * 100)
for name in embeddings_train.keys():
    row = []
    for level in ['BGA', 'Midcat', 'Category']:
        row += [gap_results[name][level][2], probe_results[name][level], nmi_results[name][level]]
    print(f"{name:<12} | {row[0]:9.4f} {row[1]:8.2f}% {row[2]:8.2f}% | "
          f"{row[3]:9.4f} {row[4]:8.2f}% {row[5]:8.2f}% | "
          f"{row[6]:9.4f} {row[7]:8.2f}% {row[8]:8.2f}%")

# ── Bar chart (gap + probe acc 둘 다) ───────────────────────────────────────
print("\n[5/5] Creating visualizations...")
levels = ['BGA', 'Midcat', 'Category']
colors = {'Baseline': '#4C72B0', 'UALB': '#DD8452', 'FTM-only': '#937860',
         'TT-only': '#8C8C8C', 'FTM+TT': '#55A868'}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (metric_name, metric_dict) in zip(axes, [('Cosine Gap', {n: {l: gap_results[n][l][2] for l in levels} for n in gap_results}),
                                                    ('Linear Probe Acc (%)', probe_results)]):
    x = np.arange(len(levels))
    n_bars = len(metric_dict)
    w = 0.8 / n_bars
    for i, (name, res) in enumerate(metric_dict.items()):
        vals = [res[l] for l in levels]
        offset = (i - (n_bars - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=name, color=colors.get(name, '#9E9E9E'),
               alpha=0.85, edgecolor='black', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=11)
    ax.set_title(metric_name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('ESC-50: Baseline vs UALB vs FTM-only vs TT-only vs FTM+TT', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {OUTPUT_DIR}/ablation_comparison.png")

np.save(OUTPUT_DIR / 'results.npy', {'gap': gap_results, 'probe': probe_results, 'nmi': nmi_results})
print(f"✓ Saved: {OUTPUT_DIR}/results.npy")
print("\n✓ Done!")