"""
WP2 Ablation - FTM only (Tree-Triplet 없이)

HSSN 논문 Table 5의 "L_FTM만" 행을 재현.
total_loss = L_FTM   (Tree-Triplet 항 자체를 계산하지 않음 — 연산 절약 + 깔끔한 분리)

데이터: 단일 라벨(species_number==1) subset (train_hssn_singlelabel.py와 동일 기준)
"""

import sys
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

#from NatureLM.anuraset.anuraset_dataset import AnuraSetEmbeddingDataset
from birdnet.anuraset.anuraset_dataset_unknown import AnuraSetEmbeddingDataset
from NatureLM.anuraset.hierarchy_tree import HierarchyTree
from models.hierarchical_model import HierarchicalModel
from focal_tree_min_loss import FocalTreeMinLoss

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH    = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings.npy'
META_PATH     = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
SAVE_DIR      = Path('/user/gaad2403/Bachelor-Thesis/results/birdnet/anuraset/wp2_ftm_only_singlelabel_unknown')  # ← 별도 폴더
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (train_hssn_singlelabel.py와 동일 — ablation은 다른 게 loss뿐이어야 공정) ──
BATCH_SIZE   = 256
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TRIPLET_DIM  = 128
VAL_SPLIT    = 0.1
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FTM_GAMMA    = 2.0

print(f"Device: {DEVICE}")
print("=" * 60)
print("Ablation: FTM only (no Tree-Triplet)")
print("=" * 60)

# ── Dataset ────────────────────────────────────────────────────────────────
full_train_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='train', single_label_only=True)

val_size   = int(len(full_train_ds) * VAL_SPLIT)
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(full_train_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Hierarchy (FTM용 V만 필요, ψ/tree_distance는 필요 없음) ────────────────
tree = HierarchyTree(TAXONOMY_PATH,
                      full_train_ds.family_enc, full_train_ds.genus_enc, full_train_ds.species_enc)
print(f"[HierarchyTree] n_nodes = {tree.n_nodes}")

# ── Model & Loss ───────────────────────────────────────────────────────────
# 모델 구조는 full(FTM+TT) 버전과 동일하게 둠 (triplet_head는 그냥 안 쓰임/학습 안 됨)
# -> ablation에서 "구조"가 아니라 "손실 구성"만 변수가 되도록 통제
model = HierarchicalModel(input_dim=1024, hidden_dim=1024,
                          triplet_dim=TRIPLET_DIM, n_hierarchy_nodes=tree.n_nodes).to(DEVICE)

ftm_loss_fn = FocalTreeMinLoss(
    ancestor_matrix=tree.ancestor_matrix(),
    descendant_matrix=tree.descendant_matrix(),
    gamma=FTM_GAMMA
).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ──────────────────────────────────────────────────────────
best_val_loss = float('inf')
history = {'train_ftm_loss': [], 'val_ftm_loss': []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    ftm_losses = []

    for batch_idx, (emb, hier_labels) in enumerate(train_loader):
        emb         = emb.to(DEVICE)
        hier_labels = hier_labels.to(DEVICE)

        _, s = model(emb)   # z(triplet 출력)는 이 ablation에서 안 씀

        gt = tree.build_ground_truth(
            hier_labels[:, 0], hier_labels[:, 1], hier_labels[:, 2],
            full_train_ds.family_enc, full_train_ds.genus_enc, full_train_ds.species_enc
        ).to(DEVICE)
        ftm_loss = ftm_loss_fn(s, gt)

        optimizer.zero_grad()
        ftm_loss.backward()
        optimizer.step()

        ftm_losses.append(ftm_loss.item())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | FTM={ftm_loss:.4f}")

    scheduler.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for emb, hier_labels in val_loader:
            emb, hier_labels = emb.to(DEVICE), hier_labels.to(DEVICE)
            _, s = model(emb)
            gt = tree.build_ground_truth(
                hier_labels[:, 0], hier_labels[:, 1], hier_labels[:, 2],
                full_train_ds.family_enc, full_train_ds.genus_enc, full_train_ds.species_enc
            ).to(DEVICE)
            val_losses.append(ftm_loss_fn(s, gt).item())

    train_ftm = np.mean(ftm_losses)
    val_ftm   = np.mean(val_losses)
    history['train_ftm_loss'].append(train_ftm)
    history['val_ftm_loss'].append(val_ftm)

    print(f"Epoch {epoch:3d}/{EPOCHS} | FTM={train_ftm:.4f} | ValFTM={val_ftm:.4f}")

    if val_ftm < best_val_loss:
        best_val_loss = val_ftm
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_ftm_loss': val_ftm}, SAVE_DIR / 'best_model.pt')
        print(f"  -> Saved best model (val_ftm_loss={val_ftm:.4f})")

np.save(SAVE_DIR / 'history.npy', history)
print(f"\nDone. Best val FTM loss: {best_val_loss:.4f}")
print(f"Results saved to: {SAVE_DIR}")