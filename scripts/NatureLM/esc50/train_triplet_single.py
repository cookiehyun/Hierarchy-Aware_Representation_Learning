"""
WP3 Training - Method 2: HSSN (ESC-50, faithful reproduction)

기존 train_triplet_loss.py(ESC-50)에서 HMLC를 완전히 제거하고,
AnuraSet의 train_hssn_singlelabel.py와 동일하게 L_FTM + beta(t)*L_TT로 학습.
"""

import sys
import math
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from esc50.esc50_dataset import ESC50EmbeddingDataset
from esc50.hierarchy_utils import ESC50Hierarchy
from esc50.esc50_hierarchy_tree import ESC50HierarchyTree
from models.hierarchical_model import HierarchicalModel
from tree_triplet_loss import TreeTripletLoss
from focal_tree_min_loss import FocalTreeMinLoss

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_esc50/embeddings.npy'
META_PATH  = '/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv'
SAVE_DIR   = Path('/user/gaad2403/Bachelor-Thesis/results/esc50_wp3_hssn')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = [1, 2, 3, 4]

# ── Hyperparameters ────────────────────────────────────────────────────────
BATCH_SIZE   = 256
EPOCHS       = 450
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TRIPLET_DIM  = 128
TEMPERATURE  = 0.07
VAL_SPLIT    = 0.1
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRIPLET_MARGIN_EPSILON = 0.1
MAX_HIERARCHY_DEPTH    = 3
USE_HIERARCHY_MARGIN    = True
MAX_TRIPLETS_PER_CLASS  = 200
FTM_GAMMA               = 2.0

def beta_schedule(epoch, total_epochs):
    return 0.25 * (1 - math.cos(math.pi * epoch / total_epochs))

print(f"Device: {DEVICE}")

# ── Dataset ────────────────────────────────────────────────────────────────
full_train_ds = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TRAIN_FOLDS)

val_size   = int(len(full_train_ds) * VAL_SPLIT)
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(full_train_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Hierarchy ──────────────────────────────────────────────────────────────
ps_hierarchy = ESC50Hierarchy(META_PATH)
tree = ESC50HierarchyTree(META_PATH, full_train_ds.bga_enc,
                          full_train_ds.midcat_enc, full_train_ds.category_enc)
print(f"[HierarchyTree] n_nodes = {tree.n_nodes}")

# ── Model & Losses ─────────────────────────────────────────────────────────
model = HierarchicalModel(input_dim=4096, hidden_dim=1024,
                          triplet_dim=TRIPLET_DIM, n_hierarchy_nodes=tree.n_nodes).to(DEVICE)

triplet_loss_fn = TreeTripletLoss(
    temperature=TEMPERATURE, m_epsilon=TRIPLET_MARGIN_EPSILON,
    max_hierarchy_depth=MAX_HIERARCHY_DEPTH, use_hierarchy_margin=USE_HIERARCHY_MARGIN
).to(DEVICE)

ftm_loss_fn = FocalTreeMinLoss(
    ancestor_matrix=tree.ancestor_matrix(), descendant_matrix=tree.descendant_matrix(),
    gamma=FTM_GAMMA
).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ──────────────────────────────────────────────────────────
best_val_loss = float('inf')
history = {'train_ftm_loss': [], 'train_triplet_loss': [], 'train_total_loss': [],
           'val_ftm_loss': [], 'beta': []}

for epoch in range(1, EPOCHS + 1):
    beta = beta_schedule(epoch, EPOCHS)

    model.train()
    ftm_losses, triplet_losses, total_losses = [], [], []

    for batch_idx, (emb, hier_labels) in enumerate(train_loader):
        emb         = emb.to(DEVICE)
        hier_labels = hier_labels.to(DEVICE)   # [bsz,3] -> [bga, midcat, category]

        z, s = model(emb)

        gt = tree.build_ground_truth(
            hier_labels[:, 0], hier_labels[:, 1], hier_labels[:, 2],
            full_train_ds.bga_enc, full_train_ds.midcat_enc, full_train_ds.category_enc
        ).to(DEVICE)
        ftm_loss = ftm_loss_fn(s, gt)

        tree_distances = ps_hierarchy.batch_tree_distance(
            hier_labels[:, 2], hier_labels[:, 0], hier_labels[:, 1],
            full_train_ds.category_enc, full_train_ds.bga_enc, full_train_ds.midcat_enc
        ).to(DEVICE)
        triplet_loss = triplet_loss_fn(
            z, hier_labels[:, 2], hier_labels[:, 0], hier_labels[:, 1],
            tree_distances=tree_distances, max_triplets_per_class=MAX_TRIPLETS_PER_CLASS
        )

        total_loss = ftm_loss + beta * triplet_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ftm_losses.append(ftm_loss.item())
        triplet_losses.append(triplet_loss.item())
        total_losses.append(total_loss.item())

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                  f"FTM={ftm_loss:.4f} | Triplet={triplet_loss:.4f} | beta={beta:.3f} | Total={total_loss:.4f}")

    scheduler.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for emb, hier_labels in val_loader:
            emb, hier_labels = emb.to(DEVICE), hier_labels.to(DEVICE)
            _, s = model(emb)
            gt = tree.build_ground_truth(
                hier_labels[:, 0], hier_labels[:, 1], hier_labels[:, 2],
                full_train_ds.bga_enc, full_train_ds.midcat_enc, full_train_ds.category_enc
            ).to(DEVICE)
            val_losses.append(ftm_loss_fn(s, gt).item())

    train_ftm = np.mean(ftm_losses)
    train_tt  = np.mean(triplet_losses)
    train_tot = np.mean(total_losses)
    val_ftm   = np.mean(val_losses)

    history['train_ftm_loss'].append(train_ftm)
    history['train_triplet_loss'].append(train_tt)
    history['train_total_loss'].append(train_tot)
    history['val_ftm_loss'].append(val_ftm)
    history['beta'].append(beta)

    print(f"Epoch {epoch:3d}/{EPOCHS} | FTM={train_ftm:.4f} | Triplet={train_tt:.4f} | "
          f"beta={beta:.3f} | Total={train_tot:.4f} | ValFTM={val_ftm:.4f}")

    if val_ftm < best_val_loss:
        best_val_loss = val_ftm
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_ftm_loss': val_ftm}, SAVE_DIR / 'best_model.pt')
        print(f"  -> Saved best model (val_ftm_loss={val_ftm:.4f})")

np.save(SAVE_DIR / 'history.npy', history)
print(f"\nDone. Best val FTM loss: {best_val_loss:.4f}")
print(f"Results saved to: {SAVE_DIR}")