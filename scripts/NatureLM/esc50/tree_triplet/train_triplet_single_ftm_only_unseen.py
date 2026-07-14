"""
WP3 Ablation - FTM only, ESC-50, NatureLM, UNSEEN-CATEGORY hold-out variant.
"""

import sys
import json
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from NatureLM.esc50.esc50_dataset_unseen import ESC50EmbeddingDataset  # 🆕
from NatureLM.esc50.esc50_hierarchy_tree import ESC50HierarchyTree
from models.hierarchical_model import HierarchicalModel
from focal_tree_min_loss import FocalTreeMinLoss

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_esc50/embeddings.npy'
META_PATH  = '/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv'
SPLIT_PATH = '/user/gaad2403/Bachelor-Thesis/data/unseen_category_split.json'  # 🆕
SAVE_DIR   = Path('/user/gaad2403/Bachelor-Thesis/results/NatureLM/esc50/esc50_wp3_ftm_only_unseen')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = [1, 2, 3, 4]

# ── Hyperparameters ────────────────────────────────────────────────────────
BATCH_SIZE   = 256
EPOCHS       = 450
LR           = 1e-3
WEIGHT_DECAY = 1e-4
TRIPLET_DIM  = 128
VAL_SPLIT    = 0.1
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FTM_GAMMA    = 2.0

print(f"Device: {DEVICE}")
print("=" * 60)
print("Ablation: FTM only (ESC-50, NatureLM), UNSEEN-CATEGORY hold-out")
print("=" * 60)

with open(SPLIT_PATH) as f:
    split = json.load(f)
exclude_categories = split['all_held_out_categories']
print(f"Excluding {len(exclude_categories)} held-out categories from training: {exclude_categories}")

full_train_ds = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TRAIN_FOLDS,
                                      exclude_categories=exclude_categories)  # 🆕

val_size   = int(len(full_train_ds) * VAL_SPLIT)
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(full_train_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

tree = ESC50HierarchyTree(META_PATH, full_train_ds.bga_enc,
                          full_train_ds.midcat_enc, full_train_ds.category_enc)
print(f"[HierarchyTree] n_nodes = {tree.n_nodes}")

# ── Model & Loss (input_dim=4096 for NatureLM) ─────────────────────────────
model = HierarchicalModel(input_dim=4096, hidden_dim=1024,
                          triplet_dim=TRIPLET_DIM, n_hierarchy_nodes=tree.n_nodes).to(DEVICE)

ftm_loss_fn = FocalTreeMinLoss(
    ancestor_matrix=tree.ancestor_matrix(),
    descendant_matrix=tree.descendant_matrix(),
    gamma=FTM_GAMMA
).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_loss = float('inf')
history = {'train_ftm_loss': [], 'val_ftm_loss': []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    ftm_losses = []

    for batch_idx, (emb, hier_labels) in enumerate(train_loader):
        emb         = emb.to(DEVICE)
        hier_labels = hier_labels.to(DEVICE)

        _, s = model(emb)

        gt = tree.build_ground_truth(
            hier_labels[:, 0], hier_labels[:, 1], hier_labels[:, 2],
            full_train_ds.bga_enc, full_train_ds.midcat_enc, full_train_ds.category_enc
        ).to(DEVICE)
        ftm_loss = ftm_loss_fn(s, gt)

        optimizer.zero_grad()
        ftm_loss.backward()
        optimizer.step()

        ftm_losses.append(ftm_loss.item())

        if (batch_idx + 1) % 5 == 0:
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
                full_train_ds.bga_enc, full_train_ds.midcat_enc, full_train_ds.category_enc
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