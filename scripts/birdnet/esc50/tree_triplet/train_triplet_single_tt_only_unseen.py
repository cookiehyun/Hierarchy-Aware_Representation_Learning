"""
WP3 Ablation - Tree-Triplet only, ESC-50, BirdNET, UNSEEN-CATEGORY hold-out variant.
"""

import sys
import json
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from birdnet.esc50.esc50_dataset_unseen import ESC50EmbeddingDataset  # 🆕
from birdnet.esc50.hierarchy_utils import ESC50Hierarchy
from birdnet.esc50.esc50_hierarchy_tree import ESC50HierarchyTree
from models.hierarchical_model import HierarchicalModel
from tree_triplet_loss_self import TreeTripletLoss

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_esc50_birdnet/embeddings.npy'
META_PATH  = '/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv'
SPLIT_PATH = '/user/gaad2403/Bachelor-Thesis/data/unseen_category_split.json'  # 🆕
SAVE_DIR   = Path('/user/gaad2403/Bachelor-Thesis/results/birdnet/esc50/esc50_wp3_tt_only_self_unseen')
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

print(f"Device: {DEVICE}")
print("=" * 60)
print("Ablation: Tree-Triplet only (ESC-50, BirdNET), UNSEEN-CATEGORY hold-out")
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

ps_hierarchy = ESC50Hierarchy(META_PATH)
tree = ESC50HierarchyTree(META_PATH, full_train_ds.bga_enc,
                          full_train_ds.midcat_enc, full_train_ds.category_enc)

# ── Model & Loss ───────────────────────────────────────────────────────────
model = HierarchicalModel(input_dim=1024, hidden_dim=1024,
                          triplet_dim=TRIPLET_DIM, n_hierarchy_nodes=tree.n_nodes).to(DEVICE)

triplet_loss_fn = TreeTripletLoss(
    temperature=TEMPERATURE,
    m_epsilon=TRIPLET_MARGIN_EPSILON,
    max_hierarchy_depth=MAX_HIERARCHY_DEPTH,
    use_hierarchy_margin=USE_HIERARCHY_MARGIN
).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_loss = float('inf')
history = {'train_triplet_loss': [], 'val_triplet_loss': []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    triplet_losses = []

    for batch_idx, (emb, hier_labels) in enumerate(train_loader):
        emb         = emb.to(DEVICE)
        hier_labels = hier_labels.to(DEVICE)

        z, _ = model(emb)

        tree_distances = ps_hierarchy.batch_tree_distance(
            hier_labels[:, 2], hier_labels[:, 0], hier_labels[:, 1],
            full_train_ds.category_enc, full_train_ds.bga_enc, full_train_ds.midcat_enc
        ).to(DEVICE)
        triplet_loss = triplet_loss_fn(
            z, hier_labels[:, 2], hier_labels[:, 0], hier_labels[:, 1],
            tree_distances=tree_distances, max_triplets_per_class=MAX_TRIPLETS_PER_CLASS
        )

        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()

        triplet_losses.append(triplet_loss.item())

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | Triplet={triplet_loss:.4f}")

    scheduler.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for emb, hier_labels in val_loader:
            emb, hier_labels = emb.to(DEVICE), hier_labels.to(DEVICE)
            z, _ = model(emb)
            tree_distances = ps_hierarchy.batch_tree_distance(
                hier_labels[:, 2], hier_labels[:, 0], hier_labels[:, 1],
                full_train_ds.category_enc, full_train_ds.bga_enc, full_train_ds.midcat_enc
            ).to(DEVICE)
            val_losses.append(triplet_loss_fn(
                z, hier_labels[:, 2], hier_labels[:, 0], hier_labels[:, 1],
                tree_distances=tree_distances, max_triplets_per_class=MAX_TRIPLETS_PER_CLASS
            ).item())

    train_tt = np.mean(triplet_losses)
    val_tt   = np.mean(val_losses)
    history['train_triplet_loss'].append(train_tt)
    history['val_triplet_loss'].append(val_tt)

    print(f"Epoch {epoch:3d}/{EPOCHS} | Triplet={train_tt:.4f} | ValTriplet={val_tt:.4f}")

    if val_tt < best_val_loss:
        best_val_loss = val_tt
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_triplet_loss': val_tt}, SAVE_DIR / 'best_model.pt')
        print(f"  -> Saved best model (val_triplet_loss={val_tt:.4f})")

np.save(SAVE_DIR / 'history.npy', history)
print(f"\nDone. Best val Triplet loss: {best_val_loss:.4f}")
print(f"Results saved to: {SAVE_DIR}")