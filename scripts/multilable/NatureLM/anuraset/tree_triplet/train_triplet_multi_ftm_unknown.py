"""
WP4 Ablation - FTM only, multi-label AnuraSet (BirdNET embeddings)

Reproduces the "L_FTM only" row of HSSN Table 5, multi-label version,
using BirdNET (1024-dim) embeddings instead of NatureLM (4096-dim).
total_loss = L_FTM   (Tree-Triplet term not computed at all)
"""

import sys
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from multilabel.NatureLM.anuraset.anuraset_dataset_un import AnuraSetMultiLabelDataset  # 🆕 unseen/unknown-capable
from multilabel.multilabel_hierarchy_tree import MultiLabelHierarchyTree
from models.hierarchical_model import HierarchicalModel
from focal_tree_min_loss import FocalTreeMinLoss

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
SAVE_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/multilable/NatureLM/anuraset/wp4_ftm_only_multilabel_unknown')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (same as the other ablations -- only the loss changes) ─
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
TRIPLET_DIM = 128
VAL_SPLIT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FTM_GAMMA = 2.0

print(f"Device: {DEVICE}")
print("=" * 60)
print("Ablation: FTM only (no Tree-Triplet), multi-label, BirdNET")
print("=" * 60)

# ── Dataset ────────────────────────────────────────────────────────────────
full_train_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='train', include_negative=True)

val_size = int(len(full_train_ds) * VAL_SPLIT)
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(full_train_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Hierarchy ──────────────────────────────────────────────────────────────
tree = MultiLabelHierarchyTree(TAXONOMY_PATH, full_train_ds.species_cols,
                               full_train_ds.genus_cols, full_train_ds.family_cols)
print(f"[HierarchyTree] n_nodes = {tree.n_nodes}")

# ── Model & Loss (input_dim=4096 for BirdNET) ──────────────────────────────
# same model structure as the FTM+TT run, triplet_head just isn't trained here
model = HierarchicalModel(input_dim=4096, hidden_dim=1024,
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

    for batch_idx, (emb, family, genus, species) in enumerate(train_loader):
        emb = emb.to(DEVICE)
        family, genus, species = family.to(DEVICE), genus.to(DEVICE), species.to(DEVICE)

        _, s = model(emb)  # triplet output z isn't used in this ablation

        gt = tree.ground_truth(family, genus, species).to(DEVICE)
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
        for emb, family, genus, species in val_loader:
            emb = emb.to(DEVICE)
            family, genus, species = family.to(DEVICE), genus.to(DEVICE), species.to(DEVICE)
            _, s = model(emb)
            gt = tree.ground_truth(family, genus, species).to(DEVICE)
            val_losses.append(ftm_loss_fn(s, gt).item())

    train_ftm = np.mean(ftm_losses)
    val_ftm = np.mean(val_losses)
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