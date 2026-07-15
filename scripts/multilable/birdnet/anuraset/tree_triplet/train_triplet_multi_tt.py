"""
WP4 Ablation - Tree-Triplet only, multi-label AnuraSet (BirdNET embeddings)

Reproduces the "L_TT only" row of HSSN Table 5, multi-label version,
using BirdNET (1024-dim) embeddings instead of NatureLM (4096-dim).
total_loss = L_TT   (FTM term not computed at all)
Uses the "_self" (bug-fixed) MultiLabelTreeTripletLoss, same as the
NatureLM multi-label version.
"""

import sys
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from multilabel.NatureLM.anuraset.anuraset_dataset import AnuraSetMultiLabelDataset
from multilabel.multilabel_hierarchy_tree import MultiLabelHierarchyTree
from multilabel.species_distance_matrix import build_species_distance_matrix
from multilabel.multilabel_tree_triplet_loss_self import MultiLabelTreeTripletLoss
from models.hierarchical_model import HierarchicalModel

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings.npy'
META_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
SAVE_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/multilable/birdnet/anuraset/wp4_tt_only_multilabel_self')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (same as the other ablations -- only the loss changes) ─
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
TRIPLET_DIM = 128
VAL_SPLIT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRIPLET_MARGIN_EPSILON = 0.1
MAX_HIERARCHY_DEPTH = 3
MAX_TRIPLETS_PER_SPECIES = 200

print(f"Device: {DEVICE}")
print("=" * 60)
print("Ablation: Tree-Triplet only (no FTM), multi-label, BirdNET")
print("=" * 60)

# ── Dataset ────────────────────────────────────────────────────────────────
full_train_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='train')

val_size = int(len(full_train_ds) * VAL_SPLIT)
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(full_train_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Hierarchy (n_hierarchy_nodes needed only to keep the model structure
#    identical across ablations; hierarchy_head isn't trained here) ────────
tree = MultiLabelHierarchyTree(TAXONOMY_PATH, full_train_ds.species_cols,
                               full_train_ds.genus_cols, full_train_ds.family_cols)
species_dist_matrix = build_species_distance_matrix(TAXONOMY_PATH, full_train_ds.species_cols)

# ── Model & Loss (input_dim=1024 for BirdNET) ──────────────────────────────
model = HierarchicalModel(input_dim=1024, hidden_dim=1024,
                          triplet_dim=TRIPLET_DIM, n_hierarchy_nodes=tree.n_nodes).to(DEVICE)

triplet_loss_fn = MultiLabelTreeTripletLoss(
    species_distance_matrix=species_dist_matrix,
    m_epsilon=TRIPLET_MARGIN_EPSILON,
    max_hierarchy_depth=MAX_HIERARCHY_DEPTH,
    max_triplets_per_species=MAX_TRIPLETS_PER_SPECIES
).to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ──────────────────────────────────────────────────────────
best_val_loss = float('inf')
history = {'train_triplet_loss': [], 'val_triplet_loss': []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    triplet_losses = []

    for batch_idx, (emb, family, genus, species) in enumerate(train_loader):
        emb = emb.to(DEVICE)
        species = species.to(DEVICE)

        z, _ = model(emb)  # hierarchy score s isn't used in this ablation

        triplet_loss = triplet_loss_fn(z, species)

        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()

        triplet_losses.append(triplet_loss.item())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | Triplet={triplet_loss:.4f}")

    scheduler.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for emb, family, genus, species in val_loader:
            emb = emb.to(DEVICE)
            species = species.to(DEVICE)
            z, _ = model(emb)
            val_losses.append(triplet_loss_fn(z, species).item())

    train_tt = np.mean(triplet_losses)
    val_tt = np.mean(val_losses)
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