"""
WP4 Training - Method 2: HSSN (multi-label AnuraSet, BirdNET embeddings)

Same objective as the NatureLM multi-label version:
    L_total = L_FTM (Eq.6) + beta(t) * L_TT (Eq.7)
    beta: 0 -> 0.5 cosine annealing

Only the embedding source changes (BirdNET, 1024-dim, input_dim=1024).
Uses the "_self" (bug-fixed) MultiLabelTreeTripletLoss.
"""

import sys
import math
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
from focal_tree_min_loss import FocalTreeMinLoss

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings.npy'
META_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
SAVE_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/multilabel/birdnet/anuraset/wp4_hssn_multilabel_self')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (same as NatureLM multi-label version) ─────────────────
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
FTM_GAMMA = 2.0


def beta_schedule(epoch, total_epochs):
    return 0.25 * (1 - math.cos(math.pi * epoch / total_epochs))


print(f"Device: {DEVICE}")

# ── Dataset ────────────────────────────────────────────────────────────────
full_train_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='train')

val_size = int(len(full_train_ds) * VAL_SPLIT)
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(
    full_train_ds, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Hierarchy setup ────────────────────────────────────────────────────────
tree = MultiLabelHierarchyTree(TAXONOMY_PATH, full_train_ds.species_cols,
                               full_train_ds.genus_cols, full_train_ds.family_cols)
print(f"[HierarchyTree] n_nodes = {tree.n_nodes} "
      f"(family={tree.n_family}, genus={tree.n_genus}, species={tree.n_species})")

species_dist_matrix = build_species_distance_matrix(TAXONOMY_PATH, full_train_ds.species_cols)

# ── Model & Losses (input_dim=1024 for BirdNET) ────────────────────────────
model = HierarchicalModel(input_dim=1024, hidden_dim=1024,
                          triplet_dim=TRIPLET_DIM, n_hierarchy_nodes=tree.n_nodes).to(DEVICE)

ftm_loss_fn = FocalTreeMinLoss(
    ancestor_matrix=tree.ancestor_matrix(),
    descendant_matrix=tree.descendant_matrix(),
    gamma=FTM_GAMMA
).to(DEVICE)

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
history = {'train_ftm_loss': [], 'train_triplet_loss': [], 'train_total_loss': [],
           'val_ftm_loss': [], 'beta': []}

for epoch in range(1, EPOCHS + 1):
    beta = beta_schedule(epoch, EPOCHS)

    model.train()
    ftm_losses, triplet_losses, total_losses = [], [], []

    for batch_idx, (emb, family, genus, species) in enumerate(train_loader):
        emb = emb.to(DEVICE)
        family, genus, species = family.to(DEVICE), genus.to(DEVICE), species.to(DEVICE)

        z, s = model(emb)

        gt = tree.ground_truth(family, genus, species).to(DEVICE)
        ftm_loss = ftm_loss_fn(s, gt)

        triplet_loss = triplet_loss_fn(z, species)

        total_loss = ftm_loss + beta * triplet_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ftm_losses.append(ftm_loss.item())
        triplet_losses.append(triplet_loss.item())
        total_losses.append(total_loss.item())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                  f"FTM={ftm_loss:.4f} | Triplet={triplet_loss:.4f} | beta={beta:.3f} | "
                  f"Total={total_loss:.4f}")

    scheduler.step()

    # validation on FTM only, same as single-label version
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
    train_tt = np.mean(triplet_losses)
    train_tot = np.mean(total_losses)
    val_ftm = np.mean(val_losses)

    history['train_ftm_loss'].append(train_ftm)
    history['train_triplet_loss'].append(train_tt)
    history['train_total_loss'].append(train_tot)
    history['val_ftm_loss'].append(val_ftm)
    history['beta'].append(beta)

    print(f"Epoch {epoch:3d}/{EPOCHS} | FTM={train_ftm:.4f} | Triplet={train_tt:.4f} | "
          f"beta={beta:.3f} | Total={train_tot:.4f} | ValFTM={val_ftm:.4f}")

    if val_ftm < best_val_loss:
        best_val_loss = val_ftm
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_ftm_loss': val_ftm,
        }, SAVE_DIR / 'best_model.pt')
        print(f"  -> Saved best model (val_ftm_loss={val_ftm:.4f})")

np.save(SAVE_DIR / 'history.npy', history)
print(f"\nDone. Best val FTM loss: {best_val_loss:.4f}")
print(f"Results saved to: {SAVE_DIR}")