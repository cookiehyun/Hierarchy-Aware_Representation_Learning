"""
WP2 Training - Method 1: UALB, single-label subset, BirdNET, UNSEEN-SPECIES hold-out.

Same as train_ualb_unseen_holdout.py (NatureLM version) except input_dim=1024
and BirdNET embedding paths. Reuses the SAME unseen_species_split.json --
the hold-out choice is taxonomy-based, not embedding-based.
"""

import sys
import json
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path

from birdnet.anuraset.anuraset_dataset_unseen import AnuraSetEmbeddingDataset  # 🆕
from models.projection_head import ProjectionHead
from hmlc_loss_l0 import HMLCWithSpeciesLevel as HMLC

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings.npy'
META_PATH  = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings_metadata.csv'
SPLIT_PATH = '/user/gaad2403/Bachelor-Thesis/data/unseen_species_split.json'  # 🆕 (shared w/ NatureLM)
SAVE_DIR   = Path('/user/gaad2403/Bachelor-Thesis/results/birdnet/anuraset/wp2_ualb_singlelabel_l0_unseen')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (identical to the standard BirdNET run) ─────────────────
BATCH_SIZE   = 256
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PROJ_DIM     = 128
TEMPERATURE  = 0.07
LOSS_TYPE    = 'hmce'
VAL_SPLIT    = 0.1
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")

with open(SPLIT_PATH) as f:
    split = json.load(f)
exclude_pairs = split['all_held_out_pairs']
print(f"Excluding {len(exclude_pairs)} held-out species from training: "
      f"{[p['code'] for p in exclude_pairs]}")

full_train_ds = AnuraSetEmbeddingDataset(EMBED_PATH, META_PATH, subset='train',
                                         single_label_only=True,
                                         exclude_pairs=exclude_pairs)  # 🆕

val_size   = int(len(full_train_ds) * VAL_SPLIT)
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(
    full_train_ds, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ── Model & Loss (input_dim=1024 for BirdNET) ──────────────────────────────
model   = ProjectionHead(input_dim=1024, proj_dim=PROJ_DIM).to(DEVICE)
loss_fn = HMLC(temperature=TEMPERATURE, loss_type=LOSS_TYPE, layer_penalty=torch.exp)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for emb, labels in train_loader:
        emb, labels = emb.to(DEVICE), labels.to(DEVICE)
        z1 = model(emb)
        z2 = model(emb + 0.01 * torch.randn_like(emb))
        z = torch.stack([z1, z2], dim=1)
        loss = loss_fn(z, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    scheduler.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for emb, labels in val_loader:
            emb, labels = emb.to(DEVICE), labels.to(DEVICE)
            z1 = model(emb)
            z2 = model(emb + 0.01 * torch.randn_like(emb))
            z = torch.stack([z1, z2], dim=1)
            val_losses.append(loss_fn(z, labels.float()).item())

    train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    print(f"Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss}, SAVE_DIR / 'best_model.pt')
        print(f"  -> Saved best model (val_loss={val_loss:.4f})")

np.save(SAVE_DIR / 'history.npy', history)
print(f"\nDone. Best val loss: {best_val_loss:.4f}")
print(f"Results: {SAVE_DIR}")