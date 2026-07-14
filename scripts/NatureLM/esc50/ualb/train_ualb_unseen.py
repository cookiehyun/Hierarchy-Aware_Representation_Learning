"""
WP3 Training - Method 1: UALB, ESC-50, NatureLM, UNSEEN-CATEGORY hold-out variant.
Same as the BirdNET version except input_dim=4096 and NatureLM embedding paths.
Reuses the SAME unseen_category_split.json (category-name-based, not
embedding-based).
"""

import sys
import json
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path

from NatureLM.esc50.esc50_dataset_unseen import ESC50EmbeddingDataset  # 🆕
from models.projection_head import ProjectionHead
from hmlc_loss_l0 import HMLCWithSpeciesLevel as HMLC

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_esc50/embeddings.npy'
META_PATH  = '/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv'
SPLIT_PATH = '/user/gaad2403/Bachelor-Thesis/data/unseen_category_split.json'  # 🆕 (shared w/ BirdNET)
SAVE_DIR   = Path('/user/gaad2403/Bachelor-Thesis/results/NatureLM/esc50/esc50_wp3_ualb_l0_unseen')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FOLDS = [1, 2, 3, 4]

# ── Hyperparameters (identical to the standard NatureLM ESC-50 run) ─────────
BATCH_SIZE   = 256
EPOCHS       = 450
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
exclude_categories = split['all_held_out_categories']
print(f"Excluding {len(exclude_categories)} held-out categories from training: {exclude_categories}")

full_train_ds = ESC50EmbeddingDataset(EMBED_PATH, META_PATH, folds=TRAIN_FOLDS,
                                      exclude_categories=exclude_categories)  # 🆕

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

# ── Model & Loss (input_dim=4096 for NatureLM) ─────────────────────────────
model   = ProjectionHead(input_dim=4096, proj_dim=PROJ_DIM).to(DEVICE)
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