"""
WP4 Training - Method 1: Use All The Labels (HMLC), multi-label AnuraSet
(BirdNET embeddings), negative ("unknown"/no-call) samples included.

Same as wp4_ualb_multilabel.py (BirdNET, normal) except
include_negative=True, mirroring the NatureLM _unknown variant.

Note (per prior analysis): MultiLabelHMLC's positive mask is built by
comparing whether two samples' label vectors are exactly equal
(torch.all(labels_i == labels_j)). Negative samples (all-zero vectors)
are therefore all mutually "equal" and get treated as one implicit shared
class -- this matches the original UALB/HiMulCon algorithm (Algorithm 1
in the paper) applied faithfully, but the original DeepFashion dataset
never had an all-zero-label case to test this against, so this behavior
on AnuraSet negatives has not been validated by the original paper. Worth
flagging to Rida alongside these results.
"""

import sys
sys.path.append("/fs/dss/home/gaad2403/Bachelor-Thesis/src")

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path

from multilabel.NatureLM.anuraset.anuraset_dataset_un import AnuraSetMultiLabelDataset  # 🆕 include_negative-capable
from models.projection_head import ProjectionHead
from multilabel.multilabel_hmlc_loss import MultiLabelHMLC

# ── Paths (BirdNET) ─────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings.npy'
META_PATH = '/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
SAVE_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/multilabel/birdnet/anuraset/wp4_ualb_multilabel_unknown')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters (same as NatureLM multi-label version) ─────────────────
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PROJ_DIM = 128
TEMPERATURE = 0.07
VAL_SPLIT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
print("=" * 60)
print("UALB (HMLC), multi-label, BirdNET, negative included")
print("=" * 60)

# ── Dataset ────────────────────────────────────────────────────────────────
full_train_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='train',
                                          include_negative=True)

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

# ── Model & Loss (input_dim=1024 for BirdNET) ──────────────────────────────
model = ProjectionHead(input_dim=1024, proj_dim=PROJ_DIM).to(DEVICE)
loss_fn = MultiLabelHMLC(temperature=TEMPERATURE, layer_penalty=torch.exp)

optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                            momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def run_epoch(loader, train):
    model.train() if train else model.eval()
    losses = []
    for emb, family, genus, species in loader:
        emb = emb.to(DEVICE)
        family, genus, species = family.to(DEVICE), genus.to(DEVICE), species.to(DEVICE)

        with torch.set_grad_enabled(train):
            z1 = model(emb)
            z2 = model(emb + 0.01 * torch.randn_like(emb))
            z = torch.stack([z1, z2], dim=1)

            # fine to coarse, so species gets the strongest penalty
            loss = loss_fn(z, [species, genus, family])

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
    return np.mean(losses)


# ── Training loop ──────────────────────────────────────────────────────────
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True)
    scheduler.step()
    val_loss = run_epoch(val_loader, train=False)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    print(f"Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, SAVE_DIR / 'best_model.pt')
        print(f"  -> Saved best model (val_loss={val_loss:.4f})")

np.save(SAVE_DIR / 'history.npy', history)
print(f"\nDone. Best val loss: {best_val_loss:.4f}")
print(f"Results: {SAVE_DIR}")