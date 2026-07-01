import torch
import torchaudio
import pandas as pd
import numpy as np
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from NatureLM.models import NatureLM

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR    = Path("/user/gaad2403/bachelor_thesis")
AUDIO_DIR   = BASE_DIR / "ESC-50-master/audio"
LABELS_PATH = Path("/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv")
OUTPUT_DIR  = BASE_DIR / "embeddings_esc50"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE  = 32
TARGET_SR   = 16000
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class ESC50Dataset(Dataset):
    def __init__(self, labels_df, audio_dir, target_sr=16000):
        self.df        = labels_df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.target_sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename   = row["filename"]
        audio_path = self.audio_dir / filename
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform  = resampler(waveform)

        return {
            "filename" : filename,
            "waveform" : waveform.squeeze(0),
            "category" : row["category"],     # fine-grained (50 classes)
            "bga_label": row["bga_label"],     # biophony/geophony/anthrophony
            "fold"     : int(row["fold"]),
        }


def collate_fn(batch):
    max_len   = max(b["waveform"].shape[0] for b in batch)
    waveforms = torch.zeros(len(batch), max_len)
    for i, b in enumerate(batch):
        wlen = b["waveform"].shape[0]
        waveforms[i, :wlen] = b["waveform"]

    return {
        "filename" : [b["filename"]  for b in batch],
        "waveform" : waveforms,
        "category" : [b["category"]  for b in batch],
        "bga_label": [b["bga_label"] for b in batch],
        "fold"     : [b["fold"]      for b in batch],
    }


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("Loading model...")
    t0 = time.time()
    model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
    model = model.to(DEVICE).eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    labels_df = pd.read_csv(LABELS_PATH)
    print(f"Total samples: {len(labels_df)}")
    print(labels_df["bga_label"].value_counts())

    dataset    = ESC50Dataset(labels_df, AUDIO_DIR, TARGET_SR)
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )

    all_embeddings = []
    all_filenames  = []
    all_categories = []
    all_bga_labels = []
    all_folds      = []

    print(f"Extracting embeddings (batch_size={BATCH_SIZE})...")
    t_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        waveform = batch["waveform"].to(DEVICE)

        with torch.no_grad():
            audio_embeds, audio_atts = model.encode_audio(waveform)
            mask      = audio_atts.bool().unsqueeze(-1)
            embedding = (audio_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        all_embeddings.append(embedding.float().cpu().numpy())
        all_filenames.extend(batch["filename"])
        all_categories.extend(batch["category"])
        all_bga_labels.extend(batch["bga_label"])
        all_folds.extend(batch["fold"])

        if (batch_idx + 1) % 10 == 0:
            elapsed   = time.time() - t_start
            processed = (batch_idx + 1) * BATCH_SIZE
            total     = len(dataset)
            eta       = elapsed / processed * (total - processed)
            print(f"[{processed}/{total}] elapsed: {elapsed/60:.1f}min  ETA: {eta/60:.1f}min")

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"\nDone! Total: {len(embeddings)} embeddings, shape: {embeddings.shape}")
    print(f"Total time: {(time.time()-t_start)/60:.1f}min")

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    meta_out = pd.DataFrame({
        "filename" : all_filenames,
        "category" : all_categories,
        "bga_label": all_bga_labels,
        "fold"     : all_folds,
    })
    meta_out.to_csv(OUTPUT_DIR / "embeddings_metadata.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  embeddings.npy            {embeddings.nbytes/1e6:.1f} MB")
    print(f"  embeddings_metadata.csv")


if __name__ == "__main__":
    main()