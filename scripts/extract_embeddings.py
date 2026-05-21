import torch
import torchaudio
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from NatureLM.models import NatureLM

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR       = Path("/user/gaad2403/bachelor_thesis")
AUDIO_DIR      = BASE_DIR / "anuraset/anuraset/audio"
METADATA_PATH  = BASE_DIR / "anuraset/anuraset/metadata.csv"
TAXONOMY_PATH  = Path("/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv")
OUTPUT_DIR     = BASE_DIR / "embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE     = 32
TARGET_SR      = 16000
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# 42개 종 코드 목록
SPECIES_COLS = [
    "SPHSUR","BOABIS","SCIPER","DENNAH","LEPLAT","RHIICT","BOALEP","BOAFAB",
    "PHYCUV","DENMIN","ELABIC","BOAPRA","DENCRU","BOALUN","BOAALB","PHYMAR",
    "PITAZU","PHYSAU","LEPFUS","DENNAN","PHYALB","LEPLAB","SCIFUS","BOARAN",
    "SCIFUV","AMEPIC","LEPPOD","ADEDIP","ELAMAT","PHYNAT","LEPELE","RHISCI",
    "SCINAS","LEPNOT","ADEMAR","BOAALM","PHYDIS","RHIORN","LEPFLA","SCIRIZ",
    "DENELE","SCIALT"
]

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class AnuraSetDataset(Dataset):
    def __init__(self, metadata_df, audio_dir, taxonomy_df, target_sr=16000):
        self.df         = metadata_df.reset_index(drop=True)
        self.audio_dir  = Path(audio_dir)
        self.target_sr  = target_sr

        # taxonomy 매핑: code → family, genus, species
        self.tax = taxonomy_df.set_index("code")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_name = row["sample_name"]
        site        = row["site"]

        fname = row["fname"]
        min_t = int(row["min_t"])
        max_t = int(row["max_t"])
        audio_path = self.audio_dir / site / f"{fname}_{min_t}_{max_t}.wav"
        waveform, sr = torchaudio.load(audio_path)

        # mono 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 리샘플링
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform  = resampler(waveform)

        # species labels (multi-label)
        species_labels = row[SPECIES_COLS].values.astype(np.float32)

        # taxonomy 레이블 (첫 번째 active species 기준)
        active = [c for c in SPECIES_COLS if row[c] == 1]
        if active and active[0] in self.tax.index:
            t = self.tax.loc[active[0]]
            family = t["family"]
            genus  = t["genus"]
            species = t["species"]
        else:
            family = genus = species = "unknown"

        return {
            "sample_name"   : sample_name,
            "waveform"      : waveform.squeeze(0),   # [samples]
            "species_labels": species_labels,         # [42]
            "family"        : family,
            "genus"         : genus,
            "species"       : species,
            "subset"        : row["subset"],
        }


def collate_fn(batch):
    """가변 길이 waveform을 zero-pad해서 배치로 묶기"""
    max_len   = max(b["waveform"].shape[0] for b in batch)
    waveforms = torch.zeros(len(batch), max_len)
    for i, b in enumerate(batch):
        wlen = b["waveform"].shape[0]
        waveforms[i, :wlen] = b["waveform"]

    return {
        "sample_name"   : [b["sample_name"]    for b in batch],
        "waveform"      : waveforms,
        "species_labels": torch.tensor(np.stack([b["species_labels"] for b in batch])),
        "family"        : [b["family"]          for b in batch],
        "genus"         : [b["genus"]           for b in batch],
        "species"       : [b["species"]         for b in batch],
        "subset"        : [b["subset"]          for b in batch],
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

    # 데이터 로드
    metadata = pd.read_csv(METADATA_PATH)
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    print(f"Total samples: {len(metadata)}")

    dataset    = AnuraSetDataset(metadata, AUDIO_DIR, taxonomy, TARGET_SR)
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 4,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )

    # 결과 저장용 리스트
    all_embeddings    = []
    all_sample_names  = []
    all_species_labels= []
    all_families      = []
    all_genera        = []
    all_species       = []
    all_subsets       = []

    print(f"Extracting embeddings (batch_size={BATCH_SIZE})...")
    t_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        waveform = batch["waveform"].to(DEVICE)  # [B, samples]

        with torch.no_grad():
            audio_embeds, audio_atts = model.encode_audio(waveform)
            # audio_embeds: [B, T, 4096]
            # attention-weighted mean pooling
            mask      = audio_atts.bool().unsqueeze(-1)
            embedding = (audio_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            # embedding: [B, 4096]

        all_embeddings.append(embedding.float().cpu().numpy())
        all_sample_names.extend(batch["sample_name"])
        all_species_labels.append(batch["species_labels"].numpy())
        all_families.extend(batch["family"])
        all_genera.extend(batch["genus"])
        all_species.extend(batch["species"])
        all_subsets.extend(batch["subset"])

        # 진행상황 출력 (100배치마다)
        if (batch_idx + 1) % 100 == 0:
            elapsed   = time.time() - t_start
            processed = (batch_idx + 1) * BATCH_SIZE
            total     = len(dataset)
            eta       = elapsed / processed * (total - processed)
            print(f"[{processed}/{total}] elapsed: {elapsed/60:.1f}min  ETA: {eta/60:.1f}min")

    # numpy로 합치기
    embeddings     = np.concatenate(all_embeddings,     axis=0)  # [N, 4096]
    species_labels = np.concatenate(all_species_labels, axis=0)  # [N, 42]

    print(f"\nDone! Total: {len(embeddings)} embeddings, shape: {embeddings.shape}")
    print(f"Total time: {(time.time()-t_start)/60:.1f}min")

    # ── 저장 ──
    # 1. embeddings (numpy)
    np.save(OUTPUT_DIR / "embeddings.npy",      embeddings)
    np.save(OUTPUT_DIR / "species_labels.npy",  species_labels)

    # 2. 메타데이터 (CSV)
    meta_out = pd.DataFrame({
        "sample_name" : all_sample_names,
        "family"      : all_families,
        "genus"       : all_genera,
        "species"     : all_species,
        "subset"      : all_subsets,
    })
    # species_labels 컬럼도 추가
    for i, col in enumerate(SPECIES_COLS):
        meta_out[col] = species_labels[:, i]

    meta_out.to_csv(OUTPUT_DIR / "embeddings_metadata.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  embeddings.npy          {embeddings.nbytes/1e6:.1f} MB")
    print(f"  species_labels.npy")
    print(f"  embeddings_metadata.csv")


if __name__ == "__main__":
    main()