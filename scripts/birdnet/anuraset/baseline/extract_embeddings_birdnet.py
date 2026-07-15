import birdnet
import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# [GPU 설정] TensorFlow GPU 인식 및 메모리 동적 할당
# ─────────────────────────────────────────────
# 0번 GPU만 보이도록 명시 (SLURM이 할당한 GPU 자동 바인딩을 방해하지 않음)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리를 처음부터 다 먹지 않고 필요할 때만 늘려 쓰도록 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ Found {len(gpus)} GPU(s). TensorFlow will use GPU acceleration.")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("⚠️ No GPU found. Running on CPU.")

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR       = Path("/user/gaad2403/bachelor_thesis")
AUDIO_DIR      = BASE_DIR / "anuraset/anuraset/audio"
METADATA_PATH  = BASE_DIR / "anuraset/anuraset/metadata.csv"
TAXONOMY_PATH  = Path("/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv")
OUTPUT_DIR     = BASE_DIR / "embeddings_anuraset_birdnet"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128  # GPU를 쓰므로 배치 사이즈를 32에서 64나 128로 키우셔도 좋습니다.
MODEL_DIM  = 1024

SPECIES_COLS = [
    "SPHSUR","BOABIS","SCIPER","DENNAH","LEPLAT","RHIICT","BOALEP","BOAFAB",
    "PHYCUV","DENMIN","ELABIC","BOAPRA","DENCRU","BOALUN","BOAALB","PHYMAR",
    "PITAZU","PHYSAU","LEPFUS","DENNAN","PHYALB","LEPLAB","SCIFUS","BOARAN",
    "SCIFUV","AMEPIC","LEPPOD","ADEDIP","ELAMAT","PHYNAT","LEPELE","RHISCI",
    "SCINAS","LEPNOT","ADEMAR","BOAALM","PHYDIS","RHIORN","LEPFLA","SCIRIZ",
    "DENELE","SCIALT"
]

def main():
    print("=" * 60)
    print("BirdNET AnuraSet Embedding Extraction (GPU Accelerated)")
    print("=" * 60)
    
    print("\n[1/4] Loading BirdNET model...")
    t0 = time.time()
    # "tf" 백엔드로 로드하면 위에서 설정한 TensorFlow GPU 환경을 그대로 이어받습니다.
    model = birdnet.load("acoustic", "2.4", "tf")
    print(f"✓ Model loaded in {time.time()-t0:.1f}s")
    print(f"  • Embedding dimension: {model.get_embeddings_dim()}")
    print(f"  • Sample rate: {model.get_sample_rate()}")
    print(f"  • Segment size: {model.get_segment_size_s()}s")

    print("\n[2/4] Loading metadata...")
    metadata = pd.read_csv(METADATA_PATH)
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    print(f"✓ Loaded {len(metadata)} samples")
    
    tax_dict = taxonomy.set_index("code")[["family", "genus", "species"]].to_dict("index")

    def get_audio_path(row):
        return str(AUDIO_DIR / row["site"] / f"{row['fname']}_{int(row['min_t'])}={int(row['max_t'])}.wav")
    
    # ⚠️ 원본 코드에 오타가 날 수 있는 파일명 템플릿('_' vs '=') 주의
    # 만약 원본 파일명이 fname_min_t_max_t.wav 라면 아래처럼 사용하세요.
    audio_paths = [str(AUDIO_DIR / row["site"] / f"{row['fname']}_{int(row['min_t'])}_{int(row['max_t'])}.wav") for _, row in metadata.iterrows()]

    all_embeddings     = []
    all_sample_names   = []
    all_species_labels = []
    all_families       = []
    all_genera         = []
    all_species        = []
    all_subsets        = []
    failed_files       = []

    print(f"\n[3/4] Extracting embeddings (batch_size={BATCH_SIZE})...")
    t_start = time.time()

    # 미리 다중 라벨 배열을 메모리에 올려 속도 개선
    species_labels_np = metadata[SPECIES_COLS].values.astype(np.float32)

    for batch_idx in tqdm(range(0, len(audio_paths), BATCH_SIZE), desc="Processing batches"):
        batch_end = min(batch_idx + BATCH_SIZE, len(audio_paths))
        batch_paths = audio_paths[batch_idx : batch_end]
        batch_rows  = metadata.iloc[batch_idx : batch_end]

        try:
            # BirdNET GPU 연산 수행 부
            result = model.encode(batch_paths)
            embeddings = result.embeddings.mean(axis=1) # [num_files, 1024]
            
            if embeddings is None or embeddings.shape[0] == 0:
                raise ValueError(f"Empty embeddings returned for batch {batch_idx}")
            
            all_embeddings.append(embeddings)
            
            # 리스트 append 오버헤드 최소화
            all_sample_names.extend(batch_rows["sample_name"].tolist())
            all_subsets.extend(batch_rows["subset"].tolist())
            
            # 슬라이싱으로 라벨 일괄 추가
            all_species_labels.append(species_labels_np[batch_idx:batch_end])
            
            # Taxonomy 매핑
            for _, row in batch_rows.iterrows():
                active = [c for c in SPECIES_COLS if row[c] == 1]
                if active and active[0] in tax_dict:
                    t = tax_dict[active[0]]
                    all_families.append(t["family"])
                    all_genera.append(t["genus"])
                    all_species.append(t["species"])
                else:
                    all_families.append("unknown")
                    all_genera.append("unknown")
                    all_species.append("unknown")

        except Exception as e:
            print(f"\n⚠️  Batch {batch_idx} failed: {str(e)}")
            print(f"   Attempting single-file fallback for {len(batch_paths)} files...")
            
            for idx, (fpath, (_, row)) in enumerate(zip(batch_paths, batch_rows.iterrows())):
                try:
                    result = model.encode([fpath])
                    emb = result.embeddings.mean(axis=1)
                    if emb is not None and emb.shape[0] > 0:
                        all_embeddings.append(emb)
                        all_sample_names.append(row["sample_name"])
                        all_subsets.append(row["subset"])
                        all_species_labels.append(species_labels_np[batch_idx + idx : batch_idx + idx + 1])
                        
                        active = [c for c in SPECIES_COLS if row[c] == 1]
                        if active and active[0] in tax_dict:
                            t = tax_dict[active[0]]
                            all_families.append(t["family"])
                            all_genera.append(t["genus"])
                            all_species.append(t["species"])
                        else:
                            all_families.append("unknown")
                            all_genera.append("unknown")
                            all_species.append("unknown")
                    else:
                        failed_files.append(row["sample_name"])
                except Exception as e2:
                    failed_files.append(row["sample_name"])
            continue

    if not all_embeddings:
        print("\n✗ ERROR: No embeddings extracted!")
        return
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    species_labels = np.concatenate(all_species_labels, axis=0)
    
    print(f"\n{'='*60}")
    print(f"✓ Extraction complete! Total time: {(time.time()-t_start)/60:.1f}min")
    print(f"  • Embeddings shape: {embeddings.shape}")    
    if failed_files:
        print(f"\n⚠️  Failed files ({len(failed_files)}):")
        for fname in failed_files[:10]:  # 최대 10개만 표시
            print(f"    - {fname}")
        if len(failed_files) > 10:
            print(f"    ... and {len(failed_files)-10} more")

    # ─────────────────────────────────────────────
    # 저장
    # ─────────────────────────────────────────────
    print(f"\n[Saving results to {OUTPUT_DIR}/]")
    
    # 1. embeddings (numpy)
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    print(f"  ✓ embeddings.npy ({embeddings.shape})")

    # 2. species_labels (numpy)
    np.save(OUTPUT_DIR / "species_labels.npy", species_labels)
    print(f"  ✓ species_labels.npy ({species_labels.shape})")

    # 3. 메타데이터 (CSV) — NatureLM과 동일 구조
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
    print(f"  ✓ embeddings_metadata.csv ({len(meta_out)} rows)")

    # 4. 요약 정보
    summary = {
        "model": "BirdNET-v2.4",
        "dataset": "AnuraSet",
        "num_samples": len(embeddings),
        "embedding_dim": MODEL_DIM,
        "extraction_time_min": round((time.time()-t_start)/60, 1),
        "failed_samples": len(failed_files),
        "family_count": len(set(all_families)) - (1 if "unknown" in all_families else 0),
        "genus_count": len(set(all_genera)) - (1 if "unknown" in all_genera else 0),
        "species_count": len(set(all_species)) - (1 if "unknown" in all_species else 0),
    }
    with open(OUTPUT_DIR / "summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print(f"  ✓ summary.txt")

    print(f"\n{'='*60}")
    print("✓ All done!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()