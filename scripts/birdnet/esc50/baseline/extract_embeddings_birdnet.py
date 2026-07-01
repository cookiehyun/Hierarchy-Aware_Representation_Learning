import birdnet
import pandas as pd
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR    = Path("/user/gaad2403/bachelor_thesis")
AUDIO_DIR   = BASE_DIR / "ESC-50-master/audio"
LABELS_PATH = Path("/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv")
OUTPUT_DIR  = BASE_DIR / "embeddings_esc50_birdnet"  # ← 명확한 이름
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
MODEL_DIM  = 1024  # BirdNET embedding dimension


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BirdNET ESC-50 Embedding Extraction")
    print("=" * 60)
    
    # 모델 로드
    print("\n[1/3] Loading BirdNET model...")
    t0 = time.time()
    model = birdnet.load("acoustic", "2.4", "tf")
    print(f"✓ Model loaded in {time.time()-t0:.1f}s")
    print(f"  • Embedding dimension: {model.get_embeddings_dim()}")
    print(f"  • Sample rate: {model.get_sample_rate()}")
    print(f"  • Segment size: {model.get_segment_size_s()}s")

    # 데이터 로드
    print("\n[2/3] Loading metadata...")
    labels_df = pd.read_csv(LABELS_PATH)
    print(f"✓ Loaded {len(labels_df)} samples")
    print(f"\nBGA distribution:")
    for label, count in labels_df["bga_label"].value_counts().items():
        print(f"  • {label:15s}: {count:4d}")

    # 오디오 파일 경로 리스트 생성
    audio_paths = [str(AUDIO_DIR / row["filename"]) for _, row in labels_df.iterrows()]
    
    # 결과 저장용 리스트
    all_embeddings = []
    all_filenames  = []
    all_categories = []
    all_bga_labels = []
    all_folds      = []
    failed_files   = []

    # 임베딩 추출
    print(f"\n[3/3] Extracting embeddings (batch_size={BATCH_SIZE})...")
    t_start = time.time()

    for batch_idx in tqdm(range(0, len(audio_paths), BATCH_SIZE), 
                          desc="Processing batches"):
        batch_end = min(batch_idx + BATCH_SIZE, len(audio_paths))
        batch_paths = audio_paths[batch_idx : batch_end]
        batch_rows  = labels_df.iloc[batch_idx : batch_end]

        try:
            # BirdNET batch encoding: file_paths -> AcousticFileEncodingResult
            result = model.encode(batch_paths)
            
            # result.embeddings shape: [num_files, num_segments, 1024]
            # → average pooling over segments: [num_files, 1024]
            embeddings = result.embeddings.mean(axis=1)
            
            if embeddings is None or embeddings.shape[0] == 0:
                raise ValueError(f"Empty embeddings returned for batch {batch_idx}")
            
            all_embeddings.append(embeddings)
            all_filenames.extend(batch_rows["filename"].tolist())
            all_categories.extend(batch_rows["category"].tolist())
            all_bga_labels.extend(batch_rows["bga_label"].tolist())
            all_folds.extend(batch_rows["fold"].tolist())

        except Exception as e:
            print(f"\n⚠️  Batch {batch_idx} failed: {str(e)}")
            print(f"   Attempting single-file fallback for {len(batch_paths)} files...")
            
            # Fallback: 단일 파일로 하나씩 처리
            for i, (fpath, (_, row)) in enumerate(zip(batch_paths, batch_rows.iterrows())):
                try:
                    result = model.encode([fpath])
                    embeddings = result.embeddings.mean(axis=1)  # [1, 1024]
                    if embeddings is not None and embeddings.shape[0] > 0:
                        all_embeddings.append(embeddings)
                        all_filenames.append(row["filename"])
                        all_categories.append(row["category"])
                        all_bga_labels.append(row["bga_label"])
                        all_folds.append(row["fold"])
                    else:
                        failed_files.append(row["filename"])
                except Exception as e2:
                    print(f"   × {row['filename']}: {str(e2)}")
                    failed_files.append(row["filename"])
            continue

    # 임베딩 병합
    if not all_embeddings:
        print("\n✗ ERROR: No embeddings extracted!")
        return
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    print(f"\n{'='*60}")
    print(f"✓ Extraction complete!")
    print(f"  • Total: {len(embeddings)} embeddings")
    print(f"  • Shape: {embeddings.shape} (expected: [N, {MODEL_DIM}])")
    print(f"  • Size: {embeddings.nbytes/1e6:.1f} MB")
    print(f"  • Time: {(time.time()-t_start)/60:.1f}min")
    
    if failed_files:
        print(f"\n⚠️  Failed files ({len(failed_files)}):")
        for fname in failed_files:
            print(f"    - {fname}")

    # ─────────────────────────────────────────────
    # 저장
    # ─────────────────────────────────────────────
    print(f"\n[Saving results to {OUTPUT_DIR}/]")
    
    # 1. embeddings (numpy)
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    print(f"  ✓ embeddings.npy ({embeddings.shape})")

    # 2. 메타데이터 (CSV) — NatureLM과 동일 구조
    meta_out = pd.DataFrame({
        "filename" : all_filenames,
        "category" : all_categories,
        "bga_label": all_bga_labels,
        "fold"     : all_folds,
    })
    meta_out.to_csv(OUTPUT_DIR / "embeddings_metadata.csv", index=False)
    print(f"  ✓ embeddings_metadata.csv ({len(meta_out)} rows)")

    # 3. 요약 정보
    summary = {
        "model": "BirdNET-v2.4",
        "dataset": "ESC-50",
        "num_samples": len(embeddings),
        "embedding_dim": MODEL_DIM,
        "extraction_time_min": round((time.time()-t_start)/60, 1),
        "failed_samples": len(failed_files),
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