"""Extract one embedding per recording with a frozen backbone (paper Appendix).

  python scripts/extract_embeddings.py --model naturelm --dataset anuraset   # naturelm-env
  python scripts/extract_embeddings.py --model birdnet  --dataset esc50      # birdnet-env

NatureLM: mono, 16 kHz, whole clip through model.encode_audio(), attention-masked mean
over frames (4096-d). BirdNET: file paths to model.encode(), mean over its 3 s segments
(1024-d). Writes embeddings.npy + embeddings_metadata.csv to paths.EMBEDDING_DIRS.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd

from hierbio import paths


def anuraset_items():
    meta = pd.read_csv(paths.ANURASET_METADATA)
    tax = pd.read_csv(paths.ANURASET_TAXONOMY).set_index('code')
    species_cols = [c for c in meta.columns if c in tax.index]
    files = [str(paths.ANURASET_AUDIO / r.site / f'{r.fname}_{int(r.min_t)}_{int(r.max_t)}.wav')
             for r in meta.itertuples()]
    # family / genus / species columns = first active species ('unknown' if none); only
    # meaningful for single-label samples, multi-label code reads the species columns.
    first = [next((c for c in species_cols if row[c] == 1), None) for _, row in meta.iterrows()]
    out = pd.DataFrame({
        'sample_name': meta['sample_name'],
        'family': [tax.at[c, 'family'] if c else 'unknown' for c in first],
        'genus': [tax.at[c, 'genus'] if c else 'unknown' for c in first],
        'species': [tax.at[c, 'species'] if c else 'unknown' for c in first],
        'subset': meta['subset'],
    })
    out = pd.concat([out, meta[species_cols].reset_index(drop=True)], axis=1)
    return files, out


def esc50_items():
    labels = pd.read_csv(paths.ESC50_LABELS)
    files = [str(paths.ESC50_AUDIO / f) for f in labels['filename']]
    return files, labels[['filename', 'category', 'bga_label', 'fold']]


def encode_naturelm(files, batch_size=32):
    import torch
    import torchaudio
    from NatureLM.models import NatureLM

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NatureLM.from_pretrained('EarthSpeciesProject/NatureLM-audio').to(device).eval()
    out = []
    for i in range(0, len(files), batch_size):
        waves = []
        for f in files[i:i + batch_size]:
            w, sr = torchaudio.load(f)
            w = w.mean(dim=0) if w.shape[0] > 1 else w[0]
            waves.append(torchaudio.functional.resample(w, sr, 16000) if sr != 16000 else w)
        batch = torch.zeros(len(waves), max(len(w) for w in waves))
        for j, w in enumerate(waves):
            batch[j, :len(w)] = w
        with torch.no_grad():
            emb, att = model.encode_audio(batch.to(device))
            mask = att.bool().unsqueeze(-1)
            out.append(((emb * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy())
        if (i // batch_size) % 100 == 0:
            print(f'  {i + len(waves)}/{len(files)}', flush=True)
    return np.concatenate(out)


def encode_birdnet(files, batch_size=128):
    import birdnet
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    model = birdnet.load('acoustic', '2.4', 'tf')
    out = []
    for i in range(0, len(files), batch_size):
        out.append(model.encode(files[i:i + batch_size]).embeddings.mean(axis=1))
        if (i // batch_size) % 50 == 0:
            print(f'  {i + batch_size}/{len(files)}', flush=True)
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['naturelm', 'birdnet'])
    p.add_argument('--dataset', required=True, choices=['anuraset', 'esc50'])
    args = p.parse_args()

    files, meta = {'anuraset': anuraset_items, 'esc50': esc50_items}[args.dataset]()
    t0 = time.time()
    emb = {'naturelm': encode_naturelm, 'birdnet': encode_birdnet}[args.model](files)
    assert len(emb) == len(meta)
    out_dir = paths.EMBEDDING_DIRS[(args.model, args.dataset)]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'embeddings.npy', emb)
    meta.to_csv(out_dir / 'embeddings_metadata.csv', index=False)
    print(f'{emb.shape} in {(time.time() - t0) / 60:.1f} min -> {out_dir}')


if __name__ == '__main__':
    main()
