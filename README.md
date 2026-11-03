# Hierarchy-Aware Representation Learning for Bio-acoustic Data

Code for *Hierarchy-Aware Representation Learning for Bio-acoustic Data* (Ku, Saghir, Gouvêa, Sonntag).

Frozen **NatureLM-audio** (4096-d) and **BirdNET** (1024-d) embeddings get a small trainable
head with one of four hierarchical objectives, and are compared with flat baselines on

- **AnuraSet**: family (5) → genus (12) → species (42), single- and multi-label
- **ESC-50**: BGA (3) → mid-category (12) → category (50), single-label

| method | objective | head | evaluated by |
|---|---|---|---|
| `baseline` | none (frozen embedding) | – | linear probe |
| `hmce` | HiMulConE + species level (Eq. 1–3) | MLP → 128, L2 | linear probe |
| `ftm` | Focal Tree-Min (Eq. 4–5) | MLP → \|V\| sigmoid | own node scores |
| `tt` | Tree-Triplet + same-class positives (Eq. 6) | MLP → 128, L2 | linear probe |
| `ftm_tt` | FTM + β(t)·TT (Eq. 7) | shared trunk, both heads | linear probe |
| `flat_linear`, `flat_mlp` | cross-entropy on the finest level | linear / same MLP | own logits |
| `mlbce`, `focal_mlbce` | (focal) BCE on all nodes, no tree constraint | same MLP | own node scores |

The flat controls use the same MLP, optimiser and model selection as the hierarchical heads,
so the only difference is the loss.

## Layout

```text
src/hierbio/
  paths.py       all filesystem locations (env: HIERBIO_DATA_ROOT, HIERBIO_RESULTS)
  data.py        load_task(embedding, dataset, label_mode, variant) -> train/test splits
  hierarchy.py   3-level tree: node order, ancestor matrix, tree distance psi
  losses.py      HMCE, FTM, Tree-Triplet (single + multi-label)
  models.py      ProjectionHead, HierarchicalModel, FlatHead
  train.py       one training loop for every method
  evaluate.py    Acc / macro-F1 / micro-F1, Gap, NMI, hP/hR/hF1, consistency, unseen AUROC
scripts/
  run.py                  train + evaluate one (task, method, seed)  -> metrics.json
  aggregate.py            mean ± std tables over seeds
  extract_embeddings.py   NatureLM / BirdNET embedding extraction
  build_esc50_hierarchy.py  ESC-50 -> BGA labels (data/esc50_bga_labels.csv)
  plot_tsne.py            before / after t-SNE
slurm/
  run.sh, grid.sh         sbatch wrappers (grid = methods x seeds job array)
  verify_paper.sh         re-evaluates the paper checkpoints with this code
data/                     taxonomy, ESC-50 hierarchy, unseen-class splits
```

## Running

```bash
# one run
python scripts/run.py --embedding naturelm --dataset anuraset --label-mode multi --method ftm --seed 0

# all methods x 5 seeds on the cluster (9 x 5 = 45 array tasks)
sbatch --array=0-44 slurm/grid.sh --embedding naturelm --dataset anuraset --label-mode single

# a subset
METHODS="baseline ftm flat_mlp" SEEDS="0 1 2" sbatch --array=0-8 slurm/grid.sh --embedding birdnet --dataset esc50

# tables
python scripts/aggregate.py --filter anuraset/multi
```

Results go to `results/runs/<protocol>/<embedding>/<dataset>/<label_mode>[_variant]/<method>/seed<k>/`.

Options:

- `--variant unknown` — AnuraSet with the "no call" recordings kept (single-label: an extra
  `unknown` class at every level; multi-label: all-zero rows).
- `--variant unseen` — single-label open-set test: the classes in `data/unseen_*_split.json`
  are removed from training; reports AUROC / recall / false-reject rate for near- and far-OOD
  classes plus the usual metrics on the held-in test classes.
- `--protocol paper` — the exact evaluation used for the paper tables (validation split
  seed 42, probes on the full training set, multi-label threshold 0.5, Gap pooled over all
  pairs). The default `v2` ties all randomness to `--seed`, fits probes on the training part
  of the split, tunes multi-label thresholds per class on the validation part, and averages
  Gap per class as in the paper's formula.
- `--checkpoint path/to/best_model.pt` — evaluate an existing checkpoint (old checkpoints load as is).
- `--save-model` — keep the trained head.

## Embeddings

```bash
python scripts/build_esc50_hierarchy.py                               # once
python scripts/extract_embeddings.py --model naturelm --dataset anuraset   # naturelm-env
python scripts/extract_embeddings.py --model birdnet --dataset esc50       # birdnet-env
```

Expected raw data under `HIERBIO_DATA_ROOT` (default `/user/gaad2403/bachelor_thesis`):
`anuraset/anuraset/{audio,metadata.csv}` and `ESC-50-master/{audio,meta}`.
NatureLM is loaded with `NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")`,
BirdNET with `birdnet.load("acoustic", "2.4", "tf")`.

## Requirements

`torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`; `torchaudio` + NatureLM-audio for
NatureLM extraction; `birdnet` + `tensorflow` for BirdNET extraction.
