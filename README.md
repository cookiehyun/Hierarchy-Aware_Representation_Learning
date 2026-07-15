# Hierarchy-Aware Representation Learning for Bioacoustic Data

This repository contains the code used to study hierarchy-aware representation learning for bioacoustic and environmental audio classification.

The experiments compare frozen embeddings from **NatureLM-Audio** and **BirdNET** using hierarchical learning objectives originally developed for computer vision.

## Overview

Two datasets and hierarchy types are used:

- **AnuraSet**: Family → Genus → Species
- **ESC-50**: BGA → Mid-category → Category

Two pretrained embedding models are compared:

- **NatureLM-Audio**: 4096-dimensional embeddings
- **BirdNET**: 1024-dimensional embeddings

The pretrained embedding models remain frozen. Only lightweight projection or hierarchy-classification heads are trained.

The evaluated methods are:

- Baseline
- UALB / Hierarchical Multi-Label Contrastive Learning
- Focal Tree-Min (`FTM-only`)
- Tree-Triplet (`TT-only`)
- Combined Focal Tree-Min and Tree-Triplet (`FTM+TT`)

Experiments include single-label, multi-label, unknown-inclusive, and unseen-class settings.

## Repository Structure

```text
├── data/
│   ├── anuraset_taxonomy.csv
│   ├── anuraset_taxonomy_unknown.csv
│   ├── esc50_bga_labels.csv
│   ├── unseen_species_split.json
│   └── unseen_category_split.json
├── scripts/
│   ├── NatureLM/
│   ├── birdnet/
│   └── multilabel/
├── src/
│   ├── NatureLM/
│   ├── birdnet/
│   ├── multilabel/
│   └── models/
└── results/
```

Large audio files, embedding arrays, model checkpoints, and logs are not included in this repository.

## Data Preparation

Download AnuraSet and ESC-50 separately.

Expected external data structure:

```text
/path/to/bioacoustic_data/
├── anuraset/
│   └── anuraset/
│       ├── audio/
│       └── metadata.csv
└── ESC-50-master/
    ├── audio/
    └── meta/
```

Before running a script, update the path variables near the top of the corresponding Python file.

Example:

```python
BASE_DIR = Path("/path/to/bioacoustic_data")
PROJECT_DIR = Path("/path/to/Bachelor-Thesis")
```

Repository metadata, taxonomy files, and evaluation splits are stored under `data/`.

## ESC-50 BGA Hierarchy

ESC-50 does not provide the soundscape hierarchy used in this study.

The original categories are reorganized as:

```text
BGA → Mid-category → Category
```

The highest-level BGA categories are:

- Biophony
- Geophony
- Anthrophony

The hierarchy file is stored at:

```text
data/esc50_bga_labels.csv
```

To regenerate it, update the paths in the script and run:

```bash
python scripts/NatureLM/esc50/baseline/build_bga_labels.py
```

The resulting CSV is used by both NatureLM and BirdNET ESC-50 scripts.

## Embedding Extraction

### NatureLM × AnuraSet

```bash
python scripts/NatureLM/anuraset/baseline/extract_embeddings.py
```

Expected output:

```text
embeddings/
├── embeddings.npy
├── species_labels.npy
└── embeddings_metadata.csv
```

### NatureLM × ESC-50

```bash
python scripts/NatureLM/esc50/baseline/extract_embeddings.py
```

Expected output:

```text
embeddings_esc50/
├── embeddings.npy
└── embeddings_metadata.csv
```

### BirdNET × AnuraSet

```bash
python scripts/birdnet/anuraset/baseline/extract_embeddings_birdnet.py
```

Expected output:

```text
embeddings_anuraset_birdnet/
├── embeddings.npy
├── species_labels.npy
├── embeddings_metadata.csv
└── summary.txt
```

### BirdNET × ESC-50

```bash
python scripts/birdnet/esc50/baseline/extract_embeddings_birdnet.py
```

Expected output:

```text
embeddings_esc50_birdnet/
├── embeddings.npy
├── embeddings_metadata.csv
└── summary.txt
```

NatureLM audio is converted to mono, resampled to 16 kHz, and pooled into one embedding per recording. BirdNET embeddings are averaged across segments.

## Training

Training scripts are organized by embedding model, dataset, and method.

Examples:

### UALB

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/ualb/train_ualb_single.py
```

### FTM + Tree-Triplet

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single.py
```

### FTM-only

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_ftm_only.py
```

### TT-only

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/tree_triplet/train_triplet_single_tt_only.py
```

Equivalent scripts are available for BirdNET, ESC-50, multi-label, unknown-inclusive, and unseen-class experiments.

## Evaluation

### NatureLM × AnuraSet Single-Label

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation.py
```

### BirdNET × AnuraSet Single-Label

```bash
PYTHONPATH=src python scripts/birdnet/anuraset/evaluation.py
```

### NatureLM × AnuraSet Multi-Label

```bash
PYTHONPATH=src python scripts/multilabel/NatureLM/anuraset/evaluation.py
```

### NatureLM × AnuraSet Unknown-Inclusive Multi-Label

```bash
PYTHONPATH=src python scripts/multilabel/NatureLM/anuraset/evaluation_unknown.py
```

### NatureLM × AnuraSet Unseen-Species

```bash
PYTHONPATH=src python scripts/NatureLM/anuraset/evaluation_unseen.py
```

Equivalent BirdNET and ESC-50 evaluation scripts are available in the corresponding directories.

Evaluation metrics include:

- Accuracy
- Macro-F1 and Micro-F1
- Hierarchical Precision, Recall, and F1
- Normalized Mutual Information
- Embedding similarity Gap
- AUROC for unseen-class detection

## Example Results

Representative results from the **NatureLM × AnuraSet multi-label** experiment are shown below.

| Method | Species Macro-F1 | Species Micro-F1 | hF1 |
|---|---:|---:|---:|
| Baseline | 36.73 | 56.36 | 62.07 |
| UALB | 42.23 | 64.77 | 73.96 |
| **FTM-only** | **54.03** | **83.12** | **86.72** |
| TT-only | 34.70 | 52.44 | 60.26 |
| FTM+TT | 35.34 | 53.91 | 60.16 |

FTM-only provides the strongest fine-grained and hierarchical classification performance in this setting.

Across experiments, hierarchy-aware learning often improves fine-grained classification, but no single method performs best in every setting. Larger embedding-space separation also does not always translate into better downstream classification.

## Requirements

Main dependencies include:

```text
torch
torchaudio
numpy
pandas
scikit-learn
matplotlib
tqdm
tensorflow
birdnet
```

NatureLM-Audio is loaded with:

```python
NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
```

BirdNET experiments use:

```python
birdnet.load("acoustic", "2.4", "tf")
```

## Notes

Several scripts were originally developed on an HPC cluster and may contain environment-specific paths.

Before execution, check:

- dataset paths,
- embedding paths,
- checkpoint paths,
- result directories,
- and Python module paths.

SLURM scripts are not required when running the Python files directly.

## Publication Status

The manuscript is currently being finalized.

**Working title:**  
*Hierarchy-Aware Representation Learning for Bioacoustic Data*

Citation and license information will be added after publication.
