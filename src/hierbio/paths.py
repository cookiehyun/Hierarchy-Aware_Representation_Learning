"""Filesystem locations. Override with environment variables instead of editing code:

    HIERBIO_DATA_ROOT   raw audio + extracted embeddings (default: <repo>/datasets)
    HIERBIO_RESULTS     where runs are written       (default: <repo>/results)
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / 'data'
DATA_ROOT = Path(os.environ.get('HIERBIO_DATA_ROOT', PROJECT_DIR / 'datasets'))
RESULTS_DIR = Path(os.environ.get('HIERBIO_RESULTS', PROJECT_DIR / 'results'))

ANURASET_TAXONOMY = DATA_DIR / 'anuraset_taxonomy.csv'
ESC50_LABELS = DATA_DIR / 'esc50_bga_labels.csv'
UNSEEN_SPLITS = {
    'anuraset': DATA_DIR / 'unseen_species_split.json',
    'esc50': DATA_DIR / 'unseen_category_split.json',
}

# Raw audio
ANURASET_AUDIO = DATA_ROOT / 'anuraset/anuraset/audio'
ANURASET_METADATA = DATA_ROOT / 'anuraset/anuraset/metadata.csv'
ESC50_AUDIO = DATA_ROOT / 'ESC-50-master/audio'
ESC50_META = DATA_ROOT / 'ESC-50-master/meta/esc50.csv'

# (embedding model, dataset) -> directory holding embeddings.npy + embeddings_metadata.csv
EMBEDDING_DIRS = {
    ('naturelm', 'anuraset'): DATA_ROOT / 'embeddings',
    ('birdnet', 'anuraset'): DATA_ROOT / 'embeddings_anuraset_birdnet',
    ('naturelm', 'esc50'): DATA_ROOT / 'embeddings_esc50',
    ('birdnet', 'esc50'): DATA_ROOT / 'embeddings_esc50_birdnet',
}
EMBEDDING_DIMS = {'naturelm': 4096, 'birdnet': 1024}
