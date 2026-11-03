"""Load frozen embeddings + hierarchical labels for every (embedding, dataset, label mode).

Every task is returned in the same form, so training and evaluation never branch on
the dataset:

    Split.X      [n, d]        frozen backbone embeddings
    Split.Y      [n, n_nodes]  node-wise multi-hot targets (coarse | mid | fine blocks)
    Split.ids    [n, 3]        (coarse, mid, fine) class ids; single-label only, -1 = class
                               not in the training hierarchy (unseen)
    Split.meta   DataFrame     the matching metadata rows

Variants:
    include_negatives  AnuraSet: keep "no call" recordings. Single-label adds an explicit
                       'unknown' class at every level; multi-label keeps them as all-zero rows.
    unseen             single-label: held-out classes (data/unseen_*_split.json) are removed
                       from train and kept in test, flagged by Split.is_unseen / near / far.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch

from . import paths
from .hierarchy import Hierarchy

ANURASET_NON_SPECIES = ['sample_name', 'family', 'genus', 'species', 'subset']
UNKNOWN = 'unknown'


@dataclass
class Split:
    X: torch.Tensor
    Y: torch.Tensor
    meta: pd.DataFrame
    ids: Optional[torch.Tensor] = None
    is_unseen: Optional[np.ndarray] = None
    is_near: Optional[np.ndarray] = None
    is_far: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.X)

    def subset(self, idx):
        idx = np.asarray(idx)
        pick = lambda a: None if a is None else a[idx]
        return Split(self.X[idx], self.Y[idx], self.meta.iloc[idx].reset_index(drop=True),
                     pick(self.ids), pick(self.is_unseen), pick(self.is_near), pick(self.is_far))


@dataclass
class Task:
    embedding: str
    dataset: str
    label_mode: str          # 'single' | 'multi'
    variant: str             # 'standard' | 'unknown' | 'unseen'
    hierarchy: Hierarchy
    train: Split
    test: Split
    epochs: int = 100
    info: dict = field(default_factory=dict)

    @property
    def input_dim(self):
        return self.train.X.shape[1]

    @property
    def single(self):
        return self.label_mode == 'single'

    @property
    def name(self):
        suffix = '' if self.variant == 'standard' else f'_{self.variant}'
        return f'{self.embedding}/{self.dataset}/{self.label_mode}{suffix}'


def train_val_indices(n, val_frac, seed):
    """Same partition as torch.utils.data.random_split(range(n), [n - n_val, n_val],
    generator=Generator().manual_seed(seed)) used by the original training scripts."""
    n_val = int(n * val_frac)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    return perm[:n - n_val], perm[n - n_val:]


def _load_embeddings(embedding, dataset):
    d = paths.EMBEDDING_DIRS[(embedding, dataset)]
    return np.load(d / 'embeddings.npy'), pd.read_csv(d / 'embeddings_metadata.csv')


def _split(X, meta, mask, hierarchy, single, fine_cols=None):
    meta = meta[mask].reset_index(drop=True)
    X = torch.from_numpy(np.ascontiguousarray(X[mask])).float()
    if single:
        names = [meta[c].values for c in hierarchy.level_names]
        ids = hierarchy.encode(names)
        return Split(X, torch.from_numpy(hierarchy.node_targets_from_ids(ids)), meta,
                     torch.from_numpy(ids).long())
    Y = hierarchy.node_targets_from_fine_multihot(meta[fine_cols].values)
    return Split(X, torch.from_numpy(Y), meta)


def _load_anuraset(embedding, label_mode, variant):
    X, meta = _load_embeddings(embedding, 'anuraset')
    species_cols = [c for c in meta.columns if c not in ANURASET_NON_SPECIES]
    n_species = meta[species_cols].sum(axis=1).values
    keep = n_species > 0
    if variant == 'unknown':
        keep |= n_species == 0
    if label_mode == 'single':
        keep &= n_species <= 1
    X, meta = X[keep], meta[keep].reset_index(drop=True)
    is_train = (meta['subset'] == 'train').values
    is_test = (meta['subset'] == 'test').values
    levels = ('family', 'genus', 'species')

    if label_mode == 'multi':
        if variant == 'unseen':
            raise NotImplementedError('unseen-class evaluation exists for single-label only')
        tax = pd.read_csv(paths.ANURASET_TAXONOMY).set_index('code')
        rows = [(tax.at[c, 'family'], tax.at[c, 'genus'], c) for c in species_cols]
        classes = [sorted({r[0] for r in rows}), sorted({r[1] for r in rows}), species_cols]
        h = Hierarchy.from_rows(levels, rows, classes)
        return h, _split(X, meta, is_train, h, False, species_cols), _split(X, meta, is_test, h, False, species_cols), {}

    info = {}
    held_out = np.zeros(len(meta), dtype=bool)
    if variant == 'unseen':
        split = json.load(open(paths.UNSEEN_SPLITS['anuraset']))
        pair = list(zip(meta['genus'], meta['species']))
        as_set = lambda key: {(p['genus'], p['species']) for p in split[key]}
        held_out = np.array([p in as_set('all_held_out_pairs') for p in pair])
        near = np.array([p in as_set('near_ood_pairs') for p in pair])
        far = np.array([p in as_set('far_ood_pairs') for p in pair])
        info['held_out'] = sorted(p['code'] for p in split['all_held_out_pairs'])

    train_meta = meta[is_train & ~held_out]
    h = Hierarchy.from_rows(levels, train_meta[list(levels)].values)
    train = _split(X, meta, is_train & ~held_out, h, True)
    test = _split(X, meta, is_test, h, True)
    if variant == 'unseen':
        test.is_unseen, test.is_near, test.is_far = held_out[is_test], near[is_test], far[is_test]
    return h, train, test, info


def _load_esc50(embedding, label_mode, variant):
    if label_mode != 'single':
        raise ValueError('ESC-50 is single-label')
    if variant == 'unknown':
        raise ValueError('ESC-50 has no negative recordings')
    X, emb_meta = _load_embeddings(embedding, 'esc50')
    meta = pd.read_csv(paths.ESC50_LABELS)
    assert (emb_meta['filename'].values == meta['filename'].values).all(), 'embedding rows out of order'
    levels = ('bga_label', 'midcat', 'category')
    is_train = meta['fold'].isin([1, 2, 3, 4]).values
    is_test = (meta['fold'] == 5).values

    info = {}
    held_out = np.zeros(len(meta), dtype=bool)
    if variant == 'unseen':
        split = json.load(open(paths.UNSEEN_SPLITS['esc50']))
        held_out = meta['category'].isin(split['all_held_out_categories']).values
        near = meta['category'].isin(split['near_ood_categories']).values
        far = meta['category'].isin(split['far_ood_categories']).values
        info['held_out'] = split['all_held_out_categories']

    h = Hierarchy.from_rows(levels, meta[is_train & ~held_out][list(levels)].values)
    train = _split(X, meta, is_train & ~held_out, h, True)
    test = _split(X, meta, is_test, h, True)
    if variant == 'unseen':
        test.is_unseen, test.is_near, test.is_far = held_out[is_test], near[is_test], far[is_test]
    return h, train, test, info


def load_task(embedding, dataset, label_mode='single', variant='standard'):
    loader = {'anuraset': _load_anuraset, 'esc50': _load_esc50}[dataset]
    h, train, test, info = loader(embedding, label_mode, variant)
    epochs = 450 if dataset == 'esc50' else 100
    task = Task(embedding, dataset, label_mode, variant, h, train, test, epochs, info)
    print(f"[data] {task.name}: train={len(train)} test={len(test)} "
          f"classes={'/'.join(map(str, h.sizes))} dim={task.input_dim}")
    return task
