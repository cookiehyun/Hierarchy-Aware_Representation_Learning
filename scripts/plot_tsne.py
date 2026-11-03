"""t-SNE of test embeddings before (frozen backbone) and after hierarchical training.

  python scripts/plot_tsne.py --embedding naturelm --dataset anuraset --level genus \
      --panel ftm=results/runs/v2/naturelm/anuraset/single/ftm/seed0/model.pt \
      --panel ftm_tt=results/NatureLM/anuraset/wp2_hssn_singlelabel_self/best_model.pt

Each --panel is method=checkpoint; the 'Before' panel is always added. Multi-label
samples are coloured by their first active class at the chosen level.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from hierbio import paths
from hierbio.data import load_task
from hierbio.evaluate import gap_single, model_outputs
from hierbio.train import DEVICE, build_model

LABELS = {'baseline': 'Before (Baseline)', 'hmce': 'HMCE', 'ftm': 'FTM-only', 'tt': 'TT-only', 'ftm_tt': 'FTM+TT'}
BG = '#E9EDF3'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embedding', default='naturelm')
    p.add_argument('--dataset', default='anuraset')
    p.add_argument('--label-mode', default='single')
    p.add_argument('--level', default=None, help='level name, e.g. family / genus / midcat (default: middle level)')
    p.add_argument('--panel', action='append', default=[], help='method=checkpoint')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=Path)
    args = p.parse_args()

    task = load_task(args.embedding, args.dataset, args.label_mode)
    h = task.hierarchy
    level = h.level_names.index(args.level) if args.level else 1
    if task.single:
        labels = np.array(h.classes[level])[task.test.ids[:, level].numpy()]
    else:
        block = task.test.Y[:, h.level_slice(level)].numpy()
        labels = np.where(block.any(1), np.array(h.classes[level])[block.argmax(1)], 'none')

    panels = [('baseline', None)] + [tuple(s.split('=', 1)) for s in args.panel]
    embs = []
    for method, ckpt in panels:
        model = None
        if ckpt:
            model = build_model(method, task).to(DEVICE)
            state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state.get('model_state_dict', state))
        emb, _ = model_outputs(method, model, task.test.X)
        gap = gap_single(emb, labels, args.seed, per_class_mean=True)
        print(f'{method}: {h.level_names[level]} Gap = {gap:.4f}')
        embs.append((method, gap, TSNE(n_components=2, perplexity=30, random_state=args.seed).fit_transform(emb)))

    classes = sorted(set(labels))
    cmap = plt.colormaps['tab20'].resampled(len(classes))
    colors = {c: cmap(i) for i, c in enumerate(classes)}
    fig, axes = plt.subplots(1, len(embs), figsize=(10 * len(embs), 10), squeeze=False)
    fig.patch.set_facecolor(BG)
    for ax, (method, gap, xy) in zip(axes[0], embs):
        ax.set_facecolor(BG)
        for c in classes:
            m = labels == c
            ax.scatter(xy[m, 0], xy[m, 1], color=colors[c], label=c, alpha=0.7, s=22, linewidths=0)
        ax.set_title(f'{LABELS.get(method, method)}  (Gap {gap:.3f})', fontsize=26, pad=14)
        ax.set_xlabel('t-SNE 1', fontsize=20)
        ax.set_ylabel('t-SNE 2', fontsize=20)
        ax.tick_params(labelsize=16)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=6,
               markerscale=3, fontsize=18, facecolor=BG)
    plt.tight_layout()
    out = args.out or (paths.RESULTS_DIR / 'figures' /
                       f'tsne_{args.embedding}_{args.dataset}_{args.label_mode}_{h.level_names[level]}.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=BG)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
