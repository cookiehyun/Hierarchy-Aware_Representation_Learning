"""Collect results/runs/**/metrics.json into mean ± std tables (one per protocol × task).

  python scripts/aggregate.py                      # all tasks, markdown to stdout
  python scripts/aggregate.py --filter anuraset/single --csv results/summary.csv
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd

from hierbio import paths
from hierbio.train import METHODS


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=paths.RESULTS_DIR / 'runs', type=Path)
    p.add_argument('--filter', default='', help='substring the task path must contain')
    p.add_argument('--csv')
    args = p.parse_args()

    groups = defaultdict(list)
    for f in sorted(args.root.glob('**/metrics.json')):
        rel = f.relative_to(args.root).parts          # protocol, emb, dataset, mode, method, run
        task, method = '/'.join(rel[:4]), rel[4]
        if args.filter in task:
            rec = json.load(open(f))
            row = flatten(rec['metrics'])
            if 'train_time_s' in rec:
                row['train_time_s'] = rec['train_time_s']
            groups[(task, method)].append(row)

    rows = []
    for (task, method), runs in groups.items():
        df = pd.DataFrame(runs)
        row = {'task': task, 'method': method, 'n_seeds': len(runs)}
        for c in df.columns:
            vals = df[c].astype(float)
            row[c] = vals.mean()
            row[c + '_std'] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        print('no runs found')
        return
    order = {m: i for i, m in enumerate(METHODS)}
    table = table.sort_values(['task', 'method'], key=lambda s: s.map(order) if s.name == 'method' else s)
    if args.csv:
        table.to_csv(args.csv, index=False)

    for task, t in table.groupby('task', sort=False):
        metrics = [c for c in t.columns if not c.endswith('_std') and c not in ('task', 'method', 'n_seeds')
                   and t[c].notna().any() and 'time' not in c]
        print(f'\n### {task}\n')
        print('| method | n | ' + ' | '.join(metrics) + ' |')
        print('|---' * (len(metrics) + 2) + '|')
        for _, r in t.iterrows():
            cells = []
            for m in metrics:
                mean, std = r[m], r.get(m + '_std')
                fmt = '{:.4f}' if 'gap' in m else '{:.2f}'
                cells.append('' if pd.isna(mean) else fmt.format(mean) + ('' if pd.isna(std) else ' ± ' + fmt.format(std)))
            print(f"| {r['method']} | {r['n_seeds']} | " + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
