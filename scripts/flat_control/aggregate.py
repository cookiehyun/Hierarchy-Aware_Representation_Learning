"""
Aggregate flat-control results into mean ± std over seeds, plus the delta of
every config vs. the flat MLP control (the number feedback #3 asks for).

Usage: python aggregate.py [naturelm|birdnet]
"""

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/flat_control/anuraset_single')
CONFIGS = ['linear', 'flat_mlp', 'mlbce', 'focal_mlbce', 'ftm']
METRICS = ['family_acc', 'genus_acc', 'species_acc',
           'family_macro_f1', 'genus_macro_f1', 'species_macro_f1',
           'hF1', 'consistency']
REFERENCE = 'flat_mlp'


def main():
    embedding = sys.argv[1] if len(sys.argv) > 1 else 'naturelm'
    res_dir = RESULTS_DIR / embedding

    runs = {}
    for config in CONFIGS:
        files = sorted(res_dir.glob(f'{config}_seed*.json'))
        if files:
            runs[config] = {m: np.array([json.load(open(f))['test'][m] for f in files]) for m in METRICS}
            runs[config]['n'] = len(files)

    print(f"\n[{embedding}] mean ± std over seeds\n")
    print(f"{'config':<12} {'n':>2} " + " ".join(f"{m:>17}" for m in METRICS))
    for config, r in runs.items():
        print(f"{config:<12} {r['n']:>2} " + " ".join(
            f"{r[m].mean():>9.2f} ± {r[m].std(ddof=1) if r['n'] > 1 else 0:>5.2f}" for m in METRICS))

    if REFERENCE in runs:
        print(f"\nDelta vs. {REFERENCE} (mean difference)\n")
        print(f"{'config':<12} " + " ".join(f"{m:>17}" for m in METRICS))
        for config, r in runs.items():
            print(f"{config:<12} " + " ".join(
                f"{r[m].mean() - runs[REFERENCE][m].mean():>+17.2f}" for m in METRICS))

    summary = {c: {m: [float(r[m].mean()), float(r[m].std(ddof=1)) if r['n'] > 1 else 0.0]
                   for m in METRICS} | {'n': r['n']} for c, r in runs.items()}
    with open(res_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {res_dir / 'summary.json'}")


if __name__ == '__main__':
    main()
