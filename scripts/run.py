"""Train one method on one task and evaluate it. Writes
results/runs/<protocol>/<embedding>/<dataset>/<label_mode>[_variant]/<method>/seed<k>/metrics.json

Examples:
  python scripts/run.py --embedding naturelm --dataset anuraset --label-mode single --method ftm --seed 0
  python scripts/run.py --embedding birdnet --dataset esc50 --method hmce --variant unseen --seed 1
  python scripts/run.py ... --method baseline                       # frozen embedding, no training
  python scripts/run.py ... --method ftm --protocol paper --checkpoint results/NatureLM/anuraset/wp2_ftm_only_singlelabel/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from hierbio import paths
from hierbio.data import load_task, train_val_indices
from hierbio.evaluate import evaluate
from hierbio.train import DEVICE, METHODS, build_model, set_seed, train

VAL_FRAC = 0.1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embedding', required=True, choices=sorted(paths.EMBEDDING_DIMS))
    p.add_argument('--dataset', required=True, choices=['anuraset', 'esc50'])
    p.add_argument('--label-mode', default='single', choices=['single', 'multi'])
    p.add_argument('--variant', default='standard', choices=['standard', 'unknown', 'unseen'])
    p.add_argument('--method', required=True, choices=list(METHODS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--protocol', default='v2', choices=['v2', 'paper'],
                   help="'paper' reproduces the published tables (split seed 42, threshold 0.5)")
    p.add_argument('--checkpoint', help='evaluate an existing best_model.pt instead of training')
    p.add_argument('--save-model', action='store_true')
    args = p.parse_args()

    task = load_task(args.embedding, args.dataset, args.label_mode, args.variant)
    split_seed = 42 if args.protocol == 'paper' else args.seed
    train_idx, val_idx = train_val_indices(len(task.train), VAL_FRAC, split_seed)

    run = f'seed{args.seed}' if args.checkpoint is None else 'checkpoint'
    out_dir = paths.RESULTS_DIR / 'runs' / args.protocol / task.name / args.method / run
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {task.name} | {args.method} | protocol={args.protocol} seed={args.seed} | {DEVICE}")

    history, model = None, None
    if args.method != 'baseline':
        if args.checkpoint:
            model = build_model(args.method, task).to(DEVICE)
            ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        else:
            model, history = train(args.method, task, train_idx, val_idx, args.seed)
            if args.save_model:
                torch.save({'model_state_dict': model.state_dict(), 'args': vars(args)}, out_dir / 'model.pt')

    set_seed(args.seed)
    metrics = evaluate(args.method, model, task, train_idx, val_idx, args.protocol, args.seed)
    record = {'args': vars(args), 'n_train': len(train_idx), 'n_val': len(val_idx), 'n_test': len(task.test),
              'metrics': metrics}
    if history is not None:
        record.update(best_epoch=history['best_epoch'], train_time_s=history['train_time_s'])
        json.dump(history, open(out_dir / 'history.json', 'w'))
    json.dump(record, open(out_dir / 'metrics.json', 'w'), indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"[run] saved {out_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
