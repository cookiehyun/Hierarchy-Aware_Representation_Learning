"""
Flat-vs-hierarchical control experiment (single-label, AnuraSet).

Feedback #3: the question is not UALB vs HSSN, but how much hierarchical
training improves over FLAT training. In the paper, FTM-only used its own
MLP head while the baseline used a linear probe, so the head capacity was
confounded with the loss. Here every config except `linear` uses the SAME
head architecture (Linear(d->1024) -> ReLU -> Linear(1024->out)), the same
optimizer/schedule, the same data split and the same model-selection rule;
only the loss differs.

Configs:
  linear       Linear(d->n_species), CE on species             (flat, linear)
  flat_mlp     MLP(d->n_species),   CE on species              (flat, MLP)   <- key control
  mlbce        MLP(d->n_nodes),     BCE on all nodes           (multi-level labels, no tree constraint)
  focal_mlbce  MLP(d->n_nodes),     focal BCE on all nodes     (+ focal term, no tree-min)
  ftm          MLP(d->n_nodes),     Focal Tree-Min             (+ tree-min = paper's FTM-only)

focal_mlbce is FocalTreeMinLoss with identity ancestor/descendant matrices,
i.e. exactly FTM with the tree-min step removed.

Flat configs predict coarser levels by marginalising species probabilities
over the taxonomy; node configs take the argmax within each level's block
(same as the paper's FTM-only evaluation).

Usage:
  python train_eval_single.py --config ftm --seed 0 --embedding naturelm
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from NatureLM.anuraset.anuraset_dataset import AnuraSetEmbeddingDataset
from NatureLM.anuraset.hierarchy_tree import HierarchyTree
from focal_tree_min_loss import FocalTreeMinLoss
from hf1 import ground_truth_singlelabel, hierarchical_f1

PROJECT_DIR   = Path('/user/gaad2403/Bachelor-Thesis')
TAXONOMY_PATH = PROJECT_DIR / 'data/anuraset_taxonomy.csv'
EMBEDDINGS = {
    'naturelm': ('/user/gaad2403/bachelor_thesis/embeddings', 4096),
    'birdnet':  ('/user/gaad2403/bachelor_thesis/embeddings_anuraset_birdnet', 1024),
}
CONFIGS = ['linear', 'flat_mlp', 'mlbce', 'focal_mlbce', 'ftm']

# Same hyperparameters as the paper's FTM-only run (train_triplet_single_ftm_only.py)
BATCH_SIZE   = 256
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM   = 1024
VAL_SPLIT    = 0.1
FTM_GAMMA    = 2.0
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_head(config, input_dim, n_out):
    if config == 'linear':
        return nn.Linear(input_dim, n_out)
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_DIM),
        nn.ReLU(inplace=True),
        nn.Linear(HIDDEN_DIM, n_out),
    )


class Taxonomy:
    """Index maps between levels + node-space ground truth, in the node order
    used by HierarchyTree (family block, genus block, species block)."""

    def __init__(self, ds, tree):
        self.tree = tree
        self.n_family, self.n_genus, self.n_species = ds.n_families, ds.n_genera, ds.n_species
        self.n_nodes = tree.n_nodes
        self.species_to_genus = np.array([
            list(ds.genus_enc.classes_).index(tree.species_to_genus[s]) for s in ds.species_enc.classes_])
        self.species_to_family = np.array([
            list(ds.family_enc.classes_).index(tree.species_to_family[s]) for s in ds.species_enc.classes_])
        self.genus_to_family = np.array([
            list(ds.family_enc.classes_).index(tree.genus_to_family[g]) for g in ds.genus_enc.classes_])

        # One-hot aggregation matrices for marginalising species probabilities
        self.M_genus = np.zeros((self.n_species, self.n_genus), dtype=np.float32)
        self.M_genus[np.arange(self.n_species), self.species_to_genus] = 1.0
        self.M_family = np.zeros((self.n_species, self.n_family), dtype=np.float32)
        self.M_family[np.arange(self.n_species), self.species_to_family] = 1.0

    def node_targets(self, hier_labels):
        """[bsz, 3] (family, genus, species ids) -> [bsz, n_nodes] multi-hot."""
        bsz = hier_labels.shape[0]
        gt = torch.zeros(bsz, self.n_nodes, device=hier_labels.device)
        rows = torch.arange(bsz, device=hier_labels.device)
        gt[rows, hier_labels[:, 0]] = 1.0
        gt[rows, self.n_family + hier_labels[:, 1]] = 1.0
        gt[rows, self.n_family + self.n_genus + hier_labels[:, 2]] = 1.0
        return gt


def make_loss(config, tax):
    if config in ('linear', 'flat_mlp'):
        return lambda out, y: F.cross_entropy(out, y[:, 2])
    if config == 'mlbce':
        return lambda out, y: F.binary_cross_entropy_with_logits(
            out, tax.node_targets(y), reduction='none').sum(dim=1).mean()
    if config == 'focal_mlbce':
        eye = torch.eye(tax.n_nodes, dtype=torch.bool)
        fn = FocalTreeMinLoss(eye, eye, gamma=FTM_GAMMA).to(DEVICE)
    elif config == 'ftm':
        fn = FocalTreeMinLoss(tax.tree.ancestor_matrix(), tax.tree.descendant_matrix(),
                              gamma=FTM_GAMMA).to(DEVICE)
    return lambda out, y: fn(torch.sigmoid(out), tax.node_targets(y))


def predict_levels(config, logits, tax):
    """logits [n, out] -> (family, genus, species) predicted ids."""
    if config in ('linear', 'flat_mlp'):
        p_species = torch.softmax(logits, dim=1).cpu().numpy()
        pred_species = p_species.argmax(axis=1)
        pred_genus = (p_species @ tax.M_genus).argmax(axis=1)
        pred_family = (p_species @ tax.M_family).argmax(axis=1)
    else:
        s = logits.cpu().numpy()  # monotone in sigmoid, argmax unchanged
        f, g = tax.n_family, tax.n_family + tax.n_genus
        pred_family = s[:, :f].argmax(axis=1)
        pred_genus = s[:, f:g].argmax(axis=1)
        pred_species = s[:, g:].argmax(axis=1)
    return pred_family, pred_genus, pred_species


@torch.no_grad()
def forward_all(model, X):
    model.eval()
    return torch.cat([model(X[i:i + 4096].to(DEVICE)) for i in range(0, len(X), 4096)])


def evaluate(config, model, X, hier, tax):
    pf, pg, ps = predict_levels(config, forward_all(model, X), tax)
    y = hier.numpy()
    res = {}
    for name, pred, col in [('family', pf, 0), ('genus', pg, 1), ('species', ps, 2)]:
        res[f'{name}_acc'] = float((pred == y[:, col]).mean() * 100)
        res[f'{name}_macro_f1'] = float(f1_score(y[:, col], pred, average='macro') * 100)

    # Tree consistency: predicted genus is the parent of predicted species, etc.
    consistent = (tax.species_to_genus[ps] == pg) & (tax.genus_to_family[pg] == pf)
    res['consistency'] = float(consistent.mean() * 100)

    preds = ground_truth_singlelabel(np.stack([pf, pg, ps], axis=1),
                                     tax.n_family, tax.n_genus, tax.n_species)
    gt = ground_truth_singlelabel(y, tax.n_family, tax.n_genus, tax.n_species)
    hP, hR, hF1 = hierarchical_f1(preds, gt, tax.tree.ancestor_matrix())
    res.update(hP=float(hP * 100), hR=float(hR * 100), hF1=float(hF1 * 100))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, choices=CONFIGS)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--embedding', default='naturelm', choices=list(EMBEDDINGS))
    args = parser.parse_args()

    out_dir = PROJECT_DIR / 'results/flat_control/anuraset_single' / args.embedding
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.config}_seed{args.seed}.json'

    print(f"config={args.config} seed={args.seed} embedding={args.embedding} device={DEVICE}")
    set_seed(args.seed)

    emb_dir, input_dim = EMBEDDINGS[args.embedding]
    embed_path, meta_path = f'{emb_dir}/embeddings.npy', f'{emb_dir}/embeddings_metadata.csv'
    full_train = AnuraSetEmbeddingDataset(embed_path, meta_path, subset='train', single_label_only=True)
    test_ds = AnuraSetEmbeddingDataset(embed_path, meta_path, subset='test', single_label_only=True,
                                       family_enc=full_train.family_enc,
                                       genus_enc=full_train.genus_enc,
                                       species_enc=full_train.species_enc)
    tree = HierarchyTree(TAXONOMY_PATH, full_train.family_enc, full_train.genus_enc, full_train.species_enc)
    tax = Taxonomy(full_train, tree)

    # Validation split varies with the seed (test set is fixed)
    perm = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(args.seed))
    n_val = int(len(full_train) * VAL_SPLIT)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_loader = DataLoader(Subset(full_train, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True,
                              generator=torch.Generator().manual_seed(args.seed))
    X_val, y_val = full_train.embeddings[val_idx], full_train.hier_labels[val_idx]
    print(f"Train: {len(train_idx)} | Val: {n_val} | Test: {len(test_ds)} | nodes={tax.n_nodes}")

    n_out = tax.n_species if args.config in ('linear', 'flat_mlp') else tax.n_nodes
    model = build_head(args.config, input_dim, n_out).to(DEVICE)
    loss_fn = make_loss(args.config, tax)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Model selection: best validation species accuracy, identical for every config
    best_val, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for emb, hier in train_loader:
            emb, hier = emb.to(DEVICE), hier.to(DEVICE)
            loss = loss_fn(model(emb), hier)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val = evaluate(args.config, model, X_val, y_val, tax)
        if val['species_acc'] > best_val:
            best_val, best_epoch = val['species_acc'], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss={np.mean(losses):.4f} | val species acc={val['species_acc']:.2f}"
                  f" | best={best_val:.2f}@{best_epoch}")

    model.load_state_dict(best_state)
    test = evaluate(args.config, model, test_ds.embeddings, test_ds.hier_labels, tax)
    print("Test:", json.dumps(test, indent=2))

    with open(out_path, 'w') as f:
        json.dump({'config': args.config, 'seed': args.seed, 'embedding': args.embedding,
                   'best_epoch': best_epoch, 'val_species_acc': best_val,
                   'train_time_s': time.time() - t0, 'test': test}, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
