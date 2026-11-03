"""One training loop for every method. Only the head and the loss differ.

Hyperparameters are the paper's (SGD lr 1e-3, momentum 0.9, wd 1e-4, cosine schedule,
batch 256, 100 epochs AnuraSet / 450 ESC-50). The checkpoint with the lowest validation
loss of the method's own objective is kept (FTM+TT: validation L_FTM).
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from . import losses as L
from .models import FlatHead, HierarchicalModel, ProjectionHead

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
FTM_GAMMA = 2.0

# method -> how it is evaluated: 'probe' = linear probe on the learned embedding,
# 'nodes' = its own per-node sigmoid scores, 'flat' = its own fine-class logits.
METHODS = {
    'baseline': 'probe',       # frozen embedding, nothing trained
    'hmce': 'probe',
    'tt': 'probe',
    'ftm_tt': 'probe',
    'ftm': 'nodes',
    'flat_linear': 'flat',     # flat controls with the same optimiser / selection
    'flat_mlp': 'flat',
    'mlbce': 'nodes',          # BCE on all nodes, no tree constraint
    'focal_mlbce': 'nodes',    # FTM without the tree-min step
}


def build_model(method, task):
    h = task.hierarchy
    if method == 'hmce':
        return ProjectionHead(task.input_dim)
    if method in ('ftm', 'tt', 'ftm_tt'):
        return HierarchicalModel(task.input_dim, h.n_nodes)
    if method in ('flat_linear', 'flat_mlp'):
        return FlatHead(task.input_dim, h.sizes[2], mlp=method == 'flat_mlp')
    if method in ('mlbce', 'focal_mlbce'):
        return FlatHead(task.input_dim, h.n_nodes, mlp=True)
    raise ValueError(method)


class Objective:
    """loss(model, X, Y, ids, epoch) -> (training loss, dict of parts, selection loss)."""

    def __init__(self, method, task):
        self.method, self.task = method, task
        h = task.hierarchy
        self.fine = h.level_slice(2)
        self.single = task.single
        if method in ('ftm', 'ftm_tt'):
            self.ftm = L.FocalTreeMinLoss(h.ancestor_matrix(), h.descendant_matrix(), FTM_GAMMA).to(DEVICE)
        if method == 'focal_mlbce':
            eye = torch.eye(h.n_nodes, dtype=torch.bool)
            self.ftm = L.FocalTreeMinLoss(eye, eye, FTM_GAMMA).to(DEVICE)
        if method in ('tt', 'ftm_tt'):
            self.tt = (L.TreeTripletLoss() if self.single
                       else L.MultiLabelTreeTripletLoss(h.fine_distance_matrix()).to(DEVICE))
        if method == 'hmce':
            self.hmce = L.HMCELoss() if self.single else L.MultiLabelHMCELoss()
        self.levels = [h.level_slice(l) for l in (2, 1, 0)]  # fine to coarse

    def __call__(self, model, X, Y, ids, epoch):
        m = self.method
        if m == 'hmce':
            z = torch.stack([model(X), model(X + 0.01 * torch.randn_like(X))], dim=1)
            loss = self.hmce(z, ids) if self.single else self.hmce(z, [Y[:, s] for s in self.levels])
            return loss, {'hmce': loss.item()}, loss
        if m in ('ftm', 'tt', 'ftm_tt'):
            z, s = model(X)
            parts, total = {}, 0.0
            if m != 'tt':
                ftm = self.ftm(s, Y)
                parts['ftm'], total = ftm.item(), ftm
            if m != 'ftm':
                tt = self.tt(z, ids) if self.single else self.tt(z, Y[:, self.fine])
                beta = 1.0 if m == 'tt' else L.beta_schedule(epoch, self.task.epochs)
                parts['tt'], parts['beta'] = tt.item(), beta
                total = total + beta * tt
            return total, parts, (tt if m == 'tt' else ftm)
        out = model(X)
        if m in ('flat_linear', 'flat_mlp'):
            loss = (F.cross_entropy(out, ids[:, 2]) if self.single else
                    F.binary_cross_entropy_with_logits(out, Y[:, self.fine], reduction='none').sum(1).mean())
        elif m == 'mlbce':
            loss = F.binary_cross_entropy_with_logits(out, Y, reduction='none').sum(1).mean()
        else:
            loss = self.ftm(torch.sigmoid(out), Y)
        return loss, {m: loss.item()}, loss


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batches(split_idx, batch_size, shuffle, generator=None, drop_last=False):
    idx = split_idx[torch.randperm(len(split_idx), generator=generator).numpy()] if shuffle else split_idx
    stop = len(idx) - (len(idx) % batch_size if drop_last else 0)
    for i in range(0, stop, batch_size):
        yield idx[i:i + batch_size]


def train(method, task, train_idx, val_idx, seed, log_every=10):
    """Returns (model with the best-validation weights, history dict)."""
    set_seed(seed)
    model = build_model(method, task).to(DEVICE)
    objective = Objective(method, task)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=task.epochs)

    tr = task.train
    X, Y = tr.X.to(DEVICE), tr.Y.to(DEVICE)
    ids = tr.ids.to(DEVICE) if tr.ids is not None else None
    take = lambda b: (X[b], Y[b], ids[b] if ids is not None else None)
    gen = torch.Generator().manual_seed(seed)

    history = {'train_loss': [], 'val_loss': [], 'parts': []}
    best = (float('inf'), None, -1)
    t0 = time.time()
    for epoch in range(1, task.epochs + 1):
        model.train()
        losses, parts = [], []
        for b in _batches(train_idx, BATCH_SIZE, True, gen, drop_last=True):
            loss, p, _ = objective(model, *take(b), epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            parts.append(p)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val = float(np.mean([objective(model, *take(b), epoch)[2].item()
                                 for b in _batches(val_idx, BATCH_SIZE, False)]))
        history['train_loss'].append(float(np.mean(losses)))
        history['val_loss'].append(val)
        history['parts'].append({k: float(np.mean([p[k] for p in parts])) for k in parts[0]})
        if val < best[0]:
            best = (val, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
        if epoch % log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{task.epochs} | train={history['train_loss'][-1]:.4f} "
                  f"| val={val:.4f} | best={best[0]:.4f}@{best[2]}")

    model.load_state_dict(best[1])
    history.update(best_epoch=best[2], best_val_loss=best[0], train_time_s=time.time() - t0)
    return model, history
