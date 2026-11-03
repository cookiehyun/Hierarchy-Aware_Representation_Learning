"""Evaluation (paper Sec. 3.1) for every method, label mode and variant.

Two protocols:
    'paper'  exactly what produced the paper tables: probes trained on the full training
             set, multi-label threshold 0.5, single-label Gap pooled over all pairs,
             everything seeded with 42.
    'v2'     probes trained on the train part of the split only, multi-label thresholds
             tuned per class on the validation part, Gap averaged per class as written in
             the paper's formula, all randomness tied to --seed.

What is evaluated per method (METHODS in train.py):
    'probe'  linear probe on the L2-normalised embedding (baseline: raw backbone embedding)
    'nodes'  the method's own per-node sigmoid scores; Gap/NMI on the trunk output h
    'flat'   the method's own fine-class logits, coarser levels by marginalisation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score, roc_auc_score

from .train import DEVICE, METHODS

PROBE_EPOCHS = 300
PROBE_LR = 0.01
N_INTER_MULT = 5
UNSEEN_PERCENTILE = 95


def l2n(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


# ── Representations ──────────────────────────────────────────────────────
@torch.no_grad()
def model_outputs(method, model, X):
    """-> (emb: L2-normalised representation for Gap / NMI / probes,
           scores: the method's own predictions or None)"""
    if method == 'baseline':
        return l2n(X.numpy()), None
    model.eval()
    embs, scores = [], []
    for i in range(0, len(X), 4096):
        x = X[i:i + 4096].to(DEVICE)
        if method == 'hmce':
            embs.append(model(x))
        elif method in ('tt', 'ftm_tt'):
            embs.append(model(x)[0])
        elif method == 'ftm':
            embs.append(model.encode_hidden(x))
            scores.append(model(x)[1])
        elif method in ('flat_linear', 'flat_mlp'):
            embs.append(model.encode_hidden(x))
            scores.append(model(x))
        else:  # mlbce, focal_mlbce
            embs.append(model.encode_hidden(x))
            scores.append(torch.sigmoid(model(x)))
    emb = l2n(torch.cat(embs).cpu().numpy())
    return emb, (torch.cat(scores).cpu().numpy() if scores else None)


# ── Embedding-space metrics ──────────────────────────────────────────────
def gap_single(emb, labels, seed, per_class_mean):
    """Gap = intra-class minus inter-class cosine similarity. Inter pairs: each class is
    compared with N_INTER_MULT x |class| random other samples.
    per_class_mean=False pools all pairs (paper code); True averages per class (paper Eq.)."""
    np.random.seed(seed)
    intra_all, inter_all, gaps = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        sub = emb[idx]
        sim = sub @ sub.T
        intra = sim[np.triu_indices(len(idx), k=1)]
        other = np.where(labels != c)[0]
        if len(other) == 0:
            continue
        pick = other[np.random.choice(len(other), min(len(idx) * N_INTER_MULT, len(other)), replace=False)]
        inter = (sub @ emb[pick].T).ravel()
        intra_all.append(intra)
        inter_all.append(inter)
        gaps.append(intra.mean() - inter.mean())
    if per_class_mean:
        return float(np.mean(gaps))
    return float(np.concatenate(intra_all).mean() - np.concatenate(inter_all).mean())


def gap_multi(emb, Y_level, seed):
    """One-vs-rest Gap per label (samples containing the label vs. samples without it),
    averaged over labels with >= 2 positives."""
    np.random.seed(seed)
    gaps = []
    for j in range(Y_level.shape[1]):
        pos = np.where(Y_level[:, j] == 1)[0]
        neg = np.where(Y_level[:, j] == 0)[0]
        if len(pos) < 2 or len(neg) == 0:
            continue
        p = emb[pos]
        intra = (p @ p.T)[np.triu_indices(len(pos), k=1)]
        pick = neg[np.random.choice(len(neg), min(len(pos) * N_INTER_MULT, len(neg)), replace=False)]
        gaps.append(intra.mean() - (p @ emb[pick].T).mean())
    return float(np.mean(gaps)) if gaps else float('nan')


def clustering_nmi(emb, labels, n_clusters, seed):
    ids = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(emb)
    return float(normalized_mutual_info_score(labels, ids) * 100)


# ── Linear probes (full batch Adam, 300 epochs) ──────────────────────────
def _fit_linear(X, n_out, loss_fn, seed):
    torch.manual_seed(seed)
    clf = nn.Linear(X.shape[1], n_out).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    for _ in range(PROBE_EPOCHS):
        opt.zero_grad()
        loss_fn(clf(Xt)).backward()
        opt.step()
    clf.eval()
    return lambda Z: clf(torch.as_tensor(Z, dtype=torch.float32, device=DEVICE)).detach().cpu().numpy()


def probe_single_level(X, y, n_classes, seed):
    """Class-balanced cross-entropy probe for one level -> predict(Z) -> logits."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    w = 1.0 / np.clip(counts, 1, None)
    w = torch.tensor(w / w.sum() * n_classes, device=DEVICE)
    yt = torch.as_tensor(y, device=DEVICE)
    return _fit_linear(X, n_classes, lambda out: F.cross_entropy(out, yt, weight=w), seed)


def probe_single_joint(X, ids, hierarchy, seed):
    """One probe over all nodes, summed per-level cross-entropy (used for hF1)."""
    yt = torch.as_tensor(ids, device=DEVICE)
    sl = [hierarchy.level_slice(l) for l in range(3)]
    loss = lambda out: sum(F.cross_entropy(out[:, s], yt[:, l]) for l, s in enumerate(sl))
    return _fit_linear(X, hierarchy.n_nodes, loss, seed)


def probe_multilabel(X, Y, seed):
    """Sigmoid + BCE probe, pos_weight = #neg / #pos per class -> predict(Z) -> logits."""
    pos = Y.sum(axis=0)
    pw = torch.tensor((len(Y) - pos) / np.clip(pos, 1, None), dtype=torch.float32, device=DEVICE)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=DEVICE)
    return _fit_linear(X, Y.shape[1], lambda out: F.binary_cross_entropy_with_logits(out, Yt, pos_weight=pw), seed)


def tune_thresholds(scores, Y, grid=np.linspace(0.05, 0.95, 19)):
    """Per-class threshold maximising F1 on held-out data; 0.5 for classes absent there."""
    th = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        if Y[:, j].sum() == 0:
            continue
        f1 = [f1_score(Y[:, j], scores[:, j] > t, zero_division=0) for t in grid]
        th[j] = grid[int(np.argmax(f1))]
    return th


def hierarchical_prf(pred_nodes, true_nodes, ancestor_matrix):
    """Kiritchenko et al. 2006: add ancestors to predictions, micro-average over all nodes."""
    anc = ancestor_matrix.numpy().astype(np.float32)
    aug = (pred_nodes @ anc > 0).astype(np.float32)
    inter = (aug * true_nodes).sum()
    hP = inter / (aug.sum() + 1e-8)
    hR = inter / (true_nodes.sum() + 1e-8)
    return float(hP * 100), float(hR * 100), float(2 * hP * hR / (hP + hR + 1e-8) * 100)


# ── Prediction helpers ───────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def flat_level_probs(fine_probs, h):
    """Marginalise fine-class probabilities onto the mid and coarse levels."""
    to_mid = np.eye(h.sizes[1])[h.fine_parent]
    to_coarse = np.eye(h.sizes[0])[h.mid_parent]
    mid = fine_probs @ to_mid
    return [mid @ to_coarse, mid, fine_probs]


def node_scores(method, scores, h):
    """The method's own scores as [n, n_nodes] in node order (flat: marginalised softmax)."""
    if METHODS[method] == 'nodes':
        return scores
    return np.concatenate(flat_level_probs(softmax(scores), h), axis=1)


# ── Main entry ───────────────────────────────────────────────────────────
def evaluate(method, model, task, train_idx, val_idx, protocol, seed):
    h = task.hierarchy
    paper = protocol == 'paper'
    s = 42 if paper else seed
    probe_idx = np.arange(len(task.train)) if paper else train_idx

    train_emb, train_scores = model_outputs(method, model, task.train.X)
    test_emb, test_scores = model_outputs(method, model, task.test.X)

    test = task.test
    known = np.ones(len(test), dtype=bool) if test.is_unseen is None else ~test.is_unseen
    res = {}
    if task.single:
        res.update(_eval_single(method, task, train_emb, train_scores, test_emb[known],
                                None if test_scores is None else test_scores[known],
                                test.ids.numpy()[known], probe_idx, paper, s))
    else:
        res.update(_eval_multi(method, task, train_emb, train_scores, test_emb, test_scores,
                               probe_idx, val_idx, paper, s))
    if test.is_unseen is not None:
        res['unseen'] = _eval_unseen(method, task, train_emb, train_scores, test_emb, test_scores,
                                     train_idx, val_idx, paper)
    return res


def _eval_single(method, task, train_emb, train_scores, test_emb, test_scores, test_ids, probe_idx, paper, seed):
    h = task.hierarchy
    train_ids = task.train.ids.numpy()
    kind = METHODS[method]
    res = {'levels': {}}
    if kind == 'probe':
        Xp, idp = train_emb[probe_idx], train_ids[probe_idx]
        pred = np.stack([probe_single_level(Xp, idp[:, l], h.sizes[l], seed)(test_emb).argmax(1)
                         for l in range(3)], axis=1)
        joint = probe_single_joint(Xp, idp, h, seed)(test_emb)
        joint_pred = np.stack([joint[:, h.level_slice(l)].argmax(1) for l in range(3)], axis=1)
    else:
        sc = node_scores(method, test_scores, h)
        pred = joint_pred = np.stack([sc[:, h.level_slice(l)].argmax(1) for l in range(3)], axis=1)

    for l, name in enumerate(h.level_names):
        y = test_ids[:, l]
        r = {'acc': float((pred[:, l] == y).mean() * 100),
             'macro_f1': float(f1_score(y, pred[:, l], average='macro') * 100)}
        if method != 'flat_linear':
            r['gap'] = gap_single(test_emb, y, seed, per_class_mean=not paper)
            r['nmi'] = clustering_nmi(test_emb, y, h.sizes[l], seed)
        res['levels'][name] = r
    hP, hR, hF1 = hierarchical_prf(h.node_targets_from_ids(joint_pred), h.node_targets_from_ids(test_ids),
                                   h.ancestor_matrix())
    res.update(hP=hP, hR=hR, hF1=hF1, consistency=float(h.is_consistent(joint_pred).mean() * 100))
    return res


def _eval_multi(method, task, train_emb, train_scores, test_emb, test_scores, probe_idx, val_idx, paper, seed):
    h = task.hierarchy
    Ytr, Yte = task.train.Y.numpy(), task.test.Y.numpy()
    kind = METHODS[method]
    sig = lambda z: 1 / (1 + np.exp(-z))

    if kind == 'probe':
        Xp = train_emb[probe_idx]
        level_probes = [probe_multilabel(Xp, Ytr[probe_idx][:, h.level_slice(l)], seed) for l in range(3)]
        joint = probe_multilabel(Xp, Ytr[probe_idx], seed)
        level_scores = lambda E: [sig(p(E)) for p in level_probes]
        joint_scores = lambda E: sig(joint(E))
    else:
        own = (lambda sc: sc) if kind == 'nodes' else (lambda sc: _flat_multi_nodes(sc, h))
        level_scores = lambda sc: [own(sc)[:, h.level_slice(l)] for l in range(3)]
        joint_scores = own

    if kind == 'probe':
        test_level, test_joint = level_scores(test_emb), joint_scores(test_emb)
        val_level = level_scores(train_emb[val_idx]) if not paper else None
        val_joint = joint_scores(train_emb[val_idx]) if not paper else None
    else:
        test_level, test_joint = level_scores(test_scores), joint_scores(test_scores)
        val_level = level_scores(train_scores[val_idx]) if not paper else None
        val_joint = joint_scores(train_scores[val_idx]) if not paper else None

    res = {'levels': {}}
    for l, name in enumerate(h.level_names):
        y = Yte[:, h.level_slice(l)]
        th = 0.5 if paper else tune_thresholds(val_level[l], Ytr[val_idx][:, h.level_slice(l)])
        p = (test_level[l] > th).astype(np.float32)
        r = {'macro_f1': float(f1_score(y, p, average='macro', zero_division=0) * 100),
             'micro_f1': float(f1_score(y, p, average='micro', zero_division=0) * 100)}
        if method != 'flat_linear':
            r['gap'] = gap_multi(test_emb, y, seed)
        res['levels'][name] = r
    th = 0.5 if paper else tune_thresholds(val_joint, Ytr[val_idx])
    hP, hR, hF1 = hierarchical_prf((test_joint > th).astype(np.float32), Yte, h.ancestor_matrix())
    res.update(hP=hP, hR=hR, hF1=hF1)
    return res


def _flat_multi_nodes(logits, h):
    """Flat multi-label head (sigmoid per fine class) -> node scores: a mid / coarse score is
    the maximum of its children's scores."""
    fine = 1 / (1 + np.exp(-logits))
    mid = np.stack([fine[:, h.fine_parent == m].max(axis=1) for m in range(h.sizes[1])], axis=1)
    coarse = np.stack([mid[:, h.mid_parent == c].max(axis=1) for c in range(h.sizes[0])], axis=1)
    return np.concatenate([coarse, mid, fine], axis=1)


def _eval_unseen(method, task, train_emb, train_scores, test_emb, test_scores, train_idx, val_idx, paper):
    """Open-set detection of held-out classes. Anomaly score = 1 - max fine-class score
    (nodes / flat methods) or 1 - max cosine similarity to a training fine-class centroid
    (probe methods). Threshold: 95th percentile of held-in anomaly scores - on the test set
    in the paper code, on the validation split in v2."""
    h, test = task.hierarchy, task.test
    train_ids = task.train.ids.numpy()
    if METHODS[method] == 'probe':
        def centroid_scores(E, level, ref=train_idx):
            C = np.stack([train_emb[ref][train_ids[ref, level] == c].mean(0) for c in range(h.sizes[level])])
            return E @ l2n(C).T
        anomaly = lambda E: 1 - centroid_scores(E, 2).max(axis=1)
        test_anom = anomaly(test_emb)
        val_anom = anomaly(train_emb[val_idx])
        coarse_pred = centroid_scores(test_emb, 0).argmax(axis=1)
    else:
        nodes = lambda sc: node_scores(method, sc, h)
        test_nodes = nodes(test_scores)
        test_anom = 1 - test_nodes[:, h.level_slice(2)].max(axis=1)
        val_anom = 1 - nodes(train_scores[val_idx])[:, h.level_slice(2)].max(axis=1)
        coarse_pred = test_nodes[:, h.level_slice(0)].argmax(axis=1)

    ref = test_anom[~test.is_unseen] if paper else val_anom
    threshold = np.percentile(ref, UNSEEN_PERCENTILE)
    flagged = test_anom > threshold
    coarse_true = test.ids.numpy()[:, 0]

    out = {}
    for tier, positives in (('near', test.is_near), ('far', test.is_far)):
        mask = positives | ~test.is_unseen
        y = positives[mask]
        r = {'auroc': float(roc_auc_score(y, test_anom[mask])),
             'recall_unseen': float(flagged[mask][y].mean()),
             'false_reject_rate': float(flagged[mask][~y].mean())}
        caught = flagged & positives
        if caught.any():
            r[f'{h.level_names[0]}_acc_given_flagged'] = float((coarse_pred[caught] == coarse_true[caught]).mean() * 100)
        out[tier] = r
    return out
