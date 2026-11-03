"""Three-level label tree (coarse -> mid -> fine) shared by every dataset.

AnuraSet: family -> genus -> species.  ESC-50: BGA -> mid-category -> category.

Node space V = [coarse classes] + [mid classes] + [fine classes], in that order.
This is the order of the FTM score vector, the node-wise ground truth and the
ancestor matrix used for hF1.
"""

import numpy as np
import torch


class Hierarchy:
    def __init__(self, level_names, classes, fine_to_mid, mid_to_coarse):
        """
        Args:
            level_names: e.g. ('family', 'genus', 'species'), coarse to fine.
            classes: three lists of class names, coarse to fine. Their order defines
                the integer ids at each level.
            fine_to_mid, mid_to_coarse: dicts mapping a class name to its parent's name.
        """
        self.level_names = tuple(level_names)
        self.classes = [list(c) for c in classes]
        self.sizes = [len(c) for c in self.classes]
        self.offsets = [0, self.sizes[0], self.sizes[0] + self.sizes[1]]
        self.n_nodes = sum(self.sizes)

        index = [{name: i for i, name in enumerate(c)} for c in self.classes]
        # parent id of every mid / fine class
        self.mid_parent = np.array([index[0][mid_to_coarse[m]] for m in self.classes[1]])
        self.fine_parent = np.array([index[1][fine_to_mid[f]] for f in self.classes[2]])
        # full path of every fine class: [n_fine, 3] (coarse, mid, fine) ids
        fine_ids = np.arange(self.sizes[2])
        self.fine_paths = np.stack([self.mid_parent[self.fine_parent], self.fine_parent, fine_ids], axis=1)
        self.index = index

    @classmethod
    def from_rows(cls, level_names, rows, classes=None):
        """Build from (coarse, mid, fine) name triples. Class order defaults to sorted
        names per level (= sklearn LabelEncoder order used by the original scripts)."""
        rows = [tuple(r) for r in rows]
        if classes is None:
            classes = [sorted({r[l] for r in rows}) for l in range(3)]
        fine_to_mid = {r[2]: r[1] for r in rows}
        mid_to_coarse = {r[1]: r[0] for r in rows}
        return cls(level_names, classes, fine_to_mid, mid_to_coarse)

    def level_slice(self, level):
        start = self.offsets[level]
        return slice(start, start + self.sizes[level])

    def encode(self, names_per_level):
        """Three sequences of class names -> [n, 3] ids. Unknown names become -1."""
        cols = [np.array([self.index[l].get(n, -1) for n in names]) for l, names in enumerate(names_per_level)]
        return np.stack(cols, axis=1)

    # ── Node-space targets ──────────────────────────────────────────────────
    def node_targets_from_ids(self, ids):
        """[n, 3] ids -> [n, n_nodes] multi-hot (one positive per level; -1 = none)."""
        ids = np.asarray(ids)
        Y = np.zeros((len(ids), self.n_nodes), dtype=np.float32)
        rows = np.arange(len(ids))
        for l in range(3):
            ok = ids[:, l] >= 0
            Y[rows[ok], self.offsets[l] + ids[ok, l]] = 1.0
        return Y

    def node_targets_from_fine_multihot(self, fine_mh):
        """[n, n_fine] multi-hot -> [n, n_nodes]: a sample is positive for a mid / coarse
        class if any of its fine classes belongs to it."""
        fine_mh = np.asarray(fine_mh, dtype=np.float32)
        to_mid = np.zeros((self.sizes[2], self.sizes[1]), dtype=np.float32)
        to_mid[np.arange(self.sizes[2]), self.fine_parent] = 1.0
        to_coarse = np.zeros((self.sizes[1], self.sizes[0]), dtype=np.float32)
        to_coarse[np.arange(self.sizes[1]), self.mid_parent] = 1.0
        mid = (fine_mh @ to_mid > 0).astype(np.float32)
        coarse = (mid @ to_coarse > 0).astype(np.float32)
        return np.concatenate([coarse, mid, fine_mh], axis=1)

    # ── Tree structure ─────────────────────────────────────────────────────
    def ancestor_matrix(self):
        """[n_nodes, n_nodes] bool, A[v, u] = True iff u is an ancestor of v (v included)."""
        A = torch.eye(self.n_nodes, dtype=torch.bool)
        o0, o1, o2 = self.offsets
        for m, c in enumerate(self.mid_parent):
            A[o1 + m, o0 + c] = True
        for f, (c, m, _) in enumerate(self.fine_paths):
            A[o2 + f, o1 + m] = True
            A[o2 + f, o0 + c] = True
        return A

    def descendant_matrix(self):
        return self.ancestor_matrix().T.contiguous()

    def fine_distance_matrix(self):
        """[n_fine, n_fine] tree distance psi between fine classes, counted in edges through
        an implicit root: 0 same fine class, 2 same mid, 4 same coarse, 6 otherwise."""
        p = self.fine_paths
        same = [p[:, l][:, None] == p[:, l][None, :] for l in range(3)]
        D = np.full((len(p), len(p)), 6, dtype=np.int64)
        D[same[0]] = 4
        D[same[1]] = 2
        D[same[2]] = 0
        return torch.from_numpy(D)

    def is_consistent(self, pred_ids):
        """[n, 3] predicted ids -> [n] bool: predicted mid is the parent of predicted fine, etc."""
        pred_ids = np.asarray(pred_ids)
        return ((self.fine_parent[pred_ids[:, 2]] == pred_ids[:, 1])
                & (self.mid_parent[pred_ids[:, 1]] == pred_ids[:, 0]))


def pairwise_tree_distance(ids):
    """[bsz, 3] ids -> [bsz, bsz] tree distance (0/2/4/6), computed from the label ids."""
    same = [ids[:, l].unsqueeze(0) == ids[:, l].unsqueeze(1) for l in range(3)]
    D = torch.full(same[0].shape, 6, dtype=torch.long, device=ids.device)
    D[same[0]] = 4
    D[same[1]] = 2
    D[same[2]] = 0
    return D
