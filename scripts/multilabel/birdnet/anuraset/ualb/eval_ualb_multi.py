"""
WP4 Evaluation - Method 1: UALB, multi-label AnuraSet

Same gap + macro F1 pipeline as the baseline multi-label eval, run twice:
once on raw NatureLM embeddings (baseline), once on UALB-projected
embeddings, so the two are directly comparable.
"""

import sys
sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from multilabel.NatureLM.anuraset.anuraset_dataset import AnuraSetMultiLabelDataset
from models.projection_head import ProjectionHead

# ── Paths ──────────────────────────────────────────────────────────────────
EMBED_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy'
META_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
MODEL_PATH = '/user/gaad2403/Bachelor-Thesis/results/multilabel/NatureLM/anuraset/wp4_ualb_multilabel/best_model.pt'
OUTPUT_DIR = Path('/user/gaad2403/Bachelor-Thesis/results/multilabel/NatureLM/anuraset/wp4_ualb_multilabel')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROBE_EPOCHS = 300
PROBE_LR = 0.01
N_INTER_MULT = 5
THRESHOLD = 0.5
SEED = 42

LEVELS = ['Family', 'Genus', 'Species']


def load_datasets():
    train_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='train')
    test_ds = AnuraSetMultiLabelDataset(EMBED_PATH, META_PATH, TAXONOMY_PATH, subset='test',
                                        species_cols=train_ds.species_cols,
                                        genus_cols=train_ds.genus_cols,
                                        family_cols=train_ds.family_cols)
    return train_ds, test_ds


def extract_baseline(ds):
    emb = ds.embeddings.numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.clip(norms, 1e-8, None)


def extract_ualb(ds, model):
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 512):
            z = model(ds.embeddings[i:i + 512].to(DEVICE))
            out.append(z.cpu().numpy())
    return np.vstack(out)  # already L2-normalized by the model's forward()


def compute_gap(embeddings, multihot_labels):
    """Groups samples by exact label combination, same convention as
    compute_gap in the baseline script."""
    np.random.seed(SEED)
    label_strings = [''.join(map(str, row.astype(int))) for row in multihot_labels]
    unique_labels = sorted(set(label_strings))

    intra_sims, inter_sims = [], []
    for lbl in unique_labels:
        idx = np.where(np.array(label_strings) == lbl)[0]
        if len(idx) < 2:
            continue
        sub = embeddings[idx]
        sim = cosine_similarity(sub)
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        intra_sims.extend(sim[mask].tolist())

        other_idx = np.where(np.array(label_strings) != lbl)[0]
        if len(other_idx) == 0:
            continue
        n_sample = min(len(idx) * N_INTER_MULT, len(other_idx))
        sample_other = other_idx[np.random.choice(len(other_idx), n_sample, replace=False)]
        inter_sims.extend(cosine_similarity(sub, embeddings[sample_other]).flatten().tolist())

    intra, inter = np.mean(intra_sims), np.mean(inter_sims)
    return intra, inter, intra - inter


def multilabel_probe(train_emb, train_labels, test_emb, test_labels, n_classes):
    torch.manual_seed(SEED)
    clf = nn.Linear(train_emb.shape[1], n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=PROBE_LR, weight_decay=1e-4)

    pos_count = train_labels.sum(axis=0)
    neg_count = train_labels.shape[0] - pos_count
    pos_weight = torch.FloatTensor(neg_count / np.clip(pos_count, 1, None)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train = torch.FloatTensor(train_emb).to(DEVICE)
    y_train = torch.FloatTensor(train_labels).to(DEVICE)
    X_test = torch.FloatTensor(test_emb).to(DEVICE)
    y_test = torch.FloatTensor(test_labels).to(DEVICE)

    clf.train()
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad()
        loss = criterion(clf(X_train), y_train)
        loss.backward()
        optimizer.step()

    clf.eval()
    with torch.no_grad():
        probs = torch.sigmoid(clf(X_test))
        preds = (probs > THRESHOLD).float().cpu().numpy()

    y_true = y_test.cpu().numpy()
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, preds, average='micro', zero_division=0)
    return macro_f1, micro_f1


def evaluate(name, train_emb, test_emb, level_labels_train, level_labels_test):
    print(f"\n--- {name} ---")
    results = {}
    for level in LEVELS:
        train_labels = level_labels_train[level]
        test_labels = level_labels_test[level]
        n_classes = train_labels.shape[1]

        intra, inter, gap = compute_gap(test_emb, test_labels)
        macro_f1, micro_f1 = multilabel_probe(train_emb, train_labels, test_emb, test_labels, n_classes)

        results[level] = {'gap': gap, 'macro_f1': macro_f1, 'micro_f1': micro_f1}
        print(f"  [{level:<8}] Gap: {gap:.4f} | Macro F1: {macro_f1 * 100:.2f}% | Micro F1: {micro_f1 * 100:.2f}%")
    return results


def main():
    print("Loading datasets...")
    train_ds, test_ds = load_datasets()

    level_labels_train = {
        'Family': train_ds.family_labels.numpy(),
        'Genus': train_ds.genus_labels.numpy(),
        'Species': train_ds.species_labels.numpy(),
    }
    level_labels_test = {
        'Family': test_ds.family_labels.numpy(),
        'Genus': test_ds.genus_labels.numpy(),
        'Species': test_ds.species_labels.numpy(),
    }

    print("\nExtracting baseline embeddings...")
    baseline_train_emb = extract_baseline(train_ds)
    baseline_test_emb = extract_baseline(test_ds)

    print("Loading UALB model...")
    model = ProjectionHead(input_dim=4096, proj_dim=128).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")

    print("Extracting UALB embeddings...")
    ualb_train_emb = extract_ualb(train_ds, model)
    ualb_test_emb = extract_ualb(test_ds, model)

    baseline_results = evaluate('Baseline', baseline_train_emb, baseline_test_emb,
                                level_labels_train, level_labels_test)
    ualb_results = evaluate('UALB', ualb_train_emb, ualb_test_emb,
                            level_labels_train, level_labels_test)

    print("\n=== Summary ===")
    print(f"{'Level':<10} {'Baseline Gap':>13} {'UALB Gap':>10} | "
          f"{'Baseline MacroF1':>17} {'UALB MacroF1':>13}")
    for level in LEVELS:
        b, u = baseline_results[level], ualb_results[level]
        print(f"{level:<10} {b['gap']:>13.4f} {u['gap']:>10.4f} | "
              f"{b['macro_f1'] * 100:>16.2f}% {u['macro_f1'] * 100:>12.2f}%")

    np.save(OUTPUT_DIR / 'wp4_ualb_multilabel_results.npy',
            {'baseline': baseline_results, 'ualb': ualb_results})
    print(f"\nSaved to: {OUTPUT_DIR}/wp4_ualb_multilabel_results.npy")


if __name__ == '__main__':
    main()