"""
Unseen-category experiment (ESC-50): hold-out selection.

Same two-tier design as AnuraSet's unseen_species_split.py:
  - Near-OOD: hold out 1 category per midcat with >=2 categories.
    Midcat stays known, only that category is unseen.
  - Far-OOD: hold out 1 whole midcat (all its categories) per BGA with
    >=2 midcats. BGA stays known, midcat+category are unseen.

Built from the actual TRAIN folds (1-4) metadata -- not just the raw
category list -- so a hold-out choice can't (by bad luck) wipe out an
entire BGA/midcat that had very few categories/samples in the train folds.
A safety check flags this if it happens anyway.
"""

import json
import random
import pandas as pd
from pathlib import Path

META_PATH = '/user/gaad2403/Bachelor-Thesis/data/esc50_bga_labels.csv'
OUT_PATH = Path('/user/gaad2403/Bachelor-Thesis/data/unseen_category_split.json')
TRAIN_FOLDS = [1, 2, 3, 4]
SEED = 42


def build_split():
    meta = pd.read_csv(META_PATH)
    train_meta = meta[meta['fold'].isin(TRAIN_FOLDS)]
    rng = random.Random(SEED)

    midcat_to_categories = train_meta.groupby('midcat')['category'].apply(lambda s: sorted(set(s))).to_dict()
    bga_to_midcats = train_meta.groupby('bga_label')['midcat'].apply(lambda s: sorted(set(s))).to_dict()
    category_to_bga = dict(zip(meta['category'], meta['bga_label']))
    category_to_midcat = dict(zip(meta['category'], meta['midcat']))

    near_ood_categories = []
    far_ood_midcats = []
    far_ood_categories = []

    # --- Near-OOD: 1 category per midcat with >=2 categories present in train ---
    eligible_midcats = [m for m, cats in midcat_to_categories.items() if len(cats) >= 2]
    for m in eligible_midcats:
        cat = rng.choice(midcat_to_categories[m])
        near_ood_categories.append(cat)

    # --- Far-OOD: 1 whole midcat per BGA with >=2 midcats present in train,
    #     excluding midcats already used for near-OOD ---
    used_midcats = set(eligible_midcats)
    for bga, midcats in bga_to_midcats.items():
        candidates = [m for m in midcats if m not in used_midcats]
        if len(midcats) >= 2 and candidates:
            chosen = rng.choice(candidates)
            far_ood_midcats.append(chosen)
            far_ood_categories.extend(midcat_to_categories[chosen])

    split = {
        'near_ood_categories': sorted(near_ood_categories),
        'far_ood_midcats': sorted(far_ood_midcats),
        'far_ood_categories': sorted(far_ood_categories),
    }
    all_categories = near_ood_categories + far_ood_categories
    split['all_held_out_categories'] = sorted(all_categories)

    # --- Safety check: make sure no BGA/midcat present in the TRAIN folds
    #     gets UNINTENTIONALLY wiped out. Far-OOD midcats are SUPPOSED to
    #     disappear entirely -- that's the tier's definition. ---
    held_out_set = set(all_categories)
    remaining_categories = set(train_meta['category'].unique()) - held_out_set
    remaining_midcats = {category_to_midcat[c] for c in remaining_categories}
    remaining_bgas = {category_to_bga[c] for c in remaining_categories}

    original_midcats = set(midcat_to_categories.keys())
    original_bgas = set(bga_to_midcats.keys())

    far_ood_midcat_set = set(far_ood_midcats)
    unintended_wiped_midcats = (original_midcats - remaining_midcats) - far_ood_midcat_set
    wiped_bgas = original_bgas - remaining_bgas

    if unintended_wiped_midcats or wiped_bgas:
        print(f"WARNING: hold-out unintentionally wipes out midcats {unintended_wiped_midcats} "
              f"and/or BGAs {wiped_bgas} from train-fold data. "
              f"Re-run with a different SEED or manually adjust the split.")
    else:
        print("Safety check passed: no UNINTENDED midcat/BGA loss "
              "(far-OOD midcats are expected to disappear by design).")

    print(f"Near-OOD categories ({len(near_ood_categories)}): {near_ood_categories}")
    print(f"Far-OOD midcats ({len(far_ood_midcats)}): {far_ood_midcats}")
    print(f"Far-OOD categories ({len(far_ood_categories)}): {far_ood_categories}")
    print(f"Total held-out categories: {len(all_categories)} / {meta['category'].nunique()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"Saved split to {OUT_PATH}")
    return split


if __name__ == '__main__':
    build_split()