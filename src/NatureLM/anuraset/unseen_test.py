"""
Unseen-species experiment: hold-out selection + dataset wrapper.

Two hold-out tiers, chosen from genera with >=2 species (so removing one
species still leaves the genus/family representable):

  - Near-OOD: hold out 1 species per qualifying genus. Genus/family remain
    fully trained; only that exact species is unseen.
  - Far-OOD:  hold out one entire genus (all its species) per family that
    has >=2 genera. Family remains trained; genus and species are unseen.

A species can only be in one tier. Run once, save the split, and reuse it
for every model (Baseline/UALB/FTM-only/TT-only/FTM+TT) so results are
comparable.
"""

import json
import random
import pandas as pd
from pathlib import Path

TAXONOMY_PATH = '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv'
META_PATH = '/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv'
OUT_PATH = Path('/user/gaad2403/Bachelor-Thesis/data/unseen_species_split.json')
SEED = 42


def build_split():
    tax = pd.read_csv(TAXONOMY_PATH)  # columns: code, genus, species, family, order
    rng = random.Random(SEED)

    # IMPORTANT: eligibility must be based on species that actually have
    # single-label samples in the training metadata, not just species that
    # exist in the taxonomy table. Some species (e.g. ELAMAT) have zero
    # single-label samples (species_number==1) even though they're a valid
    # taxonomy entry, so counting from taxonomy alone can pick a hold-out
    # that wipes out an entire genus/family from the actual training set.
    meta = pd.read_csv(META_PATH)
    species_cols_meta = [c for c in meta.columns
                         if c not in ['sample_name', 'family', 'genus', 'species', 'subset']]
    species_number = meta[species_cols_meta].sum(axis=1)
    single_label_meta = meta[species_number == 1]
    present_species_codes = set()
    for col in species_cols_meta:
        if (single_label_meta[col] == 1).any():
            present_species_codes.add(col)

    code_to_pair = {row['code']: (row['genus'], row['species']) for _, row in tax.iterrows()}
    present_pairs = {code_to_pair[c] for c in present_species_codes if c in code_to_pair}

    genus_to_species_all = tax.groupby('genus')['species'].apply(list).to_dict()
    # restrict to species that are actually present in the single-label set
    genus_to_species = {
        g: [sp for sp in sps if (g, sp) in present_pairs]
        for g, sps in genus_to_species_all.items()
    }
    genus_to_species = {g: sps for g, sps in genus_to_species.items() if sps}  # drop empty genera

    family_to_genus = tax.groupby('family')['genus'].apply(lambda s: sorted(set(s))).to_dict()
    code_lookup = {(row['genus'], row['species']): row['code'] for _, row in tax.iterrows()}

    near_ood_pairs = []      # held-out (genus, species) pairs, genus stays known
    far_ood_genus = []       # held-out genus (all its species), family stays known
    far_ood_pairs = []       # (genus, species) pairs belonging to a far-OOD genus

    # --- Near-OOD: 1 species per genus, only for genera with >=2 species
    #     PRESENT IN THE SINGLE-LABEL TRAINING DATA (not just in taxonomy) ---
    eligible_genera = [g for g, sp in genus_to_species.items() if len(sp) >= 2]
    for g in eligible_genera:
        sp = rng.choice(genus_to_species[g])
        near_ood_pairs.append({'genus': g, 'species': sp, 'code': code_lookup[(g, sp)]})

    # --- Far-OOD: 1 whole genus per family with >=2 genera that HAVE
    #     single-label samples, excluding genera already used for near-OOD ---
    used_genera = set(eligible_genera)
    for fam, genera in family_to_genus.items():
        genera_with_samples = [g for g in genera if g in genus_to_species]
        candidates = [g for g in genera_with_samples if g not in used_genera]
        if len(genera_with_samples) >= 2 and candidates:
            chosen = rng.choice(candidates)
            far_ood_genus.append(chosen)
            for sp in genus_to_species[chosen]:
                far_ood_pairs.append({'genus': chosen, 'species': sp, 'code': code_lookup[(chosen, sp)]})

    split = {
        'near_ood_pairs': near_ood_pairs,
        'far_ood_genus': far_ood_genus,
        'far_ood_pairs': far_ood_pairs,
    }
    all_pairs = near_ood_pairs + far_ood_pairs
    split['all_held_out_pairs'] = all_pairs
    split['all_held_out_codes'] = sorted(p['code'] for p in all_pairs)  # for readability/logging only

    # --- Safety check: make sure no family/genus present in the
    #     single-label training data gets UNINTENTIONALLY wiped out.
    #     Far-OOD genera are SUPPOSED to disappear entirely (that's the
    #     tier's definition), so they're excluded from this check. What we
    #     actually want to catch: (a) a near-OOD pick that accidentally
    #     empties its own genus/family, or (b) a far-OOD pick that empties
    #     the whole family (it shouldn't, since far-OOD only fires when
    #     the family has >=2 genera with samples). ---
    held_out_set = {(p['genus'], p['species']) for p in all_pairs}
    remaining_pairs = present_pairs - held_out_set
    remaining_genera = {g for g, _ in remaining_pairs}
    remaining_families = {fam for fam, genera in family_to_genus.items()
                          if any(g in remaining_genera for g in genera)}
    original_genera = set(genus_to_species.keys())
    original_families = {fam for fam, genera in family_to_genus.items()
                         if any(g in original_genera for g in genera)}

    far_ood_genus_set = set(far_ood_genus)
    unintended_wiped_genera = (original_genera - remaining_genera) - far_ood_genus_set
    wiped_families = original_families - remaining_families

    if unintended_wiped_genera or wiped_families:
        print(f"WARNING: hold-out unintentionally wipes out genera {unintended_wiped_genera} "
              f"and/or families {wiped_families} from single-label training data. "
              f"Re-run with a different SEED or manually adjust the split.")
    else:
        print("Safety check passed: no UNINTENDED genus/family loss "
              "(far-OOD genera are expected to disappear by design).")

    print(f"Near-OOD species ({len(near_ood_pairs)}): {[p['code'] for p in near_ood_pairs]}")
    print(f"Far-OOD genera ({len(far_ood_genus)}): {far_ood_genus}")
    print(f"Far-OOD species ({len(far_ood_pairs)}): {[p['code'] for p in far_ood_pairs]}")
    print(f"Total held-out species: {len(all_pairs)} / {tax['code'].nunique()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"Saved split to {OUT_PATH}")
    return split


if __name__ == '__main__':
    build_split()