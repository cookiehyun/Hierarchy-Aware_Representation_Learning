"""
Precomputes the species-species tree distance matrix (0/2/4/6, same convention
as AnuraSetHierarchy.get_tree_distance) from the taxonomy table. Built once
and reused across the whole training run, instead of recomputed per batch
like the single-label version did.
"""

import pandas as pd
import torch


def build_species_distance_matrix(taxonomy_path, species_cols):
    """
    Args:
        taxonomy_path: anuraset_taxonomy.csv (code, genus, species, family, order)
        species_cols: ordered list of species codes, e.g. dataset.species_cols
    Returns:
        [n_species, n_species] LongTensor of tree distances.
    """
    tax = pd.read_csv(taxonomy_path)
    code_to_genus = dict(zip(tax['code'], tax['genus']))
    code_to_family = dict(zip(tax['code'], tax['family']))

    n = len(species_cols)
    dist = torch.zeros(n, n, dtype=torch.long)

    for i, code_i in enumerate(species_cols):
        genus_i = code_to_genus.get(code_i)
        family_i = code_to_family.get(code_i)
        for j, code_j in enumerate(species_cols):
            if code_i == code_j:
                dist[i, j] = 0
            elif code_to_genus.get(code_j) == genus_i:
                dist[i, j] = 2
            elif code_to_family.get(code_j) == family_i:
                dist[i, j] = 4
            else:
                dist[i, j] = 6

    return dist


if __name__ == '__main__':
    import sys
    sys.path.append('/fs/dss/home/gaad2403/Bachelor-Thesis/src')
    from multilabel.NatureLM.anuraset.anuraset_dataset import AnuraSetMultiLabelDataset

    ds = AnuraSetMultiLabelDataset(
        embeddings_path='/user/gaad2403/bachelor_thesis/embeddings/embeddings.npy',
        metadata_path='/user/gaad2403/bachelor_thesis/embeddings/embeddings_metadata.csv',
        taxonomy_path='/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv',
        subset='train',
    )
    dist = build_species_distance_matrix(
        '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv', ds.species_cols)
    print(f"distance matrix shape: {dist.shape}")
    print(f"unique distance values: {dist.unique().tolist()}")
    print(f"distance is symmetric: {torch.equal(dist, dist.T)}")
    print(f"diagonal is all zero: {(dist.diagonal() == 0).all().item()}")