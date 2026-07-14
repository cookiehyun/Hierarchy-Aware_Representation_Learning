"""
Multi-label Hierarchy Tree.

Same V = Family U Genus U Species node space and same ancestor/descendant
matrix definitions as the single-label HierarchyTree, but built directly
from the dataset's species_cols/genus_cols/family_cols (species names are
6-letter codes here, not epithets, so the single-label version's taxonomy
lookups by epithet don't apply).

Ground truth for FocalTreeMinLoss doesn't need any lookup here -- it's just
family_multihot, genus_multihot and species_multihot concatenated in the
same node order used to build the ancestor/descendant matrices.
"""

import pandas as pd
import torch


class MultiLabelHierarchyTree:
    def __init__(self, taxonomy_path, species_cols, genus_cols, family_cols):
        tax = pd.read_csv(taxonomy_path)
        code_to_genus = dict(zip(tax['code'], tax['genus']))
        code_to_family = dict(zip(tax['code'], tax['family']))

        self.species_cols = species_cols
        self.genus_cols = genus_cols
        self.family_cols = family_cols

        self.n_family = len(family_cols)
        self.n_genus = len(genus_cols)
        self.n_species = len(species_cols)
        self.n_nodes = self.n_family + self.n_genus + self.n_species

        # node order: [family...] + [genus...] + [species...], must match
        # how ground truth vectors get concatenated at training time
        self.family_offset = 0
        self.genus_offset = self.n_family
        self.species_offset = self.n_family + self.n_genus

        self.ancestors = [set() for _ in range(self.n_nodes)]
        self.descendants = [set() for _ in range(self.n_nodes)]
        self._build(code_to_genus, code_to_family)

    def _build(self, code_to_genus, code_to_family):
        n = self.n_nodes
        for v in range(n):
            self.ancestors[v].add(v)
            self.descendants[v].add(v)

        for i, code in enumerate(self.species_cols):
            s_idx = self.species_offset + i
            genus_name = code_to_genus.get(code)
            family_name = code_to_family.get(code)

            if genus_name in self.genus_cols:
                g_idx = self.genus_offset + self.genus_cols.index(genus_name)
                self.ancestors[s_idx].add(g_idx)
                self.descendants[g_idx].add(s_idx)

            if family_name in self.family_cols:
                f_idx = self.family_offset + self.family_cols.index(family_name)
                self.ancestors[s_idx].add(f_idx)
                self.descendants[f_idx].add(s_idx)

        # genus -> family, using any species in that genus to find its family
        code_to_family_by_genus = {}
        for code, genus_name in code_to_genus.items():
            if genus_name not in code_to_family_by_genus:
                code_to_family_by_genus[genus_name] = code_to_family.get(code)

        for j, genus_name in enumerate(self.genus_cols):
            g_idx = self.genus_offset + j
            family_name = code_to_family_by_genus.get(genus_name)
            if family_name in self.family_cols:
                f_idx = self.family_offset + self.family_cols.index(family_name)
                self.ancestors[g_idx].add(f_idx)
                self.descendants[f_idx].add(g_idx)

    def ground_truth(self, family_multihot, genus_multihot, species_multihot):
        """Concatenates the three multi-hot tensors in the node order used
        to build the ancestor/descendant matrices. Shapes: [bsz, n_family],
        [bsz, n_genus], [bsz, n_species] -> [bsz, n_nodes]."""
        return torch.cat([family_multihot, genus_multihot, species_multihot], dim=1)

    def ancestor_matrix(self):
        n = self.n_nodes
        M = torch.zeros(n, n, dtype=torch.bool)
        for v in range(n):
            for u in self.ancestors[v]:
                M[v, u] = True
        return M

    def descendant_matrix(self):
        n = self.n_nodes
        M = torch.zeros(n, n, dtype=torch.bool)
        for v in range(n):
            for u in self.descendants[v]:
                M[v, u] = True
        return M


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
    tree = MultiLabelHierarchyTree(
        '/user/gaad2403/Bachelor-Thesis/data/anuraset_taxonomy.csv',
        ds.species_cols, ds.genus_cols, ds.family_cols,
    )
    print(f"n_nodes = {tree.n_nodes} (family={tree.n_family}, genus={tree.n_genus}, species={tree.n_species})")

    anc = tree.ancestor_matrix()
    desc = tree.descendant_matrix()
    print(f"ancestor_matrix shape: {anc.shape}, descendant_matrix shape: {desc.shape}")
    print(f"ancestor/descendant are transposes of each other: {torch.equal(anc, desc.T)}")

    # every species should have exactly 3 ancestors: itself, its genus, its family
    n_ancestors_per_species = anc[tree.species_offset:].sum(dim=1)
    print(f"species with exactly 3 ancestors: {(n_ancestors_per_species == 3).sum().item()} / {tree.n_species}")

    gt = tree.ground_truth(ds.family_labels[:4], ds.genus_labels[:4], ds.species_labels[:4])
    print(f"ground_truth shape for 4 samples: {gt.shape}")