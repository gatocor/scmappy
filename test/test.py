import scmappy
import scanpy as scp
import numpy as np
import scipy.sparse as sp


adata = scp.datasets.pbmc68k_reduced()

adata1 = adata[:350, np.random.choice(adata.n_vars, 500, replace=False)].copy()
adata2 = adata[:350, np.random.choice(adata.n_vars, 500, replace=False)].copy()

# Test numpy array
scmappy.scmap_annotate(adata1, adata2, "louvain")
scmappy.scmap_projection(adata1, adata2, "X_umap")

# Test sparse matrix
adata1.X = sp.csr_matrix(adata1.X)
adata2.X = sp.csr_matrix(adata2.X)

scmappy.scmap_annotate(adata1, adata2, "louvain")
scmappy.scmap_projection(adata1, adata2, "X_umap")
