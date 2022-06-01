# SCMAPpy

Python version of scmap as described from the original paper of [Kilesev et al. (2017)](https://www.nature.com/articles/nmeth.4644)
It integrates with scanpy objects. 

On top of the original algorithm, we implement a few additional functions.

## Usage

The purpose of this package is to map the annotations of a reference single-cell dataset (`reference`) into a target dataset (`target`). The package is implemented to be used with [AnnData objects](https://scanpy.readthedocs.io/en/latest/usage-principles.html#anndata) and [Scanpy environment](https://scanpy.readthedocs.io/en/latest/generated/scanpy.queries.enrich.html#scanpy-queries-enrich).

We implement three functions to do this pipeline of annotation.

 - `common_genes`: Function to find the common genes between both datasets.
 - `scmap`: Function to anotate the target dataset that implements the original scmap function.
 - `scmap_projection`: Function to project the target dataset into a representation of the reference dataset.
## Example

Consider a `reference` and a `target` datasets. The genes are annotated in `.var["Gene_names"]`. The annotations are in `reference.obs["Annotation"]`

The first step for the annotation is to find the common genes between both datasets.

```
reference,target = common_genes(reference,target,"Gene_names",remove_unmached=True)
```

Before proceeding to the mapping, we need to select that are going to be used for the mapping, as using the hole genome will have the curse of dimensionality. There are different flavors for selecting the genes that do not require this step, but the most typical is to use the highly varying genes. For that you have to first run the `scanpy.pp.highly_varying_genes` algorithm in the reference dataset.

```
scanpy.pp.highly_varying_genes(reference)
```

With the common genes detected and the genes on interest selected, we can proceed to map the data.

```
scmap(reference,target,"Gene_names","Annotation",algorithm_flavor="centroid",gene_selection_flavor="HVGs",similarity_threshold=.7, key_added="scmap_annotation")
```

That is it. By default there will be a new column added in `target.obs` with the annotations.

In the case that we would also to visualize where the annotated cells where projected in a representation of the data, we can use the projection function. Consider that we have a UMAP representation in `reference.obsm["X_umap"]`. We would do,

```
scmap_projection(reference,target,"Gene_names","X_umap",algorithm_flavor="centroid",gene_selection_flavor="HVGs",key_added="scmap_annotation")
```

and this will add a `target.obsm["scmap_annotation"]` with the projected cells.