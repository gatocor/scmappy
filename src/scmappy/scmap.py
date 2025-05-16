from distutils.log import error
import numpy as np
import pandas as pd
import scanpy as scp
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression

def common_genes(adata1,adata2,key_genes=None,remove_unmached=False,add_key="Common_genes",inplace=True):
    """
    def common_genes(adata1,adata2,key_genes,remove_unmached=False,add_key="Common_genes")
    
    Function that takes two Annotated Data objects and find the common genes been expressed in both datasets.

    Args:

    adata1: Annotate data to find common genes.
    adata2: Annotate data to find common genes.
    key_genes: Key in .var from both datasets to check if they have common genes.
    remove_unmached=False: If True, remove genes from both datasets that are unmatched. If False, mark genes unmatched with a new key in .var (add_key).
    add_key="Common_genes": Key to annotate the data.

    Returns:

    Annotated data objects 1 and 2 with the unmatched genes removed or marked.
    """

    if key_genes is None:
        genes1 = adata1.var_names
        genes1_set = set(adata1.var_names)
        genes2 = adata2.var_names
        genes2_set = set(adata2.var_names)
    else:
        genes1 = adata1.var[key_genes]
        genes1_set = set(adata1.var[key_genes])
        genes2 = adata2.var[key_genes]
        genes2_set = set(adata2.var[key_genes])

    if len(genes1) != len(genes1_set):
        raise ValueError("There are duplicated genes in the target dataset. Please check the adata.var_names or adata.var[key_genes] to make sure there names are unique.")
    if len(genes2) != len(genes2_set):
        raise ValueError("There are duplicated genes in the reference dataset. Please check the adata.var_names or adata.var[key_genes] to make sure there names are unique.")
    
    if set(genes1).intersection(set(genes2)) == set():
        raise ValueError("No common genes between target and reference datasets. Please check the adata.var_names or adata.var[key_genes] tomake sure there names are equivalent between both datasets.")
    else:

        if inplace:
            adata1_ = adata1
            adata2_ = adata2
        else:
            adata1_ = adata1.copy()
            adata2_ = adata2.copy()

        #Get common genes
        common_genes = list(genes1_set.intersection(genes2_set))
        common_genes_order = np.argsort(common_genes)

        #Get genes not in common
        keep = keep = [gene in common_genes for gene in genes1_set]
        adata1_.var[add_key] = keep
        keep =  keep = [gene in common_genes for gene in genes2_set]
        adata2_.var[add_key] = keep

        if remove_unmached:
            #Remove unmatched genes
            adata1_ = adata1_[:, common_genes][:, common_genes_order]
            adata2_ = adata2_[:, common_genes][:, common_genes_order]
                
        return adata1_, adata2_

def scmap_annotate(
        target_data, 
        reference_adata, 
        key_annotations,
        key_genes=None, 
        key_layer=None,
        algorithm_flavor="centroid",
        gene_selection_flavor="HVGs",
        n_genes_selected=1000,
        knn=3,
        metrics=["cosine","correlation"],
        similarity_threshold=.7,
        inplace=True,
        key_added="scmap_annotation",
        unassigned_label="Unassigned",
        verbose=False):
    """
    def scmap(target_data, 
            reference_adata, 
            key_annotations,
            key_genes=None, 
            algorithm_flavor="centroid",
            gene_selection_flavor="HVGs",
            n_genes_selected=1000,
            metrics=["cosine","pearson"],
            similarity_threshold=.7,
            inplace=True,
            key_added="scmap_annotation",
            unassigned_label="Unassigned",
            verbose=False)
    Function that implements the SCMAP algorithm from Kilesev et al. (2018) to annotate cells.
    Arguments:
     - target_data: AnnData object to be annotated. 
     - reference_adata: AnnData object to use as reference for annotation.
     - key_annotations: Key to be found in `reference_data.obs` with the labels to annotate the target data.
     - key_genes: Key to be found in `target_data.var` and `reference_data.var` for finding the common genes between both populations. 
     - key_layer: Key to be found in `target_data.layer`. If None, use the `adata.X` matrix. 
     - algorithm_flavor="centroid": Flavor of the algorithm to be chosen between "centroid" or "cells".
     - gene_selection_flavor="HVGs": Method to chose the gene subset to use for annotation. To choose between "HVGs" (high varying genes), "random" or "dropout".
     - n_genes_selected=1000: Number of genes to select for matching.
     - metrics=["cosine","pearson","spearman"]: Metrics to perform the projection. Use normalized metrics for a proper working of the algorithm.
     - similarity_threshold=.7: Proportion of consensus between neighbors of a cell to be annotated (a number between 0 and 1). Cells with a proportion of neighbors with the same fate lower than the consensus are annotated as "Unassigned".
     - inplace=True: If True, add the cells assigned to `target_data.obs` with label `key_added`.
     - key_added="scmap_annotation": If inplce is True, key added to `target_data.obs` with the Annotations.
     - unassigned_label="Unassigned": String added to this cells that do not fill the criteria of assignation.
     - verbose=False: If to print information of the steps of the algorithm.
    """

    target_data_, reference_adata_ = common_genes(target_data,reference_adata,key_genes=key_genes,remove_unmached=True,add_key="Common_genes",inplace=False)

    #Get genes to use for mapping
    if gene_selection_flavor == "HVGs":

        gene_subset = reference_adata_.var["highly_variable"].values

    elif gene_selection_flavor == "random":

        gene_subset = np.random.choice(list(range(0,reference_adata_.shape[1])),size=n_genes_selected,replace=False)

    elif gene_selection_flavor == "dropout":

        # Remove log transformation if necessary
        X = reference_adata_.X.copy()
        if 'log1p' in reference_adata_.uns.keys():
            X = np.expm1(X)

        # Make log(mean)
        m = np.log(np.mean(X,axis=1))

        # Make log(dropout fraction)
        d = np.log( (X.shape[1] - np.array((X != 0).sum(axis=1))[:,0]) / X.shape[1] )

        # Fit linear model and get residuals
        model = LinearRegression()
        model.fit(m.reshape(-1,1),d)
        res = (d - model.predict(m.reshape(-1,1)))**2

        # Get gene subset based on residuals

        gene_subset = np.argsort(-res)[:n_genes_selected]

    else:

        raise ValueError("gene_selection_flavor has to chosen between `HVGs`, `random` or `dropout`.")

    #Get matrices with selected genes
    if key_layer is None:
        X_tar = target_data_.X[:, gene_subset]
    else:
        X_tar = target_data_.layers[key_layer][:, gene_subset]

    if key_layer is None:
        X_ref = reference_adata_.X[:, gene_subset]
    else:
        X_ref = reference_adata_.layers[key_layer][:, gene_subset]

    #Convert to array if sparse
    if sp.sparse.issparse(X_tar):
        X_tar = X_tar.toarray()
    else:
        X_tar = np.asarray(X_tar)

    if sp.sparse.issparse(X_ref):
        X_ref = X_ref.toarray()
    else:
        X_ref = np.asarray(X_ref) 

    # Make knn classification
    if algorithm_flavor == "centroid":

        #Define the labels of cells and centroids
        labels_ref_cells = reference_adata_.obs[key_annotations].values.astype(str)
        labels_ref = pd.unique(labels_ref_cells)

        #Make centroids based on the median expression
        X_ref_centroids = np.zeros([len(labels_ref), X_ref.shape[1]])
        for i,label in enumerate(labels_ref):
            if X_ref[labels_ref_cells == label,:].shape[0] != 0:
                X_ref_centroids[i,:] = X_ref[labels_ref_cells == label,:].mean(axis=0)

        #Assign to the target
        X_ref = X_ref_centroids

        #Clasify cells
        fates = pd.DataFrame()
        distances = pd.DataFrame()
        for i,metric in enumerate(metrics):
            model = KNeighborsClassifier(n_neighbors=knn,metric=metric)
            model.fit(X_ref,labels_ref)
            labels_tar = model.predict(X_tar)
            distances_tar = np.max(model.kneighbors(X_tar)[1],axis=1)
            fates[metric] = labels_tar
            distances[metric] = distances_tar

        #Annotate fates
        annotation = fates.mode(axis=1).iloc[:,0]

        #Unassign fates without consensus between metrics
        annotation[np.invert(pd.isna(fates.mode(axis=1).iloc[:,-1].values))] = unassigned_label

        #Unassign fates with similarity below threshold
        annotation[distances.max(axis=1) < similarity_threshold] = unassigned_label

        annotation = annotation.values

    elif algorithm_flavor == "cell":

        #Define the labels
        labels_ref = reference_adata_.obs[key_annotations].values.astype(str)

        #Create KNN classifier model
        model = KNeighborsClassifier(n_neighbors=knn,metric=metrics[0])
        #Fit model
        model.fit(X_ref,labels_ref)
        #
        labels_tar = model.predict(X_tar)
        probabilities = model.predict_proba(X_tar)
        distances_tar = np.max(model.kneighbors(X_tar)[1],axis=1)

        #Annotate fates
        annotation = labels_tar

        #Unassign fates without consensus between metrics
        annotation[np.max(probabilities,axis=1) < similarity_threshold] = unassigned_label

        #Unassign fates with similarity below threshold
        annotation[distances_tar < similarity_threshold] = unassigned_label

    else:

        raise ValueError("algorithm_flavor has to chosen between `centroid` or `cell`.")

    # Return
    if inplace:

        target_data.obs[key_added] = annotation
        return

    else:

        return annotation
        
def scmap_projection(target_data, 
        reference_adata, 
        key_projection,
        key_genes = None, 
        key_layer=None,
        gene_selection_flavor="HVGs",
        n_genes_selected=2000,
        knn=1,
        metric="cosine",
        inplace=True,
        key_added="X_scmap_projection",
        unassigned_label="Unassigned",
        verbose=False):
    """
    def scmap(target_data, 
            reference_adata, 
            key_projection,
            key_genes=None, 
            key_layer=None,
            gene_selection_flavor="HVGs",
            n_genes_selected=2000,
            knn=1,
            metrics=["cosine","pearson"],
            inplace=True,
            key_added="X_scmap_projection",
            unassigned_label="Unassigned",
            verbose=False)

    Function that projects the SCMAP algorithm from Kilesev et al. (2018) to a reduced subspace of cells. Basically implements the KNeighborsRegressor.
    
    **Arguments**:
     - **target_data**: AnnData object to be annotated. 
     - **reference_adata**: AnnData object to use as reference for annotation.
     - **key_projection**: Key to be found in `reference_data.obs` with the labels to annotate the target data.
     - **key_genes**: Key to be found in `target_data.var` and `reference_data.var` for finding the common genes between both populations. 
     - key_layer: Key to be found in `target_data.layer`. If None, use the `adata.X` matrix. 
     - **algorithm_flavor="centroid"**: Flavor of the algorithm to be chosen between "centroid" or "cells".
     - **gene_selection_flavor="HVGs"**: Method to chose the gene subset to use for annotation. To choose between "HVGs" (high varying genes), "random" or "dropout".
     - **n_genes_selected=1000**: Number of genes to select for matching.
     - **metrics=["cosine","pearson","spearman"],**: Metrics to perform the projection. Use normalized metrics for a proper working of the algorithm.
     - **inplace=True**: If True, add the cells assigned to `target_data.obs` with label `key_added`.
     - **key_added="X_scmap_projection"**: If inplce is True, key added to `target_data.obs` with the Annotations.
     - **unassigned_label="Unassigned"**: String added to this cells that do not fill the criteria of assignation.
     - **verbose=False**: If to print information of the steps of the algorithm.
    """
    
    target_data_, reference_adata_ = common_genes(target_data,reference_adata,key_genes=key_genes,remove_unmached=True,add_key="Common_genes",inplace=False)

    #Get genes to use for mapping
    if gene_selection_flavor == "HVGs":

        gene_subset = reference_adata_.var["highly_variable"].values

    elif gene_selection_flavor == "random":

        gene_subset = np.random.choice(list(range(0,reference_adata_.shape[1])),size=n_genes_selected,replace=False)

    elif gene_selection_flavor == "dropout":

        # Remove log transformation if necessary
        X = reference_adata_.X.copy()
        if 'log1p' in reference_adata_.uns.keys():
            X = np.expm1(X)

        # Make log(mean)
        m = np.log(np.mean(X,axis=1))

        # Make log(dropout fraction)
        d = np.log( (X.shape[1] - np.array((X != 0).sum(axis=1))[:,0]) / X.shape[1] )

        # Fit linear model and get residuals
        model = LinearRegression()
        model.fit(m.reshape(-1,1),d)
        res = (d - model.predict(m.reshape(-1,1)))**2

        # Get gene subset based on residuals

        gene_subset = np.argsort(-res)[:n_genes_selected]

    else:

        raise ValueError("gene_selection_flavor has to chosen between `HVGs`, `random` or `dropout`.")

    #Get matrices with selected genes
    if key_layer is None:
        X_tar = target_data_.X[:, gene_subset]
    else:
        X_tar = target_data_.layers[key_layer][:, gene_subset]

    if key_layer is None:
        X_ref = reference_adata_.X[:, gene_subset]
    else:
        X_ref = reference_adata_.layers[key_layer][:, gene_subset]

    #Convert to array if sparse
    if sp.sparse.issparse(X_tar):
        X_tar = X_tar.toarray()
    else:
        X_tar = np.asarray(X_tar)

    if sp.sparse.issparse(X_ref):
        X_ref = X_ref.toarray()
    else:
        X_ref = np.asarray(X_ref) 

    #Create KNN classifier model
    model = KNeighborsRegressor(n_neighbors=knn,metric=metric)
    #Fit model        
    Y = reference_adata_.obsm[key_projection]
    model.fit(X_ref,Y)
    
    Y_projected = model.predict(X_tar)

    # Return
    if inplace:

        target_data.obsm[key_added] = Y_projected
        return

    else:

        return Y_projected