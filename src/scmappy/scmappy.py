from distutils.log import error
import numpy as np
import pandas as pd
import scanpy as scp
import scipy as sp
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression

def common_genes(adata1,adata2,key_genes,remove_unmached=False,add_key="Common_genes"):
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
    
    #Common genes
    data1 = adata1.var.copy()
    data2 = adata2.var.copy()
    data1[add_key] = data1[key_genes].isin(data2[key_genes]).values
    data2[add_key] = data2[key_genes].isin(data1[key_genes]).values

    #Sort
    data1.reset_index(inplace=True)
    data1.sort_values([add_key,key_genes],inplace=True)

    data2.reset_index(inplace=True)
    data2.sort_values([add_key,key_genes],inplace=True)
        
    #Eliminate not common genes
    adata1 = adata1[:,data1.index.values].copy()
    adata1.var[add_key] = data1[add_key].values
    
    adata2 = adata2[:,data2.index.values].copy()
    adata2.var[add_key] = data2[add_key].values
    
    if remove_unmached:
        adata1 = adata1[:,adata1.var[add_key]==True]
        adata2 = adata2[:,adata2.var[add_key]==True]

        adata1.var.reset_index(inplace=True)
        adata2.var.reset_index(inplace=True)
        
    return adata1, adata2

def scmap(target_data, 
        reference_adata, 
        key_genes, 
        key_annotations,
        algorithm_flavor="centroid",
        gene_selection_flavor="HVGs",
        n_genes_selected=1000,
        metrics=["cosine","correlation"],
        similarity_threshold=.7,
        inplace=True,
        key_added="scmap_annotation",
        unassigned_label="Unassigned",
        verbose=False):
    """
    def scmap(target_data, 
            reference_adata, 
            key_genes, 
            key_annotations,
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
     - key_genes: Key to be found in `target_data.var` and `reference_data.var` for finding the common genes between both populations. 
     - key_annotations: Key to be found in `reference_data.obs` with the labels to annotate the target data.
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

    #Find common genes between both datasets
    genesintarget = target_data.var[key_genes].isin(reference_adata.var[key_genes]).values
    genesinreference = reference_adata.var[key_genes].isin(target_data.var[key_genes]).values

    if verbose:
        print("Common genes:  ",np.sum(genesintarget)*100/len(genesintarget),"(%target)",np.sum(genesinreference)*100/len(genesinreference),"(%reference).")

    #Get genes to use for mapping
    
    if gene_selection_flavor == "HVGs":

        gene_subset = reference_adata.var.loc[reference_adata.var["highly_variable"].values,key_genes]

    elif gene_selection_flavor == "random":

        gene_subset = np.random.choice(reference_adata.var[key_genes].values,size=n_genes_selected,replace=False)

    elif gene_selection_flavor == "dropout":

        # Remove log transformation if necessary
        X = reference_adata.X.copy()
        if 'log1p' in reference_adata.uns.keys():
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

        gene_subset = reference_adata.var[key_genes][np.argsort(-res)[:n_genes_selected]]

    else:

        raise ValueError("gene_selection_flavor has to chosen between `HVGs`, `random` or `dropout`.")

    #Get matrices with selected genes

    retainedGenes_tar = target_data.var[key_genes].isin(gene_subset)
    order_tar = np.argsort(target_data.var[key_genes].values)
    X_tar = target_data.X[:,order_tar][:,retainedGenes_tar[order_tar]]

    retainedGenes_ref = reference_adata.var[key_genes].isin(target_data.var.loc[retainedGenes_tar,key_genes].values)
    order_ref = np.argsort(reference_adata.var[key_genes].values)
    X_ref = reference_adata.X[:,order_ref][:,retainedGenes_ref[order_ref]]

    # Make knn classification

    if algorithm_flavor == "centroid":

        #Define the labels of cells and centroids
        labels_ref_cells = reference_adata.obs[key_annotations].values.astype(str)
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
            model = KNeighborsClassifier(n_neighbors=1,metric=metric)
            model.fit(X_ref,labels_ref)
            if sp.sparse.issparse(X_tar):
                X_tar = X_tar.todense()
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
        labels_ref = reference_adata.obs[key_annotations].values.astype(str)

        #Create KNN classifier model
        model = KNeighborsClassifier(n_neighbors=3,metric=metrics[0])
        #Fit model
        if sp.sparse.issparse(X_ref):
            X_ref = X_ref.todense()
        model.fit(X_ref,labels_ref)
        #
        if sp.sparse.issparse(X_tar):
            X_tar = X_tar.todense()
        labels_tar = model.predict(X_tar)
        probabilities = model.predict_proba(X_tar)
        distances_tar = np.max(model.kneighbors(X_tar)[1],axis=1)

        #Annotate fates
        annotation = labels_tar

        #Unassign fates without consensus between metrics
        annotation[np.max(probabilities,axis=1) < 1] = unassigned_label

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
        key_genes, 
        key_projection,
        gene_selection_flavor="HVGs",
        n_genes_selected=1000,
        metric="cosine",
        inplace=True,
        key_added="scmap_projection",
        unassigned_label="Unassigned",
        verbose=False):
    """
    def scmap(target_data, 
            reference_adata, 
            key_genes, 
            key_projection,
            gene_selection_flavor="HVGs",
            n_genes_selected=1000,
            metrics=["cosine","pearson"],
            inplace=True,
            key_added="scmap_annotation",
            unassigned_label="Unassigned",
            verbose=False)

    Function that projects the SCMAP algorithm from Kilesev et al. (2018) to a reduced subspace of cells. Basically implements the KNeighborsRegressor.
    
    **Arguments**:
     - **target_data**: AnnData object to be annotated. 
     - **reference_adata**: AnnData object to use as reference for annotation.
     - **key_genes**: Key to be found in `target_data.var` and `reference_data.var` for finding the common genes between both populations. 
     - **key_projection**: Key to be found in `reference_data.obs` with the labels to annotate the target data.
     - **algorithm_flavor="centroid"**: Flavor of the algorithm to be chosen between "centroid" or "cells".
     - **gene_selection_flavor="HVGs"**: Method to chose the gene subset to use for annotation. To choose between "HVGs" (high varying genes), "random" or "dropout".
     - **n_genes_selected=1000**: Number of genes to select for matching.
     - **metrics=["cosine","pearson","spearman"],**: Metrics to perform the projection. Use normalized metrics for a proper working of the algorithm.
     - **inplace=True**: If True, add the cells assigned to `target_data.obs` with label `key_added`.
     - **key_added="scmap_annotation"**: If inplce is True, key added to `target_data.obs` with the Annotations.
     - **unassigned_label="Unassigned"**: String added to this cells that do not fill the criteria of assignation.
     - **verbose=False**: If to print information of the steps of the algorithm.
    """
    

    #Find common genes between both datasets
    genesintarget = target_data.var[key_genes].isin(reference_adata.var[key_genes]).values
    genesinreference = reference_adata.var[key_genes].isin(target_data.var[key_genes]).values

    if verbose:
        print("Common genes:  ",np.sum(genesintarget)*100/len(genesintarget),"(%target)",np.sum(genesinreference)*100/len(genesinreference),"(%reference).")

    #Get genes to use for mapping
    
    if gene_selection_flavor == "HVGs":

        gene_subset = reference_adata.var.loc[reference_adata.var["highly_variable"].values,key_genes]

    elif gene_selection_flavor == "random":

        gene_subset = np.random.choice(reference_adata.var[key_genes].values,size=n_genes_selected,replace=False)

    elif gene_selection_flavor == "dropout":

        # Remove log transformation if necessary
        X = reference_adata.X.copy()
        if 'log1p' in reference_adata.uns.keys():
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

        gene_subset = reference_adata.var[key_genes][np.argsort(-res)[:n_genes_selected]]

    else:

        raise ValueError("gene_selection_flavor has to chosen between `HVGs`, `random` or `dropout`.")

    #Get matrices with selected genes

    retainedGenes_tar = target_data.var[key_genes].isin(gene_subset)
    order_tar = np.argsort(target_data.var[key_genes].values)
    X_tar = target_data.X[:,order_tar][:,retainedGenes_tar[order_tar]]

    retainedGenes_ref = reference_adata.var[key_genes].isin(target_data.var.loc[retainedGenes_tar,key_genes].values)
    order_ref = np.argsort(reference_adata.var[key_genes].values)
    X_ref = reference_adata.X[:,order_ref][:,retainedGenes_ref[order_ref]]

    #Create KNN classifier model
    model = KNeighborsRegressor(n_neighbors=1,metric=metric)
    #Fit model
    if sp.sparse.issparse(X_ref):
        X_ref = X_ref.todense()
        
    Y = reference_adata.obsm[key_projection]
    model.fit(X_ref,Y)
    
    Y_projected = model.predict(X_tar)

    # Return
    if inplace:

        target_data.obsm[key_added] = Y_projected
        return

    else:

        return Y_projected