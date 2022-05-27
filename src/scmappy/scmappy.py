from distutils.log import error
import numpy as np
import pandas as pd
import scanpy as scp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

def scmap(target_data, 
        reference_adata, 
        key_genes, 
        key_annotations,
        algorithm_flavor="centroid",
        gene_selection_flavor="HVGs",
        n_genes_selected=1000,
        metrics=["cosine","pearson","spearman"],
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
            metrics=["cosine","pearson","spearman"],
            similarity_threshold=.7,
            inplace=True,
            key_added="scmap_annotation",
            unassigned_label="Unassigned",
            verbose=False)

    Function that implements the SCMAP algorithm from Kilesev et al. (2018) to annotate cells.

    **Arguments**:
     - **target_data**: AnnData object to be annotated. 
     - **reference_adata**: AnnData object to use as reference for annotation.
     - **key_genes**: Key to be found in `target_data.var` and `reference_data.var` for finding the common genes between both populations. 
     - **key_annotations**: Key to be found in `reference_data.obs` with the labels to annotate the target data.
     - **algorithm_flavor="centroid"**: Flavor of the algorithm to be chosen between "centroid" or "cells".
     - **gene_selection_flavor="HVGs"**: Method to chose the gene subset to use for annotation. To choose between "HVGs" (high varying genes), "random" or "dropout".
     - **n_genes_selected=1000**: Number of genes to select for matching.
     - **knn=5**: Number of knn to use with the "cell" flavor.
     - **metrics=["cosine","pearson","spearman"],**: Metrics to perform the projection. Use normalized metrics for a proper working of the algorithm.
     - **similarity_threshold=.9**: Proportion of consensus between neighbors of a cell to be annotated (a number between 0 and 1). Cells with a proportion of neighbors with the same fate lower than the consensus are annotated as "Unassigned".
     - **inplace=True**: If True, add the cells assigned to `target_data.obs` with label `key_added`.
     - **key_added="scmap_annotation"**: If inplce is True, key added to `target_data.obs` with the Annotations.
     - **unassigned_label="Unassigned"**: String added to this cells that do not fill the criteria of assignation.
     - **verbose=False**: If to print information of the steps of the algorithm.
    """

    #Find common genes between both datasets

    genesintarget = target_data.var[key_genes].isin(reference_adata.var[key_genes])

    genesinreference = reference_adata.var[key_genes].isin(target_data.var[key_genes])

    if verbose:
        print("Common genes:  ",np.sum(genesintarget)*100/len(np.sum(genesintarget)),"(%targer)",np.sum(genesinreference)*100/len(np.sum(genesinreference)),"(%reference).")

    #Get genes to use for mapping
    
    if gene_selection_flavor == "HVGs":

        gene_subset = reference_adata.var["highly_varying"].values

    elif gene_selection_flavor == "random":

        gene_subset = np.ones(reference_adata.var.shape[0]) == 0
        gene_subset[np.random.choice(range(0,reference_adata.var.shape[0]),size=n_genes_selected,replace=False)] = True

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
        model.fit(m.reshape(1,-1),d)
        res = (d - model.predict(m.reshape(1,-1)))**2

        # Get gene subset based on residuals

        gene_subset = np.ones(reference_adata.var.shape[0]) == 0
        gene_subset[np.argsort(-res)[:n_genes_selected]] = True

    else:

        raise ValueError("gene_selection_flavor has to chosen between `HVGs`, `random` or `dropout`.")

    #Get matrices with selected genes

    retainedGenes_tar = genesintarget * gene_subset 
    order_tar = np.argsort(target_data.var[key_genes].values)
    X_tar = target_data.X[:,order_tar][:,retainedGenes_tar[order_tar]]

    retainedGenes_ref = genesinreference * gene_subset 
    order_ref = np.argsort(reference_adata.var[key_genes].values)
    X_ref = reference_adata.X[:,order_ref][:,retainedGenes_ref[order_ref]]

    # Make knn classification

    if algorithm_flavor == "centroid":

        #Define the labels of cells and centroids
        labels_ref_cells = reference_adata.obs[key_annotations].values.astype(str)
        labels_ref = np.unique(labels_ref_cells)

        #Make centroids based on the median expression
        X_ref_centroids = np.zeros(len(labels_ref), X_tar.shape[1])
        for i,label in enumerate(labels_ref):
            X_ref_centroids[i,:] = np.median(X[labels_ref_cells == label],axis=0)

        #Assign to the target
        X_ref = X_ref_centroids

        #Clasify cells
        fates = pd.DataFrame()
        distances = pd.DataFrame()
        for i,metric in enumerate(metrics):
            model = KNeighborsClassifier(n_neighbors=1,metric=metric)
            model.fit(X_ref,labels_ref)
            labels_tar = model.predict(X_tar)
            fates[metric] = labels_tar
            distances_tar = np.max(model.kneighbors(X_tar)[1],axis=1)
            distances[metric] = distances_tar

        #Annotate fates
        annotation = fates.mode(axis=1).iloc[:,0]

        #Unassign fates without consensus between metrics
        annotation[np.invert(np.isnan(fates.mode(axis=1).iloc[:,1]))] = unassigned_label

        #Unassign fates with similarity below threshold
        annotation[distances.max(axis=1) < similarity_threshold] = unassigned_label

    elif algorithm_flavor == "cells":

        #Define the labels
        labels_ref = reference_adata.obs[key_annotations].values.astype(str)

        #Clasify cells
        fates = pd.DataFrame()
        distances = pd.DataFrame()

        #Create KNN classifier model
        model = KNeighborsClassifier(n_neighbors=2,metric=metrics[0])
        #Fit model
        model.fit(X_ref,labels_ref)
        #
        labels_tar = model.predict(X_tar)
        probabilities = model.predict_proba(X_tar)
        distances_tar = np.max(model.kneighbors(X_tar)[1],axis=1)

        #Annotate fates
        annotation = fates.mode(axis=1).iloc[:,0]

        #Unassign fates without consensus between metrics
        annotation[probabilities < 1] = unassigned_label

        #Unassign fates with similarity below threshold
        annotation[distances_tar < similarity_threshold] = unassigned_label

    else:

        raise ValueError("gene_selection_flavor has to chosen between `HVGs`, `random` or `dropout`.")

    # Return
    if inplace:

        target_data.obs[key_added] = annotation
        return

    else:

        return annotation
