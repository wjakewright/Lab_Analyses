import numpy as np
import pandas as pd
import scipy as sy


def population_pairwise_correlation(spikes):
    """Function to calculate the pairwise correlation between
    all pairs of neurons within a FOV

    INPUT PARAMETERS
        spikes - 2d array of the estiamted spikes (time x cells)

    """
    # Convert to dataframe
    df = pd.DataFrame(spikes)

    # Use the in-built corrlation function in pandas
    corr_matrix = df.corr("pearson")

    # Get all the pairwise correlations
    pairwise_correlations = corr_matrix.values[
        np.triu_indices_from(corr_matrix.values, 1)
    ]

    # Get the average pairwise correlation of the FOV
    avg_correlation = np.nanmean(pairwise_correlations)

    # Sort the correlation matrix into clusters
    distances = sy.cluster.hierarchy.distance.pdist(corr_matrix)
    linkage = sy.cluster.hierarchy.linkage(distances, method="complete")
    ind = sy.cluster.hierarchy.fcluster(linkage, 0.5 * distances.max(), "distance")
    columns = [df.columns.to_list()[i] for i in list((np.argsort(ind)))]
    sort_data = df.reindex(columns, axis=1)
    sort_matrix = sort_data.corr("pearson")

    return sort_matrix, pairwise_correlations, avg_correlation
