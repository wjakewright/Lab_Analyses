from Lab_Analyses.Spine_Analysis_v2.variable_spatial_clustering import \
    variable_spatial_clustering
from Lab_Analyses.Spine_Analysis_v2.variable_spatial_distribution import \
    variable_spatial_distribution


def variable_distance_dependence(
    spine_data,
    spine_positions,
    spine_flags,
    spine_groupings,
    bin_size=5,
    cluster_dist=5,
    method="nearest",
    iterations=1000,
):
    """Helper function to perform variable_spatial_distribution and variable_spatial_clustering
        together
    """

    # Spatial distribution
    variable_distribution, _ , _= variable_spatial_distribution(
        spine_data, spine_positions, spine_flags, spine_groupings, bin_size=bin_size,
    )

    # Spatial clustering
    avg_variable, shuff_variable, _ = variable_spatial_clustering(
        spine_data,
        spine_positions,
        spine_flags,
        spine_groupings,
        method=method,
        cluster_dist=cluster_dist,
        iterations=iterations,
    )

    return avg_variable, shuff_variable, variable_distribution
