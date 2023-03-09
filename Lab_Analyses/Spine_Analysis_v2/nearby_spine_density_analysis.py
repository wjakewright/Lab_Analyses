import copy

import numpy as np

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import find_present_spines
from Lab_Analyses.Spine_Analysis_v2.variable_spatial_distribution import (
    variable_spatial_distribution,
)


def nearby_spine_density_analysis(
    spine_category,
    spine_positions,
    spine_flags,
    spine_groupings,
    bin_size=5,
    cluster_dist=5,
    iterations=1000,
):
    """Function to assess the density of different types of spines (e.g., MRSs)
        along the dendrite relative to a target spine

        INPUT PARAMETERS
            spine_category - boolean array specifying if a spine belongs to a given category
                            or not
            
            spine_positions - np.array of the corresponding spine positions along
                              the dendrite
            
            spine_flags - list of the spine flags

            spine_groupings - list of the corresponding groupigns of spines along the dendrite

            bin_size - int or float speicfying the distance to bin over

            cluster_dist - int or float specifying distance that is considered local

            iterations - int specifying how many shuffles to perform
        
        OUTPUT PARAMETERS
            base_density_distribution - 2d np.array of the density of spines over the binned
                                        distances (row) relative to each spine (columns)

            cat_density_distribution - 2d np.array of the density of the category spines
                                        over the binned distance (row) relative to each spine
            
            avg_local_density - np.array of the avg spine density nearby each given spine

            shuff_local_density - 2d np.array of the shuffled values. Each row represents
                                  a shuffle and col represents each spine
            
            position_bins - np.array of the distances the data were binned over

    """
    # First get the overall spine density distribution
    present_spines = find_present_spines(spine_flags).astype(int)
    base_density_distribution, _ = variable_spatial_distribution(
        present_spines,
        spine_positions,
        spine_flags,
        spine_groupings,
        bin_size=bin_size,
        density=True,
    )

    # Get the real distribution and local density
    real_category = spine_category.astype(int)
    ## Distribution
    cat_density_distribution, _, position_bins = variable_spatial_distribution(
        real_category,
        spine_positions,
        spine_flags,
        spine_groupings,
        bin_size=bin_size,
        density=True,
    )
    ## Local density
    if bin_size != cluster_dist:
        temp_dist, _, _ = variable_spatial_distribution(
            real_category,
            spine_positions,
            spine_flags,
            spine_groupings,
            bin_size=cluster_dist,
            density=True,
        )
        avg_local_density = temp_dist[0, :]
    else:
        avg_local_density = cat_density_distribution[0, :]

    # Setup shuffle outputs
    shuff_local_density = np.zeros((iterations, len(real_category))) * np.array

    # Iterate through each shuffle
    for i in range(iterations):
        # Shuffle spine categories
        shuff_category = copy.copy(real_category)
        np.random.shuffle(shuff_category)
        shuff_distribution, _, _ = variable_spatial_distribution(
            shuff_category,
            spine_positions,
            spine_flags,
            spine_groupings,
            bin_size=cluster_dist,
            dendity=True,
        )
        shuff_local_density[i, :] = shuff_distribution[0, :]

    return (
        base_density_distribution,
        cat_density_distribution,
        avg_local_density,
        shuff_local_density,
        position_bins,
    )
