import copy

import numpy as np

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import find_present_spines


def variable_spatial_clustering(
    spine_data,
    spine_positions,
    spine_flags,
    spine_groupings,
    partner_list=None,
    method="nearest",
    cluster_dist=5,
    iterations=10000,
):
    """Function to calculate how spatially clustered a given variable is along the dendrite
    
        INPUT PARAMETERS
            spine_data - np.array of the variable to be anzlyaed
            
            spine_positions - np.array of the coresponding spine positions along the dendrite
            
            spine_flags - list of the spine flags
            
            spine_groupings - list of the corresponding groupings of spines along the dendrite
            
            partner_list - boolean array of spines to include in the clustering. Only used
                            for local average method

            method - str specifying how you want to measure the clustering. 'nearest' will
                    find the nearest neighbor distance to a spine of a specific type. 
                    'local' will simply compare the average local values to the shuffles

            cluster_dist - int specifying the distance to be considered local. Only used
                            for 'local' method
            
            iterations - int specifying how many shuffles to perform
            
        OUTPUT PARAMETERS
            real_values - np.array of the real value for each spine
            
            shuff_values - 2d np.array of the shuffled values. Each row represents
                            a shuffle and col represents each spine
            
            cluster_score - np.array of the cluster score for each spine
    """
    # Sort out the spine groupings to ensure iterable
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # Set up output variables
    real_values = np.zeros(len(spine_data)) * np.nan
    shuff_values = np.zeros((iterations, len(spine_data))) * np.nan

    # Find the present spines
    present_spines = find_present_spines(spine_flags)

    # Refine the partner list if specified
    if partner_list is not None:
        present_partners = present_spines * partner_list
    else:
        present_partners = present_spines

    # Iterate through each dendrite grouping
    for spines in spine_groupings:
        s_data = spine_data[spines]
        positions = spine_positions[spines]
        curr_present = present_spines[spines]
        curr_partners = present_partners[spines]

        # Iterate through each shuffle
        for i in range(iterations):
            # Get the real value if first shuffle
            if i == 0:
                if method == "nearest":
                    real_v = find_nearest_neighbor(s_data, positions, curr_present)
                elif method == "local":
                    real_v = get_local_avg(
                        s_data, positions, curr_present, cluster_dist, curr_partners,
                    )
                real_values[spines] = real_v

            # Shuffle data relative to positions
            shuff_data = copy.copy(s_data)
            np.random.shuffle(shuff_data)
            # Get the shuffle values
            if method == "nearest":
                shuff_v = find_nearest_neighbor(shuff_data, positions, curr_present)
            elif method == "local":
                shuff_v = get_local_avg(
                    shuff_data, positions, curr_present, cluster_dist, curr_partners
                )
            shuff_values[i, spines] = shuff_v

    # Calculate the cluster score
    shuff_medians = np.nanmedian(shuff_values, axis=0)
    cluster_scores = [
        (1 / real) / (1 / shuff) for real, shuff in zip(real_values, shuff_medians)
    ]
    cluster_scores = np.array(cluster_scores)

    return real_values, shuff_values, cluster_scores


def find_nearest_neighbor(spine_data, positions, present_spines):
    """Helper function to find the nearest neighbor for each spine. Note
        spine_data must be binary/boolean indicating spine identity for a given 
        variable
    """
    # Set up the output
    nearest_neighbor = np.zeros(len(spine_data)) * np.nan

    # Iterate through each spine
    for spine in range(len(spine_data)):
        # Skip eliminated/absent spines
        if present_spines[spine] == False:
            continue
        # Get indexes of the partner spines
        p_idxs = [x for x in range(len(spine_data)) if x != spine]
        # Remove absent spines from partner list
        partner_idxs = [i for i in p_idxs if present_spines[i] == True]

        # Get the positional information
        curr_pos = positions[spine]
        relative_pos = positions - curr_pos
        partner_pos = relative_pos[partner_idxs]
        partner_pos = np.absolute(partner_pos)

        # Find the nearest neighbor
        ## Get positions matching the condition
        matching_partners = np.nonzero(spine_data[partner_idxs])[0]
        matching_pos = partner_pos[matching_partners]
        ## Get the minimum distance
        try:
            nn = np.min(matching_pos)
        except ValueError:
            nn = np.nan
        nearest_neighbor[spine] = nn

    return nearest_neighbor


def get_local_avg(spine_data, positions, present_spines, distance, present_partners):
    """Helper function to get the average values of nearby spines"""
    # Setup the output
    local_avg = np.zeros(len(spine_data)) * np.nan

    # Iterate through each spine
    for spine in range(len(spine_data)):
        # Skip absent/eliminated spines
        if present_spines[spine] == False:
            continue
        # Find nearby spines
        target_pos = positions[spine]
        relative_pos = np.absolute(positions - target_pos)
        nearby_spines = np.nonzero(relative_pos <= distance)[0]
        nearby_spines = [
            x for x in nearby_spines if present_partners[x] == True and x != spine
        ]

        # Get the local average
        local_avg[spine] = np.nanmean(spine_data[nearby_spines])

    return local_avg

