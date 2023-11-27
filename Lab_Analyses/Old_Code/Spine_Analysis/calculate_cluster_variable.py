import copy

import numpy as np

from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes


def calculate_cluster_variable(
    spine_data,
    spine_positions,
    spine_flags,
    spine_groupings,
    method="nearest",
    iterations=10000,
):
    """Function to calculate how clustered a give variable value (e.g., LMP) along the dendrite
        Takes local average of that value around the target spine, and then compares it to a shuffled
        distribution (positions are shuffled)
        
        INPUT PARAMETERS
            spine_data - np.array of the variable to be analyzed
            
            spine_positions - list of the corresponding spine positions along the dendrite
            
            spine_flags - list of the spine flags
            
            spine_groupings - list of the corresponding groupings of spines on the same dendrite

            method - str specifying how you want to measure the clustering. "nearst" will find the 
                    nearest neighbor distance. "local" will simply compare the average local values
                    to shuffles
            
            iterations - int specifying how many shuffles to perform
        
        OUTPUT PARAMETERS
            real_values - np.array of the real value for each spine
            
            shuff_values - 2d np.array of the shuffled values. Each row represents 
                           a shuffle and col represent each spine
            
            cluster_score - np.array of the cluster score for each spine
    """
    # sort out spine groupings
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # Set up output variables
    real_values = np.zeros(len(spine_data)) * np.nan
    shuff_values = np.zeros((iterations, len(spine_data))) * np.nan

    # find indexes of eliminated spines
    el_spines = find_spine_classes(spine_flags, "Eliminated Spine")
    el_spines = np.array(el_spines)

    # Iterate through each dendrite grouping
    for spines in spine_groupings:
        s_data = spine_data[spines]
        positions = np.array(spine_positions)[spines]
        curr_el_spines = el_spines[spines]

        # Iterate through each shuffle
        for i in range(iterations):
            # Get real values if first shuffle
            if i == 0:
                if method == "nearest":
                    real_v = find_nearest_neighbor(s_data, positions, curr_el_spines)
                elif method == "local":
                    real_v = get_local_avg(s_data, positions, curr_el_spines)
                # Store real values
                real_values[spines] = real_v

            # Shuffle data relative to positions
            shuff_data = copy.copy(s_data)
            np.random.shuffle(shuff_data)
            # Get shuffle values
            if method == "nearest":
                shuff_v = find_nearest_neighbor(shuff_data, positions, curr_el_spines)
            elif method == "local":
                shuff_v = get_local_avg(shuff_data, positions, curr_el_spines)
            # Store shuffle values
            shuff_values[i, spines] = shuff_v

    # Calculate the cluster score
    shuff_medians = np.nanmedian(shuff_values, axis=0)
    cluster_scores = [
        (1 / real) / (1 / shuff) for real, shuff in zip(real_values, shuff_medians)
    ]
    cluster_scores = np.array(cluster_scores)

    return real_values, shuff_values, cluster_scores


def find_nearest_neighbor(spine_data, spine_positions, el_spines):
    """Helper function to handle finding the nearest neighbor for each spine
        Note spine_data must be a binary array indicating spine identity for 
        a given variable (MRS, enlarged, etc)"""
    # Set up output
    nearest_neighbors = np.zeros(len(spine_data)) * np.nan

    # iterate through each spine
    for spine in range(len(spine_data)):
        # Skip eliminated spine
        if el_spines[spine] == True:
            continue
        # Get indexes of the partner spines
        p_idxs = [x for x in range(len(spine_data)) if x != spine]
        # Remomve eliminated spines from partner list
        partner_idxs = [i for i in p_idxs if el_spines[i] == False]

        # Get positional information
        curr_pos = spine_positions[spine]
        relative_pos = spine_positions - curr_pos
        partner_pos = relative_pos[partner_idxs]
        partner_pos = np.absolute(partner_pos)
        # Find nearest neighbor
        ## Multiply binary array to get only positions matching the condition
        matching_partners = np.nonzero(spine_data[partner_idxs])[0]
        matching_pos = partner_pos[matching_partners]
        ## Get minimum position distance
        try:
            nearest_neighbor = np.min(matching_pos)
        except ValueError:
            nearest_neighbor = np.nan
        nearest_neighbors[spine] = nearest_neighbor

    return nearest_neighbors


def get_local_avg(spine_data, spine_positions, el_spines):
    """Helper function to get the average values of nearby spines"""
    DIST = 5

    # setup output
    local_avg = np.zeros(len(spine_data)) * np.nan

    # Iterate through each spine
    for spine in range(len(spine_data)):
        # Skip eliminated spine
        if el_spines[spine] == True:
            continue
        # Find nearby spines
        target_pos = spine_positions[spine]
        relative_pos = spine_positions - target_pos
        relative_pos = np.absolute(relative_pos)
        nearby_spines = np.nonzero(relative_pos <= DIST)[0]
        nearby_spines = [
            x for x in nearby_spines if el_spines[x] == False and x != spine
        ]

        # Get the local average
        avg = np.nanmean(spine_data[nearby_spines])
        local_avg[spine] = avg

    return local_avg

