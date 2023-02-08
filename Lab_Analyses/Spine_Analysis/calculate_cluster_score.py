import copy
import random

import numpy as np

from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes


def calculate_cluster_score(
    spine_activity,
    spine_positions,
    spine_flags,
    spine_grouping,
    constrain_matrix=None,
    partner_list=None,
    iterations=100,
):
    """Function to calculate how clustered a given spines coactivity is. For each
        event a spine is coactive, it takes the nearst neighbor distance, which is 
        then averaged across all events. This is then normalized against the chance
        distribution of the spines
        
        INPUT PARAMETERS
            spine_activity - np.array of the binarized spine activity traces. Each 
                             column represents each spine
            
            spine_positions - list of the corresponding spine positions along the
                              dendrite for each spine
            
            spine_flags - list of the spine flags

            spine_grouping - list with the corresponding groupings of spines on the
                             same dendrite

            constrain_matrix - np.array of binarized events to constrain the coactivity
                                to. (e.g., dendritic events, movement epochs)

            partner_list - boolean list specifying a subset of spines to analyze
                            the coactivity for

            iterations - int specifying how many shuffles to perform

        OUTPUT PARAMETERS
            cluster_score - np.array of the cluster score for each spine

            coactive_num - np.array of the number of coactive spines
            
    """
    # Sort out spine groupings
    if type(spine_grouping[0]) != list:
        spine_grouping = [spine_grouping]

    # Constrain data if specified
    if constrain_matrix is not None:
        if len(constrain_matrix.shape) == 1:
            constrain_matrix = constrain_matrix.reshape(-1, 1)
        activity_matrix = spine_activity * constrain_matrix
    else:
        activity_matrix = spine_activity

    # Set up output
    cluster_score = np.zeros(spine_activity.shape[1]) * np.nan
    coactive_num = np.zeros(spine_activity.shape[1]) * np.nan

    # Find indexes of eliminated spines
    el_spines = find_spine_classes(spine_flags, "Eliminated Spines")
    el_spines = np.array(el_spines)

    # Iterate through each dendrite grouping
    for spines in spine_grouping:
        s_activity = activity_matrix[:, spines]
        positions = np.array(spine_positions)[spines]
        curr_el_spines = el_spines[spines]

        # Iterate through each spine
        for spine in range(s_activity.shape[1]):
            # Skip if eliminated spine
            if curr_el_spines[spine] == True:
                continue
            curr_spine = s_activity[:, spine]
            # Get partner indexes
            p_idxs = [x for x in range(s_activity.shape[1]) if x != spine]
            # Remove eliminated spines from partner spines
            partner_idxs = [i for i in p_idxs if curr_el_spines[i] == False]
            # Subselect partners if specified
            if partner_list is not None:
                partner_idxs = [j for j in partner_idxs if partner_list[j] == True]
            partner_spines = s_activity[:, partner_idxs]
            # Get positional information
            curr_pos = positions[spine]
            relative_pos = np.array(positions) - curr_pos
            partner_pos = relative_pos[partner_idxs]
            partner_pos = np.absolute(partner_pos)
            # Get number of coactive spines and nearst neighbor distance
            nn_distance, coactive_n = find_nearest_neighbors(
                curr_spine, partner_spines, partner_pos
            )
            if np.isnan(nn_distance):
                cluster_score[spines[spine]] = np.nan
                coactive_num[spines[spine]] = 0
            # Calculate shuffled nearest neighbor distances
            all_shuff_nn_distances = []
            for i in range(iterations):
                shuff_pos = copy.copy(partner_pos)
                np.random.shuffle(shuff_pos)
                shuff_nn, _ = find_nearest_neighbors(
                    curr_spine, partner_spines, shuff_pos
                )
                all_shuff_nn_distances.append(shuff_nn)
            shuff_nn_distance = np.nanmedian(all_shuff_nn_distances)

            # Calcuate cluster score
            c_score = (1 / nn_distance) / (1 / shuff_nn_distance)

            # Store results
            cluster_score[spines[spine]] = c_score
            coactive_num[spines[spine]] = coactive_n

    return cluster_score, coactive_num


def find_nearest_neighbors(target_spine, partner_spines, partner_positions):
    """Helper function to find the average nearst neighbor distance during 
        coactivity events"""

    # Find activity periods of the target spine
    active_boundaries = np.insert(np.diff(target_spine), 0, 0, axis=0)
    active_onsets = np.nonzero(active_boundaries == 1)[0]
    active_offsets = np.nonzero(active_boundaries == -1)[0]
    if len(active_onsets) == 0:
        avg_nearest_neighbor = np.nan
        avg_num_coactive = 0
        return avg_nearest_neighbor, avg_num_coactive
    ## Check onset offset order
    if active_onsets[0] > active_offsets[0]:
        active_offsets = active_offsets[1:]
    ## Check onsets and offests are same length
    if len(active_onsets) > len(active_offsets):
        active_onsets = active_onsets[:-1]

    # Compare active epochs to other spines
    number_coactive = []
    nearest_neighbor = []
    for onset, offset in zip(active_onsets, active_offsets):
        epoch_activity = partner_spines[onset:offset, :]
        summed_activity = np.sum(epoch_activity, axis=0)
        active_partners = np.nonzero(summed_activity)[0]
        # Skip if no coactivity
        if len(active_partners) == 0:
            continue
        active_positions = partner_positions[active_partners]
        nearest_neighbor.append(np.min(active_positions))
        number_coactive.append(len(active_partners))

    avg_nearest_neighbor = np.nanmedian(nearest_neighbor)
    avg_num_coactive = np.nanmedian(number_coactive)

    return avg_nearest_neighbor, avg_num_coactive
