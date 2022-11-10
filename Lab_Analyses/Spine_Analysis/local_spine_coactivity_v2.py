import random

import numpy as np
from scipy import stats

from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes


def local_spine_coactivity_analysis(
    spine_activity,
    spine_dFoF,
    spine_calcium,
    spine_groupings,
    spine_flags,
    spine_volumes,
    spine_positions,
    movement_spines,
    non_movement_spines,
    rwd_movement_spines,
    lever_active,
    lever_unactive,
    activity_window=(-2, 4),
    cluster_dist=5,
    sampling_rate=60,
    volume_norm=None,
):
    """Function to handle the local spine coactivity analysis functions"""

    # Get distance-dependent coactivity rates
    ## Non-specified
    distance_coactivity_rate, distance_bins = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=None,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=False,
    )

    distance_coactivity_rate_norm, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=None,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=True,
    )
    ## Movement-related spines
    MRS_distance_coactivity_rate, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=movement_spines,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=False,
    )
    MRS_distance_coactivity_rate_norm, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=movement_spines,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=True,
    )
    ## Non-Movement-related spines
    nMRS_distance_coactivity_rate, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=non_movement_spines,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=False,
    )
    nMRS_distance_coactivity_rate_norm, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=non_movement_spines,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=True,
    )
    ## Local values only (< 5um)
    avg_local_coactivity_rate = distance_coactivity_rate[0, :]
    avg_local_coactivity_rate_norm = distance_coactivity_rate_norm[0, :]
    avg_MRS_local_coactivity_rate = MRS_distance_coactivity_rate[0, :]
    avg_MRS_local_coactivity_rate_norm = MRS_distance_coactivity_rate_norm[0, :]
    avg_nMRS_local_coactivity_rate = nMRS_distance_coactivity_rate[0, :]
    avg_nMRS_local_coactivity_rate_norm = nMRS_distance_coactivity_rate_norm[0, :]

    # Cluster Score
    ## Nonconstrained
    cluster_score, coactive_num = calculate_cluster_score(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=None,
        iterations=100,
    )
    MRS_cluster_score, MRS_coactive_num = calculate_cluster_score(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=movement_spines,
        iterations=100,
    )
    nMRS_cluster_score, nMRS_coactive_num = calculate_cluster_score(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=non_movement_spines,
        iterations=100,
    )
    ## Constrained to movement and nonmovement periods
    movement_cluster_score, movement_coactive_num = calculate_cluster_score(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=lever_active,
        partner_list=None,
        iterations=100,
    )
    nonmovement_cluster_score, nonmovement_coactive_num = calculate_cluster_score(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=lever_unactive,
        partner_list=None,
        iterations=100,
    )


def distance_coactivity_rate_analysis(
    spine_activity,
    spine_positions,
    flags,
    spine_grouping,
    constrain_matrix=None,
    partner_list=None,
    bin_size=5,
    sampling_rate=60,
    norm=False,
):
    """Function to calculate pairwise spine coactivity rate between all spines along the 
        same dendrite. Spine rates are then binned based on their distance from the
        target spine
        
        INPUT PARAMETERS
            spine_activity - np.array of the binarized spine activity traces.
                             Each column represents each spine
            
            spine_positions - list of the corresponding spine positions along the
                              dendrite for each spine
            
            flags - list of the spine flags
            
            spine_grouping - list with the corresponding groupings of spines on 
                             the same dendirte

            constrain_matrix- np.array of binarized events to constrain the coactivity
                                to. (e.g., dendritic events, movement periods)
            
            partner_list - boolean list specifying a subset of spines to analyze
                            coactivity rates for
            
            bin_size - int or float specifying the distance to bin over

            sampling_rate - int specifying the imaging sampling rate

            norm - boolean specifying whether or not to normalize the coactivity

        OUTPUT PARAMETERS
            coactivity_matrix - np.array of the coactivity for each spine (columns)
                                over the binned distances (rows)

            positions_bins - np.array of the distances data were binned over

    """
    # Set up the position bins
    MAX_DIST = 30
    bin_num = int(MAX_DIST / bin_size)
    position_bins = np.linspace(0, MAX_DIST, bin_num + 1)

    # Sort out the spine_groupings to make sure it is iterable
    if type(spine_grouping[0]) != list:
        spine_grouping = [spine_grouping]

    # Constrain data if if specified
    if constrain_matrix is not None:
        activity_matrix = spine_activity * constrain_matrix
    else:
        activity_matrix = spine_activity

    # Set up output variables
    coactivity_matrix = np.zeros((bin_num, spine_activity.shape[1]))

    # Find indexes of eliminated spines
    el_spines = find_spine_classes(flags, "Eliminated Spine")
    el_spines = np.array(el_spines)

    # Iterate through each dendrite grouping
    for spines in spine_grouping:
        s_activity = activity_matrix[:, spines]
        positions = np.array(spine_positions)[spines]
        curr_el_spines = el_spines[spines]

        # Iterate through each spine on this dendrite
        for spine in range(s_activity.shape[1]):
            curr_coactivity = []
            curr_spine = s_activity[:, spine]

            # Calculate coactivity with each other spine
            for partner in range(s_activity.shape[1]):
                # Don't compare spines to themselves
                if partner == spine:
                    continue
                # Don't compare eliminated spines
                if curr_el_spines[spine] == True:
                    curr_coactivity.append(np.nan)
                    continue
                if curr_el_spines[partner] == True:
                    curr_coactivity.append(np.nan)
                    continue
                # Subselect partners if specified
                if partner_list is not None:
                    if partner_list[spines[partner]] is True:
                        partner_spine = s_activity[:, partner]
                    else:
                        curr_coactivity.append(np.nan)
                else:
                    partner_spine = s_activity[:, partner]

                coactivity_rate = calculate_coactivity(
                    curr_spine, partner_spine, sampling_rate, norm=norm
                )
                curr_coactivity.append(coactivity_rate)

            # Order by positions
            curr_pos = positions[spine]
            pos = [x for idx, x in enumerate(positions) if idx != spine]
            # normalize distances relative to current position
            relative_pos = np.array(pos) - curr_pos
            # make all distances positive
            relative_pos = np.absolute(relative_pos)
            # sort coactivity and positions based on distance
            sorted_coactivity = np.array(
                [x for _, x in sorted(zip(relative_pos, curr_coactivity))]
            )
            sorted_positions = np.array(
                [y for y, _ in sorted(zip(relative_pos, curr_coactivity))]
            )
            # Bin the data
            binned_coactivity = bin_by_position(
                sorted_coactivity, sorted_positions, position_bins
            )
            coactivity_matrix[:, spines[spine]] = binned_coactivity

    return coactivity_matrix, position_bins


def calculate_coactivity(spine_1, spine_2, sampling_rate, norm):
    """Helper function to calculate spine coactivity rate between two spines"""
    duration = len(spine_1) / sampling_rate
    coactivity = spine_1 * spine_2
    events = np.nonzeor(np.diff(coactivity) == 1)[0]
    event_num = len(events)
    event_freq = event_num / duration

    if norm:
        # normalize frequency based on overall activity levels
        spine_1_freq = len(np.nonzero(np.diff(spine_1) == 1)[0]) / duration
        spine_2_freq = len(np.nonzero(np.diff(spine_2) == 1)[0]) / duration
        geo_mean = stats.gmean([spine_1_freq, spine_2_freq])
        coactivity_rate = event_freq / geo_mean
    else:
        coactivity_rate = event_freq * 60  ## convert to minutes

    return coactivity_rate


def bin_by_position(data, positions, bins):
    """Helper function to bin the data by position"""
    binned_data = []

    for i in range(len(bins)):
        if i != len(bins) - 1:
            idxs = np.nonzero((positions > bins[i]) & (positions <= bins[i + 1]))[0]
            if idxs.size == 0:
                binned_data.append(np.nan)
                continue
            binned_data.append(np.nanman(data[idxs]))

    return np.array(binned_data)


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
            partner_idxs = [i for i in p_idxs if curr_el_spines[i] is False]
            # Subselect partners if specified
            if partner_list is not None:
                partner_idxs = [j for j in partner_idxs if partner_list[i] is True]
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
            # Calculate shuffled nearest neighbor distances
            all_shuff_nn_distances = []
            for i in range(iterations):
                shuff_pos = random.sample(partner_pos, len(partner_pos))
                shuff_nn, _ = find_nearest_neighbors(
                    curr_spine, partner_spines, shuff_pos
                )
                all_shuff_nn_distances.append(shuff_nn)
            shuff_nn_distance = np.nanmean(all_shuff_nn_distances)

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

    avg_nearest_neighbor = np.nanmean(number_coactive)
    avg_num_coactive = np.nanmean(number_coactive)

    return avg_nearest_neighbor, avg_num_coactive

