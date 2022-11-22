import random

import numpy as np
from scipy import stats

from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities_v2 import (
    analyze_activity_trace,
    get_trace_coactivity_rates,
)
from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes
from Lab_Analyses.Utilities import activity_timestamps as tstamps
from Lab_Analyses.Utilities.quantify_movment_quality import quantify_movement_quality


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
    """Function to handle the local spine coactivity analysis functions
    
        INPUT PARAMETERS
            spine_activity-  2d np.array of the binarrized spine activity. Columns=spines
            
            spine_dFoF - 2d np.array of the spine dFoF traces. Columns = spines
            
            spine_calcium - 2d np.array of spine calcium traces. Columns = spines
            
            spine_groupings - list of spine groupings
            
            spine_flags - list containing the spine flags
            
            spine_volumes - list or array of the estimated spine volumes (um)
            
            spine_positions - list or array of the spine positions along their parent dendrite
            
            movement_spines - boolean list of whether each spine is a MRS
            
            non_movement_spines - boolean list of whether each spine is not a MRS
            
            rwd_movement_spines - boolean list of whether each spine is a rMRS
            
            lever_active - np.array of the binarized lever activity
            
            lever_unactive - np.array of the binarized lever inactivity
            
            activity_window - tuple specifying the time window in sec over which to analyze
            
            cluster_dist - int specifying the distance in um tht is considered local
            
            sampling_rate - int specifying the imaging sampling rate
            
            volume_norm - tuple list of constants to normalize spine dFoF and calcium by
    """

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


def absolute_local_coactivity(
    spine_activity,
    spine_dFoF,
    spine_calcium,
    spine_groupings,
    spine_flags,
    spine_volumes,
    spine_positions,
    move_spines,
    partner_list=None,
    activity_window=(-2, 4),
    cluster_dist=5,
    sampling_rate=60,
    volume_norm=None,
):
    """Function to examine absolute local coactivity, or events when any other local spine
        is coactive with the target spine
        
        INPUT PARAMETERS
            spine_activity-  2d np.array of the binarrized spine activity. Columns=spines
            
            spine_dFoF - 2d np.array of the spine dFoF traces. Columns = spines
            
            spine_calcium - 2d np.array of spine calcium traces. Columns = spines
            
            spine_groupings - list of spine groupings
            
            spine_flags - list containing the spine flags
            
            spine_volumes - list or array of the estimated spine volumes (um)
            
            spine_positions - list or array of the spine positions along their parent dendrite
            
            move_spines - list of boolean containing movement spine idxs

            partner_list - boolean list specifying a subset of spines to analyze
                            coactivity rates for
            
            activity_window - tuple specifying the time window in sec over which to analyze
            
            cluster_dist - int specifying the distance in um tht is considered local
            
            sampling_rate - int specifying the imaging sampling rate
            
            volume_norm - tuple list of constants to normalize spine dFoF and calcium by
    """

    # Sort out the spine_groupings to make sure it is iterable
    if type(spine_groupings[0]) != list:
        spine_grouping = [spine_grouping]

    el_spines = find_spine_classes(spine_flags, "Eliminated Spine")
    el_spines = np.array(el_spines)

    if volume_norm is not None:
        glu_norm_constants = volume_norm[0]
        ca_norm_constants = volume_norm[1]
    else:
        glu_norm_constants = np.array([None for x in range(spine_activity.shape[1])])
        ca_norm_constants = np.array([None for x in range(spine_activity.shape[1])])

    if partner_list is None:
        partner_list = [True for x in range(spine_activity.shape[1])]

    # Set up outputs
    nearby_coactive_spine_idxs = [None for i in range(spine_activity.shape[1])]
    frac_nearby_MRSs = np.zeros(spine_activity.shape[1]) * np.nan
    nearby_coactive_spine_volumes = np.zeros(spine_activity.shape[1]) * np.nan
    local_coactivity_rate = np.zeros(spine_activity.shape[1])
    local_coactivity_rate_norm = np.zeros(spine_activity.shape[1])
    spine_fraction_coactive = np.zeros(spine_activity.shape[1])
    local_coactivity_matrix = np.zeros(spine_activity.shape)
    spine_coactive_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_auc = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium_auc = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_traces = [None for i in range(spine_activity.shape[1])]
    spine_coactive_calcium_traces = [None for i in range(spine_activity.shape[1])]

    # Iterate through each dendrite grouping
    for spines in spine_groupings:
        # Pull current spine grouping data
        s_dFoF = spine_dFoF[:, spines]
        s_activity = spine_activity[:, spines]
        s_calcium = spine_calcium[:, spines]
        curr_positions = spine_positions[spines]
        curr_el_spines = el_spines[spines]
        curr_volumes = spine_volumes[spines]
        curr_move_spines = move_spines[spines]
        curr_glu_norm_constants = glu_norm_constants[spines]
        curr_ca_norm_constants = ca_norm_constants[spines]
        curr_partner_spines = partner_list[spines]

        # Analyze each spine individually
        for spine in range(s_dFoF.shape[1]):
            # Get positional information and nearby spine idxs
            target_position = curr_positions[spine]
            relative_positions = np.array(curr_positions) - target_position
            relative_positions = np.absolute(relative_positions)
            nearby_spines = np.nonzero(relative_positions <= cluster_dist)[0]
            nearby_spines = [
                x for x in nearby_spines if not curr_el_spines[x] and x != spine
            ]

            # Assess whether nearby spines display any coactivity
            nearby_coactive_spines = []
            for ns in nearby_spines:
                if np.sum(s_activity[:, spine] * s_activity[:, ns]):
                    nearby_coactive_spines.append(ns)
            # Refine nearby coactive spines based on partner list
            nearby_coactive_spines = [
                x for x in nearby_coactive_spines if curr_partner_spines[x] is True
            ]
            # Skip further analysis if no nearby coactive spines
            if len(nearby_coactive_spines) == 0:
                continue
            nearby_coactive_spines = np.array(nearby_coactive_spines)
            nearby_coactive_spine_idxs[spines[spine]] = spines[nearby_coactive_spines]

            # Get fraction of coactive spines that are MRSs
            nearby_move_spines = curr_move_spines[nearby_coactive_spines].astype(int)
            frac_nearby_MRSs[spines[spine]] = np.sum(nearby_move_spines) / len(
                nearby_move_spines
            )

            # Get average coactive spine volumes
            nearby_coactive_spine_volumes[spines[spine]] = np.nanmean(
                curr_volumes[nearby_coactive_spines]
            )

            # Get relative activity data
            curr_s_dFoF = s_dFoF[:, spine]
            curr_s_activity = s_activity[:, spine]
            curr_s_calcium = s_calcium[:, spine]
            nearby_s_dFoF = s_dFoF[:, nearby_coactive_spines]
            nearby_s_activity = s_activity[:, nearby_coactive_spines]
            nearby_s_calcium = s_calcium[:, nearby_coactive_spines]
            nearby_volumes = curr_volumes[nearby_coactive_spines]
            glu_constant = curr_glu_norm_constants[spine]
            ca_constant = curr_ca_norm_constants[spine]
            nearby_glu_constants = curr_glu_norm_constants[nearby_coactive_spines]
            nearby_ca_constants = curr_ca_norm_constants[nearby_coactive_spines]

            # Get local coactivity trace, where at least one spine is coactive
            combined_nearby_activity = np.sum(nearby_s_activity, axis=1)
            combined_nearby_activity[combined_nearby_activity > 1] = 1

            # Get local coactivity rates
            (
                coactivity_rate,
                coactivity_rate_norm,
                spine_frac_active,
                _,
                coactivity_trace,
            ) = get_trace_coactivity_rates(
                curr_s_activity, combined_nearby_activity, sampling_rate
            )
            # Skip further analysis if no local coactivity for current spine
            if not np.sum(coactivity_trace):
                continue

            local_coactivity_rate[spines[spine]] = coactivity_rate
            local_coactivity_rate_norm[spines[spine]] = coactivity_rate_norm
            spine_fraction_coactive[spines[spine]] = spine_frac_active
            local_coactivity_matrix[:, spines[spine]] = coactivity_trace

            # Get local coactivity timestamps
            coactivity_stamps = tstamps.get_activity_timestamps(coactivity_trace)
            if not coactivity_stamps:
                continue

            # Analyze activity traces
            ## Glutamate traces
            (s_traces, s_amp, s_auc, s_onset,) = analyze_activity_trace(
                curr_s_dFoF,
                coactivity_stamps,
                activity_window=activity_window,
                center_onset=True,
                norm_constant=glu_constant,
                sampling_rate=sampling_rate,
            )
            ## Calcium traces
            (s_ca_traces, s_ca_amp, s_ca_auc, _) = analyze_activity_trace(
                curr_s_calcium,
                coactivity_stamps,
                activity_window=activity_window,
                center_onset=True,
                norm_constant=ca_constant,
                sampling_rate=sampling_rate,
            )
            spine_coactive_amplitude[spines[spine]] = s_amp
            spine_coactive_calcium[spines[spine]] = s_ca_amp
            spine_coactive_auc[spines[spine]] = s_auc
            spine_coactive_calcium_auc[spines[spine]] = s_ca_auc
            spine_coactive_traces[spines[spine]] = s_traces
            spine_coactive_calcium_traces[spines[spine]] = s_ca_traces

            # Center timestamps around activity onset
            corrected_stamps = tstamps.timestamp_onset_correction(
                coactivity_stamps, activity_window, s_onset, sampling_rate
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

