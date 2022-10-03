"""Module to perform spine co-activity analyses"""

import numpy as np
from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities import (
    get_activity_timestamps,
    get_coactivity_rate,
    get_dend_spine_traces_and_onsets,
    nearby_spine_conjunctive_events,
)
from Lab_Analyses.Spine_Analysis.spine_movement_analysis import (
    quantify_movement_quality,
)
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    spine_volume_norm_constant,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from scipy import stats


def local_spine_coactivity_analysis(
    data,
    movement_epoch=None,
    cluster_dist=10,
    sampling_rate=60,
    zscore=False,
    volume_norm=False,
):
    """Function to analyze local spine coactivity
    
        INPUT PARAMETERS
            data - spine_data object (e.g., Dual_Channel_Spine_Data)
            
            movement_epoch - str specifying if you want to analyze only during specific
                            types of movements. Accepts - 'movement', 'rewarded', 'unrewarded',
                            'learned', and 'nonmovement'. Default is None, analyzing the entire
                            imaging session
            
            cluster_dist - int or float specifying the distance from target spine that will be
                            considered as nearby spines
            
            sampling_rate - int or float specicfying what the imaging rate is
            
            zscore - boolean of whether to zscore dFoF traces for analysis
            
            volume_nomr - boolean of whether or not to normalize activity by spine volume
            
        OUTPUT PARMATERS
            distance_coactivity_rate - 2d np.array of the normalized coactivity for each spine
                                        (columns) over the binned distances (row)

            distance_bins - np.array of the distances data were binned over

            local_correlation - np.array of the average correlation of each spine with nearby
                                coactive spines during conjunctive events
            
            local_coactivity_matrix - 2d np.array of the binarized local coactivity 
                                        (columns=spines, rows=time)
            
            local_coactivity_rate - np.array of the event rate of local coactivity events (where
                                    at least one nearby spine is coactive) for each spine
                                    
            spine_fraction_coactive - np.array of the fraction of spine activity events that 
                                      are also coacctive with at least one nearby spine
            
            local_coactive_spine_num - np.array of the average number of coactive spines across 
                                        local coactivity events for each spine
                                
            local_coactive_spine_volumes - np.array of the average volume of spines coactive during
                                            local coactivity events for each spine
            
            spine_coactive_amplitude - np.array of the average peak amplitude of activity
                                      during local coactive events for each spine
                                      
            local_coactive_amplitude - np.array of the peak average summed activity of nearby 
                                        coactive spines during local coactivity events for each spine
            
            spine_coactive_calcium - np.array of the average peak calcium amplitude during
                                    local coactivity events for each spine
            
            local_coactive_calcium - np.array of the peak average summed calcium activity of nearby
                                     spines during local coactivity events for each spine
                                     
            spine_coactive_std - np.array of the std around the peak activity amplitude during local
                                coactive events for each spine
            
            local_coactive_std - np.array of the std around the summed peak activity amplitude of
                                  nearby coactive spines during local coactive events for each spine
                                  
            spine_coactive_calcium_std - np.array of the std around the peak calcium amplitude
                                         during local coactive events for each spine
                                         
            local_coactiive_calcium_std - np.array of the std around the summe dpeak calcium amplitude
                                           of nearby coactive spines during local coactive events
            
            spine_coactive_calcium_auc - np.array of the auc of the average calcium trace during
                                         local coative events for each spine
                                         
            local_coactive_calcium_auc - np.array of the auc of the averaged summed calcium trace
                                          of nearby coactive spines during local coactive events
                                          
            spine_coactive_traces - list of 2d np.arrays of activity traces for each local coactivity
                                    event for each spine (columns = events)
            
            local_coactive_traces - list of 2d np.arrays of summed activity traces for each local
                                    coactivity event for each spine (columns=events)
                                    
            spine_coactive_calcium_traces - list of 2d np.arrays of calcium traces for each local coactivity
                                            event for each spine (coluumns=events)
            
            local_coactive_calcium_traces - list of 2d np.arrays of the summed calcium traces for each
                                            local coactivity event for each spine (columns=events)
    """

    spine_groupings = np.array(data.spine_grouping)
    spine_flags = data.spine_flags
    spine_volumes = np.array(data.corrected_spine_volume)
    spine_positions = np.array(data.spine_positions)
    spine_dFoF = data.spine_GluSnFr_processed_dFoF
    spine_calcium = data.spine_calcium_processed_dFoF
    spine_activity = data.spine_GluSnFr_activity
    dendrite_activity = data.dendrite_calcium_activity

    if zscore:
        spine_dFoF = d_utils.z_score(spine_dFoF)
        spine_calcium = d_utils.z_score(spine_calcium)

    if volume_norm:
        glu_norm_constants = spine_volume_norm_constant(
            spine_activity,
            spine_dFoF,
            data.corrected_spine_volume,
            data.imaging_parameters["Zoom"],
            sampling_rate=sampling_rate,
            iterations=1000,
        )
        ca_norm_constants = spine_volume_norm_constant(
            spine_activity,
            spine_calcium,
            data.corrected_spine_volume,
            data.imaging_parameters["Zoom"],
            sampling_rate=sampling_rate,
            iterations=1000,
        )
    else:
        glu_norm_constants = np.array([None for x in range(spine_activity.shape[1])])
        ca_norm_constants = np.array([None for x in range(spine_activity.shape[1])])

    # Get specific movement periods if specified
    if movement_epoch == "movement":
        movement = data.lever_active
    elif movement_epoch == "rewarded":
        movement = data.rewarded_movement_binary
    elif movement_epoch == "unrewarded":
        movement = data.lever_active - data.rewarded_movement_binary
    elif movement_epoch == "nonmovement":
        movement = np.absolute(data.lever_active - 1)
    elif movement_epoch == "learned":
        movement, _, _, _, _ = quantify_movement_quality(
            data.mouse_id,
            spine_activity,
            data.lever_active,
            threshold=0.5,
            sampling_rate=sampling_rate,
        )
    else:
        movement = None

    # Set up output variables
    distance_coactivity_rate = None
    distance_bins = None
    local_correlation = np.zeros(spine_activity.shape[1]) * np.nan
    local_coactivity_rate = np.zeros(spine_activity.shape[1])
    local_coactivity_matrix = np.zeros(spine_activity.shape)
    spine_fraction_coactive = np.zeros(spine_activity.shape[1])
    local_coactive_spine_num = np.zeros(spine_activity.shape[1])
    local_coactive_spine_volumes = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    local_coactive_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium = np.zeros(spine_activity.shape[1]) * np.nan
    local_coactive_calcium = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_std = np.zeros(spine_activity.shape[1]) * np.nan
    local_coactive_std = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium_std = np.zeros(spine_activity.shape[1]) * np.nan
    local_coactive_calcium_std = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium_auc = np.zeros(spine_activity.shape[1]) * np.nan
    local_coactive_calcium_auc = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_traces = [None for i in local_correlation]
    local_coactive_traces = [None for i in local_correlation]
    spine_coactive_calcium_traces = [None for i in local_correlation]
    local_coactive_calcium_traces = [None for i in local_correlation]

    # Process distance dependence coactivity rates
    distance_coactivity_rate, distance_bins = local_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        bin_size=5,
        sampling_rate=sampling_rate,
    )

    # Process spines for each parent dendrite
    for dendrite in range(dendrite_activity.shape[1]):
        # Get spines on this dendrite
        if type(spine_groupings[dendrite]) == list:
            spines = spine_groupings[dendrite]
        else:
            spines = spine_groupings

        s_dFoF = spine_dFoF[:, spines]
        s_activity = spine_activity[:, spines]
        s_calcium = spine_calcium[:, spines]
        curr_positions = spine_positions[spines]
        curr_flags = [x for i, x in enumerate(spine_flags) if i in spines]
        curr_volumes = spine_volumes[spines]
        curr_glu_norm_constants = glu_norm_constants[spines]
        curr_ca_norm_constants = ca_norm_constants[spines]

        # Refine activity matrices for only movement epochs if specified
        if movement is not None:
            s_activity = (s_activity.T * movement).T

        # Analyze each spine individually
        for spine in range(s_dFoF.shape[1]):
            # Find neighboring spines
            ## Get relative spine positions
            curr_el_spines = find_spine_classes(curr_flags, "Eliminated Spine")
            curr_el_spines = np.array(curr_el_spines)
            target_position = curr_positions[spine]
            other_positions = [
                x for idx, x in enumerate(curr_positions) if idx != spine
            ]
            relative_positions = np.array(other_positions) - target_position
            relative_positions = np.absolute(relative_positions)
            ## Find spines within cluster distance
            nearby_spines = np.nonzero(relative_positions <= cluster_dist)[0]
            ## Remove the eliminated spines. Dan't want to consider their activity here
            nearby_spines = [i for i in nearby_spines if not curr_el_spines[i]]

            # Get relevant spine activity data
            curr_s_dFoF = s_dFoF[:, spine]
            curr_s_activity = s_activity[:, spine]
            curr_s_calcium = s_calcium[:, spine]
            nearby_s_dFoF = s_dFoF[:, nearby_spines]
            nearby_s_activity = s_activity[:, nearby_spines]
            nearby_s_calcium = s_calcium[:, nearby_spines]
            nearby_volumes = curr_volumes[nearby_spines]
            glu_constant = curr_glu_norm_constants[spine]
            ca_constant = curr_ca_norm_constants[spine]
            nearby_glu_constants = curr_glu_norm_constants[nearby_spines]
            nearby_ca_constants = curr_ca_norm_constants[nearby_spines]

            # Get local coactivity trace, where at least one nearby spine is coactive with targe
            combined_nearby_activity = np.sum(nearby_s_activity, axis=1)
            combined_nearby_activity[combined_nearby_activity > 1] = 1
            curr_local_coactivity = combined_nearby_activity * curr_s_activity
            local_coactivity_matrix[:, spines[spine]] = curr_local_coactivity

            # Skip further analysis if no conjunctive coactivity for current spine
            if not np.sum(curr_local_coactivity):
                continue

            # Get local coactivity timestamps
            local_timestamps = get_activity_timestamps(curr_local_coactivity)

            # Start analyzing the local coactivity
            _, event_rate, spine_frac, _ = get_coactivity_rate(
                curr_s_activity,
                curr_local_coactivity,
                curr_local_coactivity,
                sampling_rate=sampling_rate,
            )
            local_coactivity_rate[spines[spine]] = event_rate
            spine_fraction_coactive[spines[spine]] = spine_frac

            # Analyze the activity of the target spine
            (
                s_traces,
                _,
                s_amp,
                _,
                s_std,
                _,
                _,
                _,
                _,
            ) = get_dend_spine_traces_and_onsets(
                curr_s_activity.reshape(-1, 1),
                curr_s_activity.reshape(-1, 1),
                curr_s_dFoF.reshape(-1, 1),
                curr_s_dFoF.reshape(-1, 1),
                curr_local_coactivity.reshape(-1, 1),
                norm_constants=[glu_constant],
                activity_window=(-2, 2),
                sampling_rate=sampling_rate,
            )
            (
                s_ca_traces,
                _,
                s_ca_amp,
                s_ca_auc,
                s_ca_std,
                _,
                _,
                _,
                _,
            ) = get_dend_spine_traces_and_onsets(
                curr_s_activity.reshape(-1, 1),
                curr_s_activity.reshape(-1, 1),
                curr_s_calcium.reshape(-1, 1),
                curr_s_calcium.reshape(-1, 1),
                curr_local_coactivity.reshape(-1, 1),
                norm_constants=[ca_constant],
                activity_window=(-2, 2),
                sampling_rate=sampling_rate,
            )
            spine_coactive_amplitude[spines[spine]] = s_amp[0]
            spine_coactive_calcium[spines[spine]] = s_ca_amp[0]
            spine_coactive_std[spines[spine]] = s_std[0]
            spine_coactive_calcium_std[spines[spine]] = s_ca_std[0]
            spine_coactive_calcium_auc[spines[spine]] = s_ca_auc[0]
            spine_coactive_traces[spines[spine]] = s_traces[0]
            spine_coactive_calcium_traces[spines[spine]] = s_ca_traces[0]

            # Analyze the activity of nearby coactive spines
            (
                local_corr,
                coactive_num,
                coactive_vol,
                activity_amp,
                ca_activity_amp,
                activity_std,
                ca_activity_std,
                ca_activity_auc,
                coactive_s_traces,
                coactive_s_ca_traces,
            ) = nearby_spine_conjunctive_events(
                timestamps=local_timestamps,
                spine_dFoF=curr_s_dFoF,
                nearby_dFoF=nearby_s_dFoF,
                nearby_calcium=nearby_s_calcium,
                nearby_activity=nearby_s_activity,
                dendrite_dFoF=curr_s_dFoF,
                nearby_spine_volumes=nearby_volumes,
                target_constant=glu_constant,
                glu_constants=nearby_glu_constants,
                ca_constants=nearby_ca_constants,
                activity_window=(-2, 2),
                sampling_rate=sampling_rate,
            )
            local_correlation[spines[spine]] = local_corr
            local_coactive_spine_num[spines[spine]] = coactive_num
            local_coactive_spine_volumes[spines[spine]] = coactive_vol
            local_coactive_amplitude[spines[spine]] = activity_amp
            local_coactive_calcium[spines[spine]] = ca_activity_amp
            local_coactive_std[spines[spine]] = activity_std
            local_coactive_calcium_std[spines[spine]] = ca_activity_std
            local_coactive_calcium_auc[spines[spine]] = ca_activity_auc
            local_coactive_traces[spines[spine]] = coactive_s_traces
            local_coactive_calcium_traces[spines[spine]] = coactive_s_ca_traces

    return (
        distance_coactivity_rate,
        distance_bins,
        local_correlation,
        local_coactivity_rate,
        local_coactivity_matrix,
        spine_fraction_coactive,
        local_coactive_spine_num,
        local_coactive_spine_volumes,
        spine_coactive_amplitude,
        local_coactive_amplitude,
        spine_coactive_calcium,
        local_coactive_calcium,
        spine_coactive_std,
        local_coactive_std,
        spine_coactive_calcium_std,
        local_coactive_calcium_std,
        spine_coactive_calcium_auc,
        local_coactive_calcium_auc,
        spine_coactive_traces,
        local_coactive_traces,
        spine_coactive_calcium_traces,
        local_coactive_calcium_traces,
    )


def local_coactivity_rate_analysis(
    spine_activity,
    spine_positions,
    flags,
    spine_grouping,
    bin_size=5,
    sampling_rate=60,
):
    """Function to calculate pairwise spine coactivity rate between 
        all spines along the same dendrite. Spine rates are then binned
        based on their distance from the target spine
        
        INPUT PARAMETERS
            spine_activity - np.array of the binarized spine activity traces
                            Each column represents each spine
            
            spine_positions - list of the corresponding spine positions along the dendrite
                              for each spine

            flags - list of the spine flags
            
            spine_grouping - list with the corresponding groupings of spines on
                             the same dendrite
            
            bin_size - int or float specifying the distance to bin over

        OUTPUT PARAMETERS
            coactivity_matrix - np.array of the normalized coactivity for each spine (columns)
                                over the binned distances (rows)
            
            position_bins - np.array of the distances data were binned over
            
    """
    # Set up the position bins
    MAX_DIST = 30
    bin_num = int(MAX_DIST / bin_size)
    position_bins = np.linspace(0, MAX_DIST, bin_num + 1)

    # First sort out spine_groping to make sure you can iterate
    if type(spine_grouping[0]) != list:
        spine_grouping = [spine_grouping]

    coactivity_matrix = np.zeros((bin_num, spine_activity.shape[1]))

    # find indexes of eliminated spines
    el_spines = find_spine_classes(flags, "Eliminated Spine")
    el_spines = np.array(el_spines)

    # Now iterate through each dendrite grouping
    for spines in spine_grouping:
        s_activity = spine_activity[:, spines]
        positions = np.array(spine_positions)[spines]
        curr_el_spines = el_spines[spines]

        # Go through each spine
        for i in range(s_activity.shape[1]):
            current_coactivity = []
            curr_spine = s_activity[:, i]
            # Get coactivity with each other spine
            for j in range(s_activity.shape[1]):
                # Don't compare spines to themselves
                if j == i:
                    continue
                # Don't compare eliminated spines
                if curr_el_spines[i] == True:
                    current_coactivity.append(np.nan)
                    continue
                if curr_el_spines[j] == True:
                    current_coactivity.append(np.nan)
                    continue
                test_spine = s_activity[:, j]
                co_rate = calculate_coactivity(curr_spine, test_spine, sampling_rate)
                current_coactivity.append(co_rate)

            # Order by positions
            curr_pos = positions[i]
            pos = [x for idx, x in enumerate(positions) if idx != i]
            # normalize  distances relative to current position
            relative_pos = np.array(pos) - curr_pos
            # Make all distances positive
            relative_pos = np.absolute(relative_pos)
            # Sort coactivity and position based on distance
            sorted_coactivity = np.array(
                [x for _, x in sorted(zip(relative_pos, current_coactivity))]
            )
            sorted_positions = np.array(
                [y for y, _ in sorted(zip(relative_pos, current_coactivity))]
            )
            # Bin the data
            binned_coactivity = bin_by_position(
                sorted_coactivity, sorted_positions, position_bins
            )
            coactivity_matrix[:, spines[i]] = binned_coactivity

    return coactivity_matrix, position_bins


def calculate_coactivity(spine_1, spine_2, sampling_rate):
    """Helper function to calculate spine coactivity rate"""
    duration = len(spine_1) / sampling_rate
    coactivity = spine_1 * spine_2
    events = np.nonzero(np.diff(coactivity) == 1)[0]
    event_num = len(events)
    event_freq = event_num / duration

    # Normalize frequency based on overall activity levels
    spine_1_freq = len(np.nonzero(np.diff(spine_1) == 1)[0]) / duration
    spine_2_freq = len(np.nonzero(np.diff(spine_2) == 1)[0]) / duration
    geo_mean = stats.gmean([spine_1_freq, spine_2_freq])

    coactivity_rate = event_freq / geo_mean

    return coactivity_rate


def bin_by_position(data, positions, bins):
    """Helper function to bin the data by position"""
    binned_data = []

    for i in range(len(bins)):
        if i != len(bins) - 1:
            idxs = np.where((positions > bins[i]) & (positions <= bins[i + 1]))
            binned_data.append(np.nanmean(data[idxs]))
        # else:
        #    idxs = np.where(positions > bins[i])
        #    binned_data.append(np.nanmean(data[idxs]))

    return np.array(binned_data)

