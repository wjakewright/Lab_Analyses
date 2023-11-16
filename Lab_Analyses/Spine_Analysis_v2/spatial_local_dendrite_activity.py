import numpy as np
import scipy.signal as sysignal

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    bin_by_position, find_present_spines)
from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.coactivity_functions import calculate_coactivity


def spatial_local_dendrite_activity(
    spine_activity,
    dendrite_activity,
    spine_positions,
    spine_flags,
    spine_groupings,
    nearby_spine_idxs,
    poly_dend_dFoF,
    poly_dend_positions,
    activity_window=(-2,4),
    constrain_matrix=None,
    sampling_rate=60,
):
    """Function to analyze the activity of spatial domains of local dendritic segments 
        when spines are active, coactive, or their neighbors are coactive

        INPUT PARAMETERS    
            spine_activity - 2d np.array of the binarized spine activity

            dendrite_activity - 2d np.array of the binarized global dendrite activity

            spine_positions - np.array of the spine positions along the dendrite

            spine_flags - list of the spine flags

            spine_groupings - list of the spine groupings on the different dendrites

            nearby_spine_idxs - list of the nearby spine idxs for each spine

            poly_dend_dFoF - list of 2d arrays containing the dFoF traces for each
                            dendrite poly roi (columns) on each dendrite 
                            (list items)

            poly_dend_positions - list of arrays of the positions of each poly roi
                                    along the dendrite

            activity_window - tuple specifying the activity window in sec to analyze

            constrain matrix - np.array of binarized events to constrain the 
                                activity to (e.g. movements)

            sampling_rate - int specifying the imaging sampling_rate

        OUTPUT PARAMETERS
             cocactive_local_dend_traces - list of 2d np.array of nearest local
                                            dends activity during each coactive event
            
            coactive_local_dend_amplitude - np.array of the peak amplitude of local
                                            dendritic activity during coactive events

            coactive_local_dend_amplitude_dist - 2d np.array of local dendrite amplitude
                                                binned over different distances (row) for 
                                                each spine when coactive

            noncoactive_local_dend_traces - list of 2d np.array of nearest local dends activity
                                            during noncoactive events
                                
            noncoactive_local_dend_amplitude - np.array of the peak amplitude of local
                                                dend events during noncoactive events

            noncoactive_local_dend_amplitude_dist - 2d np.array of local dendrite amplitude
                                                    binned over different distances (row) for
                                                    each spine when noncoactive

    """
    # Setup position bins
    MAX_DIST = 10
    bin_num = 10
    position_bins = np.linspace(0, MAX_DIST, 11)

    # Sort out the spine groupings
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]
    
    # Constrain activity if specified
    if constrain_matrix is not None:
        if len(constrain_matrix.shape) == 1:
            constrain_matrix = constrain_matrix.reshape(-1,1)
        activity_matrix = spine_activity * constrain_matrix
    else:
        activity_matrix = spine_activity

    # Remove events that overlap with global dendritic events
    ## Invert dendrite activity
    dendrite_inactivity = 1 - dendrite_activity
    activity_matrix = activity_matrix * dendrite_inactivity

    # Get present spines
    present_spines = find_present_spines(spine_flags)

    # Set up output variables
    coactive_local_dend_traces = [None for x in range(activity_matrix.shape[1])]
    coactive_local_dend_amplitude = np.zeros(activity_matrix.shape[1]) * np.nan
    coactive_local_dend_amplitude_dist = np.zeros((bin_num, activity_matrix.shape[1])) * np.nan
    noncoactive_local_dend_traces = [None for x in range(activity_matrix.shape[1])]
    noncoactive_local_dend_amplitude = np.zeros(activity_matrix.shape[1]) * np.nan
    noncoactive_local_dend_amplitude_dist = np.zeros((bin_num, activity_matrix.shape[1])) * np.nan

    # Iterate through each dendrite
    for i, spines in enumerate(spine_groupings):
        curr_dend_dFoF = poly_dend_dFoF[i]
        curr_dend_positions = poly_dend_positions[i]
        curr_s_activity = activity_matrix[:, spines]
        curr_present = present_spines[spines]
        curr_positions = spine_positions[spines]
        curr_nearby_spines = [nearby_spine_idxs[j] for j in spines]

        # Iterate through each spine
        for spine in range(curr_s_activity.shape[1]):
            # Skip non present spines
            if curr_present[spine] == False:
                continue
            # Find relative positions on dendrite ROIs
            ## Current spine position
            spine_pos = curr_positions[spine]
            ## Make dendrite positions relative
            dend_pos = np.array(curr_dend_positions) - spine_pos
            dend_pos = np.absolute(dend_pos)

            # Define spine coactive and noncoactive periods
            ## combine nearby spine activity
            if (curr_nearby_spines[spine] is None) or (len(curr_nearby_spines[spine]) == 0):
                combined_nearby_activity = np.zeros(curr_s_activity.shape[0])
            else:
                combined_nearby_activity = np.nansum(
                    activity_matrix[:, curr_nearby_spines[spine]], axis=1
                )
                combined_nearby_activity[combined_nearby_activity > 1] = 1
            ## Get inactivity
            combined_nearby_inactivity = 1 - combined_nearby_activity
            ## Get binary traces
            _, _, _, _, coactive = calculate_coactivity(
                curr_s_activity[:, spine],
                combined_nearby_activity,
                sampling_rate=sampling_rate,
                norm_method="mean",
            )
            _, _, _, _, noncoactive = calculate_coactivity(
                curr_s_activity[:, spine],
                combined_nearby_inactivity,
                sampling_rate=sampling_rate,
                norm_method="mean",
            )

            ## Timestamps and refine
            if np.nansum(coactive):
                coactive_stamps = t_stamps.get_activity_timestamps(coactive)
                coactive_stamps = [x[0] for x in coactive_stamps]
                coactive_stamps = t_stamps.refine_activity_timestamps(
                    coactive_stamps,
                    window=activity_window,
                    max_len=len(curr_s_activity[:, spine]),
                    sampling_rate=sampling_rate,
                )
            else:
                coactive_stamps = []
            if np.nansum(noncoactive):
                noncoactive_stamps = t_stamps.get_activity_timestamps(noncoactive)
                noncoactive_stamps = [x[0] for x in noncoactive_stamps]
                noncoactive_stamps = t_stamps.refine_activity_timestamps(
                    noncoactive_stamps,
                    window=activity_window,
                    max_len=len(curr_s_activity[:, spine]),
                    sampling_rate=sampling_rate,
                )
            else:
                noncoactive_stamps = []

            # Setup temp variables
            temp_coactive_traces = []
            temp_coactive_amps = []
            temp_noncoactive_traces = []
            temp_noncoactive_amps = []
            for i in range(curr_dend_dFoF.shape[1]):
                coactive_traces, coactive_amp = get_dend_traces(
                    coactive_stamps, curr_dend_dFoF[:, i], activity_window, sampling_rate,
                )
                noncoactive_traces, noncoactive_amp = get_dend_traces(
                    noncoactive_stamps, curr_dend_dFoF[:, i], activity_window, sampling_rate,
                )
                temp_coactive_traces.append(coactive_traces)
                temp_coactive_amps.append(coactive_amp)
                temp_noncoactive_traces.append(noncoactive_traces)
                temp_noncoactive_amps.append(noncoactive_amp)

            # Sort dendrite variables by relative position
            sorted_coactive_traces = [x for _, x in sorted(zip(dend_pos, temp_coactive_traces))]
            sorted_coactive_amps = np.array([x for _, x in sorted(zip(dend_pos, temp_coactive_amps))])
            sorted_noncoactive_traces = [x for _, x in sorted(zip(dend_pos, temp_noncoactive_traces))]
            sorted_noncoactive_amps = np.array([x for _, x in sorted(zip(dend_pos, temp_noncoactive_amps))])
            sorted_positions = np.array([y for y, _ in sorted(zip(dend_pos, temp_coactive_amps))])

            # bin amplitudes by position
            binned_coactive_amps = bin_by_position(sorted_coactive_amps, sorted_positions, position_bins)
            binned_noncoactive_amps = bin_by_position(sorted_noncoactive_amps, sorted_positions, position_bins)

            # Store values
            ## Take on the closes traces
            coactive_local_dend_traces[spines[spine]] = sorted_coactive_traces[0]
            noncoactive_local_dend_traces[spines[spine]] = sorted_noncoactive_traces[0]
            ## Closest amplitude
            coactive_local_dend_amplitude[spines[spine]] = sorted_coactive_amps[0]
            noncoactive_local_dend_amplitude[spines[spine]] = sorted_noncoactive_amps[0]
            ## Binned amplitudes
            coactive_local_dend_amplitude_dist[:, spines[spine]] = binned_coactive_amps
            noncoactive_local_dend_amplitude_dist[:, spines[spine]] = binned_noncoactive_amps

    return (
        coactive_local_dend_traces,
        coactive_local_dend_amplitude,
        coactive_local_dend_amplitude_dist,
        noncoactive_local_dend_traces,
        noncoactive_local_dend_amplitude,
        noncoactive_local_dend_amplitude_dist,
        position_bins,
    )

            



def get_dend_traces(timestamps, dend_dFoF, activity_window, sampling_rate):
    """Helper function to get teh timelocked traces and amplitudes of local 
       dendrite traces. Works with only ons set of timestamps and a single
       local dendrite dFoF trace
    """

    # Return null values if there are no timestamps
    if len(timestamps) == 0:
        traces = (
            np.zeros(((activity_window[1] - activity_window[0]) * sampling_rate, 1))
            * np.nan
        )
        amplitude = np.nan
        return traces, amplitude

    # Get the traces
    traces, mean = d_utils.get_trace_mean_sem(
        dend_dFoF.reshape(-1,1),
        ["Activity"],
        timestamps,
        activity_window,
        sampling_rate,
    )
    mean = mean["Activity"][0]
    traces = traces["Activity"]

    # Smooth the mean trace
    mean = sysignal.savgol_filter(mean, 31, 3)

    # Get max amplitude idx
    max_idx = np.argmax(mean)

    # Average amplitude around the max
    win = int((0.5 * sampling_rate) / 2)
    amplitude = np.nanmean(mean[max_idx-win:max_idx+win])

    return traces, amplitude