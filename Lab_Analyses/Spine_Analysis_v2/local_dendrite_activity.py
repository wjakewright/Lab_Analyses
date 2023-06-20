import numpy as np
import scipy.signal as sysignal

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import find_present_spines
from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.coactivity_functions import calculate_coactivity


def local_dendrite_activity(
    spine_activity,
    dendrite_activity,
    spine_positions,
    spine_flags,
    spine_groupings,
    nearby_spine_idxs,
    poly_dend_dFoF,
    poly_dend_positions,
    activity_window=(-2, 4),
    constrain_matrix=None,
    sampling_rate=60,
):
    """Function to analyze the activity of local dendritic segments when spines are
        active, coactive, or their neighbors are coactivity
        
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
            
            poly_dend_positions - list of arrays of the positions of each poly
                                  roi along the dendrite
            
            activity_window - tuple specifying the activity window in sec to analyze

            constrain_matrix - np.array of binarized events to constrain the 
                              activity to (e.g., movements)
            
            sampling_rate - int specifying the imaigng sampling rate
        
        OUTPUT PARAMETERS
            coactive_local_dend_traces - list of 2d nparray of local dends
                                        activity during each coactive event
            
            coactive_local_dend_amplitude - np.array of the peak amplitude of local
                                            dendritic activity during coactiv events
            
            noncoactive_local_dend_traces - list of 2d array of local dendcs activity
                                            when it is active, but not coactive
            
            noncoactive_local_dend_amplitude - np.array of the peak amplitude of local 
                                              dendritic events during noncoactive events
                                              
            nearby_local_dend_traces - list of 2d arrays of local dendrite activity
                                        whenever nearby spiens are active but target is not
            
            nearby_local_dend_amplitude - np.array of the peak amplitude of local dendritic
                                         events during noncoactive nearby activity
    """
    # Sort out the spine groupings
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # Constrain activity if necessary
    if constrain_matrix is not None:
        if len(constrain_matrix.shape) == 1:
            constrain_matrix = constrain_matrix.reshape(-1, 1)
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
    noncoactive_local_dend_traces = [None for x in range(activity_matrix.shape[1])]
    noncoactive_local_dend_amplitude = np.zeros(activity_matrix.shape[1]) * np.nan
    nearby_local_dend_traces = [None for x in range(activity_matrix.shape[1])]
    nearby_local_dend_amplitude = np.zeros(activity_matrix.shape[1]) * np.nan

    # Iterate through each dendrite
    for i, spines in enumerate(spine_groupings):
        curr_dend_dFoF = poly_dend_dFoF[i]
        curr_dend_positions = poly_dend_positions[i]
        curr_s_activity = activity_matrix[:, spines]
        curr_present = present_spines[spines]
        curr_positions = spine_positions[spines]
        curr_nearby_spines = [nearby_spine_idxs[j] for j in spines]

        # Not iterate through each spine
        for spine in range(curr_s_activity.shape[1]):
            # skip non present spines
            if curr_present[spine] == False:
                continue
            # First find the nearest local dendrite roi
            spine_pos = curr_positions[spine]
            ## Make dendrite positions relative
            dend_pos = np.array(curr_dend_positions) - spine_pos
            dend_pos = np.absolute(dend_pos)
            ## Find the idx and dFoF of the closest dend roi
            dend_idx = np.argmin(dend_pos)
            dend_dFoF = curr_dend_dFoF[:, dend_idx]

            # Get coactive and noncoactive timestamps
            if (curr_nearby_spines[spine] is None) or (
                len(curr_nearby_spines[spine]) == 0
            ):
                combined_nearby_activity = np.zeros(curr_s_activity.shape[0])
            else:
                combined_nearby_activity = np.nansum(
                    activity_matrix[:, curr_nearby_spines[spine]], axis=1
                )
                combined_nearby_activity[combined_nearby_activity > 1] = 1
            combined_nearby_inactivity = 1 - combined_nearby_activity
            ## Get binary trace
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

            ## timestamps and refine
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

            # Get the dendrite traces and amplitude
            ## Coactive events
            coactive_traces, coactive_amplitude = get_dend_traces(
                coactive_stamps, dend_dFoF, activity_window, sampling_rate
            )
            ## Noncoactive events
            noncoactive_traces, noncoactive_amplitude = get_dend_traces(
                noncoactive_stamps, dend_dFoF, activity_window, sampling_rate,
            )
            ## Store the results
            coactive_local_dend_traces[spines[spine]] = coactive_traces
            coactive_local_dend_amplitude[spines[spine]] = coactive_amplitude
            noncoactive_local_dend_traces[spines[spine]] = noncoactive_traces
            noncoactive_local_dend_amplitude[spines[spine]] = noncoactive_amplitude

            # Assess when nearby spines are active
            nearby_traces = []
            nearby_amplitude = []
            ## Get the activity for every nearby spine individually
            for nearby in curr_nearby_spines[spine]:
                nearby_activity = activity_matrix[:, nearby]
                spine_inactivity = 1 - curr_s_activity[:, spine]
                _, _, _, _, isolated_activity = calculate_coactivity(
                    nearby_activity,
                    spine_inactivity,
                    sampling_rate=sampling_rate,
                    norm_method="mean",
                )
                if not np.nansum(isolated_activity):
                    continue
                isolated_stamps = t_stamps.get_activity_timestamps(isolated_activity)
                isolated_stamps = [x[0] for x in isolated_stamps]
                isolated_stamps = t_stamps.refine_activity_timestamps(
                    isolated_stamps,
                    window=activity_window,
                    max_len=len(nearby_activity),
                    sampling_rate=sampling_rate,
                )
                traces, amplitude = get_dend_traces(
                    isolated_stamps, dend_dFoF, activity_window, sampling_rate
                )
                nearby_traces.append(traces)
                nearby_amplitude.append(amplitude)

            ## Pool and average across nearby spines
            if len(nearby_traces) == 0:
                continue
            if len(nearby_traces) == 1:
                nearby_local_dend_traces[spines[spine]] = nearby_traces[0]
                nearby_local_dend_amplitude[spines[spine]] = nearby_amplitude[0]
                continue
            nearby_local_dend_traces[spines[spine]] = np.hstack(nearby_traces)
            nearby_local_dend_amplitude[spines[spine]] = np.nanmean(nearby_amplitude)

    return (
        coactive_local_dend_traces,
        coactive_local_dend_amplitude,
        noncoactive_local_dend_traces,
        noncoactive_local_dend_amplitude,
        nearby_local_dend_traces,
        nearby_local_dend_amplitude,
    )


def get_dend_traces(timestamps, dend_dFoF, activity_window, sampling_rate):
    """Helper function to get the timelocked traces and amplitudes of local
        dendrite traces. Works with only one set of timestamps and a single
        local dendendrite dfoF trace
    """
    if len(timestamps) == 0:
        traces = (
            np.zeros(((activity_window[1] - activity_window[0]) * sampling_rate, 1))
            * np.nan
        )
        amplitude = np.nan
        return traces, amplitude
    # Get the traces
    traces, mean = d_utils.get_trace_mean_sem(
        dend_dFoF.reshape(-1, 1),
        ["Activity"],
        timestamps,
        activity_window,
        sampling_rate,
    )
    mean = mean["Activity"][0]
    traces = traces["Activity"]

    # Smooth mean trace
    mean = sysignal.savgol_filter(mean, 31, 3)

    # Get the max amplitude
    max_idx = np.argmax(mean)

    # Average amplitude around the max
    win = int((0.5 * sampling_rate) / 2)
    amplitude = np.nanmean(mean[max_idx - win : max_idx + win])

    return traces, amplitude

