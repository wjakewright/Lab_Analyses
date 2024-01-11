import numpy as np

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import find_present_spines
from Lab_Analyses.Utilities.activity_timestamps import (
    get_activity_timestamps,
    refine_activity_timestamps,
    timestamp_onset_correction,
)
from Lab_Analyses.Utilities.coactivity_functions import find_individual_onsets
from Lab_Analyses.Utilities.mean_trace_functions import find_peak_amplitude


def nearby_coactive_spine_activity(
    nearby_spine_idxs,
    coactivity_matrix,
    spine_flags,
    spine_activity,
    spine_dFoF,
    spine_calcium,
    offsets=None,
    norm_constants=None,
    activity_window=(-2, 4),
    sampling_rate=60,
):
    """Function to analyze the activity of nearby spines during coactive
    events

    INPUT PARAMETERS
        nearby_spine_idxs - list of arrays containing the idxs of nearby spines for each
                            spine

        coactivity_matrix - 2d np.array of the binarized coactivity trace for each
                             neuron (columns)

        spine_flags - list of the spine flags

        spine_activity - 2d np.array of the binarized activity for each spine (column)

        spine_dFoF - 2d np.array of the dFoF traces for each spine (column)

        spine_calcium - 2d np.array of the calcium_dFoF traces for each spine (column)

        offsets - np.array of values to offset coactive timestamps by for aligning the
                 nearby activity to (e.g., target spine onsets)

        norm_constants - tuple of arrays containing the norm constants for GluSnFr and Calcium

        activity_window - tuple specifying the window around which you want the activity
                          from (e.g., (-2, 4) for 2 sec before and 4 sec after)

        sampling_rate - int specifying the sampling rate

    OUTPUT PARAMETERS
        avg_coactive_spine_num -  np.array of the num of spines coactive with each spine

        sum_nearby_coactive_amplitude - np.array of the max amplitude of average
                                        summed activity of nearby spines during coactivity

        sum_nearby_coactive_calcium - np.array of the max amplitude of the average summed
                                      calcium of nearby spines during coactivity

        nearby_spine_onset - np.array of the average onset of nearby spines relative to the
                             target spine

        nearby_spine_onset_jitter - np.array of the average onset std of nearby spines
                                    relative to the target spine

        sum_nearby_coactive_traces - list of 2d arrays of the summed coactive traces for
                                     each coactive event (column) for each spine (list items)

        sum_nearby_coactive_calcium_traces - list od 2d arrays of the summmed coactive calcium
                                             traces for each coactive event for each spine


    """
    # Get window in frames
    before_f = int(activity_window[0] * sampling_rate)
    after_f = int(activity_window[1] * sampling_rate)
    center_point = np.abs(activity_window[0] * sampling_rate)
    # Set up norm constants
    if norm_constants is not None:
        glu_norm_constants = norm_constants[0]
        ca_norm_constants = norm_constants[1]
    else:
        glu_norm_constants = None
        ca_norm_constants = None

    # Set up some outputs
    avg_coactive_spine_num = np.zeros(coactivity_matrix.shape[1])
    sum_nearby_coactive_amplitude = np.zeros(coactivity_matrix.shape[1]) * np.nan
    sum_nearby_coactive_calcium = np.zeros(coactivity_matrix.shape[1]) * np.nan
    nearby_spine_onset = np.zeros(coactivity_matrix.shape[1]) * np.nan
    nearby_spine_onset_jitter = np.zeros(coactivity_matrix.shape[1]) * np.nan
    sum_nearby_coactive_traces = [None for x in range(coactivity_matrix.shape[1])]
    sum_nearby_coactive_calcium_traces = [
        None for x in range(coactivity_matrix.shape[1])
    ]

    # Get present spines
    present_spines = find_present_spines(spine_flags)

    # Iterate through each spine
    for spine in range(coactivity_matrix.shape[1]):
        # Skip if not present
        if present_spines[spine] is False:
            continue
        # Setup temporary variables
        coactive_spine_num = []
        sum_nearby_traces = []
        sum_nearby_ca_traces = []

        # Get relevant spine data
        nearby_spines = nearby_spine_idxs[spine]
        if (nearby_spines is None) or (len(nearby_spines) == 0):
            continue
        coactivity = coactivity_matrix[:, spine]
        target_activity = spine_activity[:, spine]
        nearby_activity = spine_activity[:, nearby_spines]
        nearby_dFoF = spine_dFoF[:, nearby_spines]
        nearby_calcium = spine_calcium[:, nearby_spines]
        if len(nearby_activity.shape) == 1:
            nearby_activity = nearby_activity.reshape(-1, 1)
        nearby_coactivity = nearby_activity * target_activity.reshape(-1, 1)

        # Get coactivity timestamps
        if not np.nansum(coactivity):
            continue
        # Old method
        # timestamps = get_activity_timestamps(coactivity)
        # timestamps = refine_activity_timestamps(
        #    timestamps,
        #    window=activity_window,
        #    max_len=len(coactivity),
        #    sampling_rate=sampling_rate,
        # )
        # if offsets is not None:
        #    timestamps = timestamp_onset_correction(
        #        timestamps, activity_window, offsets[spine], sampling_rate
        #    )

        ## Get timestamps of the target spine for each coactivity event
        _, timestamps = find_individual_onsets(
            target_activity,
            target_activity,
            coactivity=coactivity,
            sampling_rate=sampling_rate,
            activity_window=activity_window,
        )
        if len(timestamps) == 0:
            continue

        timestamps = refine_activity_timestamps(
            timestamps,
            window=activity_window,
            max_len=len(coactivity),
            sampling_rate=sampling_rate,
        )

        # Iterate through each coactivity event
        spine_wise_onsets = [[] for i in range(nearby_activity.shape[1])]
        for event in timestamps:
            # Temp variables
            sum_coactive_s_traces = []
            sum_coactive_ca_traces = []
            # Check activity of each nearby spine
            for nearby in range(nearby_activity.shape[1]):
                event_activity = nearby_coactivity[:, nearby][
                    event + before_f : event + after_f
                ]
                # Grab trace and onset if there is activity
                if np.nansum(event_activity):
                    activity = nearby_activity[:, nearby][
                        event + before_f : event + after_f
                    ]
                    dFoF = nearby_dFoF[:, nearby][event + before_f : event + after_f]
                    calcium = nearby_calcium[:, nearby][
                        event + before_f : event + after_f
                    ]
                    if glu_norm_constants is not None:
                        dFoF = dFoF / glu_norm_constants[nearby_spines[nearby]]
                        calcium = calcium / ca_norm_constants[nearby_spines[nearby]]
                    # Store the traces
                    sum_coactive_s_traces.append(dFoF)
                    sum_coactive_ca_traces.append(calcium)
                    # Get binary onset
                    boundaries = np.insert(np.diff(activity), 0, 0, axis=0)
                    try:
                        onset = np.nonzero(boundaries == 1)[0][0]
                    except IndexError:
                        onset = 0
                    ta = target_activity[event + before_f : event + after_f]
                    ta[:center_point] = 0
                    t_boundary = np.insert(np.diff(ta), 0, 0, axis=0)
                    try:
                        t_onset = np.nonzero(t_boundary == 1)[0][0]
                    except IndexError:
                        t_onset = center_point
                    rel_onset = (onset - t_onset) / sampling_rate
                    spine_wise_onsets[nearby].append(rel_onset)

            # Process some of the traces
            ## Skip if there are not spines
            if len(sum_coactive_s_traces) == 0:
                continue
            ## Append values if only one spine is coactive
            if len(sum_coactive_s_traces) == 1:
                coactive_spine_num.append(1)
                sum_nearby_traces.append(sum_coactive_s_traces[0])
                sum_nearby_ca_traces.append(sum_coactive_ca_traces[0])
                continue
            ## Sum multiple coactive spine traces together and append
            trace_array = np.vstack(sum_coactive_s_traces).T
            ca_trace_array = np.vstack(sum_coactive_ca_traces).T
            sum_trace = np.nansum(trace_array, axis=1)
            ca_sum_trace = np.nansum(ca_trace_array, axis=1)
            sum_nearby_traces.append(sum_trace)
            sum_nearby_ca_traces.append(ca_sum_trace)
            coactive_spine_num.append(trace_array.shape[1])

        # Check how many coactive events there are
        ## Skip if there are no coactive events
        if len(sum_nearby_traces) == 0:
            continue

        # Convert arrays into proper format
        if len(sum_nearby_traces) == 1:
            sum_nearby_traces = sum_nearby_traces[0].reshape(-1, 1)
            sum_nearby_ca_traces = sum_nearby_ca_traces[0].reshape(-1, 1)
        else:
            sum_nearby_traces = np.vstack(sum_nearby_traces).T
            sum_nearby_ca_traces = np.vstack(sum_nearby_ca_traces).T

        # Perform the final calculations
        avg_coactive_spine_num[spine] = np.nanmean(coactive_spine_num)

        # Find the mean amplitude
        ## avg traces
        avg_nearby_traces = np.nanmean(sum_nearby_traces, axis=1)
        avg_nearby_ca_traces = np.nanmean(sum_nearby_ca_traces, axis=1)
        ## Get peaks
        peak_amp, _ = find_peak_amplitude(
            [avg_nearby_traces], smooth=True, window=None, sampling_rate=sampling_rate
        )
        ca_peak_amp, _ = find_peak_amplitude(
            [avg_nearby_ca_traces],
            smooth=True,
            window=None,
            sampling_rate=sampling_rate,
        )
        sum_nearby_coactive_amplitude[spine] = peak_amp[0]
        sum_nearby_coactive_calcium[spine] = ca_peak_amp[0]

        # Handle the onset and jitter
        nearby_onsets = [np.nanmean(x) for x in spine_wise_onsets]
        nearby_jitter = [np.nanstd(x) for x in spine_wise_onsets]
        nearby_spine_onset[spine] = np.nanmean(nearby_onsets)
        nearby_spine_onset_jitter[spine] = np.nanmean(nearby_jitter)

        # Store traces
        sum_nearby_coactive_traces[spine] = sum_nearby_traces
        sum_nearby_coactive_calcium_traces[spine] = sum_nearby_ca_traces

    return (
        avg_coactive_spine_num,
        sum_nearby_coactive_amplitude,
        sum_nearby_coactive_calcium,
        nearby_spine_onset,
        nearby_spine_onset_jitter,
        sum_nearby_coactive_traces,
        sum_nearby_coactive_calcium_traces,
    )
