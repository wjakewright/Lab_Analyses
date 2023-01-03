import numpy as np

from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_onset import find_activity_onset
from Lab_Analyses.Utilities.activity_timestamps import (
    get_activity_timestamps,
    refine_activity_timestamps,
)


def movement_related_activity(
    movement_trace, dFoF_matrix, norm=None, sampling_rate=60, activity_window=(-2, 4),
):
    """Function to analyze movement_related activity 
        INPUT PARAMETERS
            movement_trace - np.array of the binarized movement trace
            
            dFoF_matrix - 2d np.array of the dFoF traces for all ROIs
            
            norm - list of constants to normalize activity traces by
            
            sampling_rate - int specifying the imaging rate
            
            activity_window - tuple specifying the period around movement to analyze in sec
    """
    center_point = int(np.absolute(activity_window[0]) * sampling_rate)

    # Get timestamps for movement onsets
    timestamps = get_activity_timestamps(movement_trace)
    timestamps = [x[0] for x in timestamps]
    timestamps = refine_activity_timestamps(
        timestamps,
        window=activity_window,
        max_len=dFoF_matrix.shape[0],
        sampling_rate=sampling_rate,
    )

    # Get individual traces and mean traces
    traces, mean_sems = d_utils.get_trace_mean_sem(
        dFoF_matrix,
        [f"roi {a}" for a in range(dFoF_matrix.shape[1])],
        timestamps,
        window=activity_window,
        sampling_rate=sampling_rate,
    )

    # Reorganize the trace outputs
    traces = list(traces.values())
    mean_traces = [x[0] for x in mean_sems.values()]

    if norm is not None:
        traces = [traces[i] / norm[i] for i in range(dFoF_matrix.shape[1])]
        mean_traces = [mean_traces[i] / norm[i] for i in range(dFoF_matrix.shape[1])]

    # Get onsets and amplitudes
    temp_onsets, amplitudes = find_activity_onset(mean_traces)

    relative_onsets = np.zeros(dFoF_matrix.shape[1])
    # Get relative onsets for each roi
    for roi in range(dFoF_matrix.shape[1]):
        if not np.isnan(amplitudes[roi]):
            rel_onset = (temp_onsets[roi] - center_point) / sampling_rate
            relative_onsets[roi] = rel_onset
        else:
            relative_onsets[roi] = np.nan
            amplitudes[roi] = 0

    return traces, amplitudes, relative_onsets
