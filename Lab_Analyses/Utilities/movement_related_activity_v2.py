import numpy as np

from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_timestamps import (
    get_activity_timestamps,
    refine_activity_timestamps,
)
from Lab_Analyses.Utilities.mean_trace_functions import (
    find_activity_onset,
    find_peak_amplitude,
)


def movement_related_activity(
    lever_active,
    activity,
    dFoF,
    norm=None,
    smooth=False,
    avg_window=None,
    sampling_rate=60,
    activity_window=(-2, 4),
):
    """Function to analyze activity traces during movements. Only focuses on movements
        during which that spine was active
        
        INPUT PARAMETERS
            lever_active - np.array of the binarized movement trace

            activity - 2d np.array of the binarized activity for all ROIs
            
            dFoF - 2d np.array of the dFoF traces for all ROIs
            
            norm - list of constants to normalize activity by

            smooth - boolean specifying whether or not to smooth the traces

            avg_window - float specifying the window you wish to average the peak
                        over. If None, it will use on the max peak amplitude
            
            sampling_rate - int specifying the imaging rate
            
            activity_window - tuple specifying the period around movement to analyze
                              in seconds
        
        OUTPUT PARAMETERS
            movement_traces - list of 2d np.arrays containing the activity during each 
                            movement (col) for each roi (item in list)
            
            movement_amplitudes - np.array of the max amplitudes of the mean traces 
                                  for each roi
            
            relative_onsets - np.array of the activity onset relative to movement onset
    """
    center_point = int(np.absolute(activity_window[0]) * sampling_rate)

    # Get timestamps for movements
    timestamps = get_activity_timestamps(lever_active)
    movement_epochs = refine_activity_timestamps(
        timestamps,
        window=activity_window,
        max_len=dFoF.shape[0],
        sampling_rate=sampling_rate,
    )

    # Setup some outputs for traces
    movement_traces = [None for i in range(dFoF.shape[1])]
    mean_traces = [None for i in range(dFoF.shape[1])]

    # Iterate through each spine
    for spine in range(dFoF.shape[1]):
        curr_dFoF = dFoF[:, spine]
        curr_activity = activity[:, spine]
        if not np.nansum(curr_activity):
            continue
        # Identify which movements the spine is active during
        active_movements = []
        for movement in movement_epochs:
            if np.nansum(curr_activity[movement[0] : movement[1]]):
                active_movements.append(movement[0])
        # Skip is spine is not active during any movements
        if len(active_movements) == 0:
            continue
        # Get the individual and mean traces
        traces, mean_sems = d_utils.get_trace_mean_sem(
            curr_dFoF.reshape(-1, 1),
            ["dFoF"],
            active_movements,
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        ## Reorganize the outputs
        traces = traces["dFoF"]
        mean_trace = mean_sems["dFoF"][0]
        # Normalize if specified
        if norm is not None:
            traces = traces / norm[spine]
            mean_trace = mean_trace / norm[spine]
        # Store the traces
        movement_traces[spine] = traces
        mean_traces[spine] = mean_trace

    # Get the max amplitudes
    movement_amplitudes, _ = find_peak_amplitude(
        mean_traces, smooth=smooth, window=avg_window, sampling_rate=sampling_rate
    )

    # Get the onsets
    onsets = find_activity_onset(mean_traces, sampling_rate=sampling_rate)
    ## make them relative to the movement onset
    relative_onsets = np.zeros(dFoF.shape[1]) * np.nan
    for i, (amp, onset) in enumerate(zip(movement_amplitudes, onsets)):
        if not np.isnan(amp):
            relative_onsets[i] = (onset - center_point) / sampling_rate
        else:
            relative_onsets[i] = np.nan

    return movement_traces, movement_amplitudes, relative_onsets

