import numpy as np
from scipy import stats

from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_onset import find_activity_onset
from Lab_Analyses.Utilities.activity_timestamps import \
    timestamp_onset_correction


def get_trace_coactivity_rates(trace_1, trace_2, sampling_rate):
    """Function to analyze the coactivity rate between two different activity
        traces. Calculates several different measures of coactivity as well as
        fraction of activity that is coactive
        
        INPUT PARAMETERS
            trace_1 - np.array of the binary activity trace
            
            trace_2 - np.array of the binary activity trace
            
            sampling_rate - int of the imaging sampling rate
            
        OUTPUT PARAMETERS
            coactivity_event_rate - float of absolute coactivity event rate
            
            coactivity_event_rate_norm - float of the normalized coactivity event rate

            trace_1_frac - float of the fraction of activity that is coactive
                            for trace_1
            
            trace_2_frac - float of the fraction of activity that is coactive
                            for trace_2
            
            coactivity_trace - np.array of the binarized coactivity between the two
            
    """
    # check if the is activity in the traces
    if (np.sum(trace_1) == 0) or (np.sum(trace_2) == 0):
        coactivity_event_rate = 0
        coactivity_event_rate_norm = 0
        trace_1_frac = 0
        trace_2_frac = 0
        coactivity_trace = np.zeros(len(trace_1))
        return (
            coactivity_event_rate,
            coactivity_event_rate_norm,
            trace_1_frac,
            trace_2_frac,
            coactivity_trace,
        )

    # Get the total time in seconds
    duration = len(trace_1) / sampling_rate

    # calculate the coactivity rate
    coactivity = trace_1 * trace_2
    ## Skip if there is no coactivity
    if not np.sum(coactivity):
        coactivity_event_rate = 0
        coactivity_event_rate_norm = 0
        trace_1_frac = 0
        trace_2_frac = 0
        coactivity_trace = np.zeros(len(trace_1))
        return (
            coactivity_event_rate,
            coactivity_event_rate_norm,
            trace_1_frac,
            trace_2_frac,
            coactivity_trace,
        )

    events = np.nonzero(np.diff(coactivity) == 1)[0]
    event_num = len(events)
    # Raw coactivity rate
    coactivity_event_rate = event_num / duration
    # normalized coactivity rate
    trace_1_event_num = len(np.nonzero(np.diff(trace_1) == 1)[0])
    trace_2_event_num = len(np.nonzero(np.diff(trace_2) == 1)[0])
    trace_1_event_rate = trace_1_event_num / duration
    trace_2_event_rate = trace_2_event_num / duration
    geo_mean = stats.gmean([trace_1_event_rate, trace_2_event_rate])
    coactivity_event_rate_norm = coactivity_event_rate / geo_mean
    # Convert raw coactivity rate into minute timescale
    coactivity_event_rate = coactivity_event_rate * 60

    # Calculate fraction coactive
    ## Break up activity trace
    active_boundaries = np.insert(np.diff(trace_1), 0, 0, axis=0)
    active_onsets = np.nonzero(active_boundaries == 1)[0]
    active_offsets = np.nonzero(active_boundaries == -1)[0]
    # Check onset offset order
    if active_onsets[0] > active_offsets[0]:
        active_offsets = active_offsets[1:]
    # Check onsets and offsets are the same length
    if len(active_onsets) > len(active_offsets):
        active_onsets = active_onsets[:-1]
    # Compare active epochs to other trace
    coactive_idxs = []
    for onset, offset in zip(active_onsets, active_offsets):
        if np.sum(trace_2[onset:offset]):
            coactive_idxs.append((onset, offset))
    # Generate coactive trace
    coactivity_trace = np.zeros(len(trace_1))
    for epoch in coactive_idxs:
        coactivity_trace[epoch[0] : epoch[1]] = 1
    # Calculate fraction coactive
    try:
        trace_1_frac = len(coactive_idxs) / trace_1_event_num
    except ZeroDivisionError:
        trace_1_frac = 0
    try:
        trace_2_frac = len(coactive_idxs) / trace_2_event_num
    except ZeroDivisionError:
        trace_2_frac = 0

    return (
        coactivity_event_rate,
        coactivity_event_rate_norm,
        trace_1_frac,
        trace_2_frac,
        coactivity_trace,
    )


def analyze_activity_trace(
    dFoF_trace,
    timestamps,
    activity_window=(-2, 4),
    center_onset=False,
    norm_constant=None,
    sampling_rate=60,
):
    """Function to analyze of the mean activity trace around specific timestamped events 
        (e.g., coactivity)
        
        INPUT PARAMETERS
            dFoF_trace - np.array of the dFoF activity trace

            timestamps - list of the event timestamps

            activity_window - tuple specifying the window around which you want to analyze
                            the activity from (e.g., (-2,4) for 2 sec before and 4 sec after)

            center_onset - boolean of whether or not you wish to center traces on the mean onset
            
            norm_constants - np.array of constants to normalize the activity by volume

            sampling_rate - int specifying the sampling rate
        
        OUTPUT PARAMETERS
            activity_traces - 2d np.array of activity around each event. columns=events, rows=time (in frames)

            activity_amplitude - float of the mean peak activity amplitude

            activity_auc - float of the area under the activity curve

            activity_onset - int specifying the activity onset within the activity window

    """
    timestamps = [x[0] for x in timestamps]
    # Get the activity around the timestamps
    activity_traces, mean_trace = d_utils.get_trace_mean_sem(
        dFoF_trace.reshape(-1, 1),
        ["Activity"],
        timestamps,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    mean_trace = mean_trace["Activity"][0]
    activity_traces = activity_traces["Activity"]
    if norm_constant is not None:
        mean_trace = mean_trace / norm_constant
        activity_traces = activity_traces / norm_constant
    # Find onset
    activity_onset, activity_amplitude = find_activity_onset(
        [mean_trace], sampling_rate=sampling_rate
    )
    activity_onset = activity_onset[0]
    activity_amplitude = activity_amplitude[0]
    # Get area under the curve
    area_trace = mean_trace[activity_onset:]
    activity_auc = np.trapz(area_trace)

    # Center around onset if specified
    if center_onset:
        c_timestamps = timestamp_onset_correction(
            timestamps, activity_window, activity_onset, sampling_rate
        )
        activity_traces, mean_trace = d_utils.get_trace_mean_sem(
            dFoF_trace.reshape(-1, 1),
            ["Activity"],
            c_timestamps,
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        activity_traces = activity_traces["Activity"][0]
        if norm_constant is not None:
            activity_traces = activity_traces / norm_constant

    return activity_traces, activity_amplitude, activity_auc, activity_onset


def analyze_nearby_coactive_spines(
    timestamps,
    nearby_dFoF,
    nearby_calcium,
    nearby_activity,
    glu_constants,
    ca_constants,
    activity_window=(-2, 4),
    sampling_rate=60,
):
    """Function to analyze the activity of nearby spines during coactivity events
    
        INPUT PARAMETERS    
            timestamps - list of tuples with the timestamps (onset, offset) of 
                        each coactivity event
            
            nearby_dFoF - 2d np.array of the nearby coactive spines (columns)
                          dFoF activity
            
            nearby_calcium - 2d np.array of the nearby coactive spines (columns)
                             calcium activity
            
            nearby_activity - 2d np.array of the nearby spines (columns) binarized
                             activity 
            
            glu_constants - np.array of GluSnFr constants for nearby spines
            
            ca_constants - np.array of the RCaMP2 constants for nearby spines
            
            activity_window - tuple specifying the window around which you want the
                             activity from (e.g., (-2,2) for 2 sec before and after
                             
            sampling_rate - int specifying the sampling rate
            
    """

