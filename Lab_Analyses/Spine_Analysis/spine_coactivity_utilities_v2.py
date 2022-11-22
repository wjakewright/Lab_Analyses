import numpy as np
from scipy import stats


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
