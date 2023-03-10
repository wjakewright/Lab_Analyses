import numpy as np
from scipy import stats

from Lab_Analyses.Utilities.activity_timestamps import get_activity_timestamps


def calculate_coactivity(
    trace_1, trace_2, norm_method="mean", duration=None, sampling_rate=60
):
    """Function to calculate the coactivity between two traces
        
        INPUT PARAMETERS
            trace_1 - np.array of binary activity of the first ROI,
            
            trace_2 - np.array of the binary activity of the second ROI

            norm_method - str specifying how to normalize coactivity. Accepts "mean"
                         which normalizes by geometric mean of both traces or "freq"
                         which normalizes only by the activity of the target trace
            
            duration - int specifying the duration of the traces. To be used
                        if traces have previously been constrained and length doesn't
                        represent the duration of the constraint. Default is None

            sampling_rate - int specifying the imaging rate
        
        OUTPUT PARAMETERS
            coactivity_rate - float of the raw coactivity rate 

            coactivity_rate_norm - float of the normalized coactivity rate

            fraction_active_1 - float of the fraction of trace 1's activity 
                                 is coactive
            
            fraction_active_2 - float of the fractoin of trace 2's activity is
                                coactive
            
            coactivity_trace - np.array of the binary coactivity trace
    """
    if duration is None:
        duration = len(trace_1)
    duration = duration / sampling_rate  # converted to seconds

    # Check if traces display activity
    if (np.sum(trace_1) == 0) or (np.sum(trace_2) == 0):
        return (0, 0, 0, 0, np.zeros(len(trace_1)))

    # Calculate the coactivity rate
    coactivity = trace_1 * trace_2

    ## Skip if no coactivity
    if not np.sum(coactivity):
        return (0, 0, 0, 0, np.zeros(len(trace_1)))

    coactive_events = np.nonzero(np.diff(coactivity) == 1)[0]
    coactive_event_num = len(coactive_events)

    ## Raw coactivity rate
    coactivity_rate = coactive_event_num / duration

    ## Normalized coactivity rate
    trace_1_event_num = len(np.nonzero(np.diff(trace_1) == 1)[0])
    trace_2_event_num = len(np.nonzero(np.diff(trace_2) == 1)[0])
    trace_1_rate = trace_1_event_num / duration
    trace_2_rate = trace_2_event_num / duration
    if norm_method == "mean":
        normalizer = stats.gmean([trace_1_rate, trace_2_rate])
    elif norm_method == "freq_1":
        normalizer = trace_1_rate
    elif norm_method == "freq_2":
        normalizer = trace_2_rate
    coactivity_rate_norm = coactivity_rate / normalizer
    ## Convert raw coactivity rate into miniute timescale
    coactivity_rate = coactivity_rate * 60

    # Calculate fraction coactive
    ## Get active periods of target trace
    timestamps = get_activity_timestamps(trace_1)
    ## Compare active epochs to other trace
    coactive_idxs = []
    for t in timestamps:
        if np.sum(trace_2[t[0] : t[1]]):
            coactive_idxs.append(t)
    ## Generate coactive trace
    coactivity_trace = np.zeros(len(trace_1))
    for epoch in coactive_idxs:
        coactivity_trace[epoch[0] : epoch[1]] = 1
    ## Calculate fractions
    fraction_active_1 = len(coactive_idxs) / trace_1_event_num
    fraction_active_2 = (
        len(coactive_idxs) / trace_2_event_num
    )  ## There is a problem with this

    return (
        coactivity_rate,
        coactivity_rate_norm,
        fraction_active_1,
        fraction_active_2,
        coactivity_trace,
    )

