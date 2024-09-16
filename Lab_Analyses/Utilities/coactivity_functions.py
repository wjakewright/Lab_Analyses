import numpy as np
import scipy.signal as sysignal
from scipy import stats

from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_timestamps import (
    get_activity_timestamps,
    refine_activity_timestamps,
)


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

    duration = float(duration)

    # Check if traces display activity
    if (np.nansum(trace_1) == 0) or (np.nansum(trace_2) == 0):
        return (0, 0, 0, 0, np.zeros(len(trace_1)))

    # Check if traces are active at the very beginning
    if trace_1[0] == 1:
        trace_1[0] = 0
    if trace_2[0] == 1:
        trace_2[0] = 0

    # Calculate the coactivity rate
    coactivity = trace_1 * trace_2

    ## Skip if no coactivity
    if not np.nansum(coactivity):
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
        if np.nansum(trace_2[t[0] : t[1]]):
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


def get_conservative_coactive_binary(trace_1, trace_2):
    """Helper function to get a conservative binary coactivity trace
    by looking for periods where there is no overlap for each event"""

    timestamps = get_activity_timestamps(trace_1)
    # Get non overlapping idxs
    coactive_idxs = []
    noncoactive_idxs = []
    for t in timestamps:
        if np.nansum(trace_2[t[0] : t[1]]):
            coactive_idxs.append(t)
        else:
            noncoactive_idxs.append(t)
    # Generate the binary coactive trace
    coactivity_trace = np.zeros(len(trace_1))
    for epoch in coactive_idxs:
        coactivity_trace[epoch[0] : epoch[1]] = 1
    # Generate binary noncoactive trace
    noncoactivity_trace = np.zeros(len(trace_1))
    for epoch in noncoactive_idxs:
        noncoactivity_trace[epoch[0] : epoch[1]] = 1

    return coactivity_trace, noncoactivity_trace


def calculate_relative_onset(
    trace_1, trace_2, coactivity=None, sampling_rate=60, activity_window=(-2, 4)
):
    """Function to calculate the relative onset of activity during coactive events

    INPUT PARAMETERS
        trace_1 - np.array of first rois binary activity

        trace_2 - np.array of the second rois binary activity

        coactivity - np.array of the binary coactivity trace. If none a trace will
                    be generated

        sampling_rate - int specifying the imaging rate

        activity_window - tuple specifying the window to analyze

    OUTPUT PARAMETERS
        avg_relative_onset - float of the avg relative onset of roi 1 to roi 2

        onset_jitter - float of the deviation of the relative onset
    """
    # Set up window in frames
    before_f = int(activity_window[0] * sampling_rate)
    after_f = int(activity_window[1] * sampling_rate)
    if coactivity is None:
        coactivity = trace_1 * trace_2

    # Get timestamps
    timestamps = get_activity_timestamps(coactivity)
    timestamps = refine_activity_timestamps(
        timestamps,
        window=activity_window,
        max_len=len(coactivity),
        sampling_rate=sampling_rate,
    )
    # Skip if no coactivity
    if len(timestamps) == 0:
        return np.nan, np.nan

    timestamps = [x[0] for x in timestamps]

    relative_onsets = np.zeros(len(timestamps)) * np.nan
    # Iterate through each event
    for i, event in enumerate(timestamps):
        event_activity = coactivity[event + before_f : event + after_f]
        ## Ensure there is coactivity
        if not np.nansum(event_activity):
            continue
        ## Find onsets of trace_1
        activity_1 = trace_1[event + before_f : event + after_f]
        boundaries_1 = np.insert(np.diff(activity_1), 0, 0, axis=0)
        try:
            onset_1 = np.nonzero(boundaries_1 == 1)[0][0]
        except IndexError:
            onset_1 = 0
        ## Find onset of trace_2
        activity_2 = trace_2[event + before_f : event + after_f]
        boundaries_2 = np.insert(np.diff(activity_2), 0, 0, axis=0)
        try:
            onset_2 = np.nonzero(boundaries_2 == 1)[0][0]
        except IndexError:
            onset_2 = 0
        ## Calculate relative onset
        rel_onset = (onset_1 - onset_2) / sampling_rate
        relative_onsets[i] = rel_onset

    # Get mean onset and jitter
    avg_relative_onset = np.nanmean(relative_onsets)
    onset_jitter = np.nanstd(relative_onsets)

    return avg_relative_onset, onset_jitter


def find_individual_onsets(
    trace_1, trace_2, coactivity=None, sampling_rate=60, activity_window=(-2, 4)
):
    """Function to find the timestamps of individual traces for each coactivity event
    INPUT PARAMETERS
        trace_1 - np.array of first rois binary activity

        trace_2 - np.array of the second rois binary activity

        coactivity - np.array of the binary coactivity trace. If none a trace will
                    be generated

        sampling_rate - int specifying the imaging rate

        activity_window - tuple specifying the window to analyze

    OUTPUT PARAMETERS
        trace_1_onsets - np.array of the individual onset timestamps

        trace_2_onsets - np.array of the individual onset timestamps

    """
    # Set up window in frames
    before_f = int(activity_window[0] * sampling_rate)
    after_f = int(activity_window[1] * sampling_rate)
    if coactivity is None:
        coactivity = trace_1 * trace_2

    # Get timestamps
    timestamps = get_activity_timestamps(coactivity)
    timestamps = refine_activity_timestamps(
        timestamps,
        window=activity_window,
        max_len=len(coactivity),
        sampling_rate=sampling_rate,
    )
    # Skip if no coactivity
    if len(timestamps) == 0:
        return np.array([]), np.array([])
    # Refine to only onsets
    timestamps = [x[0] for x in timestamps]

    # Initialize outputs
    trace_1_onsets = np.zeros(len(timestamps)) * np.nan
    trace_2_onsets = np.zeros(len(timestamps)) * np.nan

    # Iterate through each event
    for i, event in enumerate(timestamps):
        # Get coactivity around the event
        event_activity = coactivity[event + before_f : event + after_f]
        # Skip if no activity for some reason
        if not np.nansum(event_activity):
            continue
        # Get trace one activity around event
        activity_1 = trace_1[event + before_f : event + after_f]
        # Find the onset
        boundaries_1 = np.insert(np.diff(activity_1), 0, 0, axis=0)
        try:
            onset_1 = np.nonzero(boundaries_1 == 1)[0][0]
        except IndexError:
            onset_1 = 0
        # Get trace two activity around event
        activity_2 = trace_2[event + before_f : event + after_f]
        # Find the onset
        boundaries_2 = np.insert(np.diff(activity_2), 0, 0, axis=0)
        try:
            onset_2 = np.nonzero(boundaries_2 == 1)[0][0]
        except IndexError:
            onset_2 = 0
        # Calculate offset of each onset relative to the event
        offset_1 = np.absolute(before_f) - onset_1
        offset_2 = np.absolute(before_f) - onset_2
        # Calculate the onset time throughout the entire trace
        trace_1_onsets[i] = int(event - offset_1)
        trace_2_onsets[i] = int(event - offset_2)

    return trace_1_onsets.astype(int), trace_2_onsets.astype(int)


def coactivity_cross_correlation(
    activity_1,
    activity_2,
    dFoF_1,
    dFoF_2,
    activity_window=(-2, 4),
    smooth=False,
    sampling_rate=60,
):
    """
    Function to calculate the cross-correlation between traces around coactivity events

    INPUT PARAMETERS
        activity_1 - 1d np.ndarray of the binarized activity for the first trace

        activity_2 - 1d np.ndarray of the binarized activity for the second trace

        dFoF_1 - 1d np.ndarray of the dFoF for the first trace

        dFoF_2 - 1d np.ndarray of the dFoF for the second trace

        activity_window - tuple specifying the window around coactivity to calculate
                            the cross correlation

        smooth - boolean specifying whether to smooth the traces

        sampling_rate - int specifying the sampling rate

    OUTPUT PARAMETERS
        cross_corr_traces - 2d np.array of the cross correlation traces
                            around each event. Columns = events, rows = time

        cross_corr_lags - 1d np.array of the cross correlation lags converted
                            to seconds

        avg_cross_corr_peak - float of the average peak of the cross correlation
                          between the two traces

    """
    window_size = (activity_window[1] - activity_window[0]) * sampling_rate
    # Get coactive events for trace 1
    ## Cross corr will be sentered around trace 2 events
    coactive_1, _ = get_conservative_coactive_binary(activity_1, activity_2)

    # Get event onsets
    onsets = get_activity_timestamps(coactive_1)
    onsets = [x[0] for x in onsets]

    # Make sure there are coactive events
    if len(onsets) == 0:
        cross_corr_traces = None
        cross_corr_peak = np.nan
        lags = sysignal.correlation_lags(window_size, window_size) / sampling_rate
        return cross_corr_traces, lags, cross_corr_peak
    else:
        # Refine onsets if present
        onsets = refine_activity_timestamps(
            onsets,
            window=activity_window,
            max_len=(len(dFoF_1)),
            sampling_rate=sampling_rate,
        )

    # Get the dFoF traces around each event\
    traces_1, _ = d_utils.get_trace_mean_sem(
        dFoF_1.reshape(-1, 1),
        ["Trace_1"],
        timestamps=onsets,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    traces_1 = traces_1["Trace_1"]
    traces_2, _ = d_utils.get_trace_mean_sem(
        dFoF_2.reshape(-1, 1),
        ["Trace_2"],
        timestamps=onsets,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    traces_2 = traces_2["Trace_2"]

    # Set up outputs
    cross_corr_traces = []
    corr_lags = []

    # Analyze each event
    for event in range(traces_1.shape[1]):
        t1 = traces_1[:, event]
        t2 = traces_2[:, event]
        # smooth if specified
        if smooth:
            t1_smooth = sysignal.savgol_filter(t1, 15, 3)
            t2_smooth = sysignal.savgol_filter(t2, 15, 3)
            ## Normalize data
            norm_1 = np.linalg.norm(t1_smooth)
            norm_2 = np.linalg.norm(t2_smooth)
            t1_smooth = t1_smooth / norm_1
            t2_smooth = t2_smooth / norm_2
        else:
            norm_1 = np.linalg.norm(t1)
            norm_2 = np.linalg.norm(t2)
            t1_smooth = t1 / norm_1
            t2_smooth = t2 / norm_2

        # Perform cross correlation
        corr = sysignal.correlate(t1_smooth, t2_smooth)
        # Get the lags
        ## Should be all the same
        lags = sysignal.correlation_lags(len(t1_smooth), len(t2_smooth))
        ## Find index of peak cross correlation
        # peak = np.argmax(corr)
        # peak_lag = lags[peak]

        cross_corr_traces.append(corr)
        corr_lags.append(lags)

    # Organize outputs
    ## Put correlation traces into array
    cross_corr_traces = np.vstack(cross_corr_traces).T

    ## Average peaks and convert to seconds
    # avg_cross_corr_peaks = np.nanmean(cross_corr_peaks) / sampling_rate
    ## Check lags are all the same
    corr_lags = np.vstack(corr_lags)
    if (corr_lags == corr_lags[0]).all():
        cross_corr_lags = corr_lags[0] / sampling_rate
    else:
        print(corr_lags)
        raise Exception("Corr lags are not the same for each event")
    avg_cross_corr_trace = np.nanmean(cross_corr_traces, axis=1)
    avg_cross_corr_peaks = (
        cross_corr_lags[np.argmax(avg_cross_corr_trace)] / sampling_rate
    )

    return cross_corr_traces, cross_corr_lags, avg_cross_corr_peaks
