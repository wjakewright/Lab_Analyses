import numpy as np
import scipy.signal as sysignal

from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_timestamps import (
    get_activity_timestamps,
    refine_activity_timestamps,
    timestamp_onset_correction,
)


def analyze_event_activity(
    dFoF,
    timestamps,
    activity_window=(-2, 4),
    center_onset=False,
    smooth=False,
    avg_window=None,
    norm_constant=None,
    sampling_rate=60,
    peak_required=True,
):
    """Function to analyze the mean activity trace around specified events

    INPUT PARAMETERS
        dFoF - 2d np.array of the dFoF traces for each roi (columns)

        timestamps - list of the event timestamps for each roi

        activity_window - tuple specifying the window around each event to analyze

        center_onset - boolean of whether or not you wish to center traces on the
                        mean onset of the trace

        smooth - boolean specifying whether to smooth trace or not

        avg_window - float specifying the window you wish to average the peak over.
                     If None, it will use only the max peak amplitude

        norm_constants - np.array of the constants to normalize the activity by volume

        sampling_rates - int specifying the sampling rate

        peak_required - boolean specifying if there a defined peak must exist to get
                        amplitude

    OUTPUT PARAMETERS
        activity_traces - list of 2d np.array of the activity around each event.
                        columns = events, rolws=time, list items=each roi

        activity_amplitude - np.array of the peak amplitude for each spine

        activity_onset - int specifying the activity onset within the activity
                        window
    """
    # Get the traces for each spine
    activity_traces = []
    mean_traces = []
    for spine in range(dFoF.shape[1]):
        if len(timestamps[spine]) == 0:
            activity_traces.append(None)
            mean_traces.append(None)
            continue
        traces, mean = d_utils.get_trace_mean_sem(
            dFoF[:, spine].reshape(-1, 1),
            ["Activity"],
            timestamps[spine],
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        mean = mean["Activity"][0]
        traces = traces["Activity"]
        if norm_constant is not None:
            mean = mean / norm_constant[spine]
            traces = traces / norm_constant[spine]
        activity_traces.append(traces)
        mean_traces.append(mean)

    # Get the peak amplitudes
    activity_amplitude, _ = find_peak_amplitude(
        mean_traces,
        smooth=smooth,
        window=avg_window,
        sampling_rate=sampling_rate,
        peak_required=peak_required,
    )

    # Get the activity onsets
    activity_onset = find_activity_onset(mean_traces, sampling_rate)

    # Center traces if specified
    if center_onset:
        centered_traces = []
        for i, onset in enumerate(activity_onset):
            if np.isnan(onset):
                centered_traces.append(activity_traces[i])
                continue
            c_timestamps = timestamp_onset_correction(
                timestamps[i], activity_window, onset, sampling_rate
            )
            traces, _ = d_utils.get_trace_mean_sem(
                dFoF[:, i].reshape(-1, 1),
                ["Activity"],
                c_timestamps,
                window=activity_window,
                sampling_rate=sampling_rate,
            )
            traces = traces["Activity"]
            if norm_constant is not None:
                traces = traces / norm_constant[i]
            centered_traces.append(traces)
        ## Reset activity traces to be the centered traces
        activity_traces = centered_traces

    return activity_traces, activity_amplitude, activity_onset


def find_peak_amplitude(
    mean_traces, smooth=False, window=None, sampling_rate=60, peak_required=True
):
    """Function to find the peak amplitude of mean traces

    INPUT PARAMETERS
        mean_traces - list of the mean activity traces

        smooth - boolean specifying whether to smooth trace or not

        window - float specifying the window you wish to average the peak over.
                If None, it will use only the max peak amplitude

        sampling_rate - int specifying the imaging sampling rate

        peak_required - boolean specifying whether a defined peak is required
                        for finding the amplitude

    OUTPUT PARAMETERS
        trace_amplitudes - np.array of the peak amplitudes of the traces

        trace_amplitudes_idx - np.array of the idx where the peak is

    """
    DISTANCE = 0.5 * sampling_rate  ## minimum distance of 0.5 seconds

    # setup the output
    trace_amplitudes = np.zeros(len(mean_traces)) * np.nan
    trace_amplitudes_idx = np.zeros(len(mean_traces)) * np.nan

    # Iterate through each trace
    for i, trace in enumerate(mean_traces):
        if trace is None:
            continue
        if smooth:
            trace = sysignal.savgol_filter(trace, 31, 3)
        trace_med = np.nanmedian(trace)
        trace_std = np.nanstd(trace)
        trace_h = trace_med + trace_std
        trace_peaks, trace_props = sysignal.find_peaks(
            trace,
            height=trace_h,
            distance=DISTANCE,
        )
        trace_amps = trace_props["peak_heights"]
        # Find where the max amplitude is
        try:
            trace_peak = trace_peaks[np.argmax(trace_amps)]
        except ValueError:
            if peak_required is False:
                trace_peak = np.argmax(trace)
            else:
                continue
        # Get the max amplitude value
        if window:
            avg_win = int((window * sampling_rate) / 2)
            trace_amplitude = np.nanmean(
                trace[trace_peak - avg_win : trace_peak + avg_win]
            )
        else:
            trace_amplitude = trace[trace_peak] - np.min(trace)

        trace_amplitudes[i] = trace_amplitude
        trace_amplitudes_idx[i] = trace_peak

    return trace_amplitudes, trace_amplitudes_idx


def find_activity_onset(mean_traces, smooth=True, sampling_rate=60):
    """Function to find the onset of mean activity traces

    INPUT PARAMETERS
        mean_traces - list of the mean activity traces

        sampling_rate - int specifying the imaging sampling rate

    OUTPUT PARAMETERS
        activity_onsets - np.array of the activity onsets
    """
    # Set up output
    activity_onsets = np.zeros(len(mean_traces))
    # Get the idx of the max amplitudes
    peak_amps, peak_idxs = find_peak_amplitude(
        mean_traces, smooth=True, window=False, sampling_rate=sampling_rate
    )

    # Iterate through each trace
    for i, trace in enumerate(mean_traces):
        # Check if peak amplitude was found for the trace
        if np.isnan(peak_idxs[i]):
            activity_onsets[i] = np.nan
            continue
        # Smooth the trace
        if smooth:
            trace = sysignal.savgol_filter(trace, 31, 3)
        # Find the offset of the rising phase
        peak_trace = trace[: int(peak_idxs[i])]
        try:
            trace_offset = np.nonzero(peak_trace < 0.75 * peak_amps[i])[0][-1]
        except IndexError:
            activity_onsets[i] = 0
            continue
        # Get the trace velocity
        search_trace = trace[:trace_offset]
        trace_deriv = np.gradient(trace)[:trace_offset]
        # Find where the velocity goes below zero
        trace_below_zero = trace_deriv <= 0
        if np.nansum(trace_below_zero):
            onset = np.nonzero(trace_below_zero)[0][-1]
        # If derivative doesn't go below zero, find where it goes below the median
        else:
            try:
                onset = np.nonzero(search_trace < np.nanmedian(trace))[0][-1]
            except:
                onset = 0
        activity_onsets[i] = onset

    return activity_onsets


def max_and_avg_peaks(
    dFoF,
    activity,
    activity_window=(-1, 2),
    smooth=True,
    peak_required=True,
    sampling_rate=60,
):
    """Helper function to get max and average amplitudes and
        traces for individual events for each spine

    INPUT PARAMETERS
        dFoF - 2d np.ndarray of the dFoF traces with columns
                corresponding to each ROI

        activity - 2d np.ndarry of the binarized activity with columns
                    corresponding to each ROI

        activity_window - tuple specifying the window to analyze events

        smooth - boolean specifying whether to smooth the traces for
                calculating the amplitudes

        peak_required - boolean specifying if the event must have a discernable peak
                    or not to be included in analysis

        OUTPUT PARAMETERS
            max_amplitudes - np.array of the the max amplitudes for each ROI

            max_traces - list of 1d arrays of the max event traces for each ROI

            mean_amplitudes - np.ndarray of the mean amplitudes for each ROI

            mean_traces - list of 1d arrays of the averaged trace for each ROI


    """
    DISTANCE = 0.5 * sampling_rate  # constant used for peak detection
    center_point = int(np.absolute(activity_window[0]) * sampling_rate)
    # Set up outputs
    max_amplitudes = []
    max_traces = []
    mean_amplitudes = []
    mean_traces = []

    # Analyze each spine
    for spine in range(dFoF.shape[1]):
        curr_dFoF = dFoF[:, spine]
        curr_activity = activity[:, spine]
        # Get event onset times
        timestamps = get_activity_timestamps(curr_activity)
        timestamps = [x[0] for x in timestamps]
        timestamps = refine_activity_timestamps(
            timestamps, activity_window, len(curr_activity), sampling_rate
        )
        # Continue if there are not events
        if len(timestamps) == 0:
            max_amplitudes.append(np.nan)
            max_traces.append(None)
            mean_amplitudes.append(np.nan)
            mean_traces.append(None)
            continue
        # Grab the traces around each event
        traces, _ = d_utils.get_trace_mean_sem(
            curr_dFoF.reshape(-1, 1),
            ["Activity"],
            timestamps,
            activity_window,
            sampling_rate,
        )
        traces = traces["Activity"]

        # Temp variables
        temp_amps = []

        # Analyze each event
        for event in range(traces.shape[1]):
            t = traces[:, event]
            # Smooth trace
            if smooth:
                t_smooth = sysignal.savgol_filter(t, 31, 3)
            else:
                t_smooth = t

            t_smooth = t_smooth - t_smooth[0]

            # Set up find peak parameters
            med = np.nanmedian(t_smooth)
            std = np.nanstd(t_smooth)
            height = med + std
            peaks, props = sysignal.find_peaks(
                t_smooth, height=height, distance=DISTANCE
            )
            amps = props["peak_heights"]

            # Get the max amplitude
            try:
                max_peak = peaks[np.argmax(amps)]
            except:
                if peak_required is False:
                    max_peak = np.argmax(t_smooth[center_point:])
                else:
                    temp_amps.append(np.nan)
                    continue
            max_amp = t_smooth[max_peak]
            # Store value
            temp_amps.append(max_amp)

        # Find max peak amplitude and traces
        max_amplitude = np.nanmax(temp_amps)
        max_trace = traces[:, np.nanargmax(temp_amps)]
        # Find average amps and traces
        mean_amplitude = np.nanmean(temp_amps)
        mean_trace = np.nanmean(traces, axis=1)
        # Store values
        max_amplitudes.append(max_amplitude)
        max_traces.append(max_trace)
        mean_amplitudes.append(mean_amplitude)
        mean_traces.append(mean_trace)

    return np.array(max_amplitudes), max_traces, np.array(mean_amplitudes), mean_traces
