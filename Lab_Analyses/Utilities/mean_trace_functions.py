import numpy as np
import scipy.signal as sysignal


def find_peak_amplitude(mean_traces, smooth=False, window=None, sampling_rate=60):
    """Function to find the peak amplitude of mean traces
    
        INPUT PARAMETERS
            mean_traces - list of the mean activity traces

            smooth - boolean specifying whether to smooth trace or not

            window - float specifying the window you wish to average the peak over.
                    If None, it will use only the max peak amplitude
            
            sampling_rate - int specifying the imaging sampling rate
        
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
        if smooth:
            trace = sysignal.savgol_filter(trace, 31, 3)
        trace_med = np.nanmedian(trace)
        trace_std = np.nanstd(trace)
        trace_h = trace_med + trace_std
        trace_peaks, trace_props = sysignal.find_peaks(
            trace, height=trace_h, distance=DISTANCE,
        )
        trace_amps = trace_props["peak_heights"]
        # Find where the max amplitude is
        try:
            trace_peak = trace_peaks[np.argmax(trace_amps)]
        except ValueError:
            continue
        # Get the max amplitude value
        if window:
            avg_win = int((window * sampling_rate) / 2)
            trace_amplitude = np.nanmean(
                trace[trace_peak - avg_win : trace_peak + avg_win]
            )
        else:
            trace_amplitude = trace[trace_peak]

        trace_amplitudes[i] = trace_amplitude
        trace_amplitudes_idx[i] = trace_peak

    return trace_amplitudes, trace_amplitudes_idx


def find_activity_onset(mean_traces, sampling_rate=60):
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
        trace = sysignal.savgol_flter(trace, 31, 3)
        # Find the offset of the rising phase
        peak_trace = trace[: peak_idxs[i]]
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
