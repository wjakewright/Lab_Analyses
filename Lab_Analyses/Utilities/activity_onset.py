import numpy as np
import scipy.signal as sysignal


def find_activity_onset(mean_traces, sampling_rate=60):
    """Function to find the activity onset of mean activity traces
    
        INPUT PARAMETERS
            mean_traces - list of the mean activity_traces
            
            sampling_rate - int specifying the imaging sampling rate
        
        OUTPUT PARAMETERS
            activity_onsets - np.array of activity onsets
            
            trace_amplitudes - np.array of peak amplitude of the traces
    """
    DISTANCE = 0.5 * sampling_rate  ## minimum distance of 0.5 seconds

    # Set up the output
    activity_onsets = np.zeros(len(mean_traces))
    trace_amplitudes = np.zeros(len(mean_traces))

    # Find each onset
    for i, trace in enumerate(mean_traces):
        trace_med = np.nanmedian(trace)
        trace_std = np.nanstd(trace)
        trace_h = trace_med + trace_std
        trace_peaks, trace_props = sysignal.find_peaks(
            trace, height=trace_h, distance=DISTANCE,
        )
        trace_amps = trace_props["peak_heights"]
        # Get the max peak
        try:
            trace_peak = trace_peaks[np.argmax(trace_amps)]
        except ValueError:
            activity_onsets[i] = np.nan
            trace_amplitudes[i] = np.nan
            continue

        # Get the mean around the peak (0.5s window)
        window = int((0.5 * sampling_rate) / 2)
        trace_amp = np.nanmean(trace[trace_peak - window : trace_peak + window])

        # Get the offset of the rise phase
        trace_peak_trace = trace[:trace_peak]
        try:
            trace_offset = np.nonzero(trace_peak_trace < 0.75 * trace_amp)[0][-1]
        except IndexError:
            activity_onsets[i] = 0
            trace_amplitudes[i] = trace_amp
            continue
        # Get the trace velocity
        ## Smooth trace for better estimation of onset
        smooth_trace = sysignal.savgol_filter(trace, 31, 3)
        trace_search_trace = smooth_trace[:trace_offset]
        trace_deriv = np.gradient(smooth_trace)[:trace_offset]
        # Find where the velocity goes below zero
        trace_below_zero = trace_deriv <= 0
        if np.sum(trace_below_zero):
            onset = np.nonzero(trace_below_zero)[0][-1]
        # If derivative doesn't go below zero, find where it goes below the median
        else:
            try:
                onset = np.nonzero(trace_search_trace < trace_med)[0][-1]
            except:
                onset = 0
        activity_onsets[i] = onset
        trace_amplitudes[i] = trace_amp

    return activity_onsets, trace_amplitudes
