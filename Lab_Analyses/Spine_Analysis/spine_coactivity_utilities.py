from itertools import compress

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Utilities import data_utilities as d_utils
from scipy import stats


def get_coactivity_rate(spine, dendrite, coactivity, sampling_rate):
    """Helper function to calculate the coactivity frequency between a spine and 
        its parent dendrite. These rates are normalized by geometric mean of the spine
        and dendrite activity rates
        
        INPUT PARAMETERS
            spine - np.array of spine binary activity trace
            
            dendrite - np.array of dendrite binary activity traces

            coactivity - np.array of the coactivity trace to be used
            
            sampling_rate - int or float of the sampling rate
            
        OUTPUT PARAMETERS
            coactivity_event_num - int of the number of coactive events
            
            coactivity_event_rate - float of normalized coactivity event rate
            
            spine_fraction_coactive - float of fraction of spine activity events that
                                      are also coactive
            
            dend_fraction_coactive - float of fraction of dendrite activity events that
                                     are also coactive with the given spine
    """
    # Get the total time in seconds
    duration = len(spine) / sampling_rate

    # Get coactivity binary trace
    # Count coactive events
    events = np.nonzero(np.diff(coactivity) == 1)[0]
    coactivity_event_num = len(events)

    # Calculate raw event rate
    event_rate = coactivity_event_num / duration

    # Normalize rate based on spine and dendrite event rates
    spine_event_rate = len(np.nonzero(np.diff(spine) == 1)[0]) / duration
    dend_event_rate = len(np.nonzeor(np.diff(dendrite) == 1)[0]) / duration
    geo_mean = stats.gmean([spine_event_rate, dend_event_rate])
    coactivity_event_rate = event_rate / geo_mean

    # Get spine and dendrite fractions
    try:
        spine_fraction_coactive = event_rate / spine_event_rate
    except ZeroDivisionError:
        spine_fraction_coactive = 0
    try:
        dend_fraction_coactive = event_rate / dend_event_rate
    except ZeroDivisionError:
        dend_fraction_coactive = 0

    return (
        coactivity_event_num,
        coactivity_event_rate,
        spine_fraction_coactive,
        dend_fraction_coactive,
    )


def get_dend_spine_traces_and_onsets(
    dendrite_activity,
    spine_activity_matrix,
    dendrite_dFoF,
    spine_dFoF_matrix,
    coactivity,
    activity_window=(-2, 2),
    sampling_rate=60,
):
    """Helper function to help getting the activity traces of dendrites and spines
        for all dendritic or coactive events. Also gets the relative onset of spine
        activity
        
        INPUT PARAMETERS
            dendrite - np.array of dendrite binary activity trace
            
            spine_matrix = 2d np.array of spine binary activity traces (columns = spines)

            dendrite_dFoF - np.array of dendrite dFoF trace

            spine_dFoF_matrix - 2d np.array of spine dFoF activty traces (columns=spines)

            coactivity - boolean specifying whether to perform for coactivty events (True) or
                        all dendritic events (False)
            
            activity_window - tuple specifying the window around which you want the activity
                                from. E.g. (-2,2) for 2 sec before and after

            sampling_rate - int specifying the sampling rate

        OUTPUT PARAMETERS
            spine_traces - list of 2d np.array of spine activity around each
                            event. Centered around dendrite onset. 
                            columns = each event, rows = time (in frames)
            
            dend_traces - list of 2d np.array of dendrite activity around each
                            event. Centered around dendrite onset. 
                            columns = each event, rows = time (in frames)

            spine_amplitudes - np.array of the peak spine amplitude

            spine_auc - np.array of the area under the spine activity curve

            spine_std - np.array of the std of spine activity

            dend_amplitudes - np.array of the peak dendrite amplitudes

            dend_auc - np.array of the area under the dendrite activity curve

            dend_std - np.array of the std of dendrite activity
            
            relative_onsets - np.array of spine onsets relative to dendrite
    """
    # Set up outputs
    dend_traces = []
    spine_traces = []
    dend_amplitudes = []
    dend_auc = []
    dend_stds = []
    spine_amplitudes = []
    spine_auc = []
    spine_stds = []
    relative_onsets = []

    # Perform the analysis for each spine seperately
    for i in range(spine_activity_matrix.shape[1]):
        curr_spine_activity = spine_activity_matrix[:, i]
        curr_spine_dFoF = spine_dFoF_matrix[:, i]
        if coactivity:
            new_dend_activity = dendrite_activity * curr_spine_activity
        else:
            new_dend_activity = dendrite_activity
        # Get the initial timestamps
        initial_stamps = get_activity_timestamps(new_dend_activity)
        initial_stamps = [x[0] for x in initial_stamps]
        d_trace, d_mean = d_utils.get_trace_mean_sem(
            dendrite_dFoF.reshape(-1, 1),
            ["Dendrite"],
            initial_stamps,
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        initial_d_mean = list(d_mean.values())[0][0]
        d_trace = list(d_trace.values())[0]
        # Find the onset
        initial_d_onset, dend_amp = find_activity_onset([initial_d_mean])
        initial_d_onset = initial_d_onset[0]
        dend_amp = dend_amp[0]
        dmax_idx = np.where(initial_d_mean == dend_amp)
        d_std_trace = np.nanstd(d_trace, axis=1)
        dend_std = d_std_trace[dmax_idx]
        # Correct timestamps to be at the onset
        center_point = np.absolute(activity_window[0] * sampling_rate)
        offset = center_point - initial_d_onset
        onset_stamps = [x - offset for x in initial_stamps]

        # Refine the timestamps
        refined_idxs = []
        for i, stamp in enumerate(onset_stamps):
            if i == 0:
                if stamp - np.absolute(activity_window[0]) < 0:
                    refined_idxs.append(False)
                else:
                    refined_idxs.append(True)
                continue
            if i == len(onset_stamps) - 1:
                if stamp + np.absolute(activity_window[1]) >= len(curr_spine_dFoF):
                    refined_idxs.append(False)
                else:
                    refined_idxs.append(True)
                continue
            refined_idxs.append(True)
        onset_stamps = list(compress(onset_stamps, refined_idxs))

        # Get the traces centered on the dendrite onsets
        dend_trace, dend_mean = d_utils.get_trace_mean_sem(
            dendrite_dFoF.reshape(-1, 1),
            ["Dendrite"],
            onset_stamps,
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        dend_mean = list(dend_mean.values())[0][0]
        dend_trace = list(dend_trace.values())[0]
        # Get the area under curve for the mean activity trace
        d_area_trace = dend_mean[center_point:]
        ## normalize the start to zero
        d_area_trace = d_area_trace - d_area_trace[0]
        d_auc = np.trapz(d_area_trace)
        # Append dendrite values
        dend_traces.append(dend_trace)
        dend_amplitudes.append(dend_amp)
        dend_stds.append(dend_std)
        dend_auc.append(d_auc)
        # Get the traces for the current spine
        spine_trace, spine_mean = d_utils.get_trace_mean_sem(
            curr_spine_dFoF.reshape(-1, 1),
            ["Spine"],
            onset_stamps,
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        spine_mean = list(spine_mean.values())[0][0]
        spine_trace = list(spine_trace.values())[0]
        # Get the spine onsets and amplitudes and auc
        s_onset, s_amp = find_activity_onset([spine_mean])
        s_onset = s_onset[0]
        s_amp = s_amp[0]
        smax_idx = np.where(spine_mean == s_amp)
        s_std_trace = np.nanstd(spine_trace, axis=1)
        spine_std = s_std_trace[smax_idx]
        s_area_trace = spine_mean[s_onset:]
        s_area_trace = s_area_trace - s_area_trace[0]
        s_auc = np.trapz(s_area_trace)
        # Get the relative amplitude
        relative_onset = (s_onset - center_point) / sampling_rate
        # Append spine values
        spine_traces.append(spine_trace)
        spine_amplitudes.append(s_amp)
        spine_stds.append(spine_std)
        spine_auc.append(s_auc)
        relative_onsets.append(relative_onset)

    # Convert some outputs to arrays
    dend_amplitudes = np.array(dend_amplitudes)
    dend_stds = np.array(dend_stds)
    dend_auc = np.array(dend_auc)
    spine_amplitudes = np.array(spine_amplitudes)
    spine_stds = np.array(spine_stds)
    spine_auc = np.array(spine_auc)
    relative_onsets = np.array(relative_onsets)

    return (
        spine_traces,
        dend_traces,
        spine_amplitudes,
        spine_auc,
        spine_stds,
        dend_amplitudes,
        dend_auc,
        dend_stds,
        relative_onsets,
    )


def get_activity_timestamps(activity):
    """Helper function to get timestamps for activity onsets"""
    # Get activity onsets and offsets
    diff = np.insert(np.diff(activity), 0, 0)
    onset = np.nonzero(diff == 1)[0]
    offset = np.nonzero(diff == -1)[0]
    # Make sure onsets and offsets are of the same length
    if len(onset) > len(offset):
        # Drop last onset if there is no offset
        onset = onset[:-1]
    elif len(onset) < len(offset):
        # Drop first offset if there is no onset for it
        offset = offset[1:]
    # Get timestamps
    timestamps = []
    for on, off in zip(onset, offset):
        timestamps.append((on, off))

    return timestamps


def find_activity_onset(activity_means, sampling_rate=60):
    """Helper function to find the activity onsset of mean activity traces
    
        INPUT PARAMETERS
            activity_means - list of the mean activity traces
            
            sampling_rate - int specifying the sampling rate
            
        OUTPUT PARAMETERS
            activity_onsets - np.array of activity onsets

            trace_amplitudes - np.array of peak amplitude of the trace
    """
    DISTANCE = 0.5 * sampling_rate  ## minimum distance of 0.5 seconds

    # Set up the output
    activity_onsets = np.zeros(len(activity_means))
    trace_amplitudes = np.zeros(len(activity_means))
    # Find each onset
    for i, trace in enumerate(activity_means):
        trace_med = np.median(trace)
        trace_std = np.std(trace)
        trace_h = trace_med + trace_std
        trace_peaks, trace_props = sysignal.find_peaks(
            trace, height=trace_h, distance=DISTANCE,
        )
        trace_amps = trace_props["peak_heights"]
        # Get the max peak
        try:
            trace_peak = trace_peaks[np.argmax(trace_amps)]
        # If no activity, set onset to be np.nan
        except ValueError:
            activity_onsets[i] = np.nan
            continue
        # Get the offset of the rise phase
        trace_amp = np.max(trace_amps)
        trace_peak_trace = trace[:trace_peak]
        trace_offset = np.where(trace_peak_trace < 0.75 * trace_amp)[0][-1]
        # Get the trace velocity
        trace_search_trace = trace[:trace_offset]
        trace_deriv = np.gradient(trace)[:trace_offset]
        # Find where the velocity goes below zero
        trace_below_zero = trace_deriv <= 0
        if sum(trace_below_zero):
            onset = np.nonzero(trace_below_zero)[0][-1]
        # If derivative doesn't go below zero, find where it goes below the median
        else:
            try:
                onset = np.where(trace_search_trace < trace_med)[0][-1]
            except:
                onset = 0
        activity_onsets[i] = onset
        trace_amplitudes[i] = trace_amp

    return activity_onsets, trace_amplitudes
