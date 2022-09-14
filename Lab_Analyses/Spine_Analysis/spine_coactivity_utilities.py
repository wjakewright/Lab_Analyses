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
    reference_trace,
    norm_constants=None,
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

            reference_trace - 2d or 1d np.array of binary activity trace to use to get
                              timestamps 

            norm_constants - np.array of constants to normalize activity by volume
            
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
    if len(reference_trace.shape) == 1:
        reference_trace = np.hstack(
            [
                reference_trace.reshape(-1, 1)
                for x in range(spine_activity_matrix.shape[1])
            ]
        )

    # Perform the analysis for each spine seperately
    for i in range(spine_activity_matrix.shape[1]):
        curr_reference = reference_trace[:, i]
        curr_spine_dFoF = spine_dFoF_matrix[:, i]

        # Get the initial timestamps
        initial_stamps = get_activity_timestamps(curr_reference)
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
        if norm_constants is not None:
            spine_mean = spine_mean / norm_constants[i]
            spine_trace = spine_trace / norm_constants[i]
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


def nearby_spine_conjunctive_events(
    timestamps,
    spine_dFoF,
    nearby_dFoF,
    nearby_calcium,
    nearby_activity,
    dendrite_dFoF,
    nearby_spine_volumes,
    target_constant=None,
    glu_constants=None,
    ca_constants=None,
    activity_window=(-2, 2),
    sampling_rate=60,
):
    """Helper function to get the dendrite, spine, and nearby spine traces
        during conjunctive coactivity events
        
        INPUT PARAMETERS
            timestamps - list of tuples with the timestamps (onset, offset) of each 
                        conjunctive coactivity event
        
            spine_dFoF - np.array of the main spine dFoF activity
            
            nearby_dFoF - 2d np.array of the nearby spines (columns) dFoF activity

            nearby_calcium - 2d np.array of the nearby spines (columns) calcium activity

            nearby_activity - 2d np.array of the nearby spines (columns) binarized activity
            
            dendrite_dFoF - np.array of the dendrite dFoF activity
            
            nearby_spine_volumes - np.array of the spine volumes of the nearby spines

            target_constant = float or int of the GluSnFR constant to normalize activity by 
                              spine volume. No noramlization if None
            
            glu_constants - np.array of the GluSnFR constants for the nearby spines

            ca_constants - np.array of the RCaMP2 constants for the nearby spines
            
            activity_window - tuple specifying the window around which you want the activity
                              from . e.g., (-2,2) for 2sec before and after
                              
            sampling_rate - int specifying the sampling rate

        OUTPUT PARAMETERS
            avg_coactive_correlation - float or int of the average correlation of the summed
                                        nearby coactive spine activity and the target spine

            avg_coactive_num - float of the average number of nearby spines coactive during 
                                each event

            avg_coactive_volume - float of the average volume of nearby spines coactive with 
                                    with the target spine. Coactive spines are averaged for each
                                    event and then averaged across all events

            activity_amplitude - float of the average amplitude of the mean nearby spine trace. 
                                 For each event, nearby coactive spine traces are summed. These
                                 are then averaged across all trials, from which the peak is taken

            ca_activity_amplitude- float of the average calcium amplitude of the mean nearby spine
                                    trace. Calculated same as activity_amplitude

            activity_std - float of the average std around the activity peak. Measured by taking the
                            std of the summed coactive traces across each event at the time point
                            of the averaged max peak

            ca_activity_std - float of the average std around the calcium activity peak. Calculated
                              in the same manner as activity_std

            ca_activity_auc - float of the average area under the curve of the average nearby coactive
                                spine calcium traces

            sum_coactive_spine_traces - 2d np.array of the summed nearby spine activity for each coactive
                                        event (columns)
                                        
            sum_coactive_spine_ca_traces - 2d np.array of the summed nearby spine calcium activity for 
                                            each coactive event (columns)
    """
    if target_constant is not None:
        NORM = True
    else:
        NORM = False

    before_f = int(activity_window[0] * sampling_rate)
    after_f = int(activity_window[1] * sampling_rate)

    # Find dendrite onsets to center analysis around
    initial_stamps = [x[0] for x in timestamps]
    _, d_mean = d_utils.get_trace_mean_sem(
        dendrite_dFoF.reshape(-1, 1),
        ["Dendrite"],
        initial_stamps,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    d_mean = list(d_mean.values())[0][0]
    d_onset, _ = find_activity_onset([d_mean])
    d_onset = d_onset[0]
    # Correct timestamps so that they are centered on dendrite onsets
    center_point = np.absolute(activity_window[0] * sampling_rate)
    offset = center_point - d_onset
    event_stamps = [x - offset for x in initial_stamps]

    # Analyze each co-activity event
    ### Some temporary variables
    spine_nearby_correlations = []
    coactive_spine_num = []
    coacitve_spine_volumes = []
    sum_coactive_spine_traces = []
    sum_coactive_spine_ca_traces = []

    for event in event_stamps:
        # Get target spine activity
        t_spine_trace = spine_dFoF[event + before_f : event + after_f]
        if NORM:
            t_spine_trace = t_spine_trace / target_constant
        coactive_spine_traces = []
        coactive_spine_ca_traces = []
        coactive_spine_idxs = []
        # Check each nearby spine to see if coactive
        for i in range(nearby_activity.shape[1]):
            nearby_spine_a = nearby_activity[:, i]
            nearby_spine_dFoF = nearby_dFoF[:, i]
            nearby_spine_ca = nearby_calcium[:, i]
            event_activity = nearby_spine_a[event + before_f : event + after_f]
            if np.sum(event_activity):
                # If there is coactivity, append the value
                dFoF = nearby_spine_dFoF[event + before_f : event + after_f]
                calcium = nearby_spine_ca[event + before_f : event + after_f]
                if NORM:
                    dFoF = dFoF / glu_constants[i]
                    calcium = calcium / ca_constants[i]
                coactive_spine_traces.append(dFoF)
                coactive_spine_ca_traces.append(calcium)
                coactive_spine_idxs.append(i)
            else:
                continue
        # Process activity of this activity
        spine_trace_array = np.vstack(coactive_spine_traces).T
        spine_ca_trace_array = np.vstack(coactive_spine_ca_traces).T
        sum_spine_trace = np.sum(spine_trace_array, axis=1)
        sum_spine_ca_trace = np.sum(spine_ca_trace_array)
        corr, _ = stats.pearsonr(t_spine_trace, sum_spine_trace)
        num = len(coactive_spine_idxs)
        vol = np.mean(nearby_spine_volumes[coactive_spine_idxs])
        # Store values
        spine_nearby_correlations.append(corr)
        coactive_spine_num.append(num)
        coacitve_spine_volumes.append(vol)
        sum_coactive_spine_traces.append(sum_spine_trace)
        sum_coactive_spine_ca_traces.append(sum_spine_ca_trace)

    # Reformat nearby spine traces
    sum_coactive_spine_traces = np.vstack(sum_coactive_spine_traces).T
    sum_coactive_spine_ca_traces = np.vstack(sum_coactive_spine_ca_traces).T

    # Average correlations, nums, and volumes,
    avg_coactive_correlation = np.mean(spine_nearby_correlations)
    avg_coactive_num = np.mean(coactive_spine_num)
    avg_coactive_volume = np.mean(coacitve_spine_volumes)

    # Get peak, std, and auc of traces
    avg_coactive_trace = np.nanmean(sum_coactive_spine_traces, axis=1)
    std_coactive_trace = np.nanstd(sum_coactive_spine_traces, axis=1)
    avg_coactive_ca_trace = np.nanmean(sum_coactive_spine_ca_traces, axis=1)
    std_coactive_ca_trace = np.nanstd(sum_coactive_spine_ca_traces, axis=1)
    onsets, amps = find_activity_onset(
        [avg_coactive_trace, avg_coactive_ca_trace], sampling_rate=sampling_rate
    )
    ca_activity_onset = onsets[1]
    activity_amplitude = amps[0]
    ca_activity_amplitude = amps[1]
    activity_max = np.where(avg_coactive_trace == activity_amplitude)
    ca_activity_max = np.where(avg_coactive_ca_trace == ca_activity_amplitude)
    activity_std = std_coactive_trace[activity_max]
    ca_activity_std = std_coactive_ca_trace[ca_activity_max]
    area_trace = avg_coactive_ca_trace[ca_activity_onset:]
    area_trace = area_trace - area_trace[0]
    ca_activity_auc = np.trapz(area_trace)

    return (
        avg_coactive_correlation,
        avg_coactive_num,
        avg_coactive_volume,
        activity_amplitude,
        ca_activity_amplitude,
        activity_std,
        ca_activity_std,
        ca_activity_auc,
        sum_coactive_spine_traces,
        sum_coactive_spine_ca_traces,
    )

