from itertools import compress

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.quantify_movment_quality import quantify_movement_quality
from scipy import stats


def get_coactivity_rate(trace_1, trace_2, sampling_rate):
    """Function to analyze the coactivity rate between two different activity traces.
        Calculates several different measures of coactivity as well as the fraction
        of activity for each trace that is coactive. 
        
        INPUT PARAMETERS
            trace_1 - np.array of the binary activity trace
            
            trace_2 - np.array of the binary activity trace
            
            sampling_rate - int or float of the sampling rate
            
        OUTPUT PARAMETERS
            coactivity_event_rate - float of absolute coactivity event rate
            
            coactivity_event_rate_norm - float of the normalized coactivity event rate
            
            coactivity_event_rate_alt - float of absolute coactivity event rate calulated
                                        using an alternative method
            
            trace_1_frac_coactive - float of the fraction of activity that is coactive
                                    for trace_1
            
            trace_2_frac_coative - float of the fraction of activity that is coactive
                                    for trace_2
                                    
            coactivity_trace - np.array of the binarized coactivity between the two traces

            dot_corr - float of the dot_product / corr of the two traces
    """
    # Check there is activity in the traces
    if (np.sum(trace_1) == 0) or (np.sum(trace_2) == 0):
        coactivity_event_rate = 0
        coactivity_event_rate_norm = 0
        coactivity_event_rate_alt = 0
        trace_1_frac_coactive = 0
        trace_2_frac_coactive = 0
        coactivity_trace = np.zeros(len(trace_1))
        dot_corr = 0
        return (
            coactivity_event_rate,
            coactivity_event_rate_norm,
            coactivity_event_rate_alt,
            trace_1_frac_coactive,
            trace_2_frac_coactive,
            coactivity_trace,
            dot_corr,
        )
    # Get the total time in seconds
    duration = len(trace_1) / sampling_rate

    # Calculate the traditional coactivity_rate
    coactivity = trace_1 * trace_2
    if not np.sum(coactivity):
        coactivity_event_rate = 0
        coactivity_event_rate_norm = 0
        coactivity_event_rate_alt = 0
        trace_1_frac_coactive = 0
        trace_2_frac_coactive = 0
        coactivity_trace = np.zeros(len(trace_1))
        dot_corr = 0
        return (
            coactivity_event_rate,
            coactivity_event_rate_norm,
            coactivity_event_rate_alt,
            trace_1_frac_coactive,
            trace_2_frac_coactive,
            coactivity_trace,
            dot_corr,
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

    # Calculate alternaative coactivity rate and fraction coactive
    ## break up activity trace
    active_boundaries = np.insert(np.diff(trace_1), 0, 0, axis=0)
    active_onsets = np.nonzero(active_boundaries == 1)[0]
    active_offsets = np.nonzero(active_boundaries == -1)[0]
    ## Check onset offset order
    if active_onsets[0] > active_offsets[0]:
        active_offsets = active_offsets[1:]
    ## Check onsets and offests are same length
    if len(active_onsets) > len(active_offsets):
        active_onsets = active_onsets[:-1]
    # compare active epochs to other trace
    coactive_idxs = []
    for onset, offset in zip(active_onsets, active_offsets):
        if np.sum(trace_2[onset:offset]):
            coactive_idxs.append((onset, offset))
    # Calculate coactivity rate
    coactivity_event_rate_alt = len(coactive_idxs) / duration * 60
    coactivity_event_rate = coactivity_event_rate * 60
    # Generate coactivity trace
    coactivity_trace = np.zeros(len(trace_1))
    for epoch in coactive_idxs:
        coactivity_trace[epoch[0] : epoch[1]] = 1

    # Calculate fraction coactive
    try:
        trace_1_frac_coactive = len(coactive_idxs) / trace_1_event_num
    except ZeroDivisionError:
        trace_1_frac_coactive = 0
    try:
        trace_2_frac_coactive = len(coactive_idxs) / trace_2_event_num
    except ZeroDivisionError:
        trace_2_frac_coactive = 0

    # calculate dot_product / corr
    dot_product = np.dot(trace_1, trace_2)
    corr = stats.pearsonr(trace_1, trace_2)[0]
    dot_corr = dot_product / corr

    return (
        coactivity_event_rate,
        coactivity_event_rate_norm,
        coactivity_event_rate_alt,
        trace_1_frac_coactive,
        trace_2_frac_coactive,
        coactivity_trace,
        dot_corr,
    )


def get_dend_spine_traces_and_onsets(
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
    center_point = int(np.absolute(activity_window[0] * sampling_rate))
    # Perform the analysis for each spine seperately
    for i in range(spine_activity_matrix.shape[1]):
        curr_reference = reference_trace[:, i]
        curr_spine_dFoF = spine_dFoF_matrix[:, i]

        # Skip spines when there is no reference coactivity
        if not np.sum(curr_reference):
            dend_traces.append([])
            spine_traces.append([])
            dend_amplitudes.append(np.nan)
            dend_stds.append(np.nan)
            dend_auc.append(np.nan)
            spine_amplitudes.append(np.nan)
            spine_stds.append(np.nan)
            spine_auc.append(np.nan)
            relative_onsets.append(np.nan)
            continue

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
        initial_d_mean = d_mean["Dendrite"][0]
        d_trace = d_trace["Dendrite"]
        # Find the onset
        initial_d_onset, dend_amp = find_activity_onset([initial_d_mean])
        initial_d_onset = initial_d_onset[0]  # Get value inside array output
        dend_amp = dend_amp[0]  # Get value inside array output
        if not np.isnan(dend_amp):
            dmax_idx = np.nonzero(initial_d_mean == dend_amp)[0][0]
        else:
            dmax_idx = center_point
            initial_d_onset = 0
        d_std_trace = np.nanstd(d_trace, axis=1)
        dend_std = d_std_trace[dmax_idx]

        # Correct timestamps to be at the onset
        offset = center_point - initial_d_onset
        offset_stamps = [int(x - offset) for x in initial_stamps]

        # Refine the timestamps
        onset_stamps = refine_activity_timestamps(
            offset_stamps,
            activity_window,
            max_len=len(curr_spine_dFoF),
            sampling_rate=sampling_rate,
        )
        # Get the traces centered on the dendrite onsets
        dend_trace, dend_mean = d_utils.get_trace_mean_sem(
            dendrite_dFoF.reshape(-1, 1),
            ["Dendrite"],
            onset_stamps,
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        dend_mean = dend_mean["Dendrite"][0]
        dend_trace = dend_trace["Dendrite"]
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
        spine_mean = spine_mean["Spine"][0]
        spine_trace = spine_trace["Spine"]
        if norm_constants is not None:
            if norm_constants[i] is not None:
                spine_mean = spine_mean / norm_constants[i]
                spine_trace = spine_trace / norm_constants[i]
        # Get the spine onsets and amplitudes and auc
        s_onset, s_amp = find_activity_onset([spine_mean])
        s_onset = s_onset[0]
        s_amp = s_amp[0]
        # Skip spines if no peak is found
        if np.isnan(s_amp):
            spine_traces.append(spine_trace)
            spine_amplitudes.append(np.nan)
            spine_stds.append(np.nan)
            spine_auc.append(np.nan)
            relative_onsets.append(np.nan)
            continue
        s_onset = int(s_onset)
        smax_idx = np.nonzero(spine_mean == s_amp)[0][0]
        s_std_trace = np.nanstd(spine_trace, axis=1)
        spine_std = s_std_trace[smax_idx]
        s_area_trace = spine_mean[s_onset:]
        s_area_trace = s_area_trace - s_area_trace[0]
        s_auc = np.trapz(s_area_trace)
        # Get the relative onset in seconds
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


def refine_activity_timestamps(timestamps, window, max_len, sampling_rate=60):
    """Helper function to refine timestamps to make sure they fit in the window"""
    refined_idxs = []
    before = np.absolute(window[0] * sampling_rate)
    after = np.absolute(window[1] * sampling_rate)
    for stamp in timestamps:
        if stamp - before < 0:
            refined_idxs.append(False)
            continue

        if stamp + after > max_len - 1:
            refined_idxs.append(False)
            continue

        refined_idxs.append(True)

    refined_stamps = list(compress(timestamps, refined_idxs))

    return refined_stamps


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
        # If no peaks detected, set onset and amplitude to be np.nan
        except ValueError:
            activity_onsets[i] = np.nan
            trace_amplitudes[i] = np.nan
            continue
        # Get the offset of the rise phase
        trace_amp = np.max(trace_amps)
        trace_peak_trace = trace[:trace_peak]
        try:
            trace_offset = np.nonzero(trace_peak_trace < 0.75 * trace_amp)[0][-1]
        except IndexError:
            activity_onsets[i] = 0
            trace_amplitudes[i] = trace_amp
            continue
        # Get the trace velocity
        ## smooth trace for better estimation of onset
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
    initial_stamps = refine_activity_timestamps(
        initial_stamps,
        activity_window,
        max_len=len(spine_dFoF),
        sampling_rate=sampling_rate,
    )

    _, dend_mean = d_utils.get_trace_mean_sem(
        dendrite_dFoF.reshape(-1, 1),
        ["Dendrite"],
        initial_stamps,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    d_mean = dend_mean["Dendrite"][0]
    d_onset, _ = find_activity_onset([d_mean])
    d_onset = d_onset[0]
    # Correct timestamps so that they are centered on dendrite onsets
    center_point = np.absolute(activity_window[0] * sampling_rate)
    try:
        offset = int(center_point - d_onset)
    except ValueError:
        offset = 0
    if offset == center_point:
        offset = 0
    event_stamps = [int(x - offset) for x in initial_stamps]

    # Refine event stamps
    event_stamps = refine_activity_timestamps(
        event_stamps,
        activity_window,
        max_len=len(spine_dFoF),
        sampling_rate=sampling_rate,
    )

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
        try:
            spine_trace_array = np.vstack(coactive_spine_traces).T
        except:
            continue
        spine_ca_trace_array = np.vstack(coactive_spine_ca_traces).T
        sum_spine_trace = np.nansum(spine_trace_array, axis=1)
        sum_spine_ca_trace = np.nansum(spine_ca_trace_array, axis=1)
        corr, _ = stats.pearsonr(t_spine_trace, sum_spine_trace)
        num = len(coactive_spine_idxs)
        vol = np.nanmean(nearby_spine_volumes[coactive_spine_idxs])
        # Store values
        spine_nearby_correlations.append(corr)
        coactive_spine_num.append(num)
        coacitve_spine_volumes.append(vol)
        sum_coactive_spine_traces.append(sum_spine_trace)
        sum_coactive_spine_ca_traces.append(sum_spine_ca_trace)

    # Reformat nearby spine traces into arrays
    try:
        sum_coactive_spine_traces = np.vstack(sum_coactive_spine_traces).T
    except ValueError:
        avg_coactive_correlation = np.nan
        avg_coactive_num = 0
        avg_coactive_volume = np.nan
        activity_amplitude = np.nan
        ca_activity_amplitude = np.nan
        activity_std = np.nan
        ca_activity_std = np.nan
        ca_activity_auc = np.nan
        sum_coactive_spine_traces = None
        sum_coactive_spine_ca_traces = None

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

    sum_coactive_spine_ca_traces = np.vstack(sum_coactive_spine_ca_traces).T

    # Average correlations, nums, and volumes,
    avg_coactive_correlation = np.nanmean(spine_nearby_correlations)
    avg_coactive_num = np.nanmean(coactive_spine_num)
    avg_coactive_volume = np.nanmean(coacitve_spine_volumes)

    # Get peak, std, and auc of traces
    avg_coactive_trace = np.nanmean(sum_coactive_spine_traces, axis=1)
    std_coactive_trace = np.nanstd(sum_coactive_spine_traces, axis=1)
    avg_coactive_ca_trace = np.nanmean(sum_coactive_spine_ca_traces, axis=1)
    std_coactive_ca_trace = np.nanstd(sum_coactive_spine_ca_traces, axis=1)
    onsets, amps = find_activity_onset(
        [avg_coactive_trace, avg_coactive_ca_trace], sampling_rate=sampling_rate
    )
    if not np.isnan(amps[0]):
        activity_amplitude = amps[0]
        activity_max = np.nonzero(avg_coactive_trace == activity_amplitude)[0][0]
        activity_std = std_coactive_trace[activity_max]
    else:
        activity_amplitude = amps[0]
        activity_std = np.nan

    if not np.isnan(amps[1]):
        ca_activity_onset = int(onsets[1])
        ca_activity_amplitude = amps[1]
        ca_activity_max = np.nonzero(avg_coactive_ca_trace == ca_activity_amplitude)[0][
            0
        ]
        ca_activity_std = std_coactive_ca_trace[ca_activity_max]
        area_trace = avg_coactive_ca_trace[ca_activity_onset:]
        area_trace = area_trace - area_trace[0]
        ca_activity_auc = np.trapz(area_trace)
    else:
        ca_activity_onset = np.nan
        ca_activity_amplitude = amps[1]
        ca_activity_std = np.nan
        ca_activity_auc = np.nan

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


def calculate_dend_spine_freq(data, movement_epoch, sampling_rate=60):
    """Function to calculate the overall activity rate of dendrites and spines
        Can be narrowed down to during specific types of movements
        
        INPUT PARAMETERS
            data - spine_data object (e.g., Dual_Channel_Spine_Dat
            
            movement_epochs - str specifying if analysis should be confined to specific
                            movement periods. Accepts - 'movement', 'rewarded', 'unrewarded',
                            'learned', and 'nonmovement'. Default is None, analyzing the entire
                            imaging session
            
            sampling_rate - int or float specifying the imaging rate
    """
    spine_activity = data.spine_GluSnFr_activity
    dendrite_activity = data.dendrite_calcium_activity

    # Get specific movement periods if specified
    if movement_epoch == "movement":
        movement = data.lever_active
    elif movement_epoch == "rewarded":
        movement = data.rewarded_movement_binary
    elif movement_epoch == "unrewarded":
        movement = data.lever_active - data.rewarded_movement_binary
    elif movement_epoch == "nonmovement":
        movement = np.absolute(data.lever_active - 1)
    elif movement_epoch == "learned":
        movement, _, _, _, _ = quantify_movement_quality(
            data.mouse_id,
            spine_activity,
            data.lever_active,
            threshold=0.5,
            sampling_rate=sampling_rate,
        )
    else:
        movement = None

    if movement is not None:
        spine_activity = (spine_activity.T * movement).T
        dendrite_activity = (dendrite_activity.T * movement).T

    dend_activity_matrix = np.zeros(data.spine_GluSnFr_activity.shape)
    for d in range(dendrite_activity.shape[1]):
        if type(data.spine_grouping[d]) == list:
            spines = data.spine_grouping[d]
        else:
            spines = data.spine_grouping
        for s in spines:
            dend_activity_matrix[:, s] = dendrite_activity[:, d]

    spine_activity_freq = []
    dend_activity_freq = []
    for s in range(spine_activity.shape[1]):
        s_activity = spine_activity[:, s]
        d_activity = dend_activity_matrix[:, s]
        duration = len(s_activity) / sampling_rate
        s_events = np.nonzero(np.diff(s_activity) == 1)[0]
        d_events = np.nonzero(np.diff(d_activity) == 1)[0]
        s_freq = (len(s_events) / duration) * 60  # per minute
        d_freq = (len(d_events) / duration) * 60  # per minute
        spine_activity_freq.append(s_freq)
        dend_activity_freq.append(d_freq)
    spine_activity_freq = np.array(spine_activity_freq)
    dend_activity_freq = np.array(dend_activity_freq)

    return spine_activity_freq, dend_activity_freq
