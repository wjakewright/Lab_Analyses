from itertools import compress

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.movement_responsiveness import movement_responsiveness
from scipy import stats


def total_coactivity_analysis(
    data, movement_epoch=None, sampling_rate=60, zscore=False
):
    """Function to analyze spine co-activity with global dendritic activity
    
        INPUT PARAMETERS
            data - spine_data object. (e.g. Dual_Channel_Spine_Data)
            
            movement_epoch - str specifying if you want to analyze only during specific
                            types of movements. Accepts - 'movement', 'rewarded', 
                            'unrewarded' and 'nonmovement'. Default is None, analyzing
                            the entire imaging session
            
            sampling_rate - int or float specifying what the imaging rate

            zscore - boolean of whether to zscore dFoF traces for and analysis
            
        OUTPUT PARAMETERS
            global_correlation - np.array of the correlation coefficient between spine
                                and dendrite fluorescence traces
                                
            coactivity_event_num - np.array of the number of coactive events for each spine
            
            coactivity_event_rate - np.array of the normalized coactivity rate for each spine
            
            spine_fraction_coactive - np.array of the fraction of spine events that were also
                                      coactive
            
            dend_fraction_coactive - np.array of the fraction of dendritic events that were also
                                     coactive with a given spine
            
            spine_coactive_amplitude - np.array of the peak mean response of each spine during coactive
                                       events

            dend_coactive_amplitude - np.array of the peak mean response of dendritic activity during
                                      coactive events of a given spine
            
            relative_spine_coactive_amplitude - np.array of the peak mean responses of spine coactivity
                                                during coactive events of a given spine normalized to 
                                                it mean activity across all dendritic events

            relative_dend_coactive_amplitude - np.array of the peak mean responses of dendritic coactivity
                                                during coactive events of a given spine normalized to 
                                                it mean activity across all dendritic events
            
            relative_spine_onsets - np.array of the mean onset of spine activity relative to dendritic 
                                    activity for coactive events
            
            dend_triggered_spine_traces - list of 2d np.arrays of spine activity around each
                                          dendritic event. Centered around dendrite onset. 
                                          columns = each event, rows = time (in frames)
            
            dend_triggered_dend_traces - list of 2d np.array of dendrite activty around each
                                         each dendritic event. Centered around 

            coactive_spine_traces - list of 2d np.arrays of spine activity around each coactive
                                    event. Centered around corresponding dendrite onset.
                                    column = each event, rows = time (in frames)
            
            coactive_dend_traces - list of 2d np.arrays of dendrite activity around each coactive
                                    event. Centered arorund dendrite onsets. 
                                    column = each event, rows = time (in frames)
            
            coactivity_matrix - 2d np.array of the coactivity trace for each spine (columns)
    """
    # Pull some important information from data
    spine_groupings = data.spine_grouping
    spine_dFoF = data.spine_GluSnFr_processed_dFoF
    spine_activity = data.spine_GluSnFr_activity
    dendrite_dFoF = data.dendrite_calcium_processed_dFoF
    dendrite_activity = data.dendrite_calcium_activity

    if zscore:
        spine_dFoF = d_utils.z_score(spine_dFoF)
        dendrite_dFoF = d_utils.z_score(dendrite_dFoF)

    ## Get specific movement periods
    if movement_epoch == "movement":
        movement = data.lever_active
    elif movement_epoch == "rewarded":
        movement = data.rewarded_movement_binary
    elif movement_epoch == "unrewarded":
        movement = data.lever_active - data.rewarded_movement_binary
    elif movement_epoch == "nonmovement":
        movement = np.absolute(data.lever_active - 1)
    else:
        movement = None

    # Set up output variables
    global_correlation = np.zeros(spine_activity.shape[1])
    coactivity_event_num = np.zeros(spine_activity.shape[1])
    coactivity_event_rate = np.zeros(spine_activity.shape[1])
    spine_fraction_coactive = np.zeros(spine_activity.shape[1])
    dend_fraction_coactive = np.zeros(spine_activity.shape[1])
    spine_coactive_amplitude = np.zeros(spine_activity.shape[1])
    dend_coactive_amplitude = np.zeros(spine_activity.shape[1])
    relative_dend_coactive_amplitude = np.zeros(spine_activity.shape[1])
    relative_spine_coactive_amplitude = np.zeros(spine_activity.shape[1])
    relative_spine_onsets = np.zeros(spine_activity.shape[1])
    dend_triggered_spine_traces = [None for i in global_correlation]
    dend_triggered_dend_traces = [None for i in global_correlation]
    coactive_spine_traces = [None for i in global_correlation]
    coactive_dend_traces = [None for i in global_correlation]
    coactivity_matrix = np.zeros(spine_activity.shape)

    # Process spines for each parent dendrite
    for dendrite in range(dendrite_activity.shape[1]):
        # Get the spines on this dendrite
        if type(spine_groupings[dendrite]) == list:
            spines = spine_groupings[dendrite]
        else:
            spines = spine_groupings
        s_dFoF = spine_dFoF[:, spines]
        s_activity = spine_activity[:, spines]
        d_dFoF = dendrite_dFoF[:, dendrite]
        d_activity = dendrite_activity[:, dendrite]

        # Refine activity matrices for only movement epochs if specified
        if movement is not None:
            s_activity = (s_activity.T * movement).T
            d_activity = d_activity * movement

        # Analyze coactivity rates for each spine
        for spine in range(s_dFoF.shape[1]):
            # Perform correlation
            if movement is not None:
                # Correlation only during specified movements
                move_idxs = np.where(movement == 1)[0]
                corr, _ = stats.pearsonr(s_dFoF[move_idxs, spine], d_dFoF[move_idxs])
            else:
                corr, _ = stats.pearsonr(s_dFoF[:, spine], d_dFoF)
            global_correlation[spines[spine]] = corr

            # Calculate coactivity rate
            event_num, event_rate, spine_frac, dend_frac = get_coactivity_rate(
                s_activity[:, spine], d_activity, sampling_rate=sampling_rate
            )
            coactivity_event_num[spines[spine]] = event_num
            coactivity_event_rate[spines[spine]] = event_rate
            spine_fraction_coactive[spines[spine]] = spine_frac
            dend_fraction_coactive[spines[spine]] = dend_frac
            coactivity_matrix[:, spines[spine]] = d_activity * s_activity[:, spine]

        # Get amplitudes, relative_onsets and activity traces
        ### First get for all dendritic events
        (
            dt_spine_traces,
            dt_dendrite_traces,
            dt_spine_amps,
            dt_dendrite_amps,
            _,
        ) = get_dend_spine_traces_and_onsets(
            d_activity,
            s_activity,
            d_dFoF,
            s_dFoF,
            coacitivity=False,
            activity_window=(-2, 2),
            sampling_rate=sampling_rate,
        )
        ### Get for coactive events only
        (
            co_spine_traces,
            co_dendrite_traces,
            co_spine_amps,
            co_dendrite_amps,
            rel_onsets,
        ) = get_dend_spine_traces_and_onsets(
            d_activity,
            s_activity,
            d_dFoF,
            s_dFoF,
            coacitivity=True,
            activity_window=(-2, 2),
            sampling_rate=sampling_rate,
        )
        rel_dend_amps = dt_dendrite_amps - co_dendrite_amps
        rel_spine_amps = dt_spine_amps - co_spine_amps
        # Store values
        for i in range(len(dt_spine_traces)):
            spine_coactive_amplitude[spines[i]] = co_spine_amps[i]
            dend_coactive_amplitude[spines[i]] = co_dendrite_amps[i]
            relative_dend_coactive_amplitude[spines[i]] = rel_dend_amps[i]
            relative_spine_coactive_amplitude[spines[i]] = rel_spine_amps[i]
            relative_spine_onsets[spines[i]] = rel_onsets[i]
            dend_triggered_spine_traces[spines[i]] = dt_spine_traces[i]
            dend_triggered_dend_traces[spines[i]] = dt_dendrite_traces[i]
            coactive_spine_traces[spines[i]] = co_spine_traces[i]
            coactive_dend_traces[spines[i]] = co_dendrite_traces[i]

    # Return the output
    return (
        global_correlation,
        coactivity_event_num,
        coactivity_event_rate,
        spine_fraction_coactive,
        dend_fraction_coactive,
        spine_coactive_amplitude,
        dend_coactive_amplitude,
        relative_dend_coactive_amplitude,
        relative_spine_coactive_amplitude,
        relative_spine_onsets,
        dend_triggered_spine_traces,
        dend_triggered_dend_traces,
        coactive_spine_traces,
        coactive_dend_traces,
        coactivity_matrix,
    )


def get_coactivity_rate(spine, dendrite, sampling_rate):
    """Helper function to calculate the coactivity frequency between a spine and 
        its parent dendrite. These rates are normalized by geometric mean of the spine
        and dendrite activity rates
        
        INPUT PARAMETERS
            spine - np.array of spine binary activity trace
            
            dendrite - np.array of dendrite binary activity traces
            
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
    coactivity = spine * dendrite
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

            dend_ampliitudes - np.array of the peak dendrite amplitudes
            
            relative_onsets - np.array of spine onsets relative to dendrite
    """
    # Set up outputs
    dend_traces = []
    spine_traces = []
    dend_amplitudes = []
    spine_amplitudes = []
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
        _, d_mean = d_utils.get_trace_mean_sem(
            dendrite_dFoF.reshape(-1, 1),
            ["Dendrite"],
            initial_stamps,
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        initial_d_mean = list(d_mean.values())[0][0]
        # Find the onset
        initial_d_onset, dend_amp = find_activity_onset([initial_d_mean])
        initial_d_onset = initial_d_onset[0]
        dend_amp = dend_amp[0]
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
        dend_trace, _ = d_utils.get_trace_mean_sem(
            dendrite_dFoF.reshape(-1, 1),
            ["Dendrite"],
            onset_stamps,
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        dend_trace = list(dend_trace.values())[0]
        # Append dendrite values
        dend_traces.append(dend_trace)
        dend_amplitudes.append(dend_amp)
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
        # Get the spine onsets and amplitudes
        s_onset, s_amp = find_activity_onset([spine_mean])
        s_onset = s_onset[0]
        s_amp = s_amp[0]
        # Get the relative amplitude
        relative_onset = (s_onset - center_point) / sampling_rate
        # Append spine values
        spine_traces.append(spine_trace)
        spine_amplitudes.append(s_amp)
        relative_onsets.append(relative_onset)

    # Convert some outputs to arrays
    dend_amplitudes = np.array(dend_amplitudes)
    spine_amplitudes = np.array(spine_amplitudes)
    relative_onsets = np.array(relative_onsets)

    return spine_traces, dend_traces, spine_amplitudes, dend_amplitudes, relative_onsets


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


def get_coactivity_timestamps(activity, coactivity):
    """Helper function to get timestamps of activity onsets, but only
        for coactive events"""
    # Get activity onsets and offsets
    timestamps = get_activity_timestamps(activity)
    # Assess if each timestamp coincides with coactivity
    coactive_timestamps = []
    for stamp in timestamps:
        coactivity_epoch = coactivity[stamp[0] : stamp[1] + 1]
        if sum(coactivity_epoch):
            coactive_timestamps.append(stamp)

    return coactive_timestamps


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

