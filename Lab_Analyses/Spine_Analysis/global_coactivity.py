"""Module for analyzing the co-activity of spines with global activity"""

from itertools import compress

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.movement_responsiveness import movement_responsiveness
from scipy import stats


def global_coactivity_analysis(data, sampling_rate=60):
    """Function to analyze spine co-activity with global dendritic activity
    
        INPUT PARAMETERS
            data -  spind_data object. (e.g., Dual_Channel_Spine_Data)
            
            sampling_rate -  int or float specifying what the imaging rate
            
        OUTPUT PARAMETERS
            global_correlation - np.array with the correlation of each spine with its 
                                parent dendrtie
                                
            coactivity_freq - np.array with the frequency of co-activity events for each spine
        
            spine_fraction_coactive- np.array with the fraction of total spine activity that are
                                    co-active with the parent_dendrite
            
            dend_fraction_coactive - np.array with the fraction of dend activity that the spine
                                    was co-active during
                                    
            coactive_amplitude - np.array with the amplitude of spine activity during dend activity
            
            coactive_spines - boolean array of whether or not a spine displayed coactivity with dend
            
            coactivity_mean_trace - dict containing the mean and sem activity trace of each spine
    """
    spine_ids = data.spine_ids
    spine_groupings = data.spine_grouping
    spine_dFoF = data.spine_GluSnFr_processed_dFoF
    spine_activity = data.spine_GluSnFr_activity
    dendrite_dFoF = data.dendrite_calcium_processed_dFoF
    dendrite_activity = data.dendrite_calcium_activity

    # Set up some of my output variables
    global_correlation = np.zeros(spine_activity.shape[1])
    coactivity_rate = np.zeros(spine_activity.shape[1])
    spine_fraction_coactive = np.zeros(spine_activity.shape[1])
    dend_fraction_coactive = np.zeros(spine_activity.shape[1])
    coactive_amplitude = np.zeros(spine_activity.shape[1])
    coactive_spines = np.zeros(spine_activity.shape[1])
    coactivity_mean_trace = {}
    coactivity_epoch_trace = {}
    for id in spine_ids:
        coactivity_mean_trace[id] = None
        coactivity_epoch_trace[id] = None

    dend_mean_sem = []

    # Now process spines for each parrent dendrite
    for i in range(dendrite_activity.shape[1]):
        if type(spine_groupings[i]) == list:
            spines = spine_groupings[i]
        else:
            spines = spine_groupings
        s_ids = np.array(spine_ids)[spines]
        s_dFoF = spine_dFoF[:, spines]
        s_activity = spine_activity[:, spines]
        d_dFoF = dendrite_dFoF[:, i]
        d_activity = dendrite_activity[:, i]

        # analyze each spine
        for j in range(s_dFoF.shape[1]):
            # Correlation
            corr, _ = stats.pearsonr(s_dFoF[:, j], d_dFoF)
            global_correlation[spines[j]] = corr
            # Coactivity rate
            coactivity_freq, spine_frac, dend_frac = get_coactivity_freq(
                s_activity[:, j], d_activity, sampling_rate=sampling_rate
            )
            coactivity_rate[spines[j]] = coactivity_freq
            spine_fraction_coactive[spines[j]] = spine_frac
            dend_fraction_coactive[spines[j]] = dend_frac

        # Get amplitude and co-activity traces
        ## Find dendrite activity preiods
        dend_diff = np.insert(np.diff(d_activity), 0, 0)
        dend_onsets = np.nonzero(dend_diff == 1)[0]
        dend_offsets = np.nonzero(dend_diff == -1)[0]
        ### Made sure onsets and offest are the same length
        if len(dend_onsets) > len(dend_offsets):
            # Drop last onset if there is no offest
            dend_onsets = dend_onsets[:-1]
        elif len(dend_onsets) < len(dend_offsets):
            # Drop first offset if there is no onset for it
            dend_offsets = dend_offsets[1:]
        timestamps = []
        for onset, offset in zip(dend_onsets, dend_offsets):
            timestamps.append((onset, offset))
        # refine timestamps
        refined_idxs = []
        for i, stamp in enumerate(timestamps):
            if i == 0:
                if stamp[0] - 120 < 0:
                    refined_idxs.append(False)
                else:
                    refined_idxs.append(True)
                continue
            if i == len(timestamps) - 1:
                if stamp[0] + 120 >= len(s_dFoF[:, 0]):
                    refined_idxs.append(False)
                else:
                    refined_idxs.append(True)
                continue

            refined_idxs.append(True)
        timestamps = list(compress(timestamps, refined_idxs))
        # Get mean activity before and during each dend activity event
        all_befores, all_durings = d_utils.get_before_during_means(
            s_dFoF, timestamps, window=1, sampling_rate=sampling_rate,
        )
        # Take the mean difference in activity
        for j, (before, during) in enumerate(zip(all_befores, all_durings)):
            diff = during - before
            mean_diff = np.mean(diff)
            coactive_amplitude[spines[j]] = mean_diff

        # Get the mean sem traces
        epoch_timestamps = [x[0] for x in timestamps]
        epoch_trace, mean_sems = d_utils.get_trace_mean_sem(
            s_dFoF,
            s_ids,
            epoch_timestamps,
            window=(-2, 2),
            sampling_rate=sampling_rate,
        )
        for key, value in mean_sems.items():
            coactivity_mean_trace[key] = value
        for key, value in epoch_trace.items():
            coactivity_epoch_trace[key] = value

        # get dend mean sem trace
        _, d_mean_sems = d_utils.get_trace_mean_sem(
            d_dFoF.reshape(-1, 1),
            [f"Dendrite {i}"],
            epoch_timestamps,
            window=(-2, 2),
            sampling_rate=sampling_rate,
        )
        dend_mean_sem.append(d_mean_sems[f"Dendrite {i}"])

        # Determine which spines are significantly coactive
        if sum(d_activity):
            sig_spines, _, _ = movement_responsiveness(s_dFoF, d_activity,)
        else:
            sig_spines = [False for x in range(s_dFoF.shape[1])]
        for j, sig in enumerate(sig_spines):
            coactive_spines[spines[j]] = sig

    return (
        global_correlation,
        coactivity_rate,
        spine_fraction_coactive,
        dend_fraction_coactive,
        coactive_amplitude,
        coactive_spines,
        coactivity_epoch_trace,
        coactivity_mean_trace,
        dend_mean_sem,
    )


def get_coactivity_freq(spine, dendrite, sampling_rate):
    """Helper function to calculate the coactivity frequency between a spine and
        its parent dendrite. These rates are normalized by geometric mean of the 
        spine and dendrite activity rates
        
        INPUT PARAMETERS
            spine - np.array of spine binary activity trace
            
            dendrite - np.array of dendrite binary activity trace

            sampling_rate - int or float of the sampling rate
            
        OUTPUT PARAMETERS
            coactivity_freq - float of co-activity frequency

            spine_frac_coactive - float of fraction of spine activity that is coactive

            dend_frac_coactive - float of fraction of dend activity the spine is active during
    
    """
    # Get the total time in secs
    duration = len(spine) / sampling_rate

    # Get coactivity binary trace
    coactivity_binary = spine * dendrite
    # Count how many co-activity events
    events = np.nonzero(np.diff(coactivity_binary) == 1)[0]
    event_num = len(events)

    # Get frequency
    event_freq = event_num / duration

    # Normalize frequency based on spine and dendrite activity
    spine_event_freq = len(np.nonzero(np.diff(spine) == 1)[0]) / duration
    dend_event_freq = len(np.nonzero(np.diff(dendrite) == 1)[0]) / duration
    geo_mean = stats.gmean([spine_event_freq, dend_event_freq])
    coactivity_freq = event_freq / geo_mean

    # get spine and dend frac
    try:
        spine_frac_coactive = event_freq / spine_event_freq
    except ZeroDivisionError:
        spine_frac_coactive = 0
    try:
        dend_frac_coactive = event_freq / dend_event_freq
    except ZeroDivisionError:
        dend_frac_coactive = 0

    return coactivity_freq, spine_frac_coactive, dend_frac_coactive


def find_activity_onset(spine_means, dend_mean, sampling_rate):
    """Function to find the activity onset for spines and their parent dendrite. 
        Also returns the difference relative to dendrite onset
        
        INPUT PARMAETERS
            spine_means - np.array of the mean traces for each spine, with each column
                          corresponding to each spine
                          
            dend_mean - np.array of the mean trace of the parent dendrite

            sampling_rate - int or float of the imaging rate
            
        OUTPUT PARAMETERS
    
    """
    DISTANCE = 0.5 * sampling_rate  ## minimum distance of 0.5 seconds

    # Get the onset for the dendrite first
    dend_med = np.median(dend_mean)
    dend_std = np.std(dend_mean)
    dend_h = dend_med + dend_std
    dend_peaks, dend_props = sysignal.find_peaks(
        dend_mean, height=dend_h, distance=DISTANCE
    )
    dend_amps = dend_props["peak_heights"]
    # get the max peak
    dend_peak = dend_peaks[np.argmax(dend_amps)]
    dend_amp = np.max(dend_amps)
    # Get the offset of the rise phase
    dend_peak_trace = dend_mean[:dend_peak]
    dend_offset = np.where(dend_peak_trace < 0.75 * dend_amp)[0][-1]
    # Get the trace velocity
    dend_search_trace = dend_mean[:dend_offset]
    dend_deriv = np.gradient(dend_mean)[:dend_offset]
    # Find where velocity goes to zero or below
    dend_below_zero = dend_deriv <= 0
    if sum(dend_below_zero):
        dend_onset = np.nonzero(dend_below_zero)[0][-1]
    # If derivative doesn't go below zero, find where it goes below median
    else:
        try:
            dend_onset = np.where(dend_search_trace < dend_med)[0][-1]
        except:
            dend_onset = 0

    # Get the absolute and relative onsets for each spine now
    spine_onsets = np.zeros(spine_means.shape[1])
    spine_relative_onsets = np.zeros(spine_means.shape[1])
    for i in range(spine_means.shape[1]):
        spine = spine_means[:, i]
        spine_med = np.median(spine)
        spine_std = np.std(spine)
        spine_h = spine_med + spine_std
        spine_peaks, spine_props = sysignal.find_peaks(
            spine, height=spine_h, distance=DISTANCE
        )
        spine_amps = spine_props["peak_heights"]
        # Get the main peak
        if len(spine_peaks) > 1:
            peak_score = []
            for peak, amp in zip(spine_peaks, spine_amps):
                score = (1 / np.absolute(peak - dend_onset)) * amp
                peak_score.append(score)
            peak_idx = np.argmax(peak_score)
            spine_peak = spine_peaks[peak_idx]
            spine_amp = spine_amps[peak_idx]
        else:
            spine_peak = spine_peaks[0]
            spine_amp = spine_amps[0]

        # Get the offset of the rise phase
        spine_peak_trace = spine[:spine_peak]
        spine_offset = np.where(spine_peak_trace < 0.75 * spine_amp)[0][-1]
        # Get the trace velocity
        spine_search_trace = spine[:spine_offset]
        spine_deriv = np.gradient(spine)[:spine_offset]
        # Find where velocity goes to zero or below
        spine_below_zero = spine_deriv <= 0
        if sum(spine_below_zero):
            spine_onset = np.nonzero(spine_below_zero)[0][-1]
        # If derivative doesn't go below zero, find where it goes below median
        else:
            try:
                spine_onset = np.where(spine_search_trace < spine_med)[0][-1]
            except:
                spine_onset = 0
        # Store values
        spine_onsets[i] = spine_onset
        # relative onset is in terms of seconds
        rel_onset = (spine_onset - dend_onset) / sampling_rate
        spine_relative_onsets[i] = rel_onset

    return spine_onsets, spine_relative_onsets, dend_onset

