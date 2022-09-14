"""Module for analyzing the co-activity of spines with global activity"""

from itertools import compress

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.movement_responsiveness import \
    movement_responsiveness
from scipy import stats


def global_coactivity_analysis(data, movements=None, sampling_rate=60):
    """Function to analyze spine co-activity with global dendritic activity
    
        INPUT PARAMETERS
            data -  spind_data object. (e.g., Dual_Channel_Spine_Data)

            movements - str specifying if you want on analyze only during movements and of
                        different types of movements. Accepts "all", "rewarded", "unrewarded", 
                        and "nonmovement".
                        Default is None, analyzing the entire imaging period
            
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

    # Get the specified movement data
    if movements == "all":
        movement = data.lever_active
    elif movements == "rewarded":
        movement = data.rewarded_movement_binary
    elif movements == "unrewarded":
        movement = data.lever_active - data.rewarded_movement_binary
    elif movements == "nonmovement":
        movement = np.absolute(data.lever_active - 1)
    else:
        movement = None

    # Set up some of my output variables
    global_correlation = np.zeros(spine_activity.shape[1])
    coactivity_rate = np.zeros(spine_activity.shape[1])
    spine_fraction_coactive = np.zeros(spine_activity.shape[1])
    dend_fraction_coactive = np.zeros(spine_activity.shape[1])
    spine_frequency = np.zeros(spine_activity.shape[1])
    dend_frequency = np.zeros(spine_activity.shape[1])
    coactive_amplitude = np.zeros(spine_activity.shape[1])
    coactive_spines = np.zeros(spine_activity.shape[1])
    coactivity_mean_trace = {}
    coactivity_epoch_trace = {}
    for id in spine_ids:
        coactivity_mean_trace[id] = None
        coactivity_epoch_trace[id] = None
    spine_onsets = np.zeros(spine_activity.shape[1])
    relative_onsets = np.zeros(spine_activity.shape[1])

    dend_mean_sem = []
    dend_onsets = []

    # Now process spines for each parrent dendrite
    for i in range(dendrite_activity.shape[1]):
        # Get the spines along this dendrite
        if type(spine_groupings[i]) == list:
            spines = spine_groupings[i]
        else:
            spines = spine_groupings
        s_ids = np.array(spine_ids)[spines]
        s_dFoF = spine_dFoF[:, spines]
        s_activity = spine_activity[:, spines]
        d_dFoF = dendrite_dFoF[:, i]
        d_activity = dendrite_activity[:, i]

        # Correct the activity matrices for only movement periods if specified
        if movement is not None:
            s_activity = (s_activity.T * movement).T
            d_activity = (d_activity.T * movement).T

        # analyze each spine
        for j in range(s_dFoF.shape[1]):
            # Correlation
            if movement is not None:
                # Correlate only the specified movement periods
                move_idxs = np.where(movement == 1)[0]
                corr, _ = stats.pearsonr(s_dFoF[move_idxs, j], d_dFoF[move_idxs])
            else:
                corr, _ = stats.pearsonr(s_dFoF[:, j], d_dFoF)
            global_correlation[spines[j]] = corr

            # Coactivity rate
            coactivity_freq, spine_frac, dend_frac, dend_freq, spine_freq = get_coactivity_freq(
                s_activity[:, j], d_activity, sampling_rate=sampling_rate
            )
            coactivity_rate[spines[j]] = coactivity_freq
            spine_fraction_coactive[spines[j]] = spine_frac
            dend_fraction_coactive[spines[j]] = dend_frac
            dend_frequency[spines[j]] = dend_freq
            spine_frequency[spines[j]] = spine_freq

        # Get amplitude and co-activity traces
        ## Find dendrite activity preiods
        dend_diff = np.insert(np.diff(d_activity), 0, 0)
        dend_on = np.nonzero(dend_diff == 1)[0]
        dend_off = np.nonzero(dend_diff == -1)[0]
        ### Made sure onsets and offest are the same length
        if len(dend_on) > len(dend_off):
            # Drop last onset if there is no offest
            dend_on = dend_on[:-1]
        elif len(dend_on) < len(dend_off):
            # Drop first offset if there is no onset for it
            dend_off = dend_off[1:]
        timestamps = []
        for onset, offset in zip(dend_on, dend_off):
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

        # Get the onsets
        means = [x[0].reshape(-1, 1) for x in mean_sems.values()]
        means = np.concatenate(means, axis=1)
        d_means = list(d_mean_sems.values())[0][0]
        s_onsets, r_onsets, d_onsets = find_activity_onset(
            means, d_means, sampling_rate
        )
        dend_onsets.append(d_onsets)
        for j in range(len(s_onsets)):
            spine_onsets[spines[j]] = s_onsets[j]
            relative_onsets[spines[j]] = r_onsets[j]

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
        spine_frequency,
        dend_frequency,
        coactive_amplitude,
        coactive_spines,
        coactivity_epoch_trace,
        coactivity_mean_trace,
        dend_mean_sem,
        spine_onsets,
        relative_onsets,
        dend_onsets,
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
            
            dend_freq

            spine_freq
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

    return coactivity_freq, spine_frac_coactive, dend_frac_coactive, dend_event_freq, spine_event_freq


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
    try:
        dend_peak = dend_peaks[np.argmax(dend_amps)]
    # If no dendrite activity, return nan arrays
    except ValueError:
        dend_onset = np.nan
        spine_onsets = np.zeros(spine_means.shape[1])
        spine_relative_onsets = np.zeros(spine_means.shape[1])
        spine_onsets[:] = np.nan
        spine_relative_onsets[:] = np.nan
        return spine_onsets, spine_relative_onsets, dend_onset
    # Get the offset of the rise phase
    dend_amp = np.max(dend_amps)
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
        # smooth spine trace for better onset estimation
        spine = sysignal.savgol_filter(spine, 31, 3)
        spine_med = np.median(spine)
        spine_std = np.std(spine)
        spine_h = spine_med + spine_std
        spine_peaks, spine_props = sysignal.find_peaks(
            spine, height=spine_h, distance=DISTANCE
        )
        spine_amps = spine_props["peak_heights"]
        # Get the main peak
        if len(spine_peaks) < 1:
            spine_onsets[i] = np.nan
            spine_relative_onsets[i] = np.nan
            continue
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
        try:
            spine_offset = np.where(spine_peak_trace < 0.75 * spine_amp)[0][-1]
        except:
            spine_onsets[i] = np.nan
            spine_relative_onsets[i] = np.nan
            continue
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

