"""Module for analyzing the co-activity of spines with global activity"""


import numpy as np
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
    for id in spine_ids:
        coactivity_mean_trace[id] = None

    dend_mean_sem = []

    # Now process spines for each parrent dendrite
    for i in range(dendrite_activity.shape[1]):
        spines = spine_groupings[i]
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
        _, mean_sems = d_utils.get_trace_mean_sem(
            s_dFoF,
            s_ids,
            epoch_timestamps,
            window=(-2, 2),
            sampling_rate=sampling_rate,
        )
        for key, value in mean_sems.items():
            coactivity_mean_trace[key] = value

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
        sig_spines, _, _ = movement_responsiveness(s_dFoF, d_activity,)
        for j, sig in enumerate(sig_spines):
            coactive_spines[spines[j]] = sig

    return (
        global_correlation,
        coactivity_rate,
        spine_fraction_coactive,
        dend_fraction_coactive,
        coactive_amplitude,
        coactive_spines,
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
    spine_frac_coactive = event_freq / spine_event_freq
    dend_frac_coactive = event_freq / dend_event_freq

    return coactivity_freq, spine_frac_coactive, dend_frac_coactive

