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

            rel_dend_coactive_amplitude - np.array of the peak mean responses of dendritic coactivity
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
    relative_spine_onsets = np.zeros(spine_activity.shape[1])
    dend_triggered_spine_traces = [None for i in global_correlation]
    dend_triggered_dend_traces = [None for i in global_correlation]
    coactive_spine_traces = [None for i in global_correlation]
    coactive_dend_traces = [None for i in global_correlation]

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

        # Analyze each spine
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
    dendrite, spine_matrix, coactivity, activity_window=(-2, 2)
):
    """Helper function to help getting the activity traces of dendrites and spines
        for all dendritic or coactive events. Also gets the relative onset of spine
        activity
        
        INPUT PARAMETERS
            dendrite - np.array of dendrite binary activity trace
            
            spine_matrix = 2d np.array of spine binary activity traces (columns = spines)

            coactivity - boolean specifying whether to perform for coactivty events (True) or
                        all dendritic events (False)
            
            activity_window - tuple specifying the window around which you want the activity
                                from. E.g. (-2,2) for 2 sec before and after

        OUTPUT PARAMETERS
            spine_traces - 2d np.array of spine activity around each
                            event. Centered around dendrite onset. 
                            columns = each event, rows = time (in frames)
            
            dend_traces - 2d np.array of dendrite activity around each
                            event. Centered around dendrite onset. 
                            columns = each event, rows = time (in frames)
            
            relative_onsets - np.array of spine onsets relative to dendrite
    """
    # Get the initial timestamps


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
