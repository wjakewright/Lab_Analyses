"""Module to handle some of the movement analysis for the spine activity data"""

import numpy as np
from Lab_Analyses.Behavior.mouse_lever_behavior import correlate_btw_sessions
from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities import (
    find_activity_onset,
    get_activity_timestamps,
    refine_activity_timestamps,
)
from Lab_Analyses.Utilities import data_utilities as d_utils


def spine_movement_activity(
    data,
    rewarded=False,
    zscore=False,
    volume_norm=False,
    sampling_rate=60,
    activity_window=(-2, 2),
):
    """Function to get spine and dendrite movement-related activity
    
        INPUT PARAMETERS
            data - dataclass of spine data (e.g., Dual_Channel_Spine_Data
            
            rewarded - boolean term of whether to use only rewarded movements
            
            zscore - boolean term of whether to zscore the activity data
            
            volume_norm - boolean term of whether or not to normalize the activity
                            based on the spines volume
            
            sampling_rate - int or float of the imaging sampling rate
            
            activity_window - tuple specifying the period around movement you to 
                                analyze in seconds
                                
        OUTPUT PARAMETERS
            dend_traces - list of 2d np.array of dendrite activity around each event.
                            Centered around movement onset. columns = each event, 
                            rows = time (in frames)
            
            spine_traces - list of 2d np.arrays of spine activity around each event.
                            Centered around movement onset. colums = each event, 
                            rows = time (in frames)
            
            dend_amplitudes - np.array of the peak dendrite amplitude 
            
            dend_std - np.array of the std of dendritic activity
            
            spine_amplitudes - np.array of the peak spine amplitude
            
            spine_std - np.array of the std of the spine activity
            
            dend_onsets - np.array of the dendrite activity onsets relative to movements
            
            spine_onsets - np.array of the spine activity onsets relative to movements
    """

    # Get relevant data
    if rewarded:
        movement_trace = data.rewarded_movement_binary
    else:
        movement_trace = data.lever_active
    spine_groupings = data.spine_grouping
    spine_volumes = np.array(data.corrected_spine_volume)
    spine_dFoF = data.spine_GluSnFr_processed_dFoF
    spine_activity = data.spine_GluSnFr_activity
    dendrite_dFoF = data.dendrite_calcium_processed_dFoF

    if zscore:
        spine_dFoF = d_utils.z_score(spine_dFoF)
        dendrite_dFoF = d_utils.z_score(dendrite_dFoF)

    if volume_norm is not None:
        norm_constants = volume_norm
    else:
        norm_constants = np.array([None for x in range(spine_activity.shape[1])])

    center_point = int(np.absolute(activity_window[0]) * sampling_rate)

    # Set up some outputs
    dend_traces = [None for i in range(spine_dFoF.shape[1])]
    spine_traces = [None for i in range(spine_dFoF.shape[1])]
    dend_amplitudes = np.zeros(spine_dFoF.shape[1])
    dend_std = np.zeros(spine_dFoF.shape[1])
    spine_amplitudes = np.zeros(spine_dFoF.shape[1])
    spine_std = np.zeros(spine_dFoF.shape[1])
    dend_onsets = np.zeros(spine_dFoF.shape[1])
    spine_onsets = np.zeros(spine_dFoF.shape[1])

    timestamps = get_activity_timestamps(movement_trace)
    timestamps = [x[0] for x in timestamps]
    timestamps = refine_activity_timestamps(
        timestamps,
        window=activity_window,
        max_len=spine_dFoF.shape[0],
        sampling_rate=sampling_rate,
    )

    # Process spines on each parent dendrite
    for dendrite in range(dendrite_dFoF.shape[1]):
        # Get spines on this dendrite
        if type(spine_groupings[dendrite]) == list:
            spines = spine_groupings[dendrite]
        else:
            spines = spine_groupings

        # Get relevant data from current spines and dendrite
        s_dFoF = spine_dFoF[:, spines]
        d_dFoF = dendrite_dFoF[:, dendrite]

        # Get movement onset timestamps

        # Get individual traces and mean traces
        s_traces, s_mean_sems = d_utils.get_trace_mean_sem(
            s_dFoF,
            [f"spine {a}" for a in range(s_dFoF.shape[1])],
            timestamps,
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        d_traces, d_mean_sem = d_utils.get_trace_mean_sem(
            d_dFoF.reshape(-1, 1),
            ["Dendrite"],
            timestamps,
            window=activity_window,
            sampling_rate=sampling_rate,
        )
        # Reorganize trace outputs
        s_traces = list(s_traces.values())
        s_means = [x[0] for x in s_mean_sems.values()]
        d_traces = d_traces["Dendrite"]
        d_mean = d_mean_sem["Dendrite"][0]

        if volume_norm is not None:
            s_traces = [s_traces[i] / norm_constants[i] for i in range(s_dFoF.shape[1])]
            s_means = [s_means[i] / norm_constants[i] for i in range(s_dFoF.shape[1])]

        # Get onsets and amplitudes
        s_onsets, s_amps = find_activity_onset(s_means)
        d_onset, dend_amp = find_activity_onset([d_mean])
        d_onset = d_onset[0]
        dend_amp = dend_amp[0]

        # Get activity std and relative onset for each spine
        for spine in range(s_dFoF.shape[1]):
            # Relative onsets
            if not np.isnan(s_amps[spine]):
                s_rel_onset = (s_onsets[spine] - center_point) / sampling_rate
                # Activity std
                s_max = np.nonzero(s_means[spine] == s_amps[spine])[0][0]
                s_stdiv = np.nanstd(s_traces[spine], axis=1)
                s_std = s_stdiv[s_max]
            else:
                s_rel_onset = np.nan
                s_amps[spine] = 0
                s_stdiv = np.nanstd(s_traces[spine], axis=1)
                s_std = s_stdiv[center_point]

            if not np.isnan(dend_amp):
                d_amp = dend_amp
                d_rel_onset = (d_onset - center_point) / sampling_rate
                d_max = np.nonzero(d_mean == d_amp)[0][0]
                d_stdiv = np.nanstd(d_traces, axis=1)
                d_std = d_stdiv[d_max]
            else:
                d_rel_onset = np.nan
                d_amp = 0
                d_stdiv = np.nanstd(d_traces, axis=1)
                d_std = d_stdiv[center_point]

            # Store outputs
            dend_traces[spines[spine]] = d_traces
            spine_traces[spines[spine]] = s_traces[spine]
            dend_amplitudes[spines[spine]] = d_amp
            dend_std[spines[spine]] = d_std
            spine_amplitudes[spines[spine]] = s_amps[spine]
            spine_std[spines[spine]] = s_std
            dend_onsets[spines[spine]] = d_rel_onset
            spine_onsets[spines[spine]] = s_rel_onset

    return (
        dend_traces,
        spine_traces,
        dend_amplitudes,
        dend_std,
        spine_amplitudes,
        spine_std,
        dend_onsets,
        spine_onsets,
    )


def spine_dendrite_movement_similarity(
    spine_movements, dendrite_movements, nearby_spine_idxs
):
    """Function to compare the similarity of movements encoded by spines and their parent
        dendrites and neighboring spines
        
        INPUT PARAMETERS
            spine_movements - list containing all of the movements for each spine
                              that it was active during
            
            dendrite_movements - list containing all of the movements that the parent
                                  dendrite of each spine was active during
                                  
            nearby_spine_idxs - list of the indexs of the nearby spines for each spine
    
        OUTPUT PARAMETERS
            spine_dendrite_corr - np.array of the median correlation between all spine-active
                                  movements and all dendrite-active movements
            
            spine_nearby_corr - np.array of the mean of the median correlations between all
                                spine-active movements and all active movements of each of 
                                its nearby spines
    """

    # Set up outputs
    spine_dendrite_corr = np.zeros(len(spine_movements)) * np.nan
    spine_nearby_corr = np.zeros(len(spine_movements)) * np.nan

    # Analyze each spine seperately
    for i in range(len(spine_movements)):
        # get relevant movements
        s_movements = spine_movements[i]
        d_movements = dendrite_movements[i]
        nearby_idxs = nearby_spine_idxs[i]
        length = s_movements.shape[1]
        # correlate spine with parent dendrite
        s_d_corr = correlate_btw_sessions(s_movements, d_movements, length=length)
        # correlation spine with each of its neighbors
        nearby_corrs = []
        for idx in nearby_idxs:
            nearby_movements = spine_movements[idx]
            s_n_corr = correlate_btw_sessions(
                s_movements, nearby_movements, length=length
            )
            nearby_corrs.append(s_n_corr)
        avg_nearby_corr = np.nanmean(nearby_corrs)
        # save outputs
        spine_dendrite_corr[i] = s_d_corr
        spine_nearby_corr[i] = avg_nearby_corr

    return spine_dendrite_corr, spine_nearby_corr
