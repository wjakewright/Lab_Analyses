"""Module to handle some of the movement analysis for the spine activity data"""

import os
from fractions import Fraction
from itertools import compress

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities import (
    find_activity_onset,
    get_activity_timestamps,
)
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    load_spine_datasets,
    spine_volume_norm_constant,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.save_load_pickle import load_pickle
from scipy import stats


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

    if volume_norm:
        norm_constants = spine_volume_norm_constant(
            spine_activity,
            spine_dFoF,
            spine_volumes,
            data.imaging_parameters["Zoom"],
            sampling_rate=sampling_rate,
            iterations=1000,
        )
    else:
        norm_constants = np.array([None for x in range(spine_activity.shape[1])])

    center_point = int(activity_window[0] * sampling_rate)

    # Set up some outputs
    dend_traces = [None for i in range(spine_dFoF.shape[1])]
    spine_traces = [None for i in range(spine_dFoF.shape[1])]
    dend_amplitudes = np.zeros(spine_dFoF.shape[1])
    dend_std = np.zeros(spine_dFoF.shape[1])
    spine_amplitudes = np.zeros(spine_dFoF.shape[1])
    spine_std = np.zeros(spine_dFoF.shape[1])
    dend_onsets = np.zeros(spine_dFoF.shape[1])
    spine_onsets = np.zeros(spine_dFoF.shape[1])

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
        timestamps = get_activity_timestamps(movement_trace)
        timestamps = [x[0] for x in timestamps]

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

        if volume_norm:
            s_traces = [s_traces[i] / norm_constants[i] for i in range(s_dFoF.shape[1])]
            s_means = [s_means[i] / norm_constants[i] for i in range(s_dFoF.shape[1])]

        # Get onsets and amplitudes
        s_onsets, s_amps = find_activity_onset(s_means)
        d_onset, d_amp = find_activity_onset([d_mean])
        d_onset = d_onset[0]
        d_amp = d_amp[0]

        # Get activity std and relative onset for each spine
        for spine in range(s_dFoF.shape[1]):
            # Relative onsets
            if not np.isnan(s_amps[spine]):
                s_rel_onset = (s_onsets[spine] - center_point) / sampling_rate
                # Activity std
                s_max = np.nonzero(s_means[spine] == s_amps[spine])[0]
                s_std = np.nanstd(s_traces[spine], axis=1)
                s_std = s_std[s_max]
            else:
                s_rel_onset = np.nan
                s_amps[spine] = 0
                s_std = s_std[center_point]

            if not np.isnan(d_amp):
                d_rel_onset = (d_onset - center_point) / sampling_rate
                d_max = np.nonzero(d_mean == d_amp)[0]
                d_std = np.nanstd(d_traces, axis=1)
                d_std = d_std[d_max]
            else:
                d_rel_onset = np.nan
                d_amp = 0
                d_std = d_std[center_point]

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


def quantify_movement_quality(
    mouse_id,
    activity_matrix,
    lever_active,
    lever_force,
    threshold=0.5,
    sampling_rate=60,
):
    """Function to assess the quality of movments during specific activity events.
        Compared to the learned movement pattern on the final day
        
        INPUT PARAMETERS
            mouse_id - str specifying the mouse id. Used to pull relevant learned movement
            
            activity_matrix - 2d np.array of the binaried activity traces. colums = different rois
            
            lever_active - np.array of the binarized lever activity
            
            lever_force - np.array of the lever force smooth
            
            threshold - float of the correlation threshold for a movement to be considered
                        a learned movement
                        
            sampling_rate - int or float of the imaging sampling rate

        OUTPUT PARAMETERS
            lever_learned_binary - np.array binarized to when learned movements occur

            all_active_movements - list of 2d np.arrays of all the movements an roi is active
                                    during (rows = movements, columns = time)

            avg_active_movements - list of np.arrays of the average movement an roi is active
                                    during

            median_movement_correlations - np.array of the median correlation of movements an 
                                            roi is active during with the learned movement pattern
            
            learned_move_resample - np.array of the learned movement pattern resampled to a frames
            
    """
    CORR_INT = 1.5

    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
    behavior_path = os.path.join(initial_path, mouse_id, "behavior")
    final_day = sorted([x[0] for x in os.walk(behavior_path)])[-1]
    load_path = os.path.join(behavior_path, final_day)
    fnames = next(os.walk(load_path))[2]
    fname = [x for x in fnames if "summarized_lever_data" in x]
    learned_file = load_pickle(fname, load_path)[0]
    learned_movement = learned_file.movement_avg
    learned_movement = learned_movement - learned_movement[0]

    # Remove the baseline period
    corr_len = learned_file.corr_matrix.shape[1]
    baseline_len = len(learned_movement) - corr_len
    learned_movement = learned_movement[baseline_len:]

    # Need to downsample the learned movement now to match the imaging rate
    frac = Fraction(sampling_rate / 1000).limit_denominator()
    n = frac.numerator
    d = frac.denominator
    learned_move_resample = sysignal.resample_poly(learned_movement, n, d)
    corr_duration = int(CORR_INT * sampling_rate)  ## 1.5 seconds
    learned_move_resample = learned_move_resample[:corr_duration]

    # Get onsets and offsets of the movements
    movement_diff = np.insert(np.diff(lever_active), 0, 0, axis=0)
    movement_onsets = np.nonzero(movement_diff == 1)[0]
    movement_offsets = np.nonzero(movement_diff == -1)[0]
    ## Make sure the onsets and offsets are the same length
    if len(movement_onsets) > len(movement_offsets):
        # Drop last onset if there is no corresponding offset
        movement_onsets = movement_onsets[:-1]
    elif len(movement_onsets) < len(movement_offsets):
        # Drop the first offset if there is no onset for it
        movement_offsets = movement_offsets[1:]

    move_idxs = []
    for onset, offset in zip(movement_onsets, movement_offsets):
        move_idxs.append((onset, offset))

    # Generate a learned movement binary trace
    lever_learned_binary = np.zeros(len(lever_active))
    for movement in move_idxs:
        force = lever_force[movement[0] : movement[0] + corr_duration]
        r = stats.pearsonr(learned_move_resample, force)[0]
        if r >= threshold:
            lever_learned_binary[movement[0] : movement[1]] = 1
        else:
            continue

    # Assess the movements for each roi
    median_movement_correlations = []
    all_active_movements = []
    avg_active_movements = []
    for i in range(activity_matrix.shape[1]):
        active_trace = activity_matrix[:, i]
        active_movements = []
        for movement in move_idxs:
            active_epoch = active_trace[movement[0] : movement[1]]
            if sum(active_epoch):
                active_move = lever_force[movement[0] : movement[0] + corr_duration]
                active_movements.append(active_move)
            else:
                pass
        try:
            a_movements = np.stack(active_movements, axis=0)
            all_active_movements.append(a_movements)
            avg_move = np.nanmean(a_movements, axis=0)
            avg_active_movements.append(avg_move)
            # correlation with the learned movement
            corrs = []
            for m in range(a_movements.shape[0]):
                corr = stats.pearsonr(learned_move_resample, a_movements[m, :])[0]
                corrs.append(corr)
            move_corr = np.nanmedian(corrs)
            median_movement_correlations.append(move_corr)
        except ValueError:
            all_active_movements.append(np.zeros(corr_duration))
            avg_active_movements.append(np.zeros(corr_duration))
            median_movement_correlations.append(np.nan)

    # convert outputs to arrays
    median_movement_correlations = np.array(median_movement_correlations)

    return (
        lever_learned_binary,
        all_active_movements,
        avg_active_movements,
        median_movement_correlations,
        learned_move_resample,
    )

