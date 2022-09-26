"""Module to handle some of the movement analysis for the spine activity data"""

import os
from fractions import Fraction
from itertools import compress

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    load_spine_datasets,
    spine_volume_norm_constant,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.save_load_pickle import load_pickle
from scipy import stats


def spine_movement_activity(
    data, rewarded=False, zscore=False, volume_norm=False, sampling_rate=60
):
    """Function to get spine and dendrite movement-related activity"""

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
        norm_constants = np.array([None for x in spine_activity.shape[1]])

    # Set up some outputs
    dend_traces = []
    spine_traces = []
    dend_amplitudes = []
    dend_std = []
    spine_amplitudes = []
    spine_std = []
    dend_onsets = []
    spine_onsets = []

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
        curr_norm_constants = norm_constants[:, spines]


def spine_movement_activity1(
    data,
    activity_type="spine_GluSnFr_processed_dFoF",
    exclude="Eliminated",
    sampling_rate=60,
    rewarded=False,
):
    """Function to get the spine activity during movement epochs. 
        Gets the mean before and during movement, as well as the each trace and mean±sem trace
        
        INPUT PARAMETERS
            data - spind_data object. (e.g., Dual_Channel_Spine_Data
            
            activity_type - str specifying what type of activity you wish to use. Must match the 
                            field name of the data object
            
            exclude - string specifying types of spines you wish to exlude from analysis
            
            sampling_rate - float or int specifying the imaging rate the data was collected with

            rewarded - boolean specifying whether to use only rewarded movements or not
            
        OUTPUT PARAMETERS
            all_befores - 
            
            all_durings - 
            
            movement_epochs - 
            
            movement_mean_sems - 
    """

    before_window = int(2 * sampling_rate)

    # Get the activity and behavior out of the object
    activity = getattr(data, activity_type)
    spine_ids = data.spine_ids
    if rewarded:
        movement = data.rewarded_movement_binary
    else:
        movement = data.lever_active

    # Get indexes of spines to analzyed
    if exclude:
        exclude_spines = find_spine_classes(data.spine_flags, exclude)
        exclude_spines = np.array([not x for x in exclude_spines])
        spine_ids = list(compress(spine_ids, exclude_spines))
        activity = activity[:, exclude_spines]

    # Get the movement onset and offset timestamps
    movement_diff = np.insert(np.diff(movement), 0, 0, axis=0)
    movement_onsets = np.nonzero(movement_diff == 1)[0]
    movement_offsets = np.nonzero(movement_diff == -1)[0]

    ## made sure onsets and offsets are the same length
    if len(movement_onsets) > len(movement_offsets):
        # Drop last onset if there is no offset
        movement_onsets = movement_onsets[:-1]
    elif len(movement_onsets) < len(movement_offsets):
        # Drop first offest if there is no onset for it
        movement_offsets = movement_offsets[1:]

    timestamps = []
    for onset, offset in zip(movement_onsets, movement_offsets):
        stamp = (onset, offset)
        timestamps.append(stamp)

    # Refine the timestamps
    refined_idxs = []
    for i, stamp in enumerate(timestamps):
        # remove first movement if to early
        if stamp[0] - before_window < 0:
            refined_idxs.append(False)
            continue
        # remove movements that go beyond activity window at end
        if stamp[0] + before_window >= len(activity[:, 0]):
            refined_idxs.append(False)
            continue
        # remove movements with another movement 1s before
        if i == 0:
            refined_idxs.append(True)
            continue
        if stamp[0] - before_window <= timestamps[i - 1][1]:
            refined_idxs.append(False)
        else:
            refined_idxs.append(True)

    timestamps = list(compress(timestamps, refined_idxs))
    epoch_timestamps = [x[0] for x in timestamps]

    # Get all befores, and durings
    all_befores, all_durings = d_utils.get_before_during_means(
        activity, timestamps, window=1, sampling_rate=sampling_rate
    )
    # Get the traces and mean±sem trace
    movement_epochs, movement_mean_sems = d_utils.get_trace_mean_sem(
        activity,
        spine_ids,
        epoch_timestamps,
        window=(-2, 2),
        sampling_rate=sampling_rate,
    )

    return all_befores, all_durings, movement_epochs, movement_mean_sems


def quantify_movment_quality(
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
    corr_duration = int(1.5 * sampling_rate)  ## 1.5 seconds
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

