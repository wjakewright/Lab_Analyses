"""Module to handle some of the movement analysis for the spine activity data"""

import os
from fractions import Fraction
from itertools import compress

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    load_spine_datasets,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.save_load_pickle import load_pickle
from scipy import stats


def spine_movement_activity(
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


def assess_movement_quality(
    spine_data,
    activity_type="spine_GluSnFr_activity",
    coactivity=False,
    exclude=None,
    sampling_rate=60,
    rewarded=False,
):
    """Function to assess the quality of the movement when spines are active. Compared
        to the learned movement pattern on the final day
        
        INPUT PARAMETERS
            spine_data - spind_data object. (e.g., Dual_Channel_Spine_Data)
            
            activity_type - str specifying what type of activity you wish to use. Must match the 
                            field name of the data object 

            coactivity - boolean specifying if you wish to analyze only coactive periods
                        Default is False
            
            exclude - string specifying types of spines you wish to exlude from analysis
            
            sampling_rate - float or int specifying the imaging rate the data was collected with

            rewarded - boolean specifying whether to use only rewarded movements or not
    """

    # Load the learned movement pattern
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
    behavior_path = os.path.join(initial_path, spine_data.mouse_id, "behavior")
    final_day = sorted([x[0] for x in os.walk(behavior_path)])[-1]
    load_path = os.path.join(behavior_path, final_day)
    fnames = next(os.walk(load_path))[2]
    fname = [x for x in fnames if "summarized_lever_data" in x]
    learned_file = load_pickle(fname, load_path)[0]
    learned_movement = learned_file.movement_avg
    learned_movement = learned_movement - learned_movement[0]
    spine_groupings = spine_data.spine_grouping

    # Remove the baseline period
    corr_len = learned_file.corr_matrix.shape[1]
    baseline_len = len(learned_movement) - corr_len
    learned_movement = learned_movement[baseline_len:]

    # Need to down sample the learrned movement now to match the imaging rate
    frac = Fraction(sampling_rate / 1000).limit_denominator()
    n = frac.numerator
    d = frac.denominator
    learned_move_resample = sysignal.resample_poly(learned_movement, n, d)
    # move_duration = len(learned_move_resample)
    corr_duration = 60
    learned_move_resample = learned_move_resample[:corr_duration]

    # Get relevant data from spine data
    if rewarded:
        lever_active = spine_data.rewarded_movement_binary
        lever_trace = spine_data.rewarded_movement_force
    else:
        lever_active = spine_data.lever_active
        lever_trace = spine_data.lever_force_smooth

    if coactivity is False:
        activity = getattr(spine_data, activity_type)
    else:
        s_activity = getattr(spine_data, activity_type)
        d_activity = spine_data.dendrite_calcium_activity
        activity = np.zeros(s_activity.shape)
        for d in range(d_activity.shape[1]):
            if type(spine_groupings[d]) == list:
                spines = spine_groupings[d]
            else:
                spines = spine_groupings
            for s in spines:
                activity[:, s] = s_activity[:, s] * d_activity[:, d]

    if exclude:
        exclude_spines = find_spine_classes(spine_data.spine_flags, exclude)
        exclude_spines = np.array([not x for x in exclude_spines])
        activity = activity[:, exclude_spines]

    # Get onsets and offsets of the movements
    movement_diff = np.insert(np.diff(lever_active), 0, 0, axis=0)
    movement_onsets = np.nonzero(movement_diff == 1)[0]
    movement_offsets = np.nonzero(movement_diff == -1)[0]
    ## make sure onsets and offsets are the same length
    if len(movement_onsets) > len(movement_offsets):
        # Drop last onset if there is no offset
        movement_onsets = movement_onsets[:-1]
    elif len(movement_onsets) < len(movement_offsets):
        # Drop first offest if there is no onset for it
        movement_offsets = movement_offsets[1:]

    move_idxs = []
    for onset, offset in zip(movement_onsets, movement_offsets):
        move_idxs.append((onset, offset))

    movement_correlations = []
    spine_movements = []
    mean_spine_movements = []
    # Assess the movements for each spine
    for i in range(activity.shape[1]):
        spine_trace = activity[:, i]
        all_spine_movements = []
        for movement in move_idxs:
            spine_epoch = spine_trace[movement[0] : movement[1]]
            if sum(spine_epoch):
                spine_move = lever_trace[movement[0] : movement[0] + corr_duration]
                spine_movements.append(spine_move)

            else:
                continue

        # Average the movements
        try:
            s_movements = np.stack(all_spine_movements, axis=0)
            spine_movements.append(s_movements)
            mean_movement = np.nanmean(s_movements, axis=0)
            mean_spine_movements.append(mean_movement)
            # Correlate with the learned movement
            corrs = []
            for m in range(s_movements.shape[0]):
                corr = stats.pearsonr(learned_move_resample, s_movements[m, :])[0]
                corrs.append(corr)
            move_corr = np.nanmedian(corrs)
            movement_correlations.append(move_corr)
        except ValueError:
            spine_movements.append(np.zeros(corr_duration))
            mean_spine_movements.append(np.zeros(corr_duration))
            movement_correlations.append(np.nan)

    # convert outputs to arrays
    movement_correlations = np.array(movement_correlations)
    mean_spine_movements = np.array(mean_spine_movements).T

    return (
        spine_movements,
        mean_spine_movements,
        movement_correlations,
        learned_move_resample,
    )

