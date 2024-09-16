import os
from fractions import Fraction

import numpy as np
import scipy.signal as sysignal
from scipy import stats

from Lab_Analyses.Behavior.mouse_lever_behavior import correlate_btw_sessions
from Lab_Analyses.Utilities.activity_timestamps import (
    get_activity_timestamps,
    refine_activity_timestamps,
)
from Lab_Analyses.Utilities.save_load_pickle import load_pickle


def quantify_movement_quality(
    mouse_id,
    activity_matrix,
    lever_active,
    lever_force,
    threshold=0.5,
    corr_duration=0.5,
    sampling_rate=60,
):
    """Function to assess the quality of movements during specific activity events.
    Compared to the learned movement pattern on the final day of training

    INPUT PARAMETERS
        mouse_id - str specifying the mouse id. Used to pull the relevant learned movement

        activity_matrix - 2d np.array of the binarized activity traces

        lever_active - np.array of the binarized lever activity

        lever_force - np.array of the lever force smoothed

        threshold - float of the correlation threshold for a movement to be considered
                    a learned movement

        corr_duration - float specifying how long (sec) of the movements to correlate

        sampling_rate - int or float specifying the imaging rate

    OUTPUT PARAMETERS
        active_movements - list of 2d arrays containing all the movements and roi is
                            active during (rows for each movement)

        movement_correlation - np.array of the median correlation of all movements with the
                                learned movement pattern for each ROI

        movement_stereotypy - np.array of the medean correlation between all active
                              movements for each roi

        movement_reliability - np.array of the the fraction of movements an roi is
                                active during

        movement_specificity - np.array of the fraction of activity events that occur
                                during movements

        LMP_reliability - np.array of the fraction of LMP movements an roi is active during

        LMP_specificity - np.array of the fraction of activity events that occurr during
                          an LMP movement

        learned_movement_resample - np.array of the LMP resampled to the imaging rate

    """
    # Set up some constants
    EXPANSION = int(0.5 * sampling_rate)  # 0.5 seconds

    # Load the learned movement pattern
    initial_path = r"G:\Analyzed_data\individual"
    behavior_path = os.path.join(initial_path, mouse_id, "behavior")
    day_range = [-1, -2, -3, -4]
    learned_movements = []
    for day in day_range:
        b_day = sorted(x[0] for x in os.walk(behavior_path))[day]
        load_path = os.path.join(behavior_path, b_day)
        fnames = next(os.walk(load_path))[2]
        fname = [x for x in fnames if "summarized_lever_data" in x]
        learned_file = load_pickle(fname, load_path)[0]
        l_movement = np.nanmean(learned_file.corr_matrix, axis=0)
        ## Center start of movement to zero
        l_movement = l_movement - l_movement[0]
        learned_movements.append(l_movement)
    learned_movement = np.nanmean(np.vstack(learned_movements), axis=0)

    # Downsample the learned movement to match imaging rate
    frac = Fraction(sampling_rate / 1000).limit_denominator()
    n = frac.numerator
    d = frac.denominator
    learned_move_resample = sysignal.resample_poly(learned_movement, n, d)
    # Truncate to the specified correlation duration
    corr_interval = int(corr_duration * sampling_rate)
    learned_move_resample = learned_move_resample[:corr_interval]

    # Expand movement intervals
    expansion_const = np.ones(EXPANSION, dtype=int)
    npad = len(expansion_const) - 1
    lever_active_padded = np.pad(
        lever_active, (npad // 2, npad - npad // 2), mode="constant"
    )
    exp_lever_active = (
        np.convolve(lever_active_padded, expansion_const, "valid")
        .astype(bool)
        .astype(int)
    )

    # Get the onsets and offests of the movements and expanded movements
    movement_epochs = get_activity_timestamps(lever_active)
    exp_movement_epochs = get_activity_timestamps(exp_lever_active)
    # Refine the onsets and offsets
    movement_epochs = refine_activity_timestamps(
        movement_epochs, (0, corr_duration), activity_matrix.shape[0], sampling_rate
    )
    exp_movement_epochs = refine_activity_timestamps(
        exp_movement_epochs, (0, corr_duration), activity_matrix.shape[0], sampling_rate
    )

    # Generate a learned movement binary trace
    LMP_num = 0
    LMP_binary = np.zeros(len(lever_active))
    for movement in movement_epochs:
        force = lever_force[movement[0] : movement[0] + corr_interval]
        force = force - force[0]
        r = stats.pearsonr(learned_move_resample, force)[0]
        if r >= threshold:
            LMP_binary[movement[0] : movement[1]] = 1
            LMP_num += 1
    # Expand the LMP trace
    LMP_padded = np.pad(LMP_binary, (npad // 2, npad - npad // 2), mode="constant")
    exp_LMP_binary = (
        np.convolve(LMP_padded, expansion_const, "valid").astype(bool).astype(int)
    )

    # Assess movements for each roi
    ## Outputs
    active_movements = []
    movement_correlation = []
    movement_stereotypy = []
    movement_reliability = []
    movement_specificity = []
    LMP_reliability = []
    LMP_specificity = []
    for i in range(activity_matrix.shape[1]):
        active_trace = activity_matrix[:, i]
        if not np.nansum(active_trace):
            active_movements.append(np.zeros(corr_interval).reshape(1, -1) * np.nan)
            movement_correlation.append(np.nan)
            movement_stereotypy.append(np.nan)
            movement_reliability.append(np.nan)
            LMP_reliability.append(np.nan)
            movement_specificity.append(np.nan)
            LMP_specificity.append(np.nan)
            continue
        # Get all the movements active during
        a_movements = []
        for movement, e_movement in zip(movement_epochs, exp_movement_epochs):
            active_epoch = active_trace[e_movement[0] : e_movement[1]]
            if np.nansum(active_epoch):
                active_move = lever_force[movement[0] : movement[0] + corr_interval]
                active_move = active_move - active_move[0]
                a_movements.append(active_move)
        # calculate correlations and reliability
        try:
            ## Organize all the movements into array
            a_movements = np.stack(a_movements, axis=0)
            active_movements.append(a_movements)
            ## Correlate with the LMP
            corrs = []
            for m in range(a_movements.shape[0]):
                corr = stats.pearsonr(learned_move_resample, a_movements[m, :])[0]
                corrs.append(corr)
            move_corr = np.nanmedian(corrs)
            movement_correlation.append(move_corr)
            ## Correlate all active movements with each other
            move_stereo = correlate_btw_sessions(
                a_movements, a_movements, corr_interval
            )
            movement_stereotypy.append(move_stereo)
            ## Calculate movement reliability
            move_reliability = a_movements.shape[0] / len(movement_epochs)
            movement_reliability.append(move_reliability)
            try:
                LMP_rel = len(np.nonzero(np.array(corrs) >= threshold)[0]) / LMP_num
                LMP_reliability.append(LMP_rel)
            except ZeroDivisionError:
                LMP_reliability.append(0)
        # Handle exceptions when there are no active movements
        except ValueError:
            active_movements.append(np.zeros(corr_interval).reshape(1, -1) * np.nan)
            movement_correlation.append(np.nan)
            movement_stereotypy.append(np.nan)
            movement_reliability.append(np.nan)
            LMP_reliability.append(np.nan)

        # Assess movement specificity
        ## get trace activity onset offset
        activity_epochs = get_activity_timestamps(active_trace)
        active_move_events = 0
        active_LMP_events = 0
        for event in activity_epochs:
            if np.nansum(exp_lever_active[event[0] : event[1]]):
                active_move_events += 1
            if np.nansum(exp_LMP_binary[event[0] : event[1]]):
                active_LMP_events += 1
        try:
            move_specificity = active_move_events / len(activity_epochs)
            LMP_spec = active_LMP_events / len(activity_epochs)
            movement_specificity.append(move_specificity)
            LMP_specificity.append(LMP_spec)
        except ZeroDivisionError:
            movement_specificity.append(0)
            LMP_specificity.append(0)

    # Convert outputs to arrays
    movement_correlation = np.array(movement_correlation)
    movement_stereotypy = np.array(movement_stereotypy)
    movement_reliability = np.array(movement_reliability)
    movement_specificity = np.array(movement_specificity)
    LMP_reliability = np.array(LMP_reliability)
    LMP_specificity = np.array(LMP_specificity)

    return (
        active_movements,
        movement_correlation,
        movement_stereotypy,
        movement_reliability,
        movement_specificity,
        LMP_reliability,
        LMP_specificity,
        learned_move_resample,
    )
