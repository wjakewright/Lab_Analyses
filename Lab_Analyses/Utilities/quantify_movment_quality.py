import os
from audioop import tostereo
from fractions import Fraction

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Utilities.save_load_pickle import load_pickle
from scipy import stats


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
        if onset + corr_duration > len(lever_force):
            continue
        move_idxs.append((onset, offset))

    # Generate a learned movement binary trace
    learned_move_num = 0
    lever_learned_binary = np.zeros(len(lever_active))
    for movement in move_idxs:
        force = lever_force[movement[0] : movement[0] + corr_duration]
        r = stats.pearsonr(learned_move_resample, force)[0]

        if r >= threshold:
            lever_learned_binary[movement[0] : movement[1]] = 1
            learned_move_num = learned_move_num + 1
        else:
            continue

    # Assess the movements for each roi
    median_movement_correlations = []
    move_frac_active = []
    learned_move_frac_active = []
    learned_move_frac_active = []
    active_move_frac_learned = []
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
            # Get some fractions of movements
            move_frac_active.append(len(active_movements) / len(move_idxs))
            learned_move_frac_active.append(
                len(np.nonzero(np.array(corrs) >= 0.5)[0]) / learned_move_num
            )
            active_move_frac_learned.append(
                len(np.nonzero(np.array(corrs) >= 0.5)[0]) / len(corrs)
            )

        except ValueError:
            all_active_movements.append(np.zeros(corr_duration))
            avg_active_movements.append(np.zeros(corr_duration))
            median_movement_correlations.append(np.nan)
            move_frac_active.append(0)
            learned_move_frac_active.append(0)
            active_move_frac_learned.append(0)

    # convert outputs to arrays
    median_movement_correlations = np.array(median_movement_correlations)
    learned_move_num = np.array(learned_move_num)

    return (
        lever_learned_binary,
        all_active_movements,
        avg_active_movements,
        median_movement_correlations,
        move_frac_active,
        learned_move_frac_active,
        active_move_frac_learned,
        learned_move_resample,
    )
