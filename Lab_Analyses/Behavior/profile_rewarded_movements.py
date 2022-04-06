"""Module to profile the rewarded movement"""

from dataclasses import dataclass

import numpy as np


def profile_rewarded_movements(
    file,
    boundary_frames,
    trial_num,
    cue_start,
    reward_times,
    trial_ends,
    movement,
    past_thresh,
):
    """Function to profile the rewarded movements. Identifies trials to use for analyses
        and which ones to ignore. Characterizes movements during the ITI and gets the 
        successful movement trace that triggered reward delivery
        
        INPUT PARAMETERS
            file - object containing the processed lever press data
            
            boundary_frames - np.array containing the boundary frames of when the lever is active
            
            trial_num - int specifying the current trial index
            
            cue_start - int indicating the time when the cue started on the current trial
            
            reward_times - list containing the reward times of all trials analyzed thus far
            
            trial_ends - list containing the time of the end of all trials analyzed thus far
            
            past_thresh - np.array of binarized movement of the current trial
            
        OUTPUT PARAMETERS
            trial_info - dataclass containing relevant trial information
            
    """
    ############################### DISCARD BAD TRIALS ############################################

    TRIAL_STOP_WINDOW = 3000
    LICK_STOP_WINDOW = 5000

    # Discard trial if the animal is moving at the start of trial
    ### Still record details about the nature of the movements
    if any(file.lever_active[cue_start - 100 : cue_start] == 1):
        trial_info = profile_movement_before_cue(
            file, trial_num, cue_start, reward_times
        )

        return trial_info

    else:
        move_duration_before_cue = 0

    # Discard trials with very brief movements or at end of session
    if len(movement) < 1000 or cue_start == trial_ends[trial_num]:
        fault = 2
        trial_info = Trial_Info(
            trial_used=False,
            trial_length=np.nan,
            cs2r=np.nan,
            reaction_time=np.nan,
            move_duration_before_cue=np.nan,
            fraction_ITI_spent_moving=np.nan,
            number_of_mvmts_since_last_trial=np.nan,
            successful_movements=np.nan,
            cue_to_reward=np.nan,
            post_success_licking=np.nan,
            fault=fault,
        )
        return trial_info

    # Find boundaries of contiguous movements
    temp = np.nonzero(boundary_frames < reward_times[trial_num])[0]

    # Discard trials without detected movements
    if temp.size == 0:
        fault = 3
        trial_info = Trial_Info(
            trial_used=False,
            trial_length=np.nan,
            cs2r=np.nan,
            reaction_time=np.nan,
            move_duration_before_cue=np.nan,
            fraction_ITI_spent_moving=np.nan,
            number_of_mvmts_since_last_trial=np.nan,
            successful_movements=np.nan,
            cue_to_reward=np.nan,
            post_success_licking=np.nan,
            fault=fault,
        )
        return trial_info

    ############################ PROFILE GOOD TRIALS ###################################

    # Characterize the ITI of successful trials
    number_of_movements_since_last_trial = len(
        np.nonzero(np.diff(file.lever_active[0:cue_start]) > 0)[0]
    )

    if trial_num > 0:
        fraction_iti_spent_moving = np.sum(
            file.lever_active[reward_times[trial_num - 1] : cue_start]
        ) / len(file.lever_active[reward_times[trial_num - 1] : cue_start])

        if fraction_iti_spent_moving == 1:
            number_of_movements_since_last_trial = 1
        else:
            number_of_movements_since_last_trial = len(
                np.nonzero(
                    np.diff(file.lever_active[reward_times[trial_num - 1] : cue_start])
                    > 0
                )[0]
            )
    else:
        fraction_iti_spent_moving = np.sum(file.lever_active[0:cue_start]) / len(
            file.lever_active[0:cue_start]
        )

        if fraction_iti_spent_moving == 1:
            number_of_movements_since_last_trial = 1
        else:
            number_of_movements_since_last_trial = len(
                np.nonzero(np.diff(file.lever_active[0:cue_start]) > 0)[0]
            )

    # Get force from cue start to reward delivery
    cue_to_reward = file.lever_force_smooth[cue_start : reward_times[trial_num] + 1]
    cs2r = len(cue_to_reward) / 1000

    # Define the beginning of a successful movement window
    if boundary_frames[temp[-1]] < 400:
        baseline_start = len(np.arange(0, boundary_frames[temp[-1]]))
    else:
        baseline_start = 400

    successful_mvmt_start = boundary_frames[temp[-1]] - baseline_start

    if successful_mvmt_start == 0:
        successful_mvmt_start = 1

    if baseline_start < 400:
        shift = np.absolute(baseline_start - 400)
    else:
        shift = 0
    shift = int(shift)

    # Add buffer for movements that extend beyond the end of the session
    if successful_mvmt_start + (TRIAL_STOP_WINDOW - shift) > len(
        file.lever_force_smooth
    ):
        ending_buffer = np.empty(
            np.absolute(
                len(file.lever_force_smooth)
                - (successful_mvmt_start + (TRIAL_STOP_WINDOW - shift))
            )
        )
        ending_buffer[:] = np.nan
        file.lever_force_smooth.append(ending_buffer)

    # Add buffer to the start to account for shift
    start_buffer = np.empty(shift)
    start_buffer[:] = np.nan

    successful_movement = np.concatenate(
        (
            start_buffer,
            file.lever_force_smooth[
                int(successful_mvmt_start)
                - 1 : int(successful_mvmt_start)
                + (TRIAL_STOP_WINDOW - shift)
            ],
        )
    )

    trial_length = len(successful_movement)
    if trial_length == 0:
        raise Exception("Error with Trial Length on Successful trial!!!")

    if np.sum(past_thresh) == 0:
        reaction_time = 0
    else:
        reaction_time = np.nonzero(past_thresh)[0][0] / 1000  # converted to seconds

    # Repeat the above steps for licking data if available
    if successful_mvmt_start + (LICK_STOP_WINDOW - shift) > len(file.lick_data_smooth):
        ending_buffer = np.empty(
            np.absolute(
                len(file.lick_data_smooth)
                - successful_mvmt_start
                + (LICK_STOP_WINDOW - shift)
            )
        )
        file.lick_data_smooth.append(ending_buffer)

    # Get post reward licks if available
    if not file.lick_data_smooth.size == 0:
        post_success_licking = np.concatenate(
            (
                start_buffer,
                file.lick_data_smooth[
                    successful_mvmt_start
                    - 1 : successful_mvmt_start
                    + (LICK_STOP_WINDOW - shift)
                ],
            )
        )
    else:
        post_success_licking = np.nan

    # Setup final output
    fault = 0
    trial_info = Trial_Info(
        trial_used=True,
        trial_length=trial_length,
        cs2r=cs2r,
        reaction_time=reaction_time,
        move_duration_before_cue=move_duration_before_cue,
        fraction_ITI_spent_moving=fraction_iti_spent_moving,
        number_of_mvmts_since_last_trial=number_of_movements_since_last_trial,
        successful_movements=successful_movement,
        cue_to_reward=cue_to_reward,
        post_success_licking=post_success_licking,
        fault=fault,
    )

    return trial_info


def profile_movement_before_cue(file, trial_num, cue_start, reward_times):
    """Helper function to profile trials ignored due to movement before trial"""
    trial_length = np.nan
    cs2r = np.nan
    reaction_time = np.nan
    fault = 1  # Used as conditional to track different types of ignored trials

    # Get indicie of the start of the movement before cue
    move_start_before_cue = np.nonzero(np.diff(file.lever_active[0:cue_start]) > 0)[0][
        -1
    ]
    move_duration_before_cue = len(np.arange(move_start_before_cue, cue_start))

    if trial_num > 0:
        fraction_iti_spent_moving = np.sum(
            file.lever_active[reward_times[trial_num - 1] : cue_start]
        ) / len(file.lever_active[reward_times[trial_num - 1] : cue_start])

        if fraction_iti_spent_moving == 1:
            number_of_movements_since_last_trial = 1
        else:
            number_of_movements_since_last_trial = len(
                np.nonzero(
                    np.diff(file.lever_active[reward_times[trial_num - 1] : cue_start])
                    > 0
                )[0]
            )

    else:
        fraction_iti_spent_moving = np.sum(file.lever_active[0:cue_start]) / len(
            file.lever_active[0:cue_start]
        )

        if fraction_iti_spent_moving == 1:
            number_of_movements_since_last_trial = 1
        else:
            number_of_movements_since_last_trial = len(
                np.nonzero(np.diff(file.lever_active[0:cue_start]) > 0)[0]
            )

    trial_info = Trial_Info(
        trial_used=False,
        trial_length=trial_length,
        cs2r=cs2r,
        reaction_time=reaction_time,
        move_duration_before_cue=move_duration_before_cue,
        fraction_ITI_spent_moving=fraction_iti_spent_moving,
        number_of_mvmts_since_last_trial=number_of_movements_since_last_trial,
        successful_movements=np.nan,
        cue_to_reward=np.nan,
        post_success_licking=np.nan,
        fault=fault,
    )

    return trial_info


# --------------------------------------------------------------------------------------------------
# ------------------------------------DATACLASSES USED----------------------------------------------
# --------------------------------------------------------------------------------------------------


@dataclass
class Trial_Info:
    """Dataclass for storing information about individual trials"""

    trial_used: bool
    trial_length: int
    cs2r: int
    reaction_time: int
    move_duration_before_cue: int
    fraction_ITI_spent_moving: float
    number_of_mvmts_since_last_trial: int
    successful_movements: np.ndarray
    cue_to_reward: np.ndarray
    post_success_licking: np.ndarray
    fault: int
