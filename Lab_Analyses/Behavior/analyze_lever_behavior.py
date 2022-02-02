"""Module to analyze lever press behavior. Gets lever press traces and 
    correlates lever pressing within and across sessions. 
    
    Takes pickle files output from process_lever_behavior.py
    
    CREATOR
        William (Jake) Wright - 2/1/2022
"""

from dataclasses import dataclass

import numpy as np
import scipy.signal as sysignal


def summarize_lever_behavior(file):
    """Function to analyze lever press behavior of a single mouse for a single sessions
    
        INPUT PARAMETERS
            file - An object (Processed_Lever_Data dataclass) containing the processed
                    lever behavior for a single session for a single mouse
    """

    maxtrialnum = 110

    # Smooth lick data if any is present
    if "Lick" in file.xsg_data.channels.keys():
        lick_data_resample = sysignal.resample_poly(
            file.xsg_data.channels["Lick"], up=1, down=10
        )
        butter = sysignal.butter(4, (5 / 500), "low")
        lick_data_smooth = sysignal.filtfilt(
            butter[0],
            butter[1],
            lick_data_resample,
            axis=0,
            padtype="odd",
            padlen=3 * (max(len(butter[1]), len(butter[0])) - 1),
        )
        file.lick_data_smooth = lick_data_smooth
    else:
        file.lick_data_smooth = []

    if not file.behavior_frames.size == 0:
        output = summarize_imaged_lever_behavior(file)


def summarize_imaged_lever_behavior(file):
    """Function to summarize lever press behavior for sessions that were imaged"""
    # Set up new attributes for the file object that will be filled up
    file.successful_movements = []
    file.cue_to_reward = []
    file.post_success_licking = []

    # Initialize some variables
    used_trial = []
    reaction_time = []
    cue_to_reward = []
    trial_length = []
    move_duration_before_cue = []
    movement_matrix = []
    number_of_movements_during_ITI_pre_ignored_trials = []
    fraction_ITI_spent_moving_pre_ignored_trials = []
    number_of_movement_during_ITI_pre_rewarded_trials = []
    fraction_ITI_spent_moving_pre_reward_trials = []

    rewards = 0
    move_at_start_fault = 0
    reward_times = []
    cue_starts = []
    trial_ends = []
    movements = []
    past_threshold_reward_trials = []
    trials = len(file.behavior_frames)
    movements_only = file.lever_force_smooth * file.lever_active
    boundary_frames = np.nonzero(
        np.diff(
            np.insert(
                file.lever_active.astype(float),
                [0, len(file.lever_active)],
                np.Inf,
                axis=0,
            )
            != 0
        )
    )[0]

    for num, trial in enumerate(file.behavior_frames):
        if not trial.states.reward.size == 0:
            rewards = rewards + 1
            # Get time of the reward for this trial
            reward_time = np.round(
                file.frame_times[np.round(trial.states.reward[0]).astype(int)] * 1000
            )
            if reward_time == 0:
                reward_time = 1
            reward_times.append(reward_time)
            # Get time of the start of the cue for this trial
            cue_start = np.round(
                file.frame_times[np.round(trial.states.cue[0]).astype(int)] * 1000
            )
            cue_starts.append(cue_start)
            # Get time of the start of the next cue for this trial
            if num < len(file.behavior_frames) - 1:
                next_cue = np.round(
                    file.frame_times[
                        np.round(file.behavior_frames[num + 1].states.cue[0]).astype(
                            int
                        )
                        * 1000
                    ]
                )
            else:
                next_cue = np.round(file.frame_times[-1]) * 1000
            trial_ends.append(next_cue)

            # Grab movement trace
            movement = file.lever_force_smooth[cue_start - 1 : next_cue]
            movements.append(movement)
            # Binarize movement trace
            past_thresh = (
                file.lever_froce_smooth[cue_start - 1 : next_cue]
                * file.lever_active[cue_start - 1 : next_cue]
            )
            past_threshold_reward_trials.append(past_thresh)

            ## Profile Rewarded Movements
            (
                file,
                used_trial_info,
                fault,
                ignored_trial_info,
            ) = profile_rewarded_movments(
                file,
                boundary_frames,
                num,
                cue_start,
                reward_times,
                trial_ends,
                movement,
                past_thresh,
            )
            if fault == 1:
                move_at_start_fault = move_at_start_fault + 1
                move_duration_before_cue.append(
                    ignored_trial_info.move_duration_before_cue
                )
                number_of_movements_during_ITI_pre_ignored_trials.append(
                    ignored_trial_info.number_of_movements_since_last_trial
                )
                fraction_ITI_spent_moving_pre_ignored_trials.append(
                    ignored_trial_info.fraction_ITI_spent_moving
                )
                number_of_movement_during_ITI_pre_rewarded_trials.append(np.nan)
                fraction_ITI_spent_moving_pre_reward_trials.append(np.nan)
            else:
                move_duration_before_cue = 0
                number_of_movements_during_ITI_pre_ignored_trials.append(np.nan)
                fraction_ITI_spent_moving_pre_ignored_trials.append(np.nan)
                number_of_movement_during_ITI_pre_rewarded_trials.append(
                    used_trial_info.number_of_movements_since_last_trial
                )
                fraction_ITI_spent_moving_pre_reward_trials.append(
                    used_trial_info.fraction_ITI_spent_moving
                )
            if fault != 0:
                continue

            trial_length.append(used_trial_info.trial_length)
            reaction_time.append(used_trial_info.reaction_time)
            cue_to_reward.append(used_trial_info.cs2r)


def profile_rewarded_movments(
    file,
    boundary_frames,
    trial_num,
    cue_start,
    reward_times,
    trial_ends,
    movement,
    past_thresh,
):
    """Function to profile the rewarded movements
    
        INPUT PARAMETERS
            file - object containing the processed lever press behavior data
            
            boundary_frames - np.array containing the boundary frames of when the lever is active
            
            trial_num - int specifying the current trial number
            
            rewards - int indicating how many rewards have been obtained thus far
            
            cue_start - int indicating the time when the cue started on the current trial
            
            reward_times - list containing the reward times of all trials analyzed thus far
            
            trial_ends - list containing the time of the end of all the trials analyzed thus far
            
            movement - np.array of the current trial's movment trace

            past_thresh - np.array of binarized movement of the current trial

        OUTPUT PARMETERS
            file - 
            
            used_trial_info - 
            
            fault - 
            
            ignored_trial_info - 
            
    """

    ## Discard trial if the animal is already moving
    ## Still record details about the nature of the movement
    if any(file.lever_active[cue_start - 101 : cue_start] == 1):
        print(f"Animal was moving at the beginning of trail {trial_num}!")
        trial_length = []
        cs2r = []
        reaction_time = []
        fault = 1  ## Used as boolean operator but also to track error types
        # Get indice of the start of movement before cue
        move_start_before_cue = np.nonzeor(np.diff(file.lever_active[0:cue_start]) > 0)[
            0
        ][-1]
        move_duration_before_cue = len(np.arange(move_start_before_cue, cue_start))
        if trial_num > 0:
            if reward_times[trial_num - 1] == 0:
                reward_times[trial_num - 1] = 1
            fraction_iti_spent_moving = np.sum(
                file.lever_active[reward_times[trial_num - 1] : cue_start]
            ) / len(file.lever_active[reward_times[trial_num - 1] : cue_start])
            if fraction_iti_spent_moving == 1:
                number_of_movements_since_last_trial = 1
            else:
                number_of_movements_since_last_trial = len(
                    np.nonzero(
                        np.diff(
                            file.lever_active[reward_times[trial_num - 1] : cue_start]
                        )
                        > 0
                    )
                )
        else:
            fraction_iti_spent_moving = np.sum(file.lever_active[0:cue_start]) / len(
                file.lever_active[0:cue_start]
            )
            if fraction_iti_spent_moving == 1:
                number_of_movements_since_last_trial = 1
            else:
                number_of_movements_since_last_trial = len(
                    np.nonzero(np.diff(file.lever_active[0:cue_start]) > 0)
                )
        used_trial_info = Used_Trial_Info(
            trial_length=trial_length,
            cs2r=cs2r,
            reaction_time=reaction_time,
            fraction_ITI_spent_moving=np.array([]),
            number_of_movements_since_last_trial=np.array([]),
        )
        ignored_trial_info = Ignored_Trial_Info(
            move_duration_before_cue=move_duration_before_cue,
            fraction_ITI_spent_moving=fraction_iti_spent_moving,
            number_of_movements_since_last_trial=number_of_movements_since_last_trial,
        )
        file.successful_movements.append(np.array([]))
        file.cue_to_reward.append(np.array([]))
        file.post_success_licking.append(np.array([]))
        return file, used_trial_info, fault, ignored_trial_info

    else:
        move_duration_before_cue = 0

    ## Characterize the ITI for successful trials
    number_of_movements_since_last_trial = len(
        np.nonzero(np.diff(file.lever_active[0:cue_start]) > 0)
    )
    if trial_num > 0:
        if reward_times[trial_num - 1] == 0:
            reward_times[trial_num - 1] = 1

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
                )
            )
    else:
        fraction_iti_spent_moving = np.sum(file.lever_active[0:cue_start]) / len(
            file.lever_active[0:cue_start]
        )
        if fraction_iti_spent_moving == 1:
            number_of_movements_since_last_trial = 1
        else:
            number_of_movements_since_last_trial = len(
                np.nonzero(np.diff(file.lever_active[0:cue_start]) > 0)
            )

    # Discard trials with very brief movements or that are at the very end of the session
    if len(movement) < 1000 or cue_start == trial_ends[trial_num]:
        file.successful_movments.append(np.array([]))
        file.cue_to_reward.append(np.array([]))
        file.post_success_licking.append(np.array([]))
        used_trial_info = Used_Trial_Info([], [], [], [], [])
        ignored_trial_info = Ignored_Trial_Info([], [], [])
        fault = 2

        return file, used_trial_info, fault, ignored_trial_info

    cue_to_reward = file.lever_force_smooth[cue_start - 1 : reward_times[trial_num + 1]]
    cs2r = len(cue_to_reward) / 1000

    ## Define the beginning of a successful movement window
    temp = np.nonzero(boundary_frames < reward_times[trial_num])[
        0
    ]  # finds boundary of contiguous movements

    # Discard trials without detected movements
    if temp.size == 0:
        file.successful_movements.append(np.array([]))
        file.post_success_licking.append(np.array([]))
        file.cue_to_rewars.append(cue_to_reward)
        used_trial_info = Used_Trial_Info([], [], [], [], [])
        ignored_trial_info = Ignored_Trial_Info([], [], [])
        fault = 2
        return file, used_trial_info, fault, ignored_trial_info

    if boundary_frames[temp[-1]] < 400:
        baseline_start = len(np.arange(0, boundary_frames[temp[-1]]))
    else:
        baseline_start = 400

    successful_mvmt_start = boundary_frames[temp[-1]] - baseline_start
    if successful_mvmt_start == 0:
        successful_mvmt_start = 1

    rise = np.nonzero(cue_to_reward > np.median(cue_to_reward))[0][0]

    if rise.size == 0:
        rise = 1
    if baseline_start < 400:
        shift = np.absolute(baseline_start - 400)
    else:
        shift = 0

    trial_stop_window = 3000
    if successful_mvmt_start + (trial_stop_window - shift) > len(
        file.lever_force_smooth
    ):
        ending_buffer = np.empty(
            np.absolute(
                len(file.lever_force_smooth)
                - (successful_mvmt_start + (trial_stop_window - shift))
            )
        )
        ending_buffer[:] = np.nan
        file.lever_force_smooth.append(ending_buffer)

    start_buffer = np.empty(shift)
    start_buffer[:] = np.nan
    successful_movement = np.concatenate(
        start_buffer,
        file.lever_force_smooth[
            successful_mvmt_start : successful_mvmt_start + (trial_stop_window - shift)
        ],
    )
    trial_length = len(successful_movement)
    if trial_length == 0:
        raise ValueError("Error with Trial Length on a successful trial")
    reaction_time = np.nonzero(past_thresh)[0][0] / 1000  ## Converted to seconds
    if reaction_time.size == 0:
        reaction_time = 0

    # Repeat for the licking data if available
    trial_stop_window = 5000
    if successful_mvmt_start + (trial_stop_window - shift) > len(file.lick_data_smooth):
        ending_buffer = np.empty(
            np.absolute(
                len(file.lick_data_smooth)
                - (successful_mvmt_start + (trial_stop_window - shift))
            )
        )
        ending_buffer[:] = np.nan
        file.lick_data_smooth.append(ending_buffer)

    if not file.lick_data_smooth.size == 0:
        post_success_licking = np.concatenate(
            start_buffer,
            file.lick_data_smooth[
                successful_mvmt_start : successful_mvmt_start
                + (trial_stop_window - shift)
            ],
        )
    else:
        post_success_licking = np.array([])

    # Setup final outputs
    fault = 0
    file.successful_movements.append(successful_movement)
    file.cue_to_reward.append(cue_to_reward)
    file.post_success_licking.append(post_success_licking)
    used_trial_info = Used_Trial_Info(
        trial_length=trial_length,
        cs2r=cs2r,
        reaction_time=reaction_time,
        fraction_ITI_spent_moving=fraction_iti_spent_moving,
        number_of_movements_since_last_trial=number_of_movements_since_last_trial,
    )
    ignored_trial_info = Ignored_Trial_Info([], [], [])

    return file, used_trial_info, fault, ignored_trial_info


@dataclass
class Session_Lever_Data:
    """Dataclass for storing the analyzed lever press data of a single sesson for 
        a single mouse"""

    trials: int
    rewards: int


@dataclass
class Used_Trial_Info:
    """Dataclass containing used trial info generated in parse_rewarded_movements function"""

    trial_length: int
    cs2r = int
    reaction_time = int
    fraction_ITI_spent_moving: float
    number_of_movements_since_last_trial = int


@dataclass
class Ignored_Trial_Info:
    """Dataclass containing ignored trial info generated in parse_rewarded_movements function"""

    move_duration_before_cue: int
    fraction_ITI_spent_moving: float
    number_of_movements_since_last_trial: int

