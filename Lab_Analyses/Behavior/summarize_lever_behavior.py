"""Module to summarize lever press behavior. Gets various behavior parameters
    and profiles the rewarded lever presses"""

import os
from dataclasses import dataclass

import numpy as np
import scipy.signal as sysignal

from Lab_Analyses.Behavior.profile_rewarded_movements import profile_rewarded_movements
from Lab_Analyses.Behavior.read_bit_code import read_bit_code
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


def summarize_lever_behavior(file, save=False, save_suffix=None):
    """Parent function to summarize lever press behavior from a single mouse for 
        a single session. Calls different functions depending if the mouse was imaged
        or not.
        
        INPUT PARAMETERS
            file - Object (processed_lever_data dataclass) containing the processed
                    lever behavior for a single session for a single mouse
            
            save - boolean specifying whether to save the output or not
                    Default is False
            
            save_suffix - string to be appended at the end of the file name.
                          used to indicated any additional information about the session.
                          Default is set to None
        
        OUTPUT PARAMETERS
            summarized_data - Session_Summary_Lever_Data dataclass with fields:

                            mouse_id - str with the id of the mouse

                            date - str with the date the data was collected
                            
                            used_trial - boolean array of which trials were used and ignored
                            
                            movement_matrix - np.array of all rewarded movements, each 
                                              row representing a single movement trace
                            
                            movement_avg - np. array of the averaged rewarded movement traces
                            
                            rewards - int specifying how many rewards were recieved
                            
                            move_at_start_faults - int specifying how many trials were ignored
                                                    due to mouse moving at start of the trial
                            
                            avg_reaction_time - float of the average reaction time for mouse to move
                            
                            avg_cue_to_reward - float of the average time from cue onset till
                                                mouse recieved the reward
                                                
                            trials - in specifying the number of trials within the session
                            
                            move_duration_before_cue - list of floats specifying how much time mouse
                                                        spent moving before cue for each trial
                            
                            number_movements_during_ITI - list of the num of movements mouse made
                                                          during the ITI for each trial
                                                          
                            fraction_ITI_spent_moving - list of the fraction of time mouse spent
                                                        moving during the ITI for each trial
        
    """
    MIN_MOVE_NUM = 0
    MIN_T = 3001
    if file is None:
        return None

    # Smooth lick data if any is present
    if "Lick" in file.xsg_data.channels.keys():
        file.lick_data_smooth = smooth_lick_data(licks=file.xsg_data.channels["Lick"])
    else:
        file.lick_data_smooth = np.array([])

    # Summarize data collected while imaging
    if not file.behavior_frames.size == 0:
        summarized_data = summarize_imaged_lever_behavior(file, MIN_MOVE_NUM, MIN_T)

    # Summarize data collected while not imaging
    else:
        summarized_data = summarize_nonimaged_lever_behavior(file, MIN_MOVE_NUM, MIN_T)

    # Save section
    if save is True:
        mouse_id = file.mouse_id
        sess_name = file.sess_name
        # Set the path
        initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
        save_path = os.path.join(initial_path, mouse_id, "behavior", sess_name)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        # Make file name
        if save_suffix is not None:
            save_name = f"{mouse_id}_{sess_name}_summarized_lever_data_{save_suffix}"
        else:
            save_name = f"{mouse_id}_{sess_name}_summarized_lever_data"
        # Save the data as a pickle file
        save_pickle(save_name, summarized_data, save_path)

    return summarized_data


def summarize_imaged_lever_behavior(file, MIN_MOVE_NUM, MIN_T):
    """Function to summarize lever data when imaged"""

    # Set up new attributes and variables
    ## Unrewarded and ignored trials will have nan values
    successful_movements = []
    movement_baselines = []
    cue_to_reward = []
    post_success_licking = []
    faults = []
    num_trials = len(file.behavior_frames)
    used_trial = []
    reaction_time = []
    cs2r = []
    trial_length = []
    move_duration_before_cue = []
    movement_matrix = []
    corr_matrix = []
    number_of_movements_during_ITI = []
    fraction_ITI_spent_moving = []

    # Setup temporary variables
    rewards = 0
    move_at_start_fault = 0
    reward_times = []
    trial_ends = []
    movements = []
    past_threshold_reward_trials = []

    # Boundary frames of when movements occur
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

    # Profile each trial
    for num, trial in enumerate(file.behavior_frames):
        # Profile only the rewarded trials
        if not trial.states.reward.size == 0:
            # Get time of the reward for current trial
            rewards = rewards + 1
            reward_time = np.round(
                file.frame_times[np.round(trial.states.reward[0]).astype(int)] * 1000
            )
            if reward_time == 0:
                reward_time = 1
            reward_time = int(reward_time)
            reward_times.append(reward_time)

            # Get the time of the start of the cue
            cue_start = int(
                np.round(
                    file.frame_times[np.round(trial.states.cue[0]).astype(int)] * 1000
                )
            )
            # Get time of next cue, indicating end of current trial
            if num < len(file.behavior_frames) - 1:
                next_cue = np.round(
                    file.frame_times[
                        np.round(file.behavior_frames[num + 1].states.cue[0]).astype(
                            int
                        )
                    ]
                    * 1000
                )
            else:
                next_cue = np.round(file.frame_times[-1]) * 1000
            next_cue = int(next_cue)
            trial_ends.append(next_cue)

            # Get movement trace
            movement = file.lever_force_smooth[cue_start - 1 : next_cue]
            movements.append(movement)

            # Get movement traces above threshold
            past_thresh = (
                file.lever_force_smooth[cue_start - 1 : next_cue]
                * file.lever_active[cue_start - 1 : next_cue]
            )
            past_threshold_reward_trials.append(past_thresh)
            ## Profile rewarded movements
            trial_info = profile_rewarded_movements(
                file,
                boundary_frames,
                num,
                cue_start,
                reward_times,
                trial_ends,
                movement,
                past_thresh,
            )
            if trial_info.fault == 1:
                move_at_start_fault = move_at_start_fault + 1
            used_trial.append(trial_info.trial_used)
            trial_length.append(trial_info.trial_length)
            cs2r.append(trial_info.cs2r)
            reaction_time.append(trial_info.reaction_time)
            number_of_movements_during_ITI.append(
                trial_info.number_of_mvmts_since_last_trial
            )
            move_duration_before_cue.append(trial_info.move_duration_before_cue)
            fraction_ITI_spent_moving.append(trial_info.fraction_ITI_spent_moving)
            successful_movements.append(trial_info.successful_movements)
            movement_baselines.append(trial_info.succ_move_baseline)
            cue_to_reward.append(trial_info.cue_to_reward)
            post_success_licking.append(trial_info.post_success_licking)
            faults.append(trial_info.fault)

        else:
            trial_ends.append(np.nan)
            reward_times.append(0)

    # Get average reaction time and cue to reward
    avg_reaction_time = np.nanmean(reaction_time)
    avg_cue_to_reward_time = np.nanmean(cs2r)
    # Remove zeros from trial length and set min trial length
    trial_length[trial_length == 0] = np.nan
    if not len(trial_length) == 0:
        min_t = np.nanmin(trial_length)
    else:
        min_t = MIN_T
    min_t = int(min_t)
    move_duration_before_cue = (
        np.array(move_duration_before_cue)[
            np.invert(np.isnan(np.asarray(move_duration_before_cue)))
        ]
        / 1000
    )

    # Generate movement matrix
    num_tracked_movements = 0
    for rewarded_trial in range(rewards):
        try:
            if isinstance(successful_movements[rewarded_trial], np.ndarray):
                # if not np.isnan(successful_movements[rewarded_trial]):
                move_array = np.array(successful_movements[rewarded_trial][: min_t + 1])
                move_array[move_array == 0] = np.nan
                movement_matrix.append(move_array)
                corr_array = move_array[movement_baselines[rewarded_trial] :]
                corr_matrix.append(corr_array)
            else:
                move_array = np.empty(min_t)
                move_array[:] = np.nan
                movement_matrix.append(move_array)
                corr_array = move_array[movement_baselines[rewarded_trial] :]
                corr_matrix.append(corr_array)
        except Exception as error:
            print(f"Movement was not tracked for trial {rewarded_trial}")
            print(error)
            move_array = np.empty(min_t)
            move_array[:] = np.nan
            movement_matrix.append(move_array)
            corr_array = move_array[movement_baselines[rewarded_trial] :]
            corr_matrix.append(corr_array)

        if sum(np.invert(np.isnan(move_array)).astype(int)) > 100:
            num_tracked_movements = num_tracked_movements + 1
    # Convert list of movements into 2d array with each trial a row
    movement_matrix = np.array(movement_matrix)
    corr_matrix = np.array(corr_matrix)

    # Set conditional for minimum num of recorded movements
    min_move_num_contingency = num_tracked_movements > MIN_MOVE_NUM
    if rewards != 0 and min_move_num_contingency:
        movement_avg = np.nanmean(movement_matrix, axis=0)
    else:
        movement_matrix = np.empty(movement_matrix.shape)
        movement_matrix[:] = np.nan
        movement_avg = np.empty(min_t)
        movement_avg[:] = np.nan
        corr_matrix = np.empty(corr_matrix.shape)
        corr_matrix[:] = np.nan

    # Generate outpt object
    Summarized_Behavior = Session_Summary_Lever_Data(
        mouse_id=file.mouse_id,
        sess_name=file.sess_name,
        date=file.date,
        used_trial=used_trial,
        movement_matrix=movement_matrix,
        corr_matrix=corr_matrix,
        movement_avg=movement_avg,
        rewards=rewards,
        move_at_start_faults=move_at_start_fault,
        avg_reaction_time=avg_reaction_time,
        avg_cue_to_reward=avg_cue_to_reward_time,
        trials=num_trials,
        move_duration_before_cue=move_duration_before_cue,
        number_of_movements_during_ITI=number_of_movements_during_ITI,
        fraction_ITI_spent_moving=fraction_ITI_spent_moving,
    )

    return Summarized_Behavior


def summarize_nonimaged_lever_behavior(file, MIN_MOVE_NUM, MIN_T):
    """Function to summarize lever data when not imaged"""

    # Set up new attributes and variables // unrewarded or ignored trials will have np.nan
    successful_movements = []
    movement_baselines = []
    cue_to_reward = []
    post_success_licking = []
    faults = []
    used_trial = []
    reaction_time = []
    cs2r = []
    trial_length = []
    move_duration_before_cue = []
    movement_matrix = []
    corr_matrix = []
    number_of_movements_during_ITI = []
    fraction_ITI_spent_moving = []

    # Setup temporary variables
    rewards = 0
    move_at_start_fault = 0
    reward_times = []
    trial_ends = []
    movements = []
    past_threshold_reward_trials = []

    # Load xsg data
    xsg_data = file.xsg_data.channels["Trial_number"]
    # Read and setup bitcode
    bit_code = read_bit_code(xsg_data)
    bitcode = bit_code[:, 1]
    num_trials = file.dispatcher_data.saved.ProtocolsSection_n_done_trials
    if bit_code.size == 0:
        raise Exception("Could not extract bitcode information")

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

    if boundary_frames[0] == 0:
        boundary_frames = boundary_frames[1:]

    bitcode_offset = bitcode - np.arange(1, len(bitcode) + 1)

    bit_trial = []

    # Profile each trial
    for num, trial in enumerate(
        file.dispatcher_data.saved_history.ProtocolsSection_parsed_events
    ):
        # Skip if trial number exceeds bitcode
        if num > len(bitcode):
            continue

        # Setup trial information
        i_bitcode = (num) - np.absolute(bitcode_offset[num])
        i_bitcode = int(i_bitcode)
        # Skip if bitcode index is negative for some reason
        if i_bitcode < 0:
            reward_times.append(0)
            trial_ends.append(np.nan)
            continue
        # Skip if trial has already been used
        if np.sum(np.isin(bit_trial, i_bitcode)):
            reward_times.append(0)
            trial_ends.append(np.nan)
            continue

        bit_trial.append(i_bitcode)

        start_trial = np.round(bit_code[i_bitcode, 0] * 1000)  # time
        t0 = trial.states.bitcode[0]
        end_trial = int(
            start_trial + np.round((trial.states.state_0[1, 0] - t0) * 1000)
        )  # time

        # Profile only the rewarded trials
        if not trial.states.reward.size == 0:
            # Get rewards and reward time
            rewards = rewards + 1
            reward_time = np.round(start_trial + (trial.states.reward[0] - t0) * 1000)
            if reward_time == 0:
                reward_time = 1
            reward_time = int(reward_time)
            reward_times.append(reward_time)

            cue_start = int(np.round(start_trial + (trial.states.cue[0] - t0) * 1000))
            # Ensure trial occurs during movement recording
            if cue_start >= len(file.lever_force_smooth) or reward_time >= len(
                file.lever_force_smooth
            ):
                reward_times.append(0)
                trial_ends.append(np.nan)
                continue

            if end_trial > len(file.lever_force_smooth):
                end_trial = len(file.lever_force_smooth)

            trial_ends.append(end_trial)
            movement = file.lever_force_smooth[cue_start - 1 : end_trial]
            movements.append(movement)

            # Get movement trace above threshold
            past_thresh = (
                file.lever_force_smooth[cue_start - 1 : end_trial]
                * file.lever_active[cue_start - 1 : end_trial]
            )
            past_threshold_reward_trials.append(past_thresh)

            # Profile the movement
            trial_info = profile_rewarded_movements(
                file,
                boundary_frames,
                num,
                cue_start,
                reward_times,
                trial_ends,
                movement,
                past_thresh,
            )

            if trial_info.fault == 1:
                move_at_start_fault = move_at_start_fault + 1
            used_trial.append(trial_info.trial_used)
            trial_length.append(trial_info.trial_length)
            cs2r.append(trial_info.cs2r)
            reaction_time.append(trial_info.reaction_time)
            number_of_movements_during_ITI.append(
                trial_info.number_of_mvmts_since_last_trial
            )
            move_duration_before_cue.append(trial_info.move_duration_before_cue)
            fraction_ITI_spent_moving.append(trial_info.fraction_ITI_spent_moving)
            successful_movements.append(trial_info.successful_movements)
            movement_baselines.append(trial_info.succ_move_baseline)
            cue_to_reward.append(trial_info.cue_to_reward)
            post_success_licking.append(trial_info.post_success_licking)
            faults.append(trial_info.fault)

        else:
            trial_ends.append(end_trial)
            reward_times.append(0)

    # Get the average reaction time and cue_to_reward
    avg_reaction_time = np.nanmean(reaction_time)
    avg_cue_to_reward_time = np.nanmean(cs2r)

    # Remove zeros from trial length and set min trial length
    trial_length[trial_length == 0] = np.nan
    if not len(trial_length) == 0:
        min_t = np.nanmin(trial_length)
    else:
        min_t = MIN_T
    min_t = int(min_t)
    move_duration_before_cue = (
        np.array(move_duration_before_cue)[
            np.invert(np.isnan(np.asarray(move_duration_before_cue)))
        ]
        / 1000
    )
    movement_baseline = movement_baselines[0]
    # Generate movement matrix
    num_tracked_movements = 0
    for rewarded_trial in range(rewards):
        try:
            if isinstance(successful_movements[rewarded_trial], np.ndarray):
                move_array = np.array(successful_movements[rewarded_trial][: min_t + 1])
                move_array[move_array == 0] = np.nan
                movement_matrix.append(move_array)
                corr_array = move_array[movement_baseline:]
                corr_matrix.append(corr_array)
            else:
                move_array = np.empty(min_t)
                move_array[:] = np.nan
                movement_matrix.append(move_array)
                corr_array = move_array[movement_baseline:]
                corr_matrix.append(corr_array)
        except Exception as error:
            print(f"Movement was not tracked for trial {rewarded_trial}")
            print(error)
            move_array = np.empty(min_t)
            move_array[:] = np.nan
            movement_matrix.append(move_array)
            corr_array = move_array[movement_baseline:]
            corr_matrix.append(corr_array)
        if sum(np.invert(np.isnan(move_array)).astype(int)) > 100:
            num_tracked_movements = num_tracked_movements + 1

    # Convert list of movements into 2d array with each trial a row
    movement_matrix = np.array(movement_matrix)
    corr_matrix = np.array(corr_matrix)

    # Set conditional for minimum number of rewarded movements
    min_move_num_contingency = num_tracked_movements > MIN_MOVE_NUM
    if rewards != 0 and min_move_num_contingency:
        movement_avg = np.nanmean(movement_matrix, axis=0)
    else:
        movement_matrix = np.empty(movement_matrix.shape)
        movement_matrix[:] = np.nan
        movement_avg = np.empty(min_t)
        movement_avg[:] = np.nan
        corr_matrix = np.empty(corr_matrix.shape)
        corr_matrix[:] = np.nan

    # Generate outpt object
    Summarized_Behavior = Session_Summary_Lever_Data(
        mouse_id=file.mouse_id,
        sess_name=file.sess_name,
        date=file.date,
        used_trial=used_trial,
        movement_matrix=movement_matrix,
        corr_matrix=corr_matrix,
        movement_avg=movement_avg,
        rewards=rewards,
        move_at_start_faults=move_at_start_fault,
        avg_reaction_time=avg_reaction_time,
        avg_cue_to_reward=avg_cue_to_reward_time,
        trials=num_trials,
        move_duration_before_cue=move_duration_before_cue,
        number_of_movements_during_ITI=number_of_movements_during_ITI,
        fraction_ITI_spent_moving=fraction_ITI_spent_moving,
    )

    return Summarized_Behavior


def smooth_lick_data(licks):
    """Function to smooth lick data"""
    lick_data_resample = sysignal.resample_poly(licks, up=1, down=10)
    butter = sysignal.butter(4, (5 / 500), "low")
    lick_data_smooth = sysignal.filtfilt(
        butter[0],
        butter[1],
        lick_data_resample,
        axis=0,
        padtype="odd",
        padlen=3 * (max(len(butter[1]), len(butter[0])) - 1),
    )
    return lick_data_smooth


# -----------------------------------------------------------------------------
# -------------------------------DATACLASS USED--------------------------------
# -----------------------------------------------------------------------------


@dataclass
class Session_Summary_Lever_Data:
    """Dataclass for storing the summarized lver press data of a single session
        for a single mouse"""

    mouse_id: str
    sess_name: str
    date: str
    used_trial: list
    movement_matrix: np.ndarray
    corr_matrix: np.ndarray
    movement_avg: np.ndarray
    rewards: int
    move_at_start_faults: int
    avg_reaction_time: float
    avg_cue_to_reward: float
    trials: int
    move_duration_before_cue: list
    number_of_movements_during_ITI: list
    fraction_ITI_spent_moving: list
