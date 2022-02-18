"""Module to summarize the lever press behavior. Gets various behavior parameters
    and profiles the rewarded lever presses
    
    Takes pickle files output from process_lever_behavior.py
    
    CREATOR
        William (Jake) Wright - 2/2/2022
"""

from dataclasses import dataclass

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Behavior.process_lever_behavior import read_bit_code
from nbformat import read

# ----------------------------------------------------------------------------------
# ------------------------SUMMARIZE LEVER PRESS BEHAVIOR----------------------------
# ----------------------------------------------------------------------------------

# ------------------------------PARENT FUNCTION ------------------------------------
def summarize_lever_behavior(file):
    """Function to summarize lever press behavior from a single mouse for a single session

    INPUT PARAMETERS
        file - Object (Processed_Lever_Data dataclass) containing the processed
                lever behavior for a single session for a single mouse

    OUTPUT PARAMETERS


    """

    # Smooth lick data if any is present
    if "Lick" in file.xsg_data.channels.keys():
        file.lick_data_smooth = smooth_lick_data(licks=file.xsg_data.channels["Lick"])
    else:
        file.lick_data_smooth = np.array([])

    # Summarize data collected while imaging
    if not file.behavior_frames.size == 0:
        summarized_data = summarize_imaged_lever_behavior(file)

    # Summarize data collected while not imaging
    else:
        summarized_data = summarize_nonimaged_lever_behavior(file)

    return summarized_data


# ------------------------------ IMAGED TRIALS -------------------------------------
def summarize_imaged_lever_behavior(file):
    """Function to summarize lever press behavior for sessions that were imaged"""

    ## Set up new attributes and variables
    ### Note for unrewarded or ignored trials the values will be np.nan
    successful_movements = []  # movement trace for rewarded movements for each trial
    cue_to_reward = []  # movement trace from cue to reward delivery
    post_success_licking = []  # licking trace following reward delivery

    faults = []
    num_trials = len(file.behavior_frames)  # Number of trials performed
    used_trial = []  # Boolean list indicating if a trial was used or ignored
    reaction_time = []  # Reaction time values
    cs2r = []  # time from cue to reward delivery
    trial_length = []  # Length of each trial
    move_duration_before_cue = []  # Duration of movement before cue
    movement_matrix = []  # 2D array of the movements for each trial
    number_of_movements_during_ITI = []  # Number of movements during ITI
    fraction_ITI_spent_moving = []  # Fraction of ITI time spent moving

    # Set up temporary variables
    rewards = 0  # Reward counter
    move_at_start_fault = 0
    reward_times = []
    trial_ends = []
    movements = []
    past_threshold_reward_trials = []
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

    # Iterate through each trial
    for num, trial in enumerate(file.behavior_frames):
        # analyze rewarded trials
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
            # Get time of the start of the cue for this trial
            cue_start = int(
                np.round(
                    file.frame_times[np.round(trial.states.cue[0]).astype(int)] * 1000
                )
            )
            # Get time of next cue, indicating end of the current trial
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
            # Binarize movement trace
            past_thresh = (
                file.lever_force_smooth[cue_start - 1 : next_cue]
                * file.lever_active[cue_start - 1 : next_cue]
            )
            past_threshold_reward_trials.append(past_thresh)
            print(num)
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
            cue_to_reward.append(trial_info.cue_to_reward)
            post_success_licking.append(trial_info.post_success_licking)
            faults.append(trial_info.fault)

        else:
            trial_ends.append(np.nan)
            reward_times.append(0)

    # Get average reaction time and cue_to_reward
    avg_reaction_time = np.nanmean(reaction_time)
    avg_cue_to_reward_time = np.nanmean(cs2r)

    # Remove zeros from trial length and set min trial length
    trial_length[trial_length == 0] = np.nan
    if not len(trial_length) == 0:
        min_t = np.nanmin(trial_length)
    else:
        min_t = 3001
    min_t = int(min_t)
    move_duration_before_cue = np.array(move_duration_before_cue)[
        np.invert(np.isnan(np.asarray(move_duration_before_cue)))
    ]

    # Generate movement matrix
    num_tracked_movements = 0
    for rewarded_trial in range(rewards):
        try:
            if isinstance(successful_movements[rewarded_trial], np.ndarray):
                # if not np.isnan(successful_movements[rewarded_trial]):
                move_array = np.array(successful_movements[rewarded_trial][: min_t + 1])
                move_array[move_array == 0] = np.nan
                movement_matrix.append(move_array)
            else:
                move_array = np.empty(min_t)
                move_array[:] = np.nan
                movement_matrix.append(move_array)
        except Exception as error:
            print(f"Movement was not tracked for trial {rewarded_trial}")
            print("")
            print(error)
            move_array = np.empty(min_t)
            move_array[:] = np.nan
            movement_matrix.append(move_array)
        if sum(np.invert(np.isnan(move_array)).astype(int)) > 100:
            num_tracked_movements = num_tracked_movements + 1
    # Convert list of movements into 2d array with each trial a row
    movement_matrix = np.array(movement_matrix)

    # Set conditional for minimum number of rewarded movements
    min_move_num_contingency = num_tracked_movements > 0
    if rewards != 0 and min_move_num_contingency:
        movement_avg = np.nanmean(movement_matrix, axis=0)
    else:
        movement_matrix = np.empty(movement_matrix.shape)
        movement_matrix[:] = np.nan
        movement_avg = np.empty(min_t)
        movement_avg[:] = np.nan

    # Plot movements
    ### Add a behavior plotting module that will handle this later

    # Generate outpt object
    Summarized_Behavior = Session_Summary_Lever_Data(
        used_trial=used_trial,
        movement_matrix=movement_matrix,
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


# ---------------------------NON-IMAGED TRIALS-------------------------------
def summarize_nonimaged_lever_behavior(file):
    """Function to summarize lever press behavior for sessions that were not imaged"""
    ## Set up new attributes and variables
    ### Note for unrewarded or ignored trials the values will be np.nan
    successful_movements = []  # movement trace for rewarded movements for each trial
    cue_to_reward = []  # movement trace from cue to reward delivery
    post_success_licking = []  # licking trace following reward delivery

    faults = []
    num_trials = len(file.behavior_frames)  # Number of trials performed
    trial_used = []
    used_trial = []  # Boolean list indicating if a trial was used or ignored
    reaction_time = []  # Reaction time values
    cs2r = []  # time from cue to reward delivery
    trial_length = []  # Length of each trial
    move_duration_before_cue = []  # Duration of movement before cue
    movement_matrix = []  # 2D array of the movements for each trial
    number_of_movements_during_ITI = []  # Number of movements during ITI
    fraction_ITI_spent_moving = []  # Fraction of ITI time spent moving

    # Set up temporary variables
    rewards = 0  # Reward counter
    move_at_start_fault = 0
    reward_times = []
    trial_ends = []
    movements = []
    past_threshold_reward_trials = []

    xsg_data = file.xsg_data.channels["Trial_number"]
    bit_code = read_bit_code(xsg_data)
    bitcode = bit_code[:, 1]
    trials = file.dispatcher_data.saved.ProtocolsSection_n_done_trials
    if bit_code.size == 0:
        raise Exception("Could not extract bitcode information")

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
    if boundary_frames[0] == 1:
        boundary_frames = boundary_frames[1:]

    bitcode_offset = bitcode - np.arange(1, len(bitcode) + 1)

    for num, trial in enumerate(
        file.dispatcher_data.saved_history.ProtocolsSection_parsed_events
    ):

        if num + 1 > len(bitcode):
            continue
        i_bitcode = (num + 1) - np.absolute(bitcode_offset[num])
        i_bitcode = int(i_bitcode)
        if i_bitcode < 0:
            continue
        if np.sum(np.isin(trial_used, i_bitcode)):
            continue
        trial_used.append(i_bitcode)

        start_trial = np.round(
            bit_code[i_bitcode, 0] * 1000
        )  # Getting the bitcode in time
        t0 = trial.states.bitcode[0]
        end_trial = int(
            start_trial + np.round((trial.states.state_0[1, 0] - t0) * 1000)
        )

        if not trial.states.bitcode[0].size == 0:
            rewards = rewards + 1
            reward_time = np.round(start_trial + (trial.states.reward[0] - t0) * 1000)
            if reward_time == 0:
                reward_time = 1
            reward_time = int(reward_time)
            reward_times.append(reward_time)

            cue_start = int(np.round(start_trial + (trial.states.cue[0] - t0) * 1000))
            if cue_start >= len(file.lever_force_smooth) or reward_time >= len(
                file.lever_force_smooth
            ):
                continue

            if end_trial > len(file.lever_force_smooth):
                end_trial = len(file.lever_force_smooth)
            trial_ends.append(end_trial)

            movement = file.lever_force_smooth[cue_start - 1 : end_trial]
            movements.append(movement)
            # Binarize movement trace
            past_thresh = (
                file.lever_force_smooth[cue_start - 1 : end_trial]
                * file.lever_active[cue_start - 1 : end_trial]
            )
            past_threshold_reward_trials.append(past_thresh)

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
            cue_to_reward.append(trial_info.cue_to_reward)
            post_success_licking.append(trial_info.post_success_licking)
            faults.append(trial_info.fault)

        else:
            trial_ends.append(np.nan)
            reward_times.append(0)

        # Get average reaction time and cue_to_reward
    avg_reaction_time = np.nanmean(reaction_time)
    avg_cue_to_reward_time = np.nanmean(cs2r)

    # Remove zeros from trial length and set min trial length
    trial_length[trial_length == 0] = np.nan
    if not len(trial_length) == 0:
        min_t = np.nanmin(trial_length)
    else:
        min_t = 3001
    min_t = int(min_t)
    move_duration_before_cue = np.array(move_duration_before_cue)[
        np.invert(np.isnan(np.asarray(move_duration_before_cue)))
    ]

    # Generate movement matrix
    num_tracked_movements = 0
    for rewarded_trial in range(rewards):
        try:
            if isinstance(successful_movements[rewarded_trial], np.ndarray):
                # if not np.isnan(successful_movements[rewarded_trial]):
                move_array = np.array(successful_movements[rewarded_trial][: min_t + 1])
                move_array[move_array == 0] = np.nan
                movement_matrix.append(move_array)
            else:
                move_array = np.empty(min_t)
                move_array[:] = np.nan
                movement_matrix.append(move_array)
        except Exception as error:
            print(f"Movement was not tracked for trial {rewarded_trial}")
            print("")
            print(error)
            move_array = np.empty(min_t)
            move_array[:] = np.nan
            movement_matrix.append(move_array)
        if sum(np.invert(np.isnan(move_array)).astype(int)) > 100:
            num_tracked_movements = num_tracked_movements + 1
    # Convert list of movements into 2d array with each trial a row
    movement_matrix = np.array(movement_matrix)

    # Set conditional for minimum number of rewarded movements
    min_move_num_contingency = num_tracked_movements > 0
    if rewards != 0 and min_move_num_contingency:
        movement_avg = np.nanmean(movement_matrix, axis=0)
    else:
        movement_matrix = np.empty(movement_matrix.shape)
        movement_matrix[:] = np.nan
        movement_avg = np.empty(min_t)
        movement_avg[:] = np.nan

    # Plot movements
    ### Add a behavior plotting module that will handle this later

    # Generate outpt object
    Summarized_Behavior = Session_Summary_Lever_Data(
        used_trial=used_trial,
        movement_matrix=movement_matrix,
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


# ---------------------------------------------------------------------------
# -------------------------PROFILE REWARDED MOVEMENTS------------------------
# ---------------------------------------------------------------------------
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
        file - object containing the processed lever_press behavior_data

        boundary_frames - np.array containing the boundary frames of when the lever is active

        trial_num - int specifying the current trial index

        cue_start - int indicating the time when the cue started on the current trial

        reward_times - list containing the reward times of all trials analyzed thus far

        trial_ends - list containing the time of the end of all trials analyzed thus far

        past_threh - np.array of binarized movement of the current trial

    OUTPUT PARAMETERS
        trial_info - dataclass containing relevant trial information
    """
    ###################### DISCARD BAD TRIALS ###########################

    ## Discard trial if the animal is already moving
    ## Still record details about the nature of the movements
    if any(file.lever_active[cue_start - 100 : cue_start] == 1):
        print(f"Animal was moving at the beginning of trial {trial_num}!")

        trial_info = profile_movement_before_cue(
            file, trial_num, cue_start, reward_times
        )

        return trial_info

    else:
        move_duration_before_cue = 0

    ## Discard trials with very brief movements or that are at the very end of the session
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

    # find boundaries of contiguous movements
    temp = np.nonzero(boundary_frames < reward_times[trial_num])[0]

    ## Discard trials without detected movements
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

    ###################### PROFILE GOOD TRIALS #########################

    # Characterize the ITI of successful trials
    number_of_movements_since_last_trial = len(
        np.nonzero(np.diff(file.lever_active[0:cue_start]) > 0)
    )

    if trial_num > 0:
        # if reward_times[trial_num - 1] == 0: #Commented out since Python is 0 indexed
        #    reward_times[trial_num - 1] = 1

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

    ## Get force from cue start to reward delivery
    cue_to_reward = file.lever_force_smooth[cue_start : reward_times[trial_num] + 1]
    cs2r = len(cue_to_reward) / 1000

    ## Define the beginning of a successful movement window
    temp = np.nonzero(boundary_frames < reward_times[trial_num])[0]

    if boundary_frames[temp[-1]] < 400:
        baseline_start = len(np.arange(0, boundary_frames[temp[-1]]))
    else:
        baseline_start = 400

    successful_mvmt_start = boundary_frames[temp[-1]] - baseline_start

    # Might need to comment this block out since it is used to index
    if successful_mvmt_start == 0:
        successful_mvmt_start = 1

    # rise = np.nonzero(cue_to_reward > np.median(cue_to_reward))[0][0]
    # if rise.size == 0:
    #    rise = 1 ## Not used elsewhere in code...

    if baseline_start < 400:
        shift = np.absolute(baseline_start - 400)
    else:
        shift = 0
    shift = int(shift)
    trial_stop_window = 3000

    # Add buffer for movements that extend beyond the end of the session
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

    # Add buffer at the start to account for shift
    start_buffer = np.empty(shift)
    start_buffer[:] = np.nan

    successful_movement = np.concatenate(
        (
            start_buffer,
            file.lever_force_smooth[
                int(successful_mvmt_start)
                - 1 : int(successful_mvmt_start)
                + (trial_stop_window - shift)
            ],
        )
    )

    trial_length = len(successful_movement)
    if trial_length == 0:
        raise ValueError("Error with Trial Length on successful trial !!!")

    reaction_time = np.nonzero(past_thresh)[0][0] / 1000  ## Converted to seconds
    if reaction_time.size == 0:
        reaction_time = 0

    # Repeat above steps for licking data if available
    trial_stop_window = 5000
    ## Generate ending buffer
    if successful_mvmt_start + (trial_stop_window - shift) > len(file.lick_data_smooth):
        ending_buffer = np.empty(
            np.absolute(
                len(file.lick_data_smooth)
                - successful_mvmt_start
                + (trial_stop_window - shift)
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
                    + (trial_stop_window - shift)
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
    """Helper function to profile trials that are ignored due to movements before cue"""
    trial_length = np.nan
    cs2r = np.nan
    reaction_time = np.nan
    fault = 1  # Used as boolean operator but can also keep track of different types of ignored trials

    # Get indice of the start of the movement before cue
    move_start_before_cue = np.nonzero(np.diff(file.lever_active[0:cue_start]) > 0)[0][
        -1
    ]
    move_duration_before_cue = len(np.arange(move_start_before_cue, cue_start))

    if trial_num > 0:
        # if reward_times[trial_num - 1] == 0: # Commented out since Python is 0 indexed
        #    reward_times[trial_num - 1] = 1
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


# ----------------------------------------------------------------------------
# --------------------------PARSE BEHAVIOR BITCODE----------------------------
# ----------------------------------------------------------------------------
def parse_behavior_bitcode(xsg_trace):
    """Function to parse behavior bitcode for unimaged sessions.
    Similar to Read_Bitcode in process_lever_behavior, but with
    some differences"""

    # Set up some parameters
    xsg_sample_rate = 10000
    threshold_value = 2
    num_bits = 12
    binary_threshold = (xsg_trace > threshold_value).astype(float)
    shift_binary_threshold = np.insert(binary_threshold[:-1], 0, np.nan)
    # Get raw times for rising edge of signals
    rising_bitcode = np.nonzero(
        (binary_threshold == 1).astype(int) & (shift_binary_threshold == 0).astype(int)
    )[0]

    # Set up the possible bits, 12 values, most significant first
    bit_values = np.arange(num_bits - 1, -1, -1, dtype=int)
    bit_values = 2 ** bit_values

    # Find the sync bitcodes: anything where the difference is larger than the
    # length of the bitcode (16ms - set as 20ms to be safe)
    bitcode_time_samples = 200 * (
        xsg_sample_rate / 1000
    )  # THIS IS DIFFERENT THAT READBITCODE
    bitcode_sync = np.nonzeror(np.diff(rising_bitcode) > bitcode_time_samples)[0]

    # Assume that the first rising edge is a sync signal
    if len(rising_bitcode) == 0:
        behavior_trials = []
    else:
        # Add first one back and shift back to rising pulse; get the bitcode index in time [HL]
        bitcode_sync = rising_bitcode[np.insert(bitcode_sync + 1, 0, 0)]
        # Initialize the


# ---------------------------------------------------------------------------
# -----------------------------HELPER FUNCTIONS------------------------------
# ---------------------------------------------------------------------------
def smooth_lick_data(licks):
    """Helper function to smooth lick data"""
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
# ------------------------Dataclasses used in module---------------------------
# -----------------------------------------------------------------------------


@dataclass
class Session_Summary_Lever_Data:
    """Dataclass for storing the analyzed lever press data of a single session for
    a single mouse"""

    used_trial: list
    movement_matrix: np.ndarray
    movement_avg: np.ndarray
    rewards: int
    move_at_start_faults: int
    avg_reaction_time: float
    avg_cue_to_reward: float
    trials: int
    move_duration_before_cue: list
    number_of_movements_during_ITI: list
    fraction_ITI_spent_moving: list


@dataclass
class Behavior_Trials:
    """Dataclass for storing behavior trial info used in parse
    behavior bitcode"""

    xsg_sec: list
    xsg_sample: list
    behavior_trial_num: list


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
