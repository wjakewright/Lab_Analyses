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
    """Function to analyze lever press behavior of a single mouse across all sessions
    
        INPUT PARAMETERS
            file - An object (Processed_Lever_Data dataclass) containing the processed
                    lever behavior for a single session for a single mouse
    """
    rewards = 0
    movestartfault = 0
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

    rewards = 0
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


def profile_rewarded_movments(
    file, boundary_frames, trial_num, rewards, cue_start, reward_times, trial_ends
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
            
        OUTPUT PARMETERS
            file - 
            
            used_trial_info - 
            
            fault - 
            
            ignored_trial_info - 
            
    """

    ## Discard trial if the animal is already moving
    ## Still record details about the nature of the movement


@dataclass
class Session_Lever_Data:
    """Dataclass for storing the analyzed lever press data of a single sesson for 
        a single mouse"""

    trials: int
    rewards: int

