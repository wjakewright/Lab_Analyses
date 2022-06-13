"""Module to align the lever traces with the activity traces for each trial"""


from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.signal as sysignal
from Lab_Analyses.Behavior.read_bit_code import read_bit_code


def align_lever_behavior(behavior_data, imaging_data):
    """Function to align lever traces with the activity traces for each trial
        
        INPUT PARAMETERS
            behavior_data - dataclass containing the Processed_Lever_Data
                            Output from process_lever_behavior
            
            imaging_data - dataclass containing the Activity_Output with activity 
                            data in it. Output from Activity_Viewer

        OUTPUT PARAMETERS
            

    """
    # Get important part of dispatcher data
    dispatcher = (
        behavior_data.dispatcher_data.saved_history.ProtocolsSection_parsed_events
    )
    # Pull bitcode from the xsg data
    trial_channel = behavior_data.xsg_data.channels["Trial_number"]
    curr_trial_list = read_bit_code(trial_channel)
    # Get bitcode trial numbers. This starts at 1 so will subtract 1 when used for indexing
    bitcode = curr_trial_list[:, 1]
    bitcode_offset = bitcode - np.arange(1, len(bitcode) + 1)

    first_trial = 0  # Zero for indexing reasons
    last_trial = len(behavior_data.behavior_frames)

    # Check for trial errors in the bitcode output
    errors = np.nonzero(np.absolute(np.diff(bitcode_offset)) > 1)[0]
    if errors:
        trial_error = errors[0] + 1
        # Check if there were issues with the trials at the beginning of the session
        if trial_error < 20:
            # Setting first trial to trial after the error trials
            first_trial = trial_error
        else:
            # Setting last trial to trial before the error trials
            last_trial = trial_error - 1  # Recheck for indexing issues

    # Check if there are errors in what determining the first trial
    counted_as_frist_trial = np.nonzero(bitcode == 1)
    if (
        len(counted_as_frist_trial) > 1
        and first_trial == 1
        and counted_as_frist_trial[-1] < 100
    ):
        first_trial = counted_as_frist_trial[-1] - 1

    trial_idxs = np.arange(first_trial, last_trial)  # trial indexes

    # Set up variables that will contain the information for each trial
    used_trials = []
    trial_lever_force_resample_time = []
    trial_lever_force_smooth_time = []
    trial_lever_active_time = []
    trial_lever_velocity_envelope_smooth_time = []

    # Go through each trial
    for i in trial_idxs:
        ## Skip certain types of trials
        if i > len(behavior_data.behavior_frames) - 1:
            continue
        if not behavior_data.imaged_trials[i]:
            continue
        if i > len(bitcode_offset) - 1:
            continue
        if (
            behavior_data.behavior_frames[i].states.state_0[1, 0]
            - behavior_data.behavior_frames[i].states.state_0[0, 1]
            <= 1
        ):
            continue

        ######### TRIAL INFORMATION #########
        i_bitcode = i - np.absolute(bitcode_offset[i])
        if i_bitcode in used_trials:
            continue
        used_trials.append(i_bitcode)

        # Getting the times for the start and end of the trial in terms of lever sampling rate
        start_trial_time = np.round(curr_trial_list[i, 0] * 1000)
        t0 = dispatcher[i].states.bitcode[0]
        end_trial_time = start_trial_time + np.round(
            (dispatcher[i].states.state_0[1, 0] - t0) * 1000
        )
        if (
            start_trial_time > len(behavior_data.lever_force_smooth) - 1
            or end_trial_time > len(behavior_data.lever_force_smooth) - 1
        ):
            continue

        # Get start and end of trials in behavior frames
        start_trial_frames = behavior_data.behavior_frames[i].states.state_0[0, 1]
        end_trial_frames = behavior_data.behavior_frames[i].states.state_0[1, 0]
        num_frames = len(np.arange(start_trial_frames, end_trial_frames))

        ############ BEHAVIOR SECTION #############
        # Get lever traces for current trial in terms of time
        lever_force_resample_time = behavior_data.lever_force_resample[
            start_trial_time:end_trial_time
        ]
        lever_force_smooth_time = behavior_data.lever_force_smooth[
            start_trial_time:end_trial_time
        ]
        lever_active_time = behavior_data.lever_active[start_trial_time:end_trial_time]
        lever_velocity_envelop_smooth_time = behavior_data.lever_velocity_envelope_smooth[
            start_trial_time:end_trial_time
        ]
        # Zero start to minimize downsampling edge effects
        lever_force_resample_time = (
            lever_force_resample_time - lever_force_resample_time[0]
        )
        lever_force_smooth_time = lever_force_smooth_time - lever_force_smooth_time[0]
        lever_velocity_envelop_smooth_time = (
            lever_velocity_envelop_smooth_time - lever_velocity_envelop_smooth_time[0]
        )

        # Downsample the lever traces to frames
        frac = Fraction(num_frames / len(lever_force_resample_time)).limit_denominator()
        n = frac.numerator
        d = frac.denominator
        force_resample_ds = sysignal.resample_poly(lever_force_resample_time, n, d)
        force_smooth_ds = sysignal.resample_poly(lever_force_smooth_time, n, d)
        active_ds = sysignal.resample_poly(lever_active_time, n, d)
        velocity_envelope_ds = sysignal.resample_poly(
            lever_velocity_envelop_smooth_time, n, d
        )

        # Fix smooth of the binarized active trace
        active_ds[active_ds >= 0.5] = 1
        active_ds[active_ds < 0.5] = 0

        # Get cue and reward information
        cue_start = int(
            behavior_data.behavior_frames[i].states.cue[0] - start_trial_frames
        )
        if cue_start == 0:
            cue_start = 1
        cue_end = int(
            behavior_data.behavior_frames[i].states.cue[1] - start_trial_frames
        )
        binary_cue = np.zeros(num_frames)
        binary_cue[cue_start, cue_end] = 1


################# DATACLASSES #################


@dataclass
class Trial_Lever_Data:
    """Dataclass to contain all the lever data for each trial.
        Includes it in terms of original sampling and in terms of image frames"""

    trial_list: list
    force_resample_time: list
    force_smooth_time: list
    active_time: list
    velocity_envelope_smooth_time: list
    force_resample_frames: list
    force_smooth_frames: list
    active_frames: list
    velocity_envelope_smooth_frames: list
