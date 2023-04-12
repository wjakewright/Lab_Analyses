"""Module to align the lever traces with the activity traces for each trial"""

import os
from dataclasses import dataclass
from fractions import Fraction
from itertools import compress

import numpy as np
import scipy.signal as sysignal

from Lab_Analyses.Behavior.read_bit_code import read_bit_code
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


def align_lever_behavior(
    behavior_data,
    imaging_data,
    save=None,
    save_path=None,
    save_suffix={"behavior": None, "imaging": None},
):
    """Function to align lever traces with the activity traces for each trial
        
        INPUT PARAMETERS
            behavior_data - dataclass containing the Processed_Lever_Data
                            Output from process_lever_behavior
            
            imaging_data - dataclass containing the Activity_Output with activity 
                            data in it. Output from Activity_Viewer

            save - str specifying which data you wish to save. Default is none

            save_suffix - dict for additional name information for behavior and/or
                        imaging datasets

        OUTPUT PARAMETERS
            aligned_data - dataclass containing 

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

    # Used only imaged trials
    i_trials = behavior_data.imaged_trials == 1
    trial_idxs = list(compress(trial_idxs, i_trials))

    # Set up variables that will contain the information for each trial
    trial_list = []
    trial_time = []
    results = []
    lever_force_resample_frames = []
    lever_force_smooth_frames = []
    lever_active_frames = []
    lever_velocity_envelope_frames = []
    rewarded_movement_binary = []
    rewarded_movement_force = []
    binary_cue_list = []
    result_delivery_list = []
    fluorescence = []
    dFoF = []
    processed_dFoF = []
    activity_trace = []
    floored_trace = []
    spikes = []

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
        if (
            behavior_data.behavior_frames[i].states.state_0[1, 0]
            > list(imaging_data.processed_dFoF.values())[0].shape[0]
        ):
            continue

        ######### TRIAL INFORMATION #########
        i_bitcode = i - np.absolute(bitcode_offset[i])
        if i_bitcode in trial_list:
            continue
        trial_list.append(i_bitcode)

        # Getting the times for the start and end of the trial in terms of lever sampling rate
        start_trial_time = int(np.round(curr_trial_list[i, 0] * 1000))
        t0 = dispatcher[i].states.bitcode[0]
        end_trial_time = int(
            start_trial_time
            + np.round((dispatcher[i].states.state_0[1, 0] - t0) * 1000)
        )
        if (
            start_trial_time > len(behavior_data.lever_force_smooth) - 1
            or end_trial_time > len(behavior_data.lever_force_smooth) - 1
        ):
            continue

        # Get start and end of trials in behavior frames
        start_trial_frames = int(behavior_data.behavior_frames[i].states.state_0[0, 1])
        end_trial_frames = int(behavior_data.behavior_frames[i].states.state_0[1, 0])
        num_frames = len(np.arange(start_trial_frames, end_trial_frames + 1))

        # Get the time of the trial frames
        times = behavior_data.frame_times[start_trial_frames : end_trial_frames + 1]

        ############ BEHAVIOR SECTION #############
        # Get lever traces for current trial in terms of time
        lever_force_resample_time = behavior_data.lever_force_resample[
            start_trial_time : end_trial_time + 1
        ]
        lever_force_smooth_time = behavior_data.lever_force_smooth[
            start_trial_time : end_trial_time + 1
        ]
        lever_active_time = behavior_data.lever_active[
            start_trial_time : end_trial_time + 1
        ]
        lever_velocity_envelop_smooth_time = behavior_data.lever_velocity_envelope_smooth[
            start_trial_time : end_trial_time + 1
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
        force_resample_frames = sysignal.resample_poly(lever_force_resample_time, n, d)
        force_smooth_frames = sysignal.resample_poly(lever_force_smooth_time, n, d)
        active_frames = sysignal.resample_poly(lever_active_time, n, d)
        velocity_envelope_frames = sysignal.resample_poly(
            lever_velocity_envelop_smooth_time, n, d
        )

        # Fix smooth of the binarized active trace
        active_frames[active_frames >= 0.5] = 1
        active_frames[active_frames < 0.5] = 0

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
        binary_cue[cue_start : cue_end + 1] = 1

        if behavior_data.behavior_frames[i].states.reward.shape[0] > 0:
            result = 1  # This is for rewarded trials
            result_start = int(
                behavior_data.behavior_frames[i].states.reward[0] - start_trial_frames
            )
            result_end = int(
                behavior_data.behavior_frames[i].states.reward[1] - start_trial_frames
            )
            try:
                # Trying to get the last movement before reward delivery
                rwd_move_start = np.nonzero(
                    np.sign(np.diff(active_frames[cue_start : result_start + 1])) == 1
                )[0][-1]
                rwd_move_start = int(rwd_move_start + cue_start)
            except:
                # If there isn't one detected, then movement must already have been going on, so set to start of trial
                rwd_move_start = int(cue_start)
            try:
                # Trying to get the first movement termination after reward delivery
                rwd_move_end = np.nonzero(
                    np.sign(
                        np.diff(active_frames[rwd_move_start : end_trial_frames + 1])
                    )
                    == -1
                )[0][0]
                rwd_move_end = int(rwd_move_end + rwd_move_start)
            except:
                # If no movement termination found, print error and set it to end of trial
                print(f"Error finding movement end for trial {i}")
                rwd_move_end = int(end_trial_frames)

            result_delivery = np.zeros(num_frames)
            result_delivery[result_start : result_end + 1] = 1
            rwd_movement_binary = np.zeros(num_frames)
            rwd_movement_binary[rwd_move_start : rwd_move_end + 1] = 1
            rwd_movement_force = np.zeros(num_frames)
            rwd_movement_force[rwd_move_start : rwd_move_end + 1] = force_smooth_frames[
                rwd_move_start : rwd_move_end + 1
            ]
            rewarded_movement_binary.append(rwd_movement_binary)
            rewarded_movement_force.append(rwd_movement_force)

        else:
            result = 0  # This is for punished trials
            result_start = int(
                behavior_data.behavior_frames[i].states.punish[0] - start_trial_frames
            )
            result_end = int(
                behavior_data.behavior_frames[i].states.punish[1] - start_trial_frames
            )

            result_delivery = np.zeros(num_frames)
            result_delivery[result_start : result_end + 1] = 1
            rwd_movement_binary = np.zeros(num_frames)
            rwd_movement_force = np.zeros(num_frames)
            rewarded_movement_binary.append(rwd_movement_binary)
            rewarded_movement_force.append(rwd_movement_force)

        ########### ACTIVITY SECTION ##############
        # Check what type of activity data is included in imaging data file

        if imaging_data.processed_fluorescence:
            try:
                fluo = align_activity(
                    imaging_data.processed_fluorescence,
                    behavior_data.behavior_frames[i],
                )
                fluorescence.append(fluo)
            except IndexError:
                continue

        if imaging_data.dFoF:
            dfof = align_activity(imaging_data.dFoF, behavior_data.behavior_frames[i])
            dFoF.append(dfof)

        if imaging_data.processed_dFoF:
            pdfof = align_activity(
                imaging_data.processed_dFoF, behavior_data.behavior_frames[i]
            )
            processed_dFoF.append(pdfof)

        if imaging_data.activity_trace:
            a_trace = align_activity(
                imaging_data.activity_trace, behavior_data.behavior_frames[i]
            )
            activity_trace.append(a_trace)

        if imaging_data.floored_trace:
            f_trace = align_activity(
                imaging_data.floored_trace, behavior_data.behavior_frames[i]
            )
            floored_trace.append(f_trace)

        if imaging_data.deconvolved_spikes:
            deconv = align_activity(
                imaging_data.deconvolved_spikes, behavior_data.behavior_frames[i]
            )
            spikes.append(deconv)

        ## Append all the behavioral results
        trial_time.append(times)
        results.append(result)
        lever_force_resample_frames.append(force_resample_frames)
        lever_force_smooth_frames.append(force_smooth_frames)
        lever_active_frames.append(active_frames)
        lever_velocity_envelope_frames.append(velocity_envelope_frames)
        binary_cue_list.append(binary_cue)
        result_delivery_list.append(result_delivery)

    ############ OUTPUT SECTION ###############
    # Generate the output dataclasses
    aligned_behavior_data = Trial_Aligned_Behavior(
        mouse_id=behavior_data.mouse_id,
        session=behavior_data.sess_name,
        date=behavior_data.date,
        trial_list=trial_list,
        trial_time=trial_time,
        result=results,
        lever_force_resample_frames=lever_force_resample_frames,
        lever_force_smooth_frames=lever_force_smooth_frames,
        lever_active_frames=lever_active_frames,
        lever_velocity_envelope_frames=lever_velocity_envelope_frames,
        rewarded_movement_binary=rewarded_movement_binary,
        rewarded_movement_force=rewarded_movement_force,
        binary_cue=binary_cue_list,
        result_delivery=result_delivery_list,
    )

    aligned_activity_data = Trial_Aligned_Activity(
        mouse_id=behavior_data.mouse_id,
        session=behavior_data.sess_name,
        date=behavior_data.date,
        imaging_parameters=imaging_data.parameters,
        ROI_ids=imaging_data.ROI_ids,
        ROI_flags=imaging_data.ROI_flags,
        ROI_positions=imaging_data.ROI_positions,
        processed_fluorescence=fluorescence,
        dFoF=dFoF,
        processed_dFoF=processed_dFoF,
        activity_trace=activity_trace,
        floored_trace=floored_trace,
        spikes=spikes,
        spine_pixel_intensity=imaging_data.spine_pixel_intensity,
        dend_segment_intensity=imaging_data.dend_segment_intensity,
        spine_volume=imaging_data.spine_volume,
        corrected_spine_pixel_intensity=imaging_data.corrected_spine_pixel_intensity,
        corrected_dend_segment_intensity=imaging_data.corrected_dend_segment_intensity,
        corrected_spine_volume=imaging_data.corrected_spine_volume,
    )

    # save data if specified
    if save:
        # Set up the save path
        if save_path is None:
            initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
            save_path = os.path.join(
                initial_path,
                behavior_data.mouse_id,
                "aligned_data",
                behavior_data.sess_name,
            )
        else:
            save_path = save_path

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        if save_suffix["behavior"]:
            b_name = f"{behavior_data.mouse_id}_{behavior_data.sess_name}_{save_suffix['behavior']}_aligned_behavior"

        else:
            b_name = (
                f"{behavior_data.mouse_id}_{behavior_data.sess_name}_aligned_behavior"
            )

        if save_suffix["imaging"]:
            i_name = f"{behavior_data.mouse_id}_{behavior_data.sess_name}_{save_suffix['imaging']}_aligned_activity"

        else:
            i_name = (
                f"{behavior_data.mouse_id}_{behavior_data.sess_name}_aligned_activity"
            )

        if save == "behavior":
            save_pickle(b_name, aligned_behavior_data, save_path)

        elif save == "imaging":
            save_pickle(i_name, aligned_activity_data, save_path)

        else:
            save_pickle(i_name, aligned_activity_data, save_path)
            save_pickle(b_name, aligned_behavior_data, save_path)

    return aligned_behavior_data, aligned_activity_data


def align_activity(activity, behavior_frames):
    """Helper function to handle aligning all the FOVs in the activity dictionary
        
        INPUT PARAMETERS
            activity - dict containing the activity for the different types of ROIs. 
                        Each key is for a different ROI type, which then contains all
                        the ROIs for that type within. 
                        This is a single field from the Activity_Output Datalcass
                        
            behavior_frame - object containing the behavior frames generated from 
                             dispatcher data
                             
        OUTPUT PARAMETERS
            tiral_activity - dict containint the activity the different types of ROIs. 
                            Same organization as the inpupt dict
    """
    # get the start and end of the trial
    start = int(behavior_frames.states.state_0[0, 1])
    end = int(behavior_frames.states.state_0[1, 0])

    # initialize output dictionary
    if type(activity) == dict:
        trial_activity = {}

        # go through each roi type
        for key, value in activity.items():
            if type(value) == np.ndarray:
                # Slicing the start and end of the trial
                trial = value[start : end + 1, :]
            else:
                trial = []
                for poly in value:
                    trial_poly = poly[start : end + 1, :]
                    trial.append(trial_poly)

            trial_activity[key] = trial

    elif type(activity) == np.ndarray:
        trial_activity = activity[start : end + 1, :]

    return trial_activity


################# DATACLASSES #################


@dataclass
class Trial_Aligned_Behavior:
    """Dataclass to contain all the lever behavior data for each trial.
        Includes it in terms of original sampling and in terms of image frames"""

    mouse_id: str
    session: str
    date: str
    trial_list: list
    trial_time: list
    result: list
    lever_force_resample_frames: list
    lever_force_smooth_frames: list
    lever_active_frames: list
    lever_velocity_envelope_frames: list
    rewarded_movement_binary: list
    rewarded_movement_force: list
    binary_cue: list
    result_delivery: list


@dataclass
class Trial_Aligned_Activity:
    """Dataclass to contain all the relevant activity data that has been 
        aligned to each trial"""

    mouse_id: str
    session: str
    date: str
    imaging_parameters: dict
    ROI_ids: dict
    ROI_flags: dict
    ROI_positions: dict
    processed_fluorescence: list
    dFoF: list
    processed_dFoF: list
    activity_trace: list
    floored_trace: list
    spikes: list
    spine_pixel_intensity: list
    dend_segment_intensity: list
    spine_volume: list
    corrected_spine_pixel_intensity: list
    corrected_dend_segment_intensity: list
    corrected_spine_volume: list

