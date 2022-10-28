"""Module to handle initial processing of spine activity data sets"""

import os
import re
from copy import copy
from dataclasses import dataclass

import numpy as np
from Lab_Analyses.Behavior.align_lever_behavior import align_lever_behavior
from Lab_Analyses.Utilities.data_utilities import join_dictionaries
from Lab_Analyses.Utilities.movement_responsiveness_v2 import movement_responsiveness
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def organize_dual_spine_data(
    mouse_id,
    channels={"GluSnFr": "GreenCh", "Calcium": "RedCh"},
    save=False,
    structural=False,
):
    """Function to handle the initial processing of dual color spine 
        activity datasets. Specifically designed to handle GluSnFR and 
        calcium activity
        
        INPUT PARAMETERS
            mouse_id - str specifying what the mouse's id is
            
            channels - tuple of strings for the different types of activity
                    to be co-processed. Will use to search for the relevant
                    files
            
            save - boolean specifying if the data is to be saved

            structural - boolean specifying if there is seperate structural data
                        to include
    """

    print(f"----------------------------------\nAnalyzing Mouse {mouse_id}")
    if len(channels) != 2:
        return "Need to have at least two channels specified"

    if save:
        align_save = ("both", "imaging")
    else:
        align_save = (None, None)

    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
    mouse_path = os.path.join(initial_path, mouse_id)
    imaging_path = os.path.join(mouse_path, "imaging")
    behavior_path = os.path.join(mouse_path, "behavior")

    # Get the number of FOVs imaged for this mouse
    FOVs = next(os.walk(imaging_path))[1]

    FOV_data = {}
    # Preprocess each FOV seperately
    for FOV in FOVs:
        print(f"- Preprocessing {FOV}")
        FOV_path = os.path.join(imaging_path, FOV)
        # Get the different imaging session periods
        periods = next(os.walk(FOV_path))[1]
        # Reorder the periods if Early, Middle, Late
        if "Early" and "Late" and "Middle" in periods:
            periods = ["Early", "Middle", "Late"]
        # Preprocess each imaging period
        period_data = {}
        for period in periods:
            print(f"-- {period}")
            period_path = os.path.join(FOV_path, period)
            fnames = next(os.walk(period_path))[2]
            fnames = [x for x in fnames if "imaging_data" in x]
            # Get the files
            GluSnFr_fname = os.path.join(
                period_path,
                [
                    x
                    for x in fnames
                    if channels["GluSnFr"] in x and "structural" not in x
                ][0],
            )
            Calcium_fname = os.path.join(
                period_path, [x for x in fnames if channels["Calcium"] in x][0]
            )
            GluSnFr_data = load_pickle([GluSnFr_fname])[0]
            Calcium_data = load_pickle([Calcium_fname])[0]

            if structural:
                structural_fname = os.path.join(
                    period_path, [x for x in fnames if "structural" in x][0]
                )
                structural_data = load_pickle([structural_fname])[0]
            else:
                structural_data = None

            # Get the matching behavioral data
            day = re.search("[0-9]{6}", os.path.basename(GluSnFr_fname)).group()
            matched_b_path = os.path.join(behavior_path, day)
            b_fnames = next(os.walk(matched_b_path))[2]
            b_fname = [x for x in b_fnames if "processed_lever_data" in x][0]
            b_fname = os.path.join(matched_b_path, b_fname)
            behavior_data = load_pickle([b_fname])[0]
            # rewrite the session name (which is by default the date)
            behavior_data.sess_name = period

            align_save_path = os.path.join(mouse_path, "aligned_data", FOV, period)
            # Align the data
            print("----- aligning data")
            aligned_behavior, aligned_GluSnFr = align_lever_behavior(
                behavior_data,
                GluSnFr_data,
                save=align_save[0],
                save_path=align_save_path,
                save_suffix={"behavior": None, "imaging": "GluSnFr"},
            )
            _, aligned_Calcium = align_lever_behavior(
                behavior_data,
                Calcium_data,
                save=align_save[1],
                save_path=align_save_path,
                save_suffix={"behavior": None, "imaging": "Calcium"},
            )

            # Group the data together
            ## Initialize output
            dual_spine_data = Dual_Channel_Spine_Data(
                mouse_id=aligned_behavior.mouse_id,
                session=period,
                date=aligned_behavior.date,
                time=np.concatenate(aligned_behavior.trial_time),
                lever_force_resample=np.concatenate(
                    aligned_behavior.lever_force_resample_frames
                ),
                lever_force_smooth=np.concatenate(
                    aligned_behavior.lever_force_smooth_frames
                ),
                lever_velocity_envelope=np.concatenate(
                    aligned_behavior.lever_velocity_envelope_frames
                ),
                lever_active=np.concatenate(aligned_behavior.lever_active_frames),
                rewarded_movement_force=np.concatenate(
                    aligned_behavior.rewarded_movement_force
                ),
                rewarded_movement_binary=np.concatenate(
                    aligned_behavior.rewarded_movement_binary
                ),
                binary_cue=np.concatenate(aligned_behavior.binary_cue),
                reward_delivery=np.array([]),
                punish_delivery=np.array([]),
                spine_ids=aligned_GluSnFr.ROI_ids["Spine"],
                spine_flags=aligned_GluSnFr.ROI_flags["Spine"],
                spine_grouping=[],
                spine_positions=aligned_GluSnFr.ROI_positions["Spine"],
                spine_GluSnFr_dFoF=np.array([]),
                spine_GluSnFr_processed_dFoF=np.array([]),
                spine_GluSnFr_activity=np.array([]),
                spine_GluSnFr_floored=np.array([]),
                spine_calcium_dFoF=np.array([]),
                spine_calcium_processed_dFoF=np.array([]),
                spine_calcium_activity=np.array([]),
                spine_calcium_floored=np.array([]),
                dendrite_calcium_dFoF=np.array([]),
                dendrite_calcium_processed_dFoF=np.array([]),
                dendrite_calcium_activity=np.array([]),
                dendrite_calcium_floored=np.array([]),
                spine_volume=aligned_GluSnFr.spine_volume,
                corrected_spine_volume=aligned_GluSnFr.corrected_spine_volume,
                movement_spines=[],
                reward_movement_spines=[],
                silent_spines=[],
                reward_silent_spines=[],
                spine_movement_activity={},
                spine_reward_activity={},
                movement_dendrites=[],
                reward_movement_dendrites=[],
                silent_dendrites=[],
                reward_silent_dendrites=[],
                dendrite_movement_activity={},
                dendrite_reward_activity={},
                imaging_parameters=aligned_GluSnFr.imaging_parameters,
            )

            # Start filling in the missing data
            ## Reward and punish information
            reward_delivery = copy(aligned_behavior.result_delivery)
            for i, outcome in enumerate(aligned_behavior.result):
                if outcome == 0:
                    reward_delivery[i] = np.zeros(len(reward_delivery[i]))
            punish_delivery = copy(aligned_behavior.result_delivery)
            for i, outcome in enumerate(aligned_behavior.result):
                if outcome == 1:
                    punish_delivery[i] = np.zeros(len(punish_delivery[i]))

            dual_spine_data.reward_delivery = np.concatenate(reward_delivery)
            dual_spine_data.punish_delivery = np.concatenate(punish_delivery)

            ## Activity
            spine_grouping = aligned_GluSnFr.imaging_parameters["Spine Groupings"]
            if not spine_grouping:
                spine_grouping = list(range(aligned_GluSnFr.dFoF[0]["Spine"].shape[1]))
            dual_spine_data.spine_grouping = spine_grouping

            GluSnFr_dFoF = join_dictionaries(aligned_GluSnFr.dFoF)
            GluSnFr_processed_dFoF = join_dictionaries(aligned_GluSnFr.processed_dFoF)
            GluSnFr_activity = join_dictionaries(aligned_GluSnFr.activity_trace)
            GluSnFr_floored = join_dictionaries(aligned_GluSnFr.floored_trace)
            calcium_dFoF = join_dictionaries(aligned_Calcium.dFoF)
            calcium_processed_dFoF = join_dictionaries(aligned_Calcium.processed_dFoF)
            calcium_activity = join_dictionaries(aligned_Calcium.activity_trace)
            calcium_floored = join_dictionaries(aligned_Calcium.floored_trace)

            dual_spine_data.spine_GluSnFr_dFoF = GluSnFr_dFoF["Spine"]
            dual_spine_data.spine_GluSnFr_processed_dFoF = GluSnFr_processed_dFoF[
                "Spine"
            ]
            dual_spine_data.spine_GluSnFr_activity = GluSnFr_activity["Spine"]
            dual_spine_data.spine_GluSnFr_floored = GluSnFr_floored["Spine"]
            dual_spine_data.spine_calcium_dFoF = calcium_dFoF["Spine"]
            dual_spine_data.spine_calcium_processed_dFoF = calcium_processed_dFoF[
                "Spine"
            ]
            dual_spine_data.spine_calcium_activity = calcium_activity["Spine"]
            dual_spine_data.spine_calcium_floored = calcium_floored["Spine"]
            dual_spine_data.dendrite_calcium_dFoF = calcium_dFoF["Dendrite"]
            dual_spine_data.dendrite_calcium_processed_dFoF = calcium_processed_dFoF[
                "Dendrite"
            ]
            dual_spine_data.dendrite_calcium_activity = calcium_activity["Dendrite"]
            dual_spine_data.dendrite_calcium_floored = calcium_floored["Dendrite"]

            # Add movement-related information
            print("----- analyzing movement responses")
            (
                movement_spines,
                silent_spines,
                spine_move_activity,
            ) = movement_responsiveness(
                dual_spine_data.spine_GluSnFr_processed_dFoF,
                dual_spine_data.lever_active,
            )
            (
                reward_movement_spines,
                reward_silent_spines,
                reward_spine_activity,
            ) = movement_responsiveness(
                dual_spine_data.spine_GluSnFr_processed_dFoF,
                dual_spine_data.rewarded_movement_binary,
            )

            (
                movement_dendrites,
                silent_dendrites,
                dendrite_move_activity,
            ) = movement_responsiveness(
                dual_spine_data.dendrite_calcium_processed_dFoF,
                dual_spine_data.lever_active,
            )
            (
                reward_movement_dendrites,
                reward_silent_dendrites,
                reward_dendrite_activity,
            ) = movement_responsiveness(
                dual_spine_data.dendrite_calcium_processed_dFoF,
                dual_spine_data.rewarded_movement_binary,
            )
            dual_spine_data.movement_spines = movement_spines
            dual_spine_data.silent_spines = silent_spines
            dual_spine_data.spine_movement_activity = spine_move_activity
            dual_spine_data.reward_movement_spines = reward_movement_spines
            dual_spine_data.reward_silent_spines = reward_silent_spines
            dual_spine_data.spine_reward_activity = reward_spine_activity
            dual_spine_data.movement_dendrites = movement_dendrites
            dual_spine_data.silent_dendrites = silent_dendrites
            dual_spine_data.dendrite_movement_activity = dendrite_move_activity
            dual_spine_data.reward_movement_dendrites = reward_movement_dendrites
            dual_spine_data.reward_silent_dendrites = reward_silent_dendrites
            dual_spine_data.dendrite_reward_activity = reward_dendrite_activity

            if structural:
                s_day = re.search(
                    "[0-9]{6}", os.path.basename(structural_fname)
                ).group()
                structural_output = Structural_Spine_Data(
                    mouse_id=aligned_behavior.mouse_id,
                    session=period,
                    date=s_day,
                    spine_ids=structural_data.ROI_ids["Spine"],
                    spine_flags=structural_data.ROI_flags["Spine"],
                    spine_positions=structural_data.ROI_positions["Spine"],
                    spine_pixel_intensity=structural_data.spine_pixel_intensity,
                    dend_segment_intensity=structural_data.dend_segment_intensity,
                    spine_volume=structural_data.spine_volume,
                    corrected_spine_pixel_intensity=structural_data.corrected_spine_pixel_intensity,
                    corrected_dend_segment_intensity=structural_data.corrected_dend_segment_intensity,
                    corrected_spine_volume=structural_data.corrected_spine_volume,
                    imaging_parameters=structural_data.parameters,
                )

            # Store the period data
            if structural:
                period_data[period] = (dual_spine_data, structural_output)
            else:
                period_data[period] = dual_spine_data

            # Save section
            if save:
                # Setup the save path
                save_path = os.path.join(mouse_path, "spine_data", FOV)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                # Make the file name and save
                fname = f"{dual_spine_data.mouse_id}_{FOV}_{period}_dual_spine_data"
                save_pickle(fname, dual_spine_data, save_path)

                # save structural data if present
                if structural:
                    sname = f"{structural_output.mouse_id}_{FOV}_{period}_followup_structure"
                    save_pickle(sname, structural_output, save_path)

        # Store the FOV data
        FOV_data[FOV] = period_data

    return FOV_data


#################### DATACLASSES #########################
@dataclass
class Dual_Channel_Spine_Data:
    """Dataclass to contain all the relevant behavioral and activity data
        for a single imaging session. Contains both GluSnFR and calcium  
        activity data together"""

    mouse_id: str
    session: str
    date: str
    time: np.ndarray
    lever_force_resample: np.ndarray
    lever_force_smooth: np.ndarray
    lever_velocity_envelope: np.ndarray
    lever_active: np.ndarray
    rewarded_movement_force: np.ndarray
    rewarded_movement_binary: np.ndarray
    binary_cue: np.ndarray
    reward_delivery: np.ndarray
    punish_delivery: np.ndarray
    spine_ids: list
    spine_flags: list
    spine_grouping: list
    spine_positions: list
    spine_GluSnFr_dFoF: np.ndarray
    spine_GluSnFr_processed_dFoF: np.ndarray
    spine_GluSnFr_activity: np.ndarray
    spine_GluSnFr_floored: np.ndarray
    spine_calcium_dFoF: np.ndarray
    spine_calcium_processed_dFoF: np.ndarray
    spine_calcium_activity: np.ndarray
    spine_calcium_floored: np.ndarray
    dendrite_calcium_dFoF: np.ndarray
    dendrite_calcium_processed_dFoF: np.ndarray
    dendrite_calcium_activity: np.ndarray
    dendrite_calcium_floored: np.ndarray
    spine_volume: list
    corrected_spine_volume: list
    movement_spines: list
    reward_movement_spines: list
    silent_spines: list
    reward_silent_spines: list
    spine_movement_activity: dict
    spine_reward_activity: dict
    movement_dendrites: list
    reward_movement_dendrites: list
    silent_dendrites: list
    reward_silent_dendrites: list
    dendrite_movement_activity: dict
    dendrite_reward_activity: dict
    imaging_parameters: dict


@dataclass
class Structural_Spine_Data:
    """Dataclass for storing only structural related information"""

    mouse_id: str
    session: str
    date: str
    spine_ids: list
    spine_flags: list
    spine_positions: list
    spine_pixel_intensity: list
    dend_segment_intensity: list
    spine_volume: list
    corrected_spine_pixel_intensity: list
    corrected_dend_segment_intensity: list
    corrected_spine_volume: list
    imaging_parameters: dict

