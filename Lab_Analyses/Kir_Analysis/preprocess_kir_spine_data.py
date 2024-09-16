import os
import re
from copy import copy
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Behavior.align_lever_behavior import align_lever_behavior
from Lab_Analyses.Utilities.check_file_exists import get_existing_files
from Lab_Analyses.Utilities.data_utilities import join_dictionaries, pad_array_to_length
from Lab_Analyses.Utilities.event_detection import event_detection
from Lab_Analyses.Utilities.get_dFoF import resmooth_dFoF
from Lab_Analyses.Utilities.movement_responsiveness_v2 import movement_responsiveness
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def organize_kir_spine_data(
    mouse_id,
    fov_type="apical",
    redetection=False,
    resmooth=False,
    reprocess=True,
    save=False,
    followup=True,
):
    """Function to handle the initial processing and organization of kir spine activity
    datasets.

        INPUT PARAMETERS
            mouse_id - str specifying the mouse to process

            fov_type - str specifying whether to process apical or basal FOVs

            redetection - boolean specifying whether to redo the event detection

            resmooth - boolean specifying whether to redo the dFoF smoothing

            reprocess - boolean specifying whether to reprocess or try to load
                        the data

            save - boolean specifying if the data is to be saved or not

            followup - boolean specifying if there is a dedicated followup
                        structural file to be included in the data

    """
    print(
        f"--------------------------------------------------\nProcessing Mouse {mouse_id}"
    )
    if save:
        align_save = "both"
    else:
        align_save = None

    # Set up the paths to load the data from
    initial_path = r"G:\Analyzed_data\individual"
    mouse_path = os.path.join(initial_path, mouse_id)
    imaging_path = os.path.join(mouse_path, "imaging")
    behavior_path = os.path.join(mouse_path, "behavior")

    # Identify the correct FOVs to process
    FOVs = next(os.walk(imaging_path))[1]
    FOVs = [x for x in FOVs if "FOV" in x]
    FOVs = [x for x in FOVs if fov_type in x]

    # Preprocess each FOV seperately
    for FOV in FOVs:
        print(f"- Preprocessing {FOV}")
        FOV_path = os.path.join(imaging_path, FOV)
        # Get the different imaging session periods
        sessions = next(os.walk(FOV_path))[1]
        # Reorder
        if ("Early" in sessions) and ("Late" in sessions) and ("Middle" in sessions):
            sessions = ["Early", "Middle", "Late"]

        # Setup containers for the data files
        GluSnFr_datasets = []
        followup_datasets = []
        behavior_datasets = []
        # Load and align the data files for each session
        for session in sessions:
            session_path = os.path.join(FOV_path, session)
            fnames = next(os.walk(session_path))[2]
            fnames = [x for x in fnames if "imaging_data" in x]
            align_save_path = os.path.join(mouse_path, "aligned_data", FOV, session)
            # check reprocessing
            if reprocess is False:
                print(f"-- Checking if {FOV} {session} aligned data exists...")
                behavior_exists = get_existing_files(
                    path=align_save_path,
                    name="aligned_behavior",
                    includes=True,
                )
                GluSnFr_exists = get_existing_files(
                    path=align_save_path,
                    name="GluSnFr_aligned",
                    includes=True,
                )
                if behavior_exists is not None and GluSnFr_exists is not None:
                    print(f"-- Loading aligned datasets")
                    aligned_behavior = load_pickle(
                        [behavior_exists], path=align_save_path
                    )[0]
                    aligned_GluSnFr = load_pickle(
                        [GluSnFr_exists],
                        path=align_save_path,
                    )[0]
                    GluSnFr_datasets.append(aligned_GluSnFr)
                    behavior_datasets.append(aligned_behavior)
                    if followup:
                        followup_fname = os.path.join(
                            session_path, [x for x in fnames if "structural" in x][0]
                        )
                        followup_datasets.append(load_pickle([followup_fname])[0])
                    else:
                        followup_datasets.append(None)
                    continue

            # Get the specific files
            ## Get activity data files
            GluSnFr_fname = os.path.join(
                session_path,
                [x for x in fnames if "GreenCh" in x and "structural" not in x][0],
            )
            GluSnFr_data = load_pickle([GluSnFr_fname])[0]
            ## Get the followup data file if exists
            if followup:
                followup_fname = os.path.join(
                    session_path, [x for x in fnames if "structural" in x][0]
                )
                followup_datasets.append(load_pickle([followup_fname])[0])
            else:
                followup_datasets.append(None)
            # Get the matching behavioral file
            day = re.search("[0-9]{6}", os.path.basename(GluSnFr_fname)).group()
            matched_b_path = os.path.join(behavior_path, day)
            b_fnames = next(os.walk(matched_b_path))[2]
            b_fname = [x for x in b_fnames if "processed_lever_data" in x][0]
            b_fname = os.path.join(matched_b_path, b_fname)
            behavior_data = load_pickle([b_fname])[0]
            #### Rewrited the session name (default is the date)
            behavior_data.sess_name = session

            # Align the imaging and behavioral data
            print(f"--- aligning {session} datasets")
            aligned_behavior, aligned_GluSnFr = align_lever_behavior(
                behavior_data,
                GluSnFr_data,
                save=align_save,
                save_path=align_save_path,
                save_suffix={"behavior": None, "imaging": "KIR"},
            )
            # Store the data
            GluSnFr_datasets.append(aligned_GluSnFr)
            behavior_datasets.append(aligned_behavior)

        # Find how much to pad the data by in order to match across days
        max_spine_num = np.max([len(x.ROI_ids["Spine"]) for x in GluSnFr_datasets])
        # Process and save each session
        for session, GluSnFr, behavior, followup_data in zip(
            sessions, GluSnFr_datasets, behavior_datasets, followup_datasets
        ):
            save_path = os.path.join(mouse_path, "spine_data", FOV)
            ## Check reprocessing
            if reprocess is False:
                print(f"-- Checking if {FOV} {session} spine data exists...")
                exists = get_existing_files(
                    path=save_path,
                    name=f"{FOV}_{session}_kir_spine_data",
                    includes=True,
                )
                if exists is not None:
                    print(f"-- {FOV} {session} data already organized")
                    continue
            # Organize the data if it doesn't exists
            print(f"--- organizing {session} datasets")
            # Pull relevant behavior data
            ## lever and cue information
            date = behavior.date
            imaging_parameters = GluSnFr.imaging_parameters
            time = np.concatenate(behavior.trial_time)
            lever_force_resample = np.concatenate(behavior.lever_force_resample_frames)
            lever_force_smooth = np.concatenate(behavior.lever_force_smooth_frames)
            lever_velocity_envelope = np.concatenate(
                behavior.lever_velocity_envelope_frames
            )
            lever_active = np.concatenate(behavior.lever_active_frames)
            rewarded_movement_force = np.concatenate(behavior.rewarded_movement_force)
            rewarded_movement_binary = np.concatenate(behavior.rewarded_movement_binary)
            binary_cue = np.concatenate(behavior.binary_cue)
            ## Reward and punishment information
            rwd_delivery = copy(behavior.result_delivery)
            pun_delivery = copy(behavior.result_delivery)
            for i, outcome in enumerate(behavior.result):
                if outcome == 0:
                    rwd_delivery[i] = np.zeros(len(rwd_delivery[i]))
                else:
                    pun_delivery[i] = np.zeros(len(pun_delivery[i]))
            reward_delivery = np.concatenate(rwd_delivery)
            punish_delivery = np.concatenate(pun_delivery)

            # Process and organize activity data
            ## Linearize activity datasets
            GluSnFr_dFoF = join_dictionaries(GluSnFr.dFoF)
            GluSnFr_processed_dFoF = join_dictionaries(GluSnFr.processed_dFoF)
            GluSnFr_activity = join_dictionaries(GluSnFr.activity_trace)
            GluSnFr_floored = join_dictionaries(GluSnFr.floored_trace)

            ## Spine related variables
            ### Structural and positional information
            spine_flags = GluSnFr.ROI_flags["Spine"]
            spine_positions = np.array(GluSnFr.ROI_positions["Spine"])
            spine_groupings = GluSnFr.imaging_parameters["Spine Groupings"]
            if len(spine_groupings) == 0:
                spine_groupings = list(range(GluSnFr.dFoF[0]["Spine"].shape[1]))
            spine_volume = np.array(GluSnFr.spine_volume)
            corrected_spine_volume = np.array(GluSnFr.corrected_spine_volume)
            if followup_data is not None:
                followup_flags = followup_data.ROI_flags["Spine"]
                followup_volume = np.array(followup_data.spine_volume)
                corrected_followup_volume = np.array(
                    followup_data.corrected_spine_volume
                )
                followup_positions = np.array(followup_data.ROI_positions["Spine"])
                followup_groupings = followup_data.parameters["Spine Groupings"]
                if len(followup_groupings) == 0:
                    followup_groupings = list(range(len(followup_flags)))
            else:
                followup_flags = [None for x in spine_flags]
                followup_volume = np.zeros(spine_volume.shape) * np.nan
                corrected_followup_volume = (
                    np.zeros(corrected_spine_volume.shape) * np.nan
                )
                followup_positions = np.zeros(spine_positions.shape) * np.nan
                followup_groupings = copy(spine_groupings)
            ### Activity
            spine_GluSnFr_dFoF = GluSnFr_dFoF["Spine"]
            if resmooth is True:
                spine_GluSnFr_processed_dFoF = resmooth_dFoF(
                    spine_GluSnFr_dFoF,
                    sampling_rate=imaging_parameters["Sampling Rate"],
                    smooth_window=0.5,
                )
            else:
                spine_GluSnFr_processed_dFoF = GluSnFr_processed_dFoF["Spine"]
            if redetection is True:
                spine_GluSnFr_activity, spine_GluSnFr_floored, _ = event_detection(
                    spine_GluSnFr_processed_dFoF,
                    threshold=2,
                    lower_threshold=1,
                    lower_limit=0.2,
                    sampling_rate=imaging_parameters["Sampling Rate"],
                    filt_poly=4,
                    sec_smooth=1,
                )
            else:
                spine_GluSnFr_activity = GluSnFr_activity["Spine"]
                spine_GluSnFr_floored = GluSnFr_floored["Spine"]

            ### Classify movements
            movement_spines, silent_spines, _ = movement_responsiveness(
                spine_GluSnFr_processed_dFoF, lever_active
            )
            (
                reward_movement_spines,
                reward_silent_spines,
                _,
            ) = movement_responsiveness(
                spine_GluSnFr_processed_dFoF, rewarded_movement_binary
            )

            # Pad spine data if it is not the longest
            if len(spine_flags) != max_spine_num:
                spine_flags = pad_spine_flags(spine_flags, max_spine_num)
                spine_positions = pad_array_to_length(spine_positions, max_spine_num)
                spine_volume = pad_array_to_length(spine_volume, max_spine_num)
                corrected_spine_volume = pad_array_to_length(
                    corrected_spine_volume, max_spine_num
                )
                spine_GluSnFr_dFoF = pad_array_to_length(
                    spine_GluSnFr_dFoF, max_spine_num, axis=1
                )
                spine_GluSnFr_processed_dFoF = pad_array_to_length(
                    spine_GluSnFr_processed_dFoF, max_spine_num, axis=1
                )
                spine_GluSnFr_activity = pad_array_to_length(
                    spine_GluSnFr_activity, max_spine_num, axis=1
                )
                spine_GluSnFr_floored = pad_array_to_length(
                    spine_GluSnFr_floored, max_spine_num, axis=1
                )
                movement_spines = pad_array_to_length(
                    movement_spines, max_spine_num, value=False
                )
                reward_movement_spines = pad_array_to_length(
                    reward_movement_spines, max_spine_num, value=False
                )
                silent_spines = pad_array_to_length(
                    silent_spines, max_spine_num, value=False
                )
                reward_silent_spines = pad_array_to_length(
                    reward_silent_spines, max_spine_num, value=False
                )
                followup_volume = pad_array_to_length(followup_volume, max_spine_num)
                corrected_followup_volume = pad_array_to_length(
                    corrected_followup_volume, max_spine_num
                )
                if followup_data is not None:
                    followup_flags = pad_spine_flags(followup_flags, max_spine_num)
                else:
                    followup_flags = pad_array_to_length(followup_flags, max_spine_num)

            # Store data in dataclass
            spine_data = Kir_Spine_Data(
                mouse_id=behavior.mouse_id,
                session=session,
                date=date,
                imaging_parameters=imaging_parameters,
                time=time,
                lever_force_resample=lever_force_resample,
                lever_force_smooth=lever_force_smooth,
                lever_velocity_envelope=lever_velocity_envelope,
                lever_active=lever_active,
                rewarded_movement_force=rewarded_movement_force,
                rewarded_movement_binary=rewarded_movement_binary,
                binary_cue=binary_cue,
                reward_delivery=reward_delivery,
                punish_delivery=punish_delivery,
                spine_flags=spine_flags,
                spine_groupings=spine_groupings,
                spine_positions=spine_positions,
                spine_volume=spine_volume,
                corrected_spine_volume=corrected_spine_volume,
                spine_GluSnFr_dFoF=spine_GluSnFr_dFoF,
                spine_GluSnFr_processed_dFoF=spine_GluSnFr_processed_dFoF,
                spine_GluSnFr_activity=spine_GluSnFr_activity,
                spine_GluSnFr_floored=spine_GluSnFr_floored,
                movement_spines=movement_spines,
                reward_movement_spines=reward_movement_spines,
                silent_spines=silent_spines,
                reward_silent_spines=reward_silent_spines,
                followup_flags=followup_flags,
                followup_volume=followup_volume,
                corrected_followup_volume=corrected_followup_volume,
                followup_positions=followup_positions,
                followup_groupings=followup_groupings,
            )

            # Save section
            if save:
                # Setup the save path
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                # make the file name and save
                fname = f"{spine_data.mouse_id}_{FOV}_{session}_kir_spine_data"
                save_pickle(fname, spine_data, save_path)


################################ HELPER FUNCTIONS #################################
def pad_spine_flags(spine_flags, length):
    """Function to pad spine flag lists to a given length

    INPUT PARAMETERS
        spine_flags - list of the spine flags

        length - int specifying the final length of the list

    OUTPUT PARAMETERS
        padded_spine_flags - list of the padded spine flags
    """
    padded_spine_flags = (spine_flags + length * [["Absent"]])[:length]
    return padded_spine_flags


############################ DATACLASS ##############################
@dataclass
class Kir_Spine_Data:
    """Dataclass containing all of the relevant behavioral and activity
    data for a single imaging session.
    """

    # General variables
    mouse_id: str
    session: str
    date: str
    imaging_parameters: dict
    # Behavior-related variables
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
    # Spine related variables
    spine_flags: list
    spine_groupings: list
    spine_positions: np.ndarray
    spine_volume: np.ndarray
    corrected_spine_volume: np.ndarray
    spine_GluSnFr_dFoF: np.ndarray
    spine_GluSnFr_processed_dFoF: np.ndarray
    spine_GluSnFr_activity: np.ndarray
    spine_GluSnFr_floored: np.ndarray
    movement_spines: list
    reward_movement_spines: list
    silent_spines: list
    reward_silent_spines: list
    # Followup data
    followup_flags: list
    followup_volume: np.ndarray
    corrected_followup_volume: np.ndarray
    followup_positions: np.ndarray
    followup_groupings: list
