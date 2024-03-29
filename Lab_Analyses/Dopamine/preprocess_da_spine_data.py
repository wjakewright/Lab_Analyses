import os
import re
from copy import copy
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Behavior.align_lever_behavior import align_lever_behavior
from Lab_Analyses.Utilities.check_file_exists import get_existing_files
from Lab_Analyses.Utilities.data_utilities import join_dictionaries
from Lab_Analyses.Utilities.event_detection import event_detection
from Lab_Analyses.Utilities.get_dFoF import resmooth_dFoF
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def organize_da_spine_data(
    mouse_id,
    fov_type="apical",
    resmooth=False,
    redetection=False,
    reprocess=True,
    save=False,
):
    """Function to organize dendritic dopamine imaging data

    INPUT PARAMETERS
        mouse_id - str specifying the mouse to process

        fov_type - str specifying whether to process apical or basal FOVs

        resmooth - boolean specifying whether to redo the dFoF smoothing

        reprocess - boolean specifying whether to reprocess or try to load
                    the data

        save - boolean specifying whether to save the data or not

    NOTE: Not coded to pad ROIs across days for new spines being added nor is it
            coded for followup structural data integration

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

        # Set up containers for data files
        DA_datasets = []
        behavior_datasets = []
        # Load and align the data files for each session
        for session in sessions:
            session_path = os.path.join(FOV_path, session)
            fnames = next(os.walk(session_path))[2]
            fnames = [x for x in fnames if "imaging_data" in x]
            align_save_path = os.path.join(mouse_path, "aligned_data", FOV, session)
            # Check reprocessing
            if reprocess is False:
                print(f"-- Checking if {FOV} {session} aligned data exists...")
                behavior_exists = get_existing_files(
                    path=align_save_path,
                    name="aligned_behavior",
                    includes=True,
                )
                DA_exists = get_existing_files(
                    path=align_save_path,
                    name="DA_aligned",
                    includes=True,
                )
                if behavior_exists is not None and DA_exists is not None:
                    print("-- Loading aligned datasets")
                    aligned_behavior = load_pickle(
                        [behavior_exists],
                        path=align_save_path,
                    )[0]
                    aligned_DA = load_pickle(
                        [DA_exists],
                        path=align_save_path,
                    )[0]
                    DA_datasets.append(aligned_DA)
                    behavior_datasets.append(aligned_behavior)

                    continue

            # Get the specific files
            ## Get the activity data files
            DA_fname = os.path.join(
                session_path, [x for x in fnames if "GreenCh" in x][0]
            )
            DA_data = load_pickle([DA_fname])[0]
            # Get the matching behavioral file
            day = re.search("[0-9]{6}", os.path.basename(DA_fname)).group()
            matched_b_path = os.path.join(behavior_path, day)
            b_fnames = next(os.walk(matched_b_path))[2]
            b_fname = [x for x in b_fnames if "processed_lever_data" in x][0]
            b_fname = os.path.join(matched_b_path, b_fname)
            behavior_data = load_pickle([b_fname])[0]
            #### Rewrite the session name (default is the data)
            behavior_data.sess_name = session

            # Align the imaging and behavioral data
            print(f"--- aligning {session} datasets")
            aligned_behavior, aligned_DA = align_lever_behavior(
                behavior_data,
                DA_data,
                save=align_save,
                save_path=align_save_path,
                save_suffix={"behavior": None, "imaging": "DA"},
            )
            # Store the data
            DA_datasets.append(aligned_DA)
            behavior_datasets.append(aligned_behavior)

        # Process and save each session
        for session, DA, behavior in zip(sessions, DA_datasets, behavior_datasets):
            save_path = os.path.join(mouse_path, "spine_data", FOV)
            ### Check reprocessing
            if reprocess is False:
                print(f"-- Checking if {FOV} {session} spine data exists")
                exists = get_existing_files(
                    path=save_path,
                    name=f"{FOV}_{session}_DA_spine_data",
                    includes=True,
                )
                if exists is not None:
                    print(f"-- {FOV} {session} data already organized")
                    continue
            # Organize the data if it doesn't exist
            print(f"--- organizing {session} datasets")
            # Pull relevant behavior data
            ## Lever and cue information
            date = behavior.date
            imaging_parameters = DA.imaging_parameters
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
            DA_dFoF = join_dictionaries(DA.dFoF)
            DA_processed_dFoF = join_dictionaries(DA.processed_dFoF)
            DA_activity = join_dictionaries(DA.activity_trace)
            DA_floored = join_dictionaries(DA.floored_trace)

            ## Spine related variables
            ### Structural and positional information
            spine_flags = DA.ROI_flags["Spine"]
            spine_positions = np.array(DA.ROI_positions["Spine"])
            spine_groupings = DA.imaging_parameters["Spine Groupings"]
            if len(spine_groupings) == 0:
                spine_groupings = list(range(DA.dFoF[0]["Spine"].shape[1]))
            spine_volume = np.array(DA.spine_volume)
            corrected_spine_volume = np.array(DA.corrected_spine_volume)
            ### Activity
            spine_DA_dFoF = DA_dFoF["Spine"]
            if resmooth is True:
                spine_DA_processed_dFoF = resmooth_dFoF(
                    spine_DA_dFoF,
                    sampling_rate=imaging_parameters["Sampling Rate"],
                    smooth_window=0.5,
                )
            else:
                spine_DA_processed_dFoF = DA_processed_dFoF["Spine"]
            if redetection is True:
                spine_DA_activity, spine_DA_floored, _ = event_detection(
                    spine_DA_processed_dFoF,
                    threshold=2,
                    lower_threshold=1,
                    lower_limit=0.2,
                    sampling_rate=imaging_parameters["Sampling Rate"],
                    filt_poly=4,
                    sec_smooth=1,
                )
            else:
                spine_DA_activity = DA_activity["Spine"]
                spine_DA_floored = DA_floored["Spine"]
            #### Dendrite poly ROIs
            dendrite_DA_positions = DA.ROI_positions["Dendrite"]
            dendrite_DA_dFoF = DA_dFoF["Dendrite Poly"]
            if resmooth is True:
                dendrite_DA_processed_dFoF = []
                for poly in dendrite_DA_dFoF["Dendrite Poly"]:
                    temp_dFoF = resmooth_dFoF(
                        poly,
                        sampling_rate=imaging_parameters["Sampling Rate"],
                        smooth_window=0.5,
                    )
                    dendrite_DA_processed_dFoF.append(temp_dFoF)
            else:
                dendrite_DA_processed_dFoF = DA_processed_dFoF["Dendrite Poly"]
            if redetection is True:
                dendrite_DA_activity = []
                dendrite_DA_floored = []
                for poly in dendrite_DA_processed_dFoF:
                    a, f = event_detection(
                        poly,
                        threshold=2,
                        lower_threshold=1,
                        lower_limit=0.2,
                        sampling_rate=imaging_parameters["Sampling Rate"],
                        filt_poly=4,
                        sec_smooth=1,
                    )
                    dendrite_DA_activity.append(a)
                    dendrite_DA_floored.append(f)
            else:
                dendrite_DA_activity = DA_activity["Dendrite Poly"]
                dendrite_DA_floored = DA_floored["Dendrite Poly"]

            # Store data in the dataclass
            spine_data = DA_Spine_Data(
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
                spine_DA_dFoF=spine_DA_dFoF,
                spine_DA_processed_dFoF=spine_DA_processed_dFoF,
                spine_DA_activity=spine_DA_activity,
                spine_DA_floored=spine_DA_floored,
                dendrite_positions=dendrite_DA_positions,
                dendrite_DA_dFoF=dendrite_DA_dFoF,
                dendrite_DA_processed_dFoF=dendrite_DA_processed_dFoF,
                dendrite_DA_activity=dendrite_DA_activity,
                dendrite_DA_floored=dendrite_DA_floored,
            )

            # Save section
            if save:
                # Set up the save path
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                # Make the file name and save
                fname = f"{spine_data.mouse_id}_{FOV}_{session}_DA_spine_data"
                save_pickle(fname, spine_data, save_path)


########################### DATACLASS #############################
@dataclass
class DA_Spine_Data:
    """Dataclass containing all of the relevant behavioral and activity
    data for a single imaging session
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
    # Spine-related variables
    spine_flags: list
    spine_groupings: list
    spine_positions: np.ndarray
    spine_volume: np.ndarray
    corrected_spine_volume: np.ndarray
    spine_DA_dFoF: np.ndarray
    spine_DA_processed_dFoF: np.ndarray
    spine_DA_activity: np.ndarray
    spine_DA_floored: np.ndarray
    # Dendrite-related variables
    dendrite_positions: list
    dendrite_DA_dFoF: list
    dendrite_DA_processed_dFoF: list
    dendrite_DA_activity: list
    dendrite_DA_floored: list
