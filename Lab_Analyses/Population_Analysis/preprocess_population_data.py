import os
import re
from copy import copy
from dataclasses import dataclass

import numpy as np
import scipy.signal as sysignal
from scipy.ndimage import uniform_filter1d

from Lab_Analyses.Behavior.align_lever_behavior_suite2p import (
    align_lever_behavior_suite2p,
)
from Lab_Analyses.Utilities.check_file_exists import get_existing_files
from Lab_Analyses.Utilities.deconvolve_calcium import oasis
from Lab_Analyses.Utilities.event_detection import event_detection
from Lab_Analyses.Utilities.get_dFoF import get_dFoF
from Lab_Analyses.Utilities.movement_responsiveness_v2 import movement_responsiveness
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def organize_population_data(
    mouse_id,
    roi_match=False,
    sensor="RCaMP2",
    zoom_factor=2,
    reprocess=True,
    save=False,
):
    """Function to handle the initial processing and organization of population data
        extracted using Suite2p. 
        
        INPUT PARAMETERS
            mouse_id - str specifying what the mouse's id is
            
            roi_match - boolean specifying whether or not to take only cells tracked
                        across all sessions
            
            reprocess - boolean specifying whether to reprocess the data or try to load it
            
            save - boolean specifying if the data is to be saved or not
    """

    print(
        f"--------------------------------------------------\nProcessing Mouse {mouse_id}"
    )
    if save:
        align_save = "both"
    else:
        align_save = None
    # Set up the paths to load the data from
    suite2p_path = r"D:\Suite2P_data\paAIP2"
    mouse_path = os.path.join(
        r"C:\Users\Jake\Desktop\Analyzed_data\individual", mouse_id
    )
    behavior_path = os.path.join(mouse_path, "behavior")
    imaging_path = os.path.join(suite2p_path, mouse_id)

    # Identify the imaging sessions
    sessions = next(os.walk(imaging_path))[1]
    # Reorder if periods
    if "Early" and "Late" and "Middle" in sessions:
        sessions = ["Early", "Middle", "Late"]

    # Set up containers for the data files
    imaging_datasets = []
    behavior_datasets = []
    # Load, process, and aligne the data files for each session
    for session in sessions:
        # First check if aligned data exists
        align_save_path = os.path.join(mouse_path, "aligned_data", session)
        if reprocess is False:
            print(f"--Checking if {session} aligned data exists...")
            behavior_exists = get_existing_files(
                path=align_save_path, name="aligned_behavior", includes=True,
            )
            imaging_exists = get_existing_files(
                path=align_save_path, name="aligned_activity", includes=True,
            )
            if behavior_exists is not None and imaging_exists is not None:
                print(f"--Loading aligned datasets")
                aligned_behavior = load_pickle([behavior_exists], path=align_save_path)[
                    0
                ]
                aligned_imaging = load_pickle([imaging_exists], path=align_save_path)[0]
                imaging_datasets.append(aligned_imaging)
                behavior_datasets.append(aligned_behavior)
                continue

        # Process data if not loading
        print(f"--- preprocessing {session} datasets")
        session_path = os.path.join(imaging_path, session)
        fnames = next(os.walk(session_path))[2]
        ## Get names of relevant files
        fluo_fname = os.path.join(
            session_path, [x for x in fnames if "F" in x and "Fneu" not in x][0]
        )
        ops_fname = os.path.join(session_path, [x for x in fnames if "ops" in x][0])
        iscell_fname = os.path.join(
            session_path, [x for x in fnames if "iscell" in x][0]
        )
        stat_fname = os.path.join(session_path, [x for x in fnames if "stat" in x][0])
        ## Load the activity files
        raw_fluoro = np.load(fluo_fname)
        ops_file = np.load(ops_fname, allow_pickle=True).item()
        iscell = np.load(iscell_fname)[:, 0].astype(int)
        iscell = iscell.astype(bool)
        stat_file = np.load(stat_fname, allow_pickle=True)

        # Refine the fluorescence with only cell data
        if roi_match:
            return "Haven't coded for roi matching"
        else:
            cells = iscell

        fluoro = raw_fluoro[cells].T

        # Calculate dFoF and processed dFoF
        dFoF = np.zeros(fluoro.shape)
        processed_dFoF = np.zeros(fluoro.shape)
        for i in range(fluoro.shape[1]):
            f, pf, _ = get_dFoF(
                data=fluoro[:, i],
                sampling_rate=int(ops_file["fs"]),
                smooth_window=0.5,
                bout_separations=None,
                artifact_frames=None,
            )
            dFoF[:, i] = f
            processed_dFoF[:, i] = pf

        # Perform event detection
        activity, floored, _ = event_detection(
            processed_dFoF,
            threshold=3,
            lower_threshold=1,
            lower_limit=0,
            sampling_rate=int(ops_file["fs"]),
            filt_poly=3,
            sec_smooth=1,
        )

        if sensor == "GCaMP6f":
            tau = 0.7
        elif sensor == "GCaMP6s":
            tau = 1.5
        elif sensor == "GCaMP7b":
            tau = 1.2
        elif sensor == "RCaMP2":
            tau = 1.5
        deconvolved = oasis(
            fluo=fluoro, batch_size=500, tau=tau, sampling_rate=int(ops_file["fs"]),
        )

        deconvolved = uniform_filter1d(deconvolved, 6, axis=0)

        # process spikes
        smooth_int = int(ops_file["fs"])
        if not smooth_int % 2:
            smooth_int = smooth_int + 1
        processed_deconvolved = np.zeros(deconvolved.shape)
        for i in range(deconvolved.shape[1]):
            smooth_spikes = sysignal.savgol_filter(deconvolved[:i], smooth_int, 3)
            processed_deconvolved[:, i] = smooth_spikes

        # Store important prameters
        parameters = {
            "Matched": roi_match,
            "Sampling Rate": int(ops_file["fs"]),
            "Sensor": sensor,
            "tau": tau,
            "Zoom": zoom_factor,
            "FOV Dimension": (ops_file["Lx"], ops_file["Ly"]),
        }
        ## Get positions
        stats = stat_file[cells]
        roi_positions = np.array([x["med"] for x in stats])

        suite2p_data = Temp_Suite2P_activity(
            fluorescence=fluoro,
            dFoF=dFoF,
            processed_dFoF=processed_dFoF,
            activity_trace=activity,
            floored_trace=floored,
            deconvolved_spikes=deconvolved,
            processed_deconvolved_spikes=processed_deconvolved,
            parameters=parameters,
            roi_positions=roi_positions,
        )

        # Load matching behavioral file
        day = re.search("[0-9]{6}", os.path.basename(fluo_fname)).group()
        matched_b_path = os.path.join(behavior_path, day)
        b_fnames = next(os.walk(matched_b_path))[2]
        b_fname = [x for x in b_fnames if "processed_lever_data" in x][0]
        b_fname = os.path.join(matched_b_path, b_fname)
        behavior_data = load_pickle([b_fname])[0]
        behavior_data.sess_name = session

        # Align the imaging and behavioral data
        print(f"--- aligning {session} datasets")
        aligned_behavior, aligned_imaging = align_lever_behavior_suite2p(
            behavior_data=behavior_data,
            imaging_data=suite2p_data,
            save=align_save,
            save_path=align_save_path,
            save_suffix={"behavior": None, "imaging": None},
        )
        behavior_datasets.append(aligned_behavior)
        imaging_datasets.append(aligned_imaging)

    # Perform further processing and organizing
    for session, behavior, imaging in zip(
        sessions, behavior_datasets, imaging_datasets
    ):
        save_path = os.path.join(mouse_path, "population_data")
        # Check reprocessing
        if reprocess is False:
            print(f"--Checking if {session} population data exists...")
            exists = get_existing_files(
                path=save_path, name=f"{session}_dual_spine_data", includes=True,
            )
            if exists is not None:
                print(f"--{session} data already organized")
                continue

        # Organize the data if it doesn't exists
        print(f"--- organizing {session} datasets")
        # Pull relevant behavioral data
        ## Lever and cue information
        date = behavior.date
        imaging_parameters = imaging.imaging_parameters
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
        ## reward and punish information
        rwd_delivery = copy(behavior.result_delivery)
        pun_delivery = copy(behavior.result_delivery)
        for i, outcome in enumerate(behavior.result):
            if outcome == 0:
                rwd_delivery[i] = np.zeros(len(rwd_delivery[i]))
            else:
                pun_delivery[i] = np.zeros(len(pun_delivery[i]))
        reward_delivery = np.concatenate(rwd_delivery)
        punish_delivery = np.concatenate(pun_delivery)

        # Pull activity data
        ## Join all trials together
        aligned_fluorescence = np.vstack(imaging.fluorescence)
        aligned_dFoF = np.vstack(imaging.dFoF)
        aligned_processed_dFoF = np.vstack(imaging.processed_dFoF)
        aligned_activity = np.vstack(imaging.activity_trace)
        aligned_floored = np.vstack(imaging.floored_trace)
        aligned_spikes = np.vstack(imaging.spikes)
        aligned_processed_spikes = np.vstack(imaging.processed_spikes)

        cell_positions = imaging.roi_positions

        # Classify movement responsiveness
        movement_cells, silent_cells, _ = movement_responsiveness(
            aligned_processed_dFoF, lever_active, permutations=1000, percentile=99
        )
        reward_movement_cells, reward_silent_cells, _ = movement_responsiveness(
            aligned_processed_dFoF,
            rewarded_movement_binary,
            permutations=1000,
            percentile=99,
        )
        movement_cells_spikes, silent_cells_spikes, _ = movement_responsiveness(
            aligned_spikes, lever_active, permutations=1000, percentile=99
        )
        (
            reward_movement_cells_spikes,
            reward_silent_cells_spikes,
            _,
        ) = movement_responsiveness(
            aligned_spikes, rewarded_movement_binary, permutations=1000, percentile=99
        )

        # Store the data in the dataclass
        population_data = Population_Data(
            mouse_id=behavior.mouse_id,
            session=session,
            date=date,
            imaging_parameters=imaging_parameters,
            time=time,
            lever_force_resample=lever_force_resample,
            lever_force_smooth=lever_force_smooth,
            lever_velocity_envelop=lever_velocity_envelope,
            lever_active=lever_active,
            rewarded_movement_force=rewarded_movement_force,
            rewarded_movement_binary=rewarded_movement_binary,
            binary_cue=binary_cue,
            reward_delivery=reward_delivery,
            punish_delivery=punish_delivery,
            cell_positions=cell_positions,
            fluorescence=aligned_fluorescence,
            dFoF=aligned_dFoF,
            processed_dFoF=aligned_processed_dFoF,
            estimated_spikes=aligned_spikes,
            processed_estimated_spikes=aligned_processed_spikes,
            activity_trace=aligned_activity,
            floored_trace=aligned_floored,
            movement_cells=movement_cells,
            silent_cells=silent_cells,
            reward_movement_cells=reward_movement_cells,
            reward_silent_cells=reward_silent_cells,
            movement_cells_spikes=movement_cells_spikes,
            silent_cells_spikes=silent_cells_spikes,
            reward_movement_cells_spikes=reward_movement_cells_spikes,
            reward_silent_cells_spikes=reward_silent_cells_spikes,
        )

        # Save section
        if save:
            # Setup the save path
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            # make the file name and save
            if roi_match:
                mname = "_matched"
            else:
                mname = ""
            fname = f"{population_data.mouse_id}_{session}{mname}_population_data"
            save_pickle(fname, population_data, save_path)


@dataclass
class Temp_Suite2P_activity:
    """Temporary dataclass to input suite2p data for alignment"""

    fluorescence: np.ndarray
    dFoF: np.ndarray
    processed_dFoF: np.ndarray
    activity_trace: np.ndarray
    floored_trace: np.ndarray
    deconvolved_spikes: np.ndarray
    processed_deconvolved_spikes: np.ndarray
    parameters: dict
    roi_positions: np.ndarray


@dataclass
class Population_Data:
    """Dataclass for containing all of the relevant behavioral and activity data
        for a single population imaging session"""

    # General variables
    mouse_id: str
    session: str
    date: str
    imaging_parameters: dict
    # Behavioral related variables
    time: np.ndarray
    lever_force_resample: np.ndarray
    lever_force_smooth: np.ndarray
    lever_velocity_envelop: np.ndarray
    lever_active: np.ndarray
    rewarded_movement_force: np.ndarray
    rewarded_movement_binary: np.ndarray
    binary_cue: np.ndarray
    reward_delivery: np.ndarray
    punish_delivery: np.ndarray
    # Activity related variables
    cell_positions: np.ndarray
    fluorescence: np.ndarray
    dFoF: np.ndarray
    processed_dFoF: np.ndarray
    estimated_spikes: np.ndarray
    processed_estimated_spikes: np.ndarray
    activity_trace: np.ndarray
    floored_trace: np.ndarray
    movement_cells: list
    silent_cells: list
    reward_movement_cells: list
    reward_silent_cells: list
    movement_cells_spikes: list
    silent_cells_spikes: list
    reward_movement_cells_spikes: list
    reward_silent_cells_spikes: list

