"""Module for analyzing and summarizing the main metrics of lever press behavior
    for a single mouse across all sessions. Stores outputs as dataclasses"""


import os
import re
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Behavior.process_lever_behavior import process_lever_behavior
from Lab_Analyses.Behavior.summarize_lever_behavior import summarize_lever_behavior
from Lab_Analyses.Utilities.check_file_exists import get_existing_files
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def analyze_mouse_lever_behavior(
    mouse_id,
    path,
    imaged,
    exp=None,
    save=False,
    save_suffix=None,
    reanalyze=False,
    ignore_dir=(),
    press_len=3,
):
    """Function to analyze the lever press of all the sessions for a single
        mouse
    
    INPUT PARAMETERS
        mouse_id - str specifying what the mouse's id is
        
        path - str of the path to the directory containing all of the behavioral
                data for a given mouse, with each session in a subdirectory
                
        imaged - boolean list specifying if the session was also imaged or not
        
        exp - str containing descriptino of the experiment

        save - boolean specifying if the data is to be saved

        save_suffix - list of str to add additional descriptor to file name 
                        each file
        
        reanalyze - boolean specifying if this is to reanalyze data

        ignore_dir - tuple of strings specifying directories/days to ignore for analysis

        press_len - float specifying how long of the lever press you wish to correlate

    OUTPUT PARAMETERS

    """
    print(f"----------------------------\nAnalyzing Mouse {mouse_id}")

    # Parent path where analyzed data is stored
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"

    # Check if file already exists
    if reanalyze is False:
        try:
            exists = get_existing_files(
                path=os.path.join(initial_path, mouse_id, "behavior"),
                name="all_lever_data",
                includes=True,
            )
            if exists is not None:
                mouse_lever_data = load_pickle(
                    fname_list=[exists.replace(".pickle", "")],
                    path=os.path.join(initial_path, mouse_id, "behavior"),
                )
                print(" - Loading previously analyzed data")
                return mouse_lever_data[0]

        except FileNotFoundError:
            pass

    # Move to the directory containing all the behavior data for the mouse
    os.chdir(path)
    directories = [x[0] for x in os.walk(".")]
    directories = sorted(directories)
    directories = directories[1:]

    if ignore_dir:
        for dir in ignore_dir:
            directories.remove(dir)

    # Initialize sessions list
    sessions = np.linspace(1, len(imaged), len(imaged), dtype=int)
    sess_names = [os.path.basename(x) for x in directories]

    # Check is save suffix is a list or not
    if save_suffix is None:
        save_suffix = [None for x in sessions]
    else:
        if len(save_suffix) == 1:
            save_suffix = [save_suffix for x in sessions]

    # Process lever press data for each session
    files = []
    for directory, im, sess, name, suffix, in zip(
        directories, imaged, sessions, sess_names, save_suffix
    ):
        print(f" - Processing session {sess}", end="\r")
        p_file = get_processed_data(
            mouse_id, directory, im, name, save, suffix, initial_path, reanalyze
        )
        files.append(p_file)
    print("")

    # Summarize lever press data for each session
    summarized_data = []
    for file, sess, name, suffix in zip(files, sessions, sess_names, save_suffix):
        print(f" - Summarizing session {sess}", end="\r")
        summed_data = get_summarized_data(
            file, name, save, suffix, initial_path, reanalyze
        )
        summarized_data.append(summed_data)

    # Pull out relevant data to store together
    # Initialize the dataclass object
    mouse_lever_data = Mouse_Lever_Data(
        mouse_id=mouse_id,
        experiment=exp,
        sessions=sessions,
        trials=[],
        rewards=[],
        used_trials=[],
        all_movements=[],
        corr_movements=[],
        average_movements=[],
        reaction_time=[],
        cue_to_reward=[],
        move_at_start_faults=[],
        move_duration_before_cue=[],
        number_movements_during_ITI=[],
        fraction_ITI_moving=[],
        correlation_matrix=np.nan,
        within_sess_corr=np.nan,
        across_sess_corr=np.nan,
    )

    for data in summarized_data:
        if data is not None:
            mouse_lever_data.trials.append(data.trials)
            mouse_lever_data.rewards.append(data.rewards)
            mouse_lever_data.used_trials.append(data.used_trial)
            mouse_lever_data.all_movements.append(data.movement_matrix)
            mouse_lever_data.corr_movements.append(data.corr_matrix)
            mouse_lever_data.average_movements.append(data.movement_avg)
            mouse_lever_data.reaction_time.append(data.avg_reaction_time)
            mouse_lever_data.cue_to_reward.append(data.avg_cue_to_reward)
            mouse_lever_data.move_at_start_faults.append(data.move_at_start_faults)
            mouse_lever_data.move_duration_before_cue.append(
                data.move_duration_before_cue
            )
            mouse_lever_data.number_movements_during_ITI.append(
                data.number_of_movements_during_ITI
            )
            mouse_lever_data.fraction_ITI_moving.append(data.fraction_ITI_spent_moving)
        else:
            mouse_lever_data.trials.append(np.nan)
            mouse_lever_data.rewards.append(np.nan)
            mouse_lever_data.used_trials.append(np.nan)
            mouse_lever_data.all_movements.append(np.nan)
            mouse_lever_data.corr_movements.append(np.nan)
            mouse_lever_data.average_movements.append(np.nan)
            mouse_lever_data.reaction_time.append(np.nan)
            mouse_lever_data.cue_to_reward.append(np.nan)
            mouse_lever_data.move_at_start_faults.append(np.nan)
            mouse_lever_data.move_duration_before_cue.append(np.nan)
            mouse_lever_data.number_movements_during_ITI.append(np.nan)
            mouse_lever_data.fraction_ITI_moving.append(np.nan)

    mouse_lever_data.correlation_matrix = correlate_lever_press(
        mouse_lever_data.corr_movements, length=press_len
    )
    mouse_lever_data.within_sess_corr = mouse_lever_data.correlation_matrix.diagonal()
    mouse_lever_data.across_sess_corr = mouse_lever_data.correlation_matrix.diagonal(
        offset=1
    )

    # Save section
    if save is True:
        mouse_id = file.mouse_id
        # Set path
        save_path = os.path.join(initial_path, mouse_id, "behavior")
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        # Make file name
        save_name = f"{mouse_id}_all_lever_data"
        # Save the data as a pickle file
        save_pickle(
            save_name, mouse_lever_data, save_path,
        )

    print(f"\nDone Analyzing Mouse {mouse_id}\n----------------------------")

    return mouse_lever_data


def correlate_lever_press(movement_matrices, length):
    """Helper function to correlate movements within and across sessions for a single mouse
    
        INPUT PARAMETERS
            movement_matricies - 2d np.array of movements
        
            length - float specifying how long of the press you wish to correlate"""

    # Initialize the correlation matrix
    length = int(length * 1000)
    correlation_matrix = np.zeros((len(movement_matrices), len(movement_matrices)))
    correlation_matrix[:] = np.nan

    # Perform pairwise correlations
    for i_idx, i in enumerate(movement_matrices):
        if type(i) is not np.ndarray:
            continue
        for j_idx, j in enumerate(movement_matrices):
            if type(j) is not np.ndarray:
                continue
            corr = correlate_btw_sessions(i, j, length)
            if i_idx == j_idx:
                correlation_matrix[i_idx, j_idx] = corr
            else:
                correlation_matrix[i_idx, j_idx] = corr
                correlation_matrix[j_idx, i_idx] = corr

    return correlation_matrix


def correlate_btw_sessions(A, B, length):
    """Helper function to perform pairwise correlations between movements from two
        different sessions. This is a vectorized approach for faster run time
        
        INPUT PARAMETERS
            A - np.array of movement matrix of the first session
            
            B - np.array of movement matrix fo the second session
            
        OUTPUT PARAMETER
            across_corr - float of the median pairwise correlation for all movements
                          in session A with those of session B
    
    """
    # Transpose movement rows to columns
    A = A[:, :length]
    B = B[:, :length]
    A = A.T
    B = B.T

    # Get number of rows in either A or B (should be same)
    n = B.shape[0]
    # Store column-wise in A and B
    sA = A.sum(axis=0)
    sB = B.sum(axis=0)

    # Vectorize and broadcast the A and B
    p1 = n * np.dot(B.T, A)
    p2 = sA * sB[:, None]
    p3 = n * ((B ** 2).sum(axis=0)) - (sB ** 2)
    p4 = n * ((A ** 2).sum(axis=0)) - (sA ** 2)

    # Compute pairwise Pearsons Correlation Coefficient as 2D array
    pcorr = (p1 - p2) / np.sqrt(p4 * p3[:, None])

    across_corr = np.nanmedian(pcorr)

    return across_corr


def get_processed_data(
    mouse_id, directory, im, sname, save, suffix, save_path, reanalyze
):
    """Helper function to get the processed data"""
    # Check if file already exists
    if reanalyze is False:
        try:
            exists = get_existing_files(
                path=os.path.join(save_path, mouse_id, "behavior", sname),
                name="processed_lever_data",
                includes=True,
            )
            if exists is not None:
                p_file = load_pickle(
                    fname_list=[exists.replace(".pickle", "")],
                    path=os.path.join(save_path, mouse_id, "behavior", sname),
                )
                return p_file[0]

        except FileNotFoundError:
            pass

    # Run processing if it is reanalyzing or file does not already exist
    fnames = os.listdir(directory)
    xsg_files = [file for file in os.listdir(directory) if file.endswith(".xsglog")]
    if len(xsg_files) == 0:
        p_file = None
        return p_file
    # Check if dispatcher file is present
    d_present = False
    for fname in fnames:
        if "data_@lever2p" in fname:
            d_present = True

    if d_present is False:
        p_file = None
        return p_file

    p_file = process_lever_behavior(
        mouse_id, directory, imaged=im, save=save, save_suffix=suffix
    )

    return p_file


def get_summarized_data(file, sname, save, suffix, save_path, reanalyze):
    """Helper function to get the summarized data"""
    # Check if file already exists
    if file is None:
        summed_data = None
        return None
    if reanalyze is False:
        try:
            exists = get_existing_files(
                path=os.path.join(save_path, file.mouse_id, "behavior", sname),
                name="summarized_lever_data",
                includes=True,
            )
        except FileNotFoundError:
            pass
        if exists is not None:
            summed_data = load_pickle(
                fname_list=[exists.replace(".pickle", "")],
                path=os.path.join(save_path, file.mouse_id, "behavior", sname),
            )
            return summed_data[0]

    # Summarize data if it is reanalyzing or file does not already exist
    if file is None:
        summed_data = None
        return summed_data

    summed_data = summarize_lever_behavior(file, save=save, save_suffix=suffix)

    return summed_data


# --------------------------------DATACLASS USED--------------------------------------------
@dataclass
class Mouse_Lever_Data:
    """Dataclass for storing processed lever press behavior data across
       all sessions for a single mouse"""

    mouse_id: str
    experiment: str
    sessions: list
    trials: list
    rewards: list
    used_trials: list
    all_movements: list
    corr_movements: list
    average_movements: list
    reaction_time: list
    cue_to_reward: list
    move_at_start_faults: list
    move_duration_before_cue: list
    number_movements_during_ITI: list
    fraction_ITI_moving: list
    correlation_matrix: np.ndarray
    within_sess_corr: np.ndarray
    across_sess_corr: np.ndarray
