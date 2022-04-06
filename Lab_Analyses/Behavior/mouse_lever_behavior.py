"""Module for analzying and summarizing the main metrics of lever press behavior
    for a single mouse across all sessions. Stores output as a dataclass"""

import os
import re
from dataclasses import dataclass

import numpy as np
from Lab_Analyses.Behavior.process_lever_behavior import process_lever_behavior
from Lab_Analyses.Behavior.summarize_lever_behavior import summarize_lever_behavior
from Lab_Analyses.Utilities.check_file_exists import check_file_exists
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def analyze_mouse_lever_behavior(
    mouse_id,
    path,
    imaged,
    sessions=None,
    exp=None,
    save=False,
    save_suffix=None,
    reanalyze_suffix=None,
):
    """Function to analyze the lever press behavior of all the sessions for a
        single mouse
        
        INPUT PARAMETERS
            mouse_id - string specifying what the mouse's ID is
            
            path - string of the path to the directory containing all of the behavioral data
                    for a given mouse, with each session in a subdirectory
                    
            imaged - boolean list specifying if the session was also imaged or not
            
            exp - string containing description of experiment
            
            sessions - a list of numbers indicating the session number for each 
                        input file. Optional. If no sessions are provided it is
                        assumed that the files are in the correct order
            
            save - boolean specifying if the data is to be saved

            save_suffix - list of str to add additonal descriptor to file name for each file

            reanalyze_suffix - str to add to descriptor for reanalyzed datasets
                        
        OUTPUT PARAMETERS
            """
    # Parent path where analyzed data is stored
    save_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"

    # Move to the directory containing all the behvior data for the mouse
    print(f"----------------------------\nAnalyzing Mouse {mouse_id}")
    os.chdir(path)
    directories = [x[0] for x in os.walk(".")]
    directories = sorted(directories)
    directories = directories[1:]

    # Make sessions if it is not input
    if sessions is None:
        sessions = np.linspace(1, len(directories), len(directories), dtype=int)

    # Process lever data for each session
    files = []
    for directory, im, sess, suffix in zip(directories, imaged, sessions, save_suffix):
        print(f" - Processing session {sess}", end="\r")
        fnames = os.listdir(directory)
        xsg_files = [file for file in os.listdir(directory) if file.endswith(".xsglog")]
        if len(xsg_files) == 0:
            p_file = None
            files.append(p_file)
            continue
        for fname in fnames:
            if "data_@lever2p" not in fname:
                p_file = None
            else:
                p_file = process_lever_behavior(
                    directory, im, save=save, save_suffix=suffix
                )
        files.append(p_file)

    print("")

    # Summarize lever press behavior for each session
    summarized_data = []
    for file, sess, suffix in zip(files, sessions, save_suffix):
        print(f" - Summarizing session {sess}", end="\r")
        if file is None:
            summarized_data.append(None)
        else:
            summed_data = summarize_lever_behavior(file, save=save, save_suffix=suffix)
        summarized_data.append(summed_data)

    # Sort the summarized sessions to be in the corred order based on sessions
    zipped_data = zip(sessions, summarized_data)
    sorted_data = sorted(zipped_data)
    ts = zip(*sorted_data)
    sessions, summarized_data = [list(t) for t in ts]

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
            mouse_lever_data.average_movements.append(np.nan)
            mouse_lever_data.reaction_time.append(np.nan)
            mouse_lever_data.cue_to_reward.append(np.nan)
            mouse_lever_data.move_at_start_faults.append(np.nan)
            mouse_lever_data.move_duration_before_cue.append(np.nan)
            mouse_lever_data.number_movements_during_ITI.append(np.nan)
            mouse_lever_data.fraction_ITI_moving.append(np.nan)

    mouse_lever_data.correlation_matrix = correlate_lever_press(
        mouse_lever_data.all_movements
    )
    mouse_lever_data.within_sess_corr = mouse_lever_data.correlation_matrix.diagonal()
    mouse_lever_data.across_sess_corr = mouse_lever_data.correlation_matrix.diagonal(
        offset=1
    )

    # Save section
    if save is True:
        mouse_id = file.mouse_id
        # Make mouse folder to for its data if it doesn't already exist
        mouse_path = os.path.join(save_path, mouse_id)
        if not os.path.isdir(mouse_path):
            os.mkdir(mouse_path)
        # Check if mouse has path for behavioral data
        behavior_path = os.path.join(mouse_path, "behavior")
        if not os.isdir(behavior_path):
            os.mkdir(behavior_path)
        # Make file name
        if save_suffix is not None:
            save_name = f"{mouse_id}_all_lever_data_{reanalyze_suffix}"
        else:
            save_name = f"{mouse_id}_all_lever_data"
        # Save the data as a pickle file
        save_pickle(
            save_name, mouse_lever_data, behavior_path,
        )

    print(f"\nDone Analyzing Mouse {mouse_id}\n----------------------------")

    return mouse_lever_data


def correlate_lever_press(movement_matrices):
    """Helper function to correlate movements within and across sessions for a single mouse"""

    # Initialize the correlation matrix
    correlation_matrix = np.zeros((len(movement_matrices), len(movement_matrices)))
    correlation_matrix[:] = np.nan

    # Perform pairwise correlations
    for i_idx, i in enumerate(movement_matrices):
        if type(i) is not np.ndarray:
            continue
        for j_idx, j in enumerate(movement_matrices):
            if type(j) is not np.ndarray:
                continue
            corr = correlate_btw_sessions(i, j)
            if i_idx == j_idx:
                correlation_matrix[i_idx, j_idx] = corr
            else:
                correlation_matrix[i_idx, j_idx] = corr
                correlation_matrix[j_idx, i_idx] = corr

    return correlation_matrix


def correlate_btw_sessions(A, B):
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
