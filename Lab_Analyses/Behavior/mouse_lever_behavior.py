""" Module to analyze and summarize the main metrics of lever press behavior
    for a single mouse across all sessions. Stores output as a dataclass.
    
    CREATOR - William (Jake) Wright 3/6/2022"""

import os
from dataclasses import dataclass

import numpy as np
from Lab_Analyses.Behavior import process_lever_behavior as plb
from Lab_Analyses.Behavior import summarize_lever_behavior as slb


def analyze_mouse_lever_behavior(id_, path, imaged, sessions=None, exp=None):
    """Function to analyze the lever press behavior all the 
        sessions for a single mouse.
        
        INPUT PARAMETERS
            id - string specifying what the mouses ID is

            path - string of the path to the directory containing all the behavioral
                    data for a given mouse, with each session in a subdirectory

            imaged - boolean specifying if the session was also imaged or not

            exp - string containing description of experiment

            sessions - a list of numbers indicating the session number for each 
                       input file. Optional. If no sessions are provide it is
                       assumeed that the files are in correct order

        OUTPUT PARAMETERS
    
    """
    ## Move to the directory containing all the behavioral data for the mouse
    os.chdir(path)
    directories = [x[0] for x in os.walk(".")]
    directories = sorted(directories)
    directories = directories[1:]

    if sessions is None:
        sessions = np.linspace(1, len(directories), len(directories), dtype=int)
    else:
        sessions = sessions
    print(sessions)
    # Process lever data for each session
    files = []
    for directory, im in zip(directories, imaged):
        print(directory)
        fnames = os.listdir(directory)
        for fname in fnames:
            if "data_@lever2p" not in fname:
                p_file = None
            else:
                p_file = plb.process_lever_press_behavior(directory, im)
        files.append(p_file)

    # Summarize lever press behavior for each session
    summarized_data = []
    for file in files:
        if file is None:
            summarized_data.append(None)
        else:
            summed_data = slb.summarize_lever_behavior(file)
        summarized_data.append(summed_data)

    # Sort the summarized session to be in correct order based on sessions
    zipped_data = zip(sessions, summarized_data)
    sorted_data = sorted(zipped_data)
    ts = zip(*sorted_data)
    sessions, summarized_data = [list(t) for t in ts]

    # Pull out relevant data to store
    mouse_lever_data = Mouse_Lever_Data(
        mouse_id=id_,
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

    return mouse_lever_data


def correlate_lever_press(movement_matrices):
    """Function to correlate movements within and across sessions for a single mouse
        
        INPUT PARAMETERS
            movement_matrices - list containing np.arrays of all the rewarded movements. 
            
        OUTPUT PARAMETERS
            correlation_matrix - np.array of the median pairwise movement correlations for
                                each pair of sessions
    """

    # Initialize the correlation matrix
    correlation_matrix = np.zeros((len(movement_matrices), len(movement_matrices)))

    # Perform pairwise correlations
    for i_idx, i in enumerate(movement_matrices):
        for j_idx, j in enumerate(movement_matrices):
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

    across_corr = np.median(pcorr)

    return across_corr


# -------------------------------------------------------------------------
# ---------------------------DATACLASSES USED------------------------------
# -------------------------------------------------------------------------


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
