""" Module to analyze and summarize the main metrics of lever press behavior
    for a single mouse across all sessions. Stores output as a dataclass."""

from dataclasses import dataclass
import numpy as np

from Lab_Analyses.Behavior import summarize_lever_behavior as slb


def analyze_mouse_lever_behavior(id, files, exp=None, sessions=None):
    """Function to analyze the lever press behavior all the 
        sessions for a single mouse.
        
        INPUT PARAMETERS
            id - string specifying what the mouses ID is

            files - a list of objects/files output from process_lever_behavior.py

            exp - string containing description of experiment
                        
            sessions - a list of numbers indicating the session number for each 
                       input file. Optional. If no sessions are provide it is
                       assumeed that the files are in correct order

        OUTPUT PARAMETERS
    
    """
    if sessions is None:
        sessions = np.arange(len(files)) + 1
    else:
        sessions = sessions

    # Summarize lever press behavior for each session
    summarized_data = []
    for file in files:
        summed_data = slb.summarize_lever_behavior(file)
        summarized_data.append(summed_data)
    
    # Pull out relevant data to store
    mouse_lever_data = Mouse_Lever_Data(mouse_id=id, experiment=exp, sessions=sessions, trials=[], rewards=[], used_trials=[], 
            all_movements=[], average_movements=[], reacton_time=[], cue_to_reward=[], move_at_start_faults=[],
            move_duration_before_cue=[], number_movements_during_ITI=[], fraction_ITI_moving=[], correlation_matrix=np.nan)
    for data in summarized_data:
        mouse_lever_data.trials.append(data.trials)
        mouse_lever_data.rewards.append(data.rewards)
        mouse_lever_data.used_trials.append(data.used_trial)
        mouse_lever_data.all_movements.append(data.movement_matrix)
        mouse_lever_data.average_movements.append(data.movement_avg)
        mouse_lever_data.reaction_time.append(data.avg_reaction_time)
        mouse_lever_data.move_at_start_faults.append(data.move_at_start_faults)
        mouse_lever_data.move_duration_before_cue.append(data.move_duration_before_cue)
        mouse_lever_data.number_movements_during_ITI.append(data.number_of_movements_during_ITI)
        mouse_lever_data.fraction_ITI_moving.append(data.fraction_ITI_spent_moving)
    
    correlation_matrix = correlate_lever_press(mouse_lever_data.all_movements)


def correlate_lever_press(movement_matrices):
    """Function to correlate movements within and across sessions for a single mouse
        
        INPUT PARAMETERS
            movement_matrices - list containing np.arrays of all the rewarded movements. """



#-------------------------------------------------------------------------
#---------------------------DATACLASSES USED------------------------------
#-------------------------------------------------------------------------

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
