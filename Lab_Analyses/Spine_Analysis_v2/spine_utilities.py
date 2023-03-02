import os

import numpy as np

from Lab_Analyses.Utilities.data_utilities import pad_array_to_length
from Lab_Analyses.Utilities.save_load_pickle import load_pickle


def find_stable_spines(spine_flags):
    """Function to find stable spines from a single session
    
        INPUT PARAMETERS
            spine_flags - list of the spine flags
        
        OUTPUT PARAMETERS
            stable_spines - boolean np.array of whether each spine is stable
    """
    stable_spines = []
    for flag in spine_flags:
        if "New Spine" in flag or "Eliminated Spine" in flag or "Absent" in flag:
            stable_spines.append(False)
            continue
        else:
            stable_spines.append(True)

    return np.array(stable_spines)


def find_stable_spines_across_days(spine_flag_list):
    """Function to find stable spines across multiple sessions
    
        INPUT PARAMETERS
            spine_flag_list - list of lists containing all the spine flags
            
        OUTPUT PARAMETERS
            stable_spines - boolean array of whether or not each spine is stable
                            across all sessions
    """
    # Find stable spines for each day
    daily_stable_spines = []
    for spine_flags in spine_flag_list:
        stable = find_stable_spines(spine_flags)
        daily_stable_spines.append(stable)

    # Find stable spines across all days
    ## Pad arrays first
    max_len = np.max([len(x) for x in daily_stable_spines])
    padded_spines = []
    for ds in daily_stable_spines:
        if len(ds) != max_len:
            padded_spines.append(
                pad_array_to_length(ds, max_len, value=False).astype(bool)
            )
        else:
            padded_spines.append(ds)
    stable_spines = np.prod(np.vstack(daily_stable_spines), axis=0)

    return stable_spines


def find_present_spines(spine_flags):
    """Function to find spines present during the imaging session. Excludes eliminated
        and absent spines
        
        INPUT PARAMETERS
            spine_flags - list of spine flags
            
        OUTPUT PARAMETERS
            present_spines - boolean array of whether or not each spine is present
    """
    # Initialize output
    present_spines = []
    for flag in spine_flags:
        if "Eliminated Spine" in flag or "Absent" in flag:
            present_spines.append(False)
        else:
            present_spines.append(True)

    return np.array(present_spines)


def find_spine_classes(spine_flags, spine_class):
    """Function to find specific types of spines based on their flags
    
        INPUT PARAMETERS
            spine_flags - list of the spine flags
            
            spine_class - str specifying what type of spine you want to find
        
        OUTPUT PARAMETERS
            classed_spines - boolean array of whether or not each spine is stable
    """
    # Initialize output
    classed_spines = np.zeros(len(spine_flags)).astype(bool)
    # Find the specific spine classes
    for i, spine in enumerate(spine_flags):
        if spine_class in spine:
            classed_spines[i] = True

    return classed_spines


def load_spine_datasets(mouse_id, days, fov_type):
    """Function to handel loading all the spine datasets for a mouse
    
        INPUT PARAMETERS
            mouse_id - str specifying which mouse to load
            
            days - list of str specifying which days to load. Used
                    to search for filenames
            
            fov_type - str specifying whether to load apical or basal FOVs

        OUTPUT PARAMETERS
            mouse_data - dict of dict containing data for each FOV (upper dict) 
                        for each imaged data (lower dict)
    """
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"

    data_path = os.path.join(initial_path, mouse_id, "spine_data")
    FOVs = next(os.walk(data_path))[1]
    FOVs = [x for x in FOVs if fov_type in x]

    mouse_data = {}
    for FOV in FOVs:
        FOV_path = os.path.join(data_path, FOV)
        FOV_data = {}
        fnames = next(os.walk(FOV_path))[2]
        for day in days:
            load_name = [x for x in fnames if day in x][0]
            data = load_pickle([load_name], path=FOV_path)[0]
            FOV_data[day] = data
        mouse_data[FOV] = FOV_data

    return mouse_data

