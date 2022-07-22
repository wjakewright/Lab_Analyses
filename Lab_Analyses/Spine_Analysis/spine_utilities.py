"""Module containing some regularly used functions for spine analysis"""

import os

import numpy as np
from Lab_Analyses.Utilities.save_load_pickle import load_pickle


def pad_spine_data(spine_data_list, pad_value=np.nan):
    """Function to pad spine data to account for differences when new spines
        are added in later days
        
        INPUT PARAMETERS
            spine_data_list - list of the spine data from each day that is to 
                             be padded. 
            
            pad_value - value you wish to pad the data with
                             
        OUTPUT PARAMETERS
            padded_spine_data - list of the spine data now padded
            
    """

    # Find the maximum size of the spine data
    data_type = type(spine_data_list[0])
    sizes = []
    # do this differently depending on the nature of input data
    ## Convert list to array
    if data_type == list:
        for i, spine_data in enumerate(spine_data_list):
            spine_data_list[i] = np.array(spine_data)

    ## Handle one dimension arrays
    if type(spine_data_list[0]) == np.ndarray and len(spine_data_list[0].shape) == 1:
        individual_type = type(spine_data_list[0][0])
        for spine_data in spine_data_list:
            sizes.append(len(spine_data))
        max_size = max(sizes)
        padded_spine_data = [
            np.pad(x, (0, max_size - len(x)), "constant", constant_values=(pad_value))
            for x in spine_data_list
        ]

        if data_type == list:
            padded_spine_data = [list(x) for x in padded_spine_data]

    ## Handle two dimensional arrays
    elif type(spine_data_list[0]) == np.ndarray and len(spine_data_list[0].shape) == 2:
        for spine_data in spine_data_list:
            sizes.append(spine_data.shape[1])
        max_size = max(sizes)
        padded_spine_data = []
        for spine_data in spine_data_list:
            diff = max_size - spine_data.shape[1]
            if diff > 0:
                z = np.zeros((spine_data.shape[0], diff))
                z[:] = np.nan
                padded_spine_data.append(np.concatenate((spine_data, z), axis=1))
            else:
                padded_spine_data.append(spine_data)

    else:
        return "Input data is not in the correct format!!!"

    return padded_spine_data


def find_stable_spines(spine_flag_list):
    """Function to find all the spines that are stable throughout all analyzed days
        
        INPUT PARAMETERS
            spine_flag_list - list of all the spine flags for all the spine ROIs
            
        OUTPUT PARAMETERS
            stable_spines - boolean array of whether or not each spine is stable
    """
    # find stable spines for each day
    daily_stable_spines = []
    for spine_flags in spine_flag_list:
        stable_s = []
        for spine in spine_flags:
            if "New Spine" in spine or "Eliminated Spine" in spine:
                stable = False
            else:
                stable = True
            stable_s.append(stable)
        daily_stable_spines.append(np.array(stable_s))

    # find spines stable across all days
    ### Make all spines the same length
    daily_stable_spines = pad_spine_data(daily_stable_spines, False)
    stable_spines = np.prod(np.vstack(daily_stable_spines), axis=0)

    return stable_spines


def find_spine_classes(spine_flags, spine_class):
    """Function to find specific types of spines based on their flags
        INPUT PARAMETERS
            spine_flag_list - list of all the spine flags for all the spine ROIs
            
            spine_class - str specifying what type of spine you want to finde

        OUTPUT PARAMETERS
            classed_spines - boolean array of whether or not each spine is stable
    """
    # Initialize output
    classed_spines = [False for x in spine_flags]

    # find the speicific spine classes
    for i, spine in enumerate(spine_flags):
        if spine_class in spine:
            classed_spines[i] = True

    return classed_spines


def load_spine_datasets(mouse_id, days, followup):
    """Function to handle loading all the spine datasets for a mouse
    
        INPUT PARAMETERS
            mouse_id - str specifying which mouse to load
            
            days - list of strings specifying which days to load. Used
                    to search filenames
                    
            followup - boolean of whether or not to also load followup data
    """

    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"

    data_path = os.path.join(initial_path, mouse_id, "spine_data")
    FOVs = next(os.walk(data_path))[1]

    mouse_data = {}
    for FOV in FOVs:
        FOV_path = os.path.join(data_path, FOV)
        datasets = []
        fnames = next(os.walk(FOV_path))[2]
        for day in days:
            load_name = [x for x in fnames if day in x and "followup not in x"][0]
            data = load_pickle([load_name], path=FOV_path)[0]
            datasets.append(data)
        used_days = list(days)
        # Add followup data if specified
        if followup:
            followup_datasets = []
            for day in days:
                followup_name = [x for x in fnames if day in x and "followup" in x][0]
                followup_data = load_pickle([followup_name], path=FOV_path)[0]
                followup_datasets.append(followup_data)
            datasets = [
                sub[item]
                for item in range(len(followup_datasets))
                for sub in [datasets, followup_datasets]
            ]
            pre_days = [f"Pre {day}" for day in days]
            post_days = [f"Post {day}" for day in days]
            used_days = [
                sub[item]
                for item in range(len(post_days))
                for sub in [pre_days, post_days]
            ]
        FOV_data = {}
        for data, day in zip(datasets, used_days):
            FOV_data[day] = data

        mouse_data[day] = FOV_data

    return mouse_data

