"""Module containing some regularly used functions for spine analysis"""

import os

import numpy as np
import scipy.optimize as syop
import scipy.signal as sysignal
from Lab_Analyses.Spine_Analysis.global_coactivity_v2 import get_activity_timestamps
from Lab_Analyses.Utilities import data_utilities as d_utils
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

        mouse_data[FOV] = FOV_data

    return mouse_data


def spine_volume_norm_constant(
    activity_traces, dFoF_traces, volumes, zoom_factor, sampling_rate, iterations=1000,
):
    """Function to generate a normalization constant to normalize spine activity
        by its volume
        
        INPUT PARAMETERS
            activity_trace - np.array of all spines binarized activity
            
            dFoF_trace - np.array of all spine's dFoF trace
            
            volume - int or float of all spine's volume
            
            zoom_factor - int or float of the zoom used when imaging

            sampling_rate - int or float of the imaging sampling rate

            interations - int of how many interations of constants to test
            
        OUTPUT PARAMETERS
            norm_constant - np.array of the normalization constant for each spine
            
    """
    DISTANCE = 0.5 * sampling_rate
    # First generate an averaged activity trace
    max_amplitudes = []
    for i in range(activity_traces.shape[1]):
        activity_stamps = get_activity_timestamps(activity_traces[:, i])
        activity_stamps = [x[0] for x in activity_stamps]
        _, mean_trace = d_utils.get_trace_mean_sem(
            dFoF_traces[:, i].reshape(-1, 1),
            ["Spine"],
            activity_stamps,
            window=(-2, 2),
            sampling_rate=sampling_rate,
        )
        # Find max peak amplitude
        trace_med = np.median(mean_trace)
        trace_std = np.std(mean_trace)
        trace_h = trace_med + trace_std
        _, trace_props = sysignal.find_peaks(
            mean_trace, height=trace_h, distance=DISTANCE,
        )
        trace_amps = trace_props["peak_heights"]
        max_amp = np.max(trace_amps)
        max_amplitudes.append(max_amp)

    # Convert Volume to um from pixels
    pix_to_um = zoom_factor / 2
    um_volumes = []
    for volume in volumes:
        um_volume = volume / pix_to_um
        um_volumes.append(um_volume)

    # Convert values to arrays
    max_amplitudes = np.array(max_amplitudes)
    um_volumes = np.array(um_volumes)

    # Estimate minimum constant
    obj_function = lambda C: norm_objective_function(max_amplitudes, um_volumes, C)
    test_constants = []
    for i in range(iterations):
        tc = obj_function(i)
        test_constants.append(tc)
    x0 = np.nanmean(test_constants)
    # Find minimum constant
    constant = syop.minimize(obj_function, x0, bounds=(0, np.inf))
    # Apply min constant to each volume
    norm_constant = um_volumes + constant

    return norm_constant


def norm_objective_function(x, y, con):
    """Helper function to genterate the objective function for spine volume normalization"""
    new_x = x / (y + con)
    new_y = y + con

    x_input = np.vstack([new_x, np.ones(len(new_x))]).T
    _, c = np.linalg.lstsq(x_input, new_y, rcond=0)[0]

    return np.absolute(c)

