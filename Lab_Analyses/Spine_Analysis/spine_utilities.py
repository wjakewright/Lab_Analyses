"""Module containing some regularly used functions for spine analysis"""

import os
from re import L

import numpy as np
import scipy.optimize as syop
import scipy.signal as sysignal
from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities import (
    get_activity_timestamps,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.save_load_pickle import load_pickle
from scipy import stats


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
    activity_traces, dFoF_traces, um_volumes, sampling_rate, iterations=1000,
):
    """Function to generate a normalization constant to normalize spine activity
        by its volume
        
        INPUT PARAMETERS
            activity_trace - np.array of all spines binarized activity
            
            dFoF_trace - np.array of all spine's dFoF trace
            
            volume - int or float of all spine's volume converted to um

            sampling_rate - int or float of the imaging sampling rate

            interations - int of how many interations of constants to test
            
        OUTPUT PARAMETERS
            norm_constants - np.array of the normalization constant for each spine
            
    """
    DISTANCE = 0.5 * sampling_rate
    # First generate an averaged activity trace
    max_amplitudes = []
    for i in range(activity_traces.shape[1]):
        if not np.sum(activity_traces[:, i]):
            max_amplitudes.append(0)
            continue
        activity_stamps = get_activity_timestamps(activity_traces[:, i])
        activity_stamps = [x[0] for x in activity_stamps]
        _, mean_trace = d_utils.get_trace_mean_sem(
            dFoF_traces[:, i].reshape(-1, 1),
            ["Spine"],
            activity_stamps,
            window=(-2, 2),
            sampling_rate=sampling_rate,
        )
        mean_trace = mean_trace["Spine"][0]
        # Find max peak amplitude
        trace_med = np.median(mean_trace)
        trace_std = np.std(mean_trace)
        trace_h = trace_med + trace_std
        _, trace_props = sysignal.find_peaks(
            mean_trace, height=trace_h, distance=DISTANCE,
        )
        trace_amps = trace_props["peak_heights"]
        try:
            max_amp = np.max(trace_amps)
        except ValueError:
            max_amp = 0
        max_amplitudes.append(max_amp)

    # Convert values to arrays
    max_amplitudes = np.array(max_amplitudes)
    um_volumes = np.array(um_volumes)
    # um_volumes = np.array(volumes)

    # Test initial slope
    _, p = stats.pearsonr(max_amplitudes, um_volumes)
    if p > 0.05:
        norm_constants = np.ones(len(um_volumes))
        return norm_constants

    # Estimate minimum constant
    obj_function = lambda C: norm_objective_function(max_amplitudes, um_volumes, C)
    test_constants = []
    for i in range(iterations):
        tc = obj_function(i)
        test_constants.append(tc)
    x0 = np.nanargmin(test_constants)
    # Find minimum constant
    constant = syop.minimize(obj_function, x0, bounds=[(0, np.inf)]).x
    # Apply min constant to each volume
    norm_constants = um_volumes + constant

    return norm_constants


def batch_spine_volume_norm_constant(mice_list, day, activity_type):
    # Set up final output dict
    constant_dict = {}

    mouse_ids = []
    FOV_list = []
    flags = []
    activity_traces = []
    dFoF_traces = []
    volumes_um = []
    sampling_rates = []
    # Pool data across all mice togetehr
    for mouse in mice_list:
        datasets = load_spine_datasets(mouse, [day], followup=False)
        mouse_dict = {}
        for FOV, dataset in datasets.items():
            data = dataset[day]
            mouse_dict[FOV] = []
            sampling_rates.append(data.imaging_parameters["Sampling Rate"])
            activity = data.spine_GluSnFr_activity
            if activity_type == "GluSnFr":
                dFoF = data.spine_GluSnFr_processed_dFoF
            elif activity_type == "Calcium":
                dFoF = data.spine_calcium_processed_dFoF
            else:
                return "Improper activity type input !!!"
            volume = data.spine_volume
            pix_to_um = data.imaging_parameters["Zoom"] / 2
            for v in volume:
                um_volume = (np.sqrt(v) / pix_to_um) ** 2
                mouse_ids.append(mouse)
                FOV_list.append(FOV)
                volumes_um.append(um_volume)
            activity_traces.append(activity)
            dFoF_traces.append(dFoF)
            for f in data.spine_flags:
                flags.append(f)
        constant_dict[mouse] = mouse_dict

    max_len = np.max([a.shape[0] for a in activity_traces])
    temp_traces = [np.zeros((max_len, a.shape[1])) for a in activity_traces]
    padded_a_traces = []
    padded_d_traces = []
    for t, a, d in zip(temp_traces, activity_traces, dFoF_traces):
        td = np.copy(t)
        t[list(range(a.shape[0])), :] = a
        td[list(range(a.shape[0])), :] = d
        padded_a_traces.append(t)
        padded_d_traces.append(td)

    activity_traces = np.hstack(padded_a_traces)
    dFoF_traces = np.hstack(padded_d_traces)
    volumes_um = np.array(volumes_um)
    sampling_rate = np.unique(sampling_rates)
    if len(sampling_rate) != 1:
        return "Different sampling rates between datasets !!!"

    # Set up the output
    norm_constants = np.ones(activity_traces.shape[1])
    # Exclude eliminated spines
    el_spines = find_spine_classes(flags, "Eliminated Spine")
    spine_idxs = np.nonzero([not i for i in el_spines])[0]
    good_activity = activity_traces[:, spine_idxs]
    good_dFoF = dFoF_traces[:, spine_idxs]
    good_volumes = volumes_um[spine_idxs]

    good_constants = spine_volume_norm_constant(
        good_activity, good_dFoF, good_volumes, sampling_rate, iterations=1000
    )
    norm_constants[spine_idxs] = good_constants

    for mouse, value in constant_dict.items():
        for fov in value.keys():
            mouse_idxs = [m == mouse for m in mouse_ids]
            fov_idxs = [f == fov for f in FOV_list]
            idxs = np.nonzero(np.array(mouse_idxs) * np.array(fov_idxs))[0]
            constant_dict[mouse][fov] = norm_constants[idxs]

    return constant_dict


def norm_objective_function(x, y, con):
    """Helper function to genterate the objective function for spine volume normalization"""
    new_x = x / (y + con)
    new_y = y

    x_input = np.vstack([new_x, np.ones(len(new_x))]).T
    m, _ = np.linalg.lstsq(x_input, new_y, rcond=0)[0]

    return np.absolute(m)

