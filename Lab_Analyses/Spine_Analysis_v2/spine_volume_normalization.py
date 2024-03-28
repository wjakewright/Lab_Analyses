import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    find_present_spines,
    load_spine_datasets,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_timestamps import get_activity_timestamps
from Lab_Analyses.Utilities.data_utilities import pad_array_to_length
from Lab_Analyses.Utilities.mean_trace_functions import find_peak_amplitude
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def batch_spine_volume_normalization(
    mice_list, day, fov_type, activity_type, zscore, plot=False
):
    """Function to handle calculating constants to normalize activity traces
    based on spine volume

    INPUT PARAMETERS
        mice_list - list of str specifying the mice to analyze

        day - str speicifying what session to analyze

        fov_type - str specifying whether to analyze apical or basal FOVs

        activity_type - str specifying what type of data (GluSnFr or Calcium)
                        to analyze

        zscore - boolean specifying if the input data is zscored

        plot - boolean specifying whether to plot the before and after relationship
                between activity and volume

    OUTPUT PARAMETERS
        constant_dict - dict of dict containing the constants for each mouse and
                        each FOV spines
    """
    # Initialize the output
    constant_dict = {}

    # Temporary variables
    mouse_ids = []
    FOV_list = []
    flags = []
    activity_traces = []
    dFoF_traces = []
    volumes_um = []
    sampling_rates = []
    # Pool data across all mice together
    for mouse in mice_list:
        ## Load the data
        datasets = load_spine_datasets(mouse, [day], fov_type=fov_type)
        mouse_dict = {}
        ## Go through each FOV
        for FOV, dataset in datasets.items():
            data = dataset[day]
            mouse_dict[FOV] = []  ## This is to make an empty nested dict
            sampling_rates.append(data.imaging_parameters["Sampling Rate"])
            ## Pull relevant activity data
            activity = data.spine_GluSnFr_activity
            if activity_type == "GluSnFr":
                dFoF = data.spine_GluSnFr_processed_dFoF
            elif activity_type == "Calcium":
                dFoF = data.spine_calcium_processed_dFoF
            else:
                return "Improper activity type input !!!"
            ## Convert volumes into um area
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
        constant_dict[mouse] = mouse_dict  ## This is just to make an empty nested dict

    # Pad activity arrays in order to concatenate
    max_len = np.max([a.shape[0] for a in activity_traces])
    a_traces = []
    d_traces = []
    for a, d in zip(activity_traces, dFoF_traces):
        padded_a = pad_array_to_length(a, max_len, axis=0, value=0)
        padded_d = pad_array_to_length(d, max_len, axis=0, value=0)
        a_traces.append(padded_a)
        d_traces.append(padded_d)
    activity_traces = np.hstack(a_traces)
    dFoF_traces = np.hstack(d_traces)
    volumes_um = np.array(volumes_um)
    sampling_rate = np.unique(sampling_rates)
    if len(sampling_rate) != 1:
        return "Different sampling rates between datasets!!!"

    # Setup container
    norm_constants = np.ones(activity_traces.shape[1])
    # Exlude eliminated and absent spines
    present_spines = find_present_spines(flags)
    good_activity = activity_traces[:, present_spines]
    good_dFoF = dFoF_traces[:, present_spines]
    good_volumes = volumes_um[present_spines]
    # Calculate the constants
    good_constants = spine_volume_norm_constant(
        good_activity,
        good_dFoF,
        good_volumes,
        sampling_rate,
        activity_type,
        zscore,
        iterations=1000,
        plot=plot,
    )
    norm_constants[present_spines] = good_constants
    # Store the constants
    for mouse, value in constant_dict.items():
        for fov in value.keys():
            mouse_idxs = [m == mouse for m in mouse_ids]
            fov_idxs = [f == fov for f in FOV_list]
            idxs = np.nonzero(np.array(mouse_idxs) * np.array(fov_idxs))[0]
            constant_dict[mouse][fov] = norm_constants[idxs]

    # Save section
    save_path = r"G:\Analyzed_data\grouped\Dual_Spine_Imaging\Normalization_Constants"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if zscore:
        zname = "zscore"
    else:
        zname = "dFoF"
    fname = f"{fov_type}_{zname}_{activity_type}_normalization_constants"
    save_pickle(fname, constant_dict, save_path)

    return constant_dict


def spine_volume_norm_constant(
    activity_traces,
    dFoF_traces,
    um_volumes,
    sampling_rate,
    activity_type,
    zscore,
    iterations=1000,
    plot=False,
):
    """Function to generate a normalization constant to normalize spine activity
    by its volume if there is a relationship between them

    INPUT PARAMETERS
        activity_trace - np.array of all spines binarized activity

        dFoF_traces - np.array of all spines' dFoF traces

        volume - np.array of all the spines volumes converted to um

        sampling_rate - int specifying the imaging rate

        activity_type - str specifying the type of activity

        zscore - boolean specifying whether to zscore the traces or not

        iterations - int specifying how many iterations to perform

        plot - boolean indicating whether to plot the relation ship between
                activity and volume before and after correction

    OUTPUT PARAMTERS
        norm_constants - np.array of the normalization constants for each spine
    """
    # Find the max amplitudes for each trace when it is active
    max_amplitudes = np.zeros(activity_traces.shape[1])
    if zscore:
        dFoF_traces = d_utils.z_score(dFoF_traces)
    ## Go through each trace
    for i in range(activity_traces.shape[1]):
        ### If not activity skip
        if not np.sum(activity_traces[:, i]):
            continue
        ### Get mean trace when the spine is active
        activity_stamps = get_activity_timestamps(activity_traces[:, i])
        activity_stamps = [x[0] for x in activity_stamps]
        _, mean_trace = d_utils.get_trace_mean_sem(
            dFoF_traces[:, i].reshape(-1, 1),
            ["Spine"],
            activity_stamps,
            window=(-2, 2),
            sampling_rate=sampling_rate,
        )
        mean_trace = mean_trace["Spine"]
        ### Find the peak amplitude
        max_amp, _ = find_peak_amplitude(
            mean_trace, smooth=True, window=False, sampling_rate=sampling_rate
        )
        max_amplitudes[i] = max_amp[0]

    # Remove any nan values from amplitudes
    non_nan = np.nonzero(~np.isnan(max_amplitudes))[0]
    maximum_amplitudes = max_amplitudes[non_nan]
    volumes = um_volumes[non_nan]

    # Test initial slope
    r, p = stats.pearsonr(volumes, maximum_amplitudes)
    if (p > 0.05) or (r < 0):
        norm_constants = np.ones(len(um_volumes))
        if plot:
            plot_norm_constants(
                max_amplitudes, um_volumes, norm_constants, activity_type
            )
            return norm_constants

    # Estimate minimum constant
    objective_function = lambda C: norm_objective_function(
        maximum_amplitudes, volumes, C
    )
    test_constants = []
    for i in range(iterations):
        tc = objective_function(i)
        test_constants.append(tc)
    x0 = np.nanargmin(test_constants)
    # Find minimum constant from estimate
    constant = optimize.minimize(objective_function, x0, bounds=[(0, np.inf)]).x
    # Apply minimum constant to each volume
    norm_constants = um_volumes + constant
    if plot:
        plot_norm_constants(max_amplitudes, um_volumes, norm_constants, activity_type)

    return norm_constants


def norm_objective_function(x, y, con):
    """Helper function to generate the objective function for
    spine volume normalization"""

    new_x = x / (y + con)
    new_y = y

    x_input = np.vstack([new_x, np.ones(len(new_x))]).T
    m, _ = np.linalg.lstsq(x_input, new_y, rcond=0)[0]

    return np.absolute(m)


def plot_norm_constants(max_amplitudes, volumes, norm_constants, activity_type):
    """Function to plot the before and after relationship between
    amplitudes and spine volumes"""

    # Construct the plot
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.5), constrained_layout=True)
    fig.suptitle(activity_type)

    # Plot the initial amplitudes and volumes
    plot_scatter_correlation(
        x_var=volumes,
        y_var=max_amplitudes,
        CI=95,
        title="Non-normalized amplitudes",
        xtitle="Spine area (um)",
        ytitle="Max amplitude (dFoF)",
        xlim=None,
        ylim=None,
        marker_size=10,
        face_color="mediumblue",
        edge_color="white",
        line_color="mediumblue",
        s_alpha=0.8,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes[0],
        save=False,
        save_path=None,
    )
    # normalize the max amplitudes by the constant
    norm_amplitudes = max_amplitudes / norm_constants
    plot_scatter_correlation(
        x_var=volumes,
        y_var=norm_amplitudes,
        CI=95,
        title="Normalized amplitudes",
        xtitle="Spine area (um)",
        ytitle="Norm. max amplitude (dFoF)",
        xlim=None,
        ylim=None,
        marker_size=10,
        face_color="forestgreen",
        edge_color="white",
        line_color="forestgreen",
        s_alpha=0.8,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes[1],
        save=False,
        save_path=None,
    )


def load_norm_constants(fov_type, activity_type, signal_type):
    """Helper function to search and load for the normalization constants

    INPUT PARAMETERS
        fov_type - str specifying the type of FOV

        activity_type - boolean specifying whether the activity type is zscore
                        or dFoF

        signal_type - str specifying if the signal is GluSnFr or Calcium

    OUTPUT PARAMETERS
        normalization_constants - nested dict of the normalization constants
    """
    load_path = r"G:\Analyzed_data\grouped\Dual_Spine_Imaging\Normalization_Constants"
    if activity_type:
        aname = "zscore"
    else:
        aname = "dFoF"
    fname = f"{fov_type}_{aname}_{signal_type}_normalization_constants"
    load_name = os.path.join(load_path, fname)

    normalization_constants = load_pickle([load_name])[0]

    return normalization_constants
