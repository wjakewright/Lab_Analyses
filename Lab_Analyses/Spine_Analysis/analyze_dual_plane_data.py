import fractions
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sysignal
import seaborn as sns
from scipy import optimize, stats
from sklearn import preprocessing

from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities import (
    get_trace_coactivity_rates,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_timestamps import get_activity_timestamps
from Lab_Analyses.Utilities.event_detection import event_detection
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle

sns.set()
sns.set_style("ticks")
warnings.simplefilter("error", optimize.OptimizeWarning)


def analyze_dual_plane_data(mouse_list, save=False, save_path=None):
    """Function to handle the analysis of dual plane somatic and 
        dendritic imaging datasets. Combines data across mice and FOVs
        
        INPUT PARAMETERS
            mouse_list - list of str specifying the mice we wish to analyze
            
            save - boolean of whether or not to save the output
    """
    # Load datasets
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
    datasets = []
    for mouse in mouse_list:
        top_path = os.path.join(initial_path, mouse, "dual_plane")
        FOVs = next(os.walk(top_path))[1]
        for FOV in FOVs:
            FOV_path = os.path.join(top_path, FOV)
            fname = f"{mouse}_{FOV}_dual_soma_dend_data"
            file = os.path.join(FOV_path, fname)
            data = load_pickle([file])[0]
            datasets.append(data)

    # Set up temp_output
    temp_output = defaultdict(list)
    coactive_dendrite_traces = []
    noncoactive_dendrite_traces = []
    coactive_somatic_traces = []
    noncoactive_somatic_traces = []
    coactive_dendrite_traces_norm = []
    noncoactive_dendrite_traces_norm = []
    coactive_somatic_traces_norm = []
    noncoactive_somatic_traces_norm = []

    # Analyze each dataset seperately
    for dataset in datasets:
        sampling_rate = dataset.imaging_parameters["Sampling Rate"]
        # Check somatic and dendritic recordings are same length
        dl = len(dataset.dend_processed_dFoF[:, 0])
        sl = len(dataset.soma_processed_dFoF)
        if dl > sl:
            dend_dFoF = dataset.dend_processed_dFoF[:sl, :]
            # dend_activity = dataset.dend_activity[:sl, :]
            somatic_dFoF = dataset.soma_processed_dFoF
            # somatic_activity = dataset.soma_activity
        elif dl < sl:
            dend_dFoF = dataset.dend_processed_dFoF
            # dend_activity = dataset.dend_activity
            somatic_dFoF = dataset.soma_processed_dFoF[:dl]
            # somatic_activity = dataset.soma_activity[:dl]
        else:
            dend_dFoF = dataset.dend_processed_dFoF
            # dend_activity = dataset.dend_activity
            somatic_dFoF = dataset.soma_processed_dFoF
            # somatic_activity = dataset.soma_activity
        # duplicate somatic activity to pair with each dendrite
        soma_dFoF = np.zeros(dend_dFoF.shape)
        # soma_activity = np.zeros(dend_activity.shape)
        for i in range(dend_dFoF.shape[1]):
            soma_dFoF[:, i] = somatic_dFoF.flatten()
            # soma_activity[:, i] = somatic_activity.flatten()

        # Generate normalized traces
        scalar = preprocessing.MinMaxScaler()
        dend_dFoF_norm = scalar.fit_transform(dend_dFoF)
        soma_dFoF_norm = scalar.fit_transform(soma_dFoF)

        dend_activity, _, _ = event_detection(
            dend_dFoF,
            threshold=2,
            lower_threshold=1,
            lower_limit=0.2,
            sampling_rate=sampling_rate,
        )
        soma_activity, _, _ = event_detection(
            soma_dFoF,
            threshold=2,
            lower_threshold=0,
            lower_limit=None,
            sampling_rate=sampling_rate,
        )

        # store traces
        temp_output["dend_activity"].append(dend_activity)
        temp_output["soma_activity"].append(soma_activity)
        temp_output["dend_dFoF_traces"].append(dend_dFoF)
        temp_output["soma_dFoF_traces"].append(soma_dFoF)
        temp_output["dend_dFoF_norm_traces"].append(dend_dFoF_norm)
        temp_output["soma_dFoF_norm_traces"].append(soma_dFoF_norm)

        # Analyze each dendrite seperately
        for dend in range(dend_dFoF.shape[1]):
            d_dFoF = dend_dFoF[:, dend]
            d_dFoF_norm = dend_dFoF_norm[:, dend]
            d_activity = dend_activity[:, dend]
            s_dFoF = soma_dFoF[:, dend]
            s_dFoF_norm = soma_dFoF_norm[:, dend]
            s_activity = soma_activity[:, dend]

            # Calculate fraction of events coactive
            (_, _, frac_dend_act, _, _) = get_trace_coactivity_rates(
                d_activity, s_activity, sampling_rate=sampling_rate
            )
            (_, _, frac_soma_act, _, _) = get_trace_coactivity_rates(
                s_activity, d_activity, sampling_rate=sampling_rate
            )
            # Store fraction active
            temp_output["frac_dend_active"].append(frac_dend_act)
            temp_output["frac_soma_active"].append(frac_soma_act)

            # Get dend activity timestamps
            dend_events = get_activity_timestamps(d_activity)
            dend_events = [x[0] for x in dend_events]

            # Get amps and taus of paired events
            (dend_amps, soma_amps, dend_tau, soma_tau) = analyze_paired_events(
                dend_events,
                d_dFoF,
                s_dFoF,
                activity_window=(-1, 4),
                sampling_rate=sampling_rate,
            )
            (
                dend_amps_norm,
                soma_amps_norm,
                dend_tau_norm,
                soma_tau_norm,
            ) = analyze_paired_events(
                dend_events,
                d_dFoF_norm,
                s_dFoF_norm,
                activity_window=(-1, 4),
                sampling_rate=sampling_rate,
            )
            # Store amps and taus
            temp_output["dend_amps"].append(dend_amps)
            temp_output["soma_amps"].append(soma_amps)
            temp_output["dend_tau"].append(dend_tau)
            temp_output["soma_tau"].append(soma_tau)
            temp_output["dend_amps_norm"].append(dend_amps_norm)
            temp_output["soma_amps_norm"].append(soma_amps_norm)
            temp_output["dend_tau_norm"].append(dend_tau_norm)
            temp_output["soma_tau_norm"].append(soma_tau_norm)

            # Get the coactive and noncoactive traces
            (
                coactive_dend_traces,
                noncoactive_dend_traces,
                coactive_soma_traces,
                noncoactive_soma_traces,
            ) = get_coactive_vs_noncoactive_traces(
                d_activity,
                s_activity,
                d_dFoF,
                s_dFoF,
                activity_window=(-1, 2),
                sampling_rate=sampling_rate,
            )
            (
                coactive_dend_traces_norm,
                noncoactive_dend_traces_norm,
                coactive_soma_traces_norm,
                noncoactive_soma_traces_norm,
            ) = get_coactive_vs_noncoactive_traces(
                d_activity,
                s_activity,
                d_dFoF_norm,
                s_dFoF_norm,
                activity_window=(-1, 2),
                sampling_rate=sampling_rate,
            )
            coactive_dendrite_traces.append(coactive_dend_traces)
            noncoactive_dendrite_traces.append(noncoactive_dend_traces)
            coactive_somatic_traces.append(coactive_soma_traces)
            noncoactive_somatic_traces.append(noncoactive_soma_traces)
            coactive_dendrite_traces_norm.append(coactive_dend_traces_norm)
            noncoactive_dendrite_traces_norm.append(noncoactive_dend_traces_norm)
            coactive_somatic_traces_norm.append(coactive_soma_traces_norm)
            noncoactive_somatic_traces_norm.append(noncoactive_soma_traces_norm)

    # Merge data across FOVs and mice
    output_dict = {}
    for key, value in temp_output.items():
        if type(value[0]) == list:
            output_dict[key] = [y for x in value for y in x]
        elif type(value[0]) == np.ndarray:
            if len(value[0].shape) == 2:
                pad_value = pad_arrays(value)
                output_dict[key] = np.hstack(pad_value)
            elif len(value[0].shape) == 1:
                output_dict[key] = value
        else:
            output_dict[key] = np.array(value)

    # Store data in dataclass
    soma_dendrite_activity = Soma_Dendrite_Activity_Data(
        dend_activity=output_dict["dend_activity"],
        soma_activity=output_dict["soma_activity"],
        dend_dFoF_traces=output_dict["dend_dFoF_traces"],
        soma_dFoF_traces=output_dict["soma_dFoF_traces"],
        dend_dFoF_norm_traces=output_dict["dend_dFoF_norm_traces"],
        soma_dFoF_norm_traces=output_dict["soma_dFoF_norm_traces"],
        frac_dend_active=output_dict["frac_dend_active"],
        frac_soma_active=output_dict["frac_soma_active"],
        dend_amplitudes=output_dict["dend_amps"],
        soma_amplitudes=output_dict["soma_amps"],
        dend_tau=output_dict["dend_tau"],
        soma_tau=output_dict["soma_tau"],
        dend_amplitudes_norm=output_dict["dend_amps_norm"],
        soma_amplitudes_norm=output_dict["soma_amps_norm"],
        dend_tau_norm=output_dict["dend_tau_norm"],
        soma_tau_norm=output_dict["soma_tau_norm"],
        coactive_dend_traces=coactive_dendrite_traces,
        noncoactive_dend_traces=noncoactive_dendrite_traces,
        coactive_soma_traces=coactive_somatic_traces,
        noncoactive_soma_traces=noncoactive_somatic_traces,
        coactive_dend_traces_norm=coactive_dendrite_traces_norm,
        noncoactive_dend_traces_norm=noncoactive_dendrite_traces_norm,
        coactive_soma_traces_norm=coactive_somatic_traces_norm,
        noncoactive_soma_traces_norm=noncoactive_somatic_traces_norm,
        sampling_rate=sampling_rate,
    )

    # Save Section
    if save:
        if save_path is None:
            save_path = r"C:Users\Desktop\Analyzed_data\grouped\dual_plane_imaging"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_name = f"dual_plane_soma_dendrite_activity_data"
        save_pickle(save_name, soma_dendrite_activity, save_path)

    return soma_dendrite_activity


def analyze_paired_events(
    timestamps, dend_dFoF, soma_dFoF, activity_window, sampling_rate
):
    """Function to analyze the amplitudes and decay tau of paired somatic and dendritic
        events
        
        INPUT PARAMETERS
            timestamps - list of timestamps for event onsets
            
            dend_dFoF - np.array of the dendritic dFoF
            
            soma_dFoF - np.array of the somatic dFoF
            
            activity_window - tuple specifying the time window before and after you 
                              want to analyze in seconds
            
            sampling_rate - int specifying the imaging sampling rate

        OUTPUT PARAMETERS
            dend_amps - np.array of peak amp for each dendritic event

            soma_amps - np.array of the peak amp for each somatic event

            dend_tau - np.array of decay taus for each dendritic event

            soma_tau - np.array of decay taus for each somatic 

            dend_exp - list of np.arrays of the exponential fits for each event

            soma_exp - list of np.arrays of the exponential fits for each event
    
    """
    DISTANCE = 0.5 * sampling_rate
    # Get the traces around timestamps
    dend_traces, _ = d_utils.get_trace_mean_sem(
        dend_dFoF.reshape(-1, 1),
        ["Dend"],
        timestamps,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    soma_traces, _ = d_utils.get_trace_mean_sem(
        soma_dFoF.reshape(-1, 1),
        ["Soma"],
        timestamps,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    dend_traces = dend_traces["Dend"]
    soma_traces = soma_traces["Soma"]

    # Set up output
    dend_amps = np.zeros(dend_traces.shape[1])
    soma_amps = np.zeros(dend_traces.shape[1])
    dend_tau = np.zeros(dend_traces.shape[1])
    soma_tau = np.zeros(dend_traces.shape[1])
    dend_exp = [None for x in range(dend_traces.shape[1])]
    soma_exp = [None for x in range(dend_traces.shape[1])]

    for i in range(dend_traces.shape[1]):
        d_trace = dend_traces[:, i]
        s_trace = soma_traces[:, i]
        # Find peaks
        d_med = np.nanmedian(d_trace)
        s_med = np.nanmedian(s_trace)
        d_std = np.nanstd(d_trace)
        s_std = np.nanstd(s_trace)
        d_height = d_med + d_std
        s_height = s_med + s_std

        d_peaks, d_props = sysignal.find_peaks(
            d_trace, height=d_height, distance=DISTANCE,
        )
        s_peaks, s_props = sysignal.find_peaks(
            s_trace, height=s_height, distance=DISTANCE,
        )
        d_amps = d_props["peak_heights"]
        s_amps = s_props["peak_heights"]

        d_max_amp = np.max(d_amps)
        d_max_peak = d_peaks[np.argmax(d_amps)]
        try:
            s_max_amp = np.max(s_amps)
            s_max_peak = s_peaks[np.argmax(s_amps)]
        except ValueError:
            s_max_amp = np.max(s_trace)
            s_max_peak = np.argmax(s_trace)

        # Find the trace tau
        ## First smooth the traces
        d_smooth = sysignal.savgol_filter(d_trace, 15, 3)
        s_smooth = sysignal.savgol_filter(s_trace, 15, 3)
        ## Get trace after the peak
        d_decay_trace = d_smooth[d_max_peak:]
        s_decay_trace = s_smooth[s_max_peak:]
        # ## Fit exponential
        # dx = np.arange(len(d_decay_trace)) + 7
        # sx = np.arange(len(s_decay_trace)) + 7
        # dy = d_decay_trace / np.max(d_decay_trace)
        # sy = s_decay_trace / np.max(s_decay_trace)

        # try:
        #     d_params, _ = optimize.curve_fit(monoExp, dx, dy, maxfev=5000)
        #     s_params, _ = optimize.curve_fit(monoExp, sx, sy, maxfev=5000)

        #     d_m, d_t, d_b = d_params
        #     s_m, s_t, s_b = s_params
        #     d_exp = monoExp(dx, d_m, d_t, d_b)
        #     s_exp = monoExp(sx, s_m, s_t, s_b)

        #     ## Get the tau in seconds
        #     d_tau = (1 / d_t) / sampling_rate
        #     s_tau = (1 / s_t) / sampling_rate
        # except:
        #     d_exp = None
        #     s_exp = None
        #     d_tau = np.nan
        #     s_tau = np.nan
        ## Find time to where it goes below half max
        try:
            d_half_max = np.nonzero(d_decay_trace < 0.5 * d_max_amp)[0][0]
        except IndexError:
            d_half_max = len(d_decay_trace)
        try:
            s_half_max = np.nonzero(s_decay_trace < 0.5 * s_max_amp)[0][0]
        except IndexError:
            s_half_max = len(s_decay_trace)
        ## conver to seconds
        d_tau = d_half_max / sampling_rate
        s_tau = s_half_max / sampling_rate

        # Store data
        dend_amps[i] = d_max_amp
        soma_amps[i] = s_max_amp
        dend_tau[i] = d_tau
        soma_tau[i] = s_tau
        # dend_exp[i] = d_exp
        # soma_exp[i] = s_exp

    return dend_amps, soma_amps, dend_tau, soma_tau


def monoExp(x, m, t, b):
    """Helper function to generate an exponential decay"""
    return m * np.exp(-t * x) + b


def pad_arrays(array_list):
    """Helper function to pad arrays for horizontal concatenation"""
    # Get max length if arrays
    lengths = [x.shape[0] for x in array_list]
    max_len = np.max(lengths)
    max_idx = np.argmax(lengths)
    # pad arrays
    new_arrays = []
    for i, arr in enumerate(array_list):
        if i == max_idx:
            new_arrays.append(arr)
            continue
        diff = max_len - arr.shape[0]
        pad_arr = np.pad(
            arr, ((0, diff), (0, 0)), mode="constant", constant_values=np.nan
        )
        new_arrays.append(pad_arr)

    return new_arrays


def get_coactive_vs_noncoactive_traces(
    dend_activity, soma_activity, dend_dFoF, soma_dFoF, activity_window, sampling_rate,
):
    """Function to get the somatic and dendritic traces when there is
        coactivity vs only dendritic activity
        
        INPUT PARAMETERS    
            dend_activity - np.array of binary dendrite activity
            
            soma_activity - np.array of binary somatic activity
            
            dend_dFoF - np.array of dendritic dFoF
            
            soma_dFoF - np.array of somatic dFoF
            
        OUTPUT PARAMETERS
            coactive_dend_traces - np.array of coactive dend traces
            
            noncoactive_dend_traces - np.array of noncoactive dend traces
            
            coactive_soma_traces - np.array of coactive soma traces
            
            noncoactive_soma_traces - np.array of noncoactive soma traces
    """
    boundary = 2 * sampling_rate
    # Get active epochs
    active_boundaries = np.insert(np.diff(dend_activity), 0, 0, axis=0)
    active_onsets = np.nonzero(active_boundaries == 1)[0]
    active_offsets = np.nonzero(active_boundaries == -1)[0]
    # Check onset ofset order
    if active_onsets[0] > active_offsets[0]:
        active_offsets = active_offsets[1:]
    # Check onsets and offsets are the same length
    if len(active_onsets) > len(active_offsets):
        active_onsets = active_onsets[:-1]
    # Refine onsets to remove events within 2 sec of each other
    r_onsets = []
    r_offsets = []
    for i in range(len(active_onsets)):
        if i == len(active_onsets) - 1:
            r_onsets.append(active_onsets[i])
            r_offsets.append(active_offsets[i])
            continue
        if active_onsets[i] + boundary <= active_onsets[i + 1]:
            r_onsets.append(active_onsets[i])
            r_offsets.append(active_offsets[i])
    # Compare active epochs to other trace
    coactive_idxs = []
    noncoactive_idxs = []
    for onset, offset in zip(r_onsets, r_offsets):
        if np.sum(soma_activity[onset:offset]):
            coactive_idxs.append(onset)
        else:
            noncoactive_idxs.append(onset)

    # Get the traces
    coactive_dend_traces, _ = d_utils.get_trace_mean_sem(
        dend_dFoF.reshape(-1, 1),
        ["Dend"],
        coactive_idxs,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    coactive_soma_traces, _ = d_utils.get_trace_mean_sem(
        soma_dFoF.reshape(-1, 1),
        ["Soma"],
        coactive_idxs,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    coactive_dend_traces = coactive_dend_traces["Dend"]
    coactive_soma_traces = coactive_soma_traces["Soma"]
    noncoactive_dend_traces, _ = d_utils.get_trace_mean_sem(
        dend_dFoF.reshape(-1, 1),
        ["Dend"],
        noncoactive_idxs,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    noncoactive_soma_traces, _ = d_utils.get_trace_mean_sem(
        soma_dFoF.reshape(-1, 1),
        ["Soma"],
        noncoactive_idxs,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    noncoactive_dend_traces = noncoactive_dend_traces["Dend"]
    noncoactive_soma_traces = noncoactive_soma_traces["Soma"]

    return (
        coactive_dend_traces,
        noncoactive_dend_traces,
        coactive_soma_traces,
        noncoactive_soma_traces,
    )


@dataclass
class Soma_Dendrite_Activity_Data:
    """Dataclass for storing analyzed dual plane somatic and dendritic activity
        data. Includes methods for plotting results"""

    dend_activity: np.ndarray
    soma_activity: np.ndarray
    dend_dFoF_traces: np.ndarray
    soma_dFoF_traces: np.ndarray
    dend_dFoF_norm_traces: np.ndarray
    soma_dFoF_norm_traces: np.ndarray
    frac_dend_active: np.ndarray
    frac_soma_active: np.ndarray
    dend_amplitudes: list
    soma_amplitudes: list
    dend_tau: list
    soma_tau: list
    dend_amplitudes_norm: list
    soma_amplitudes_norm: list
    dend_tau_norm: list
    soma_tau_norm: list
    coactive_dend_traces: list
    noncoactive_dend_traces: list
    coactive_soma_traces: list
    noncoactive_soma_traces: list
    coactive_dend_traces_norm: list
    noncoactive_dend_traces_norm: list
    coactive_soma_traces_norm: list
    noncoactive_soma_traces_norm: list
    sampling_rate: int

    def plot_soma_dend_traces(
        self, norm=True, plot_binary=False, subselect=None, save=False, save_path=None
    ):
        """Method to plot somatic and dendritic traces over each other
            INPUT
                norm - boolean specifying if you wish to use the normalized traces or
                        not
                
                subselect - tuple specifying a range you wish to subselect for plotting

                save - boolean of whether or not to save the figure

                save_path - str specifying where to save the data
        """
        dend_activity = self.dend_activity
        soma_activity = self.soma_activity
        if norm:
            title = "Normalized Somatic and Dendritic Traces"
            ytitle = "Normalized " + "\u0394" + "F/F\u2080"
            dend_traces = self.dend_dFoF_norm_traces
            soma_traces = self.soma_dFoF_norm_traces
        else:
            title = "Somatic and Dendritic Traces"
            ytitle = "\u0394" + "F/F\u2080"
            dend_traces = self.dend_dFoF_traces
            soma_traces = self.soma_dFoF_traces

        row_num = dend_traces.shape[1]
        fig_size = (10, 4 * row_num)
        fig = plt.figure(figsize=fig_size)
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle(title)

        count = 1
        for i in range(dend_traces.shape[1]):
            d_trace = dend_traces[:, i]
            s_trace = soma_traces[:, i]
            d_activity = dend_activity[:, i]
            s_activity = soma_activity[:, i]
            ax = fig.add_subplot(row_num, 1, count)
            if subselect is None:
                x = np.arange(len(d_trace)) / self.sampling_rate
                ax.plot(x, d_trace, color="darkorange", label="Dendrite")
                ax.plot(x, s_trace, color="black", label="Soma")
                if plot_binary:
                    ax.plot(x, d_activity, color="red", label="Dend binary")
                    ax.plot(x, s_activity, color="blue", label="Soma binary")
            else:
                x = (
                    np.arange(len(d_trace[subselect[0] : subselect[1]]))
                    / self.sampling_rate
                )
                ax.plot(
                    x,
                    d_trace[subselect[0] : subselect[1]],
                    color="darkorange",
                    label="Dendrite",
                )
                ax.plot(
                    x,
                    s_trace[subselect[0] : subselect[1]],
                    color="black",
                    label="Soma",
                )
                if plot_binary:
                    ax.plot(
                        x,
                        d_activity[subselect[0] : subselect[1]],
                        color="red",
                        label="Dend binary",
                    )
                    ax.plot(
                        x,
                        s_activity[subselect[0] : subselect[1]],
                        color="blue",
                        label="Soma binary",
                    )
            ax.set_title(f"Dendrite {i+1}", fontsize=10)
            plt.xlabel("Time (s)")
            plt.ylabel(ytitle)
            plt.legend()
            count = count + 1

        fig.tight_layout()
        if save:
            if save_path is None:
                save_path = r"C:\Users\Jake\Desktop\Figures"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            fname = os.path.join(save_path, title)
            plt.savefig(fname + ".pdf")

    def plot_fraction_coactive(
        self, mean_type="mean", err_type="sem", save=False, save_path=None
    ):
        """Method to plot the fraction of dendritic and somatic events coactive with each
            other"""

        #  Set up data
        soma_points = self.frac_soma_active
        dend_points = self.frac_dend_active
        # Caluculate relevant means and errors
        ## Means
        if mean_type == "mean":
            soma_mean = np.nanmean(soma_points)
            dend_mean = np.nanmean(dend_points)
        elif mean_type == "median":
            soma_mean = np.nanmedian(soma_points)
            dend_mean = np.nanmedian(dend_points)
        else:
            return "Only accepts mean and median for mean_type!!!"
        ## Error
        if err_type == "sem":
            soma_err = stats.sem(soma_points, nan_policy="omit")
            dend_err = stats.sem(dend_points, nan_policy="omit")
        elif err_type == "std":
            soma_err = np.nanstd(soma_points)
            dend_err = np.nanstd(dend_points)
        elif err_type == "CI":
            num = len(dend_points)
            if num <= 30:
                s_ci = stats.t.interval(
                    alpha=0.95,
                    df=len(soma_points) - 1,
                    loc=np.nanmean(soma_points),
                    scale=stats.sem(soma_points, nan_policy="omit"),
                )
                d_ci = stats.t.interval(
                    alpha=0.95,
                    df=len(dend_points) - 1,
                    loc=np.nanmean(dend_points),
                    scale=stats.sem(dend_points, nan_policy="omit"),
                )
                soma_err = np.array([s_ci[0], s_ci[1]]).reshape(-1, 1)
                dend_err = np.array([d_ci[0], d_ci[1]]).reshape(-1, 1)
            else:
                s_ci = stats.norm.interval(
                    alpha=0.95,
                    loc=np.nanmean(soma_points),
                    scale=stats.sem(soma_points, nan_policy="omit"),
                )
                d_ci = stats.norm.interval(
                    alpha=0.95,
                    loc=np.nanmean(dend_points),
                    scale=stats.sem(dend_points, nan_policy="omit"),
                )
                soma_err = np.array([s_ci[0], s_ci[1]]).reshape(-1, 1)
                dend_err = np.array([d_ci[0], d_ci[1]]).reshape(-1, 1)
        else:
            return "Only accepts sem, std, and CI for err_type!!!"

        data_points = {"Soma": soma_points, "Dendrite": dend_points}
        data_points = pd.DataFrame(data_points)
        data_means = np.array([soma_mean, dend_mean])
        if err_type != "CI":
            data_err = np.array([soma_err, dend_err])
        else:
            data_err = np.hstack((soma_err, dend_err))

        # make the figure
        fig = plt.figure(figsize=(3, 4))
        ax = fig.add_subplot()
        ax.set_title("Fraction of Events Coactive")
        # Plot the points
        sns.stripplot(data=data_points, palette=["black", "darkorange"], size=4)
        # Plot the means
        ax.bar(
            x=[0, 1],
            height=data_means,
            yerr=data_err,
            width=0.5,
            color=["black", "darkorange"],
            edgecolor="black",
            ecolor="black",
            linewidth=0,
            alpha=0.5,
        )
        sns.despine()
        ax.set_ylabel("Fraction of events coactive")
        ax.set_xticklabels(labels=["Soma", "Dendrite"])

        fig.tight_layout()

        # Save section
        if save:
            if save_path is None:
                save_path = r"C:\Users\Jake\Desktop\Figures"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            fname = os.path.join(save_path, "fraction_of_events_coactive")
            plt.savefig(fname + ".pdf")

    def plot_between_correlations(
        self, data_type, individual=True, save=False, save_path=None
    ):
        """Method to plot correlations between dendritic and somatic events"""
        COL_NUM = 3
        # Get relevant data
        if data_type == "amplitude":
            soma_data = self.soma_amplitudes
            dend_data = self.dend_amplitudes
            title = "Event Amplitudes"
            a_title = "\u0394" + "F/F\u2080"
        elif data_type == "amplitude_norm":
            soma_data = self.soma_amplitudes_norm
            dend_data = self.dend_amplitudes_norm
            title = "Normalized Event Amplitudes"
            a_title = "Normalized" + "\u0394" + "F/F\u2080"
        elif data_type == "tau":
            soma_data = self.soma_tau
            dend_data = self.dend_tau
            title = "Event Decay"
            a_title = f"Time to {fractions.Fraction(1,2)} amplitude (s)"
        elif data_type == "tau_norm":
            soma_data = self.soma_tau_norm
            dend_data = self.dend_tau_norm
            title = "Event Decay"
            a_title = f"Time to {fractions.Fraction(1,2)} amplitude (s)"

        if individual is not True:
            soma_data = np.concatenate(soma_data)
            soma_data = [soma_data]
            dend_data = np.concatenate(dend_data)
            dend_data = [dend_data]

        # Set up the figure and subplot
        num = len(soma_data)
        row_num = num // COL_NUM
        row_num += num % COL_NUM
        figsize = (10, 4 * row_num)
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle(title, fontweight="bold")

        # Make each correlation plot
        count = 1
        for i, (s_data, d_data) in enumerate(zip(soma_data, dend_data)):
            ax = fig.add_subplot(row_num, COL_NUM, count)
            # remove nan values
            s_nan = np.nonzero(~np.isnan(s_data))[0]
            d_nan = np.nonzero(~np.isnan(d_data))[0]
            non_nans = np.intersect1d(s_nan, d_nan)
            s_data = s_data[non_nans]
            d_data = d_data[non_nans]
            # perform correlation of the data
            corr, p = stats.pearsonr(s_data, d_data)
            subtitle = f"Dendrite {i+1}\n r = {np.around(corr, decimals=5)}   p = {p}"
            ax.set_title(subtitle, style="italic")
            scatter_kws = {
                "facecolor": "darkorange",
                "edgecolor": "none",
                "linewidth": 0,
                "alpha": 0.3,
                "s": 15,
            }
            line_kws = {"linewidth": 1, "color": "darkorange"}
            sns.regplot(
                x=s_data,
                y=d_data,
                ci=None,
                scatter_kws=scatter_kws,
                line_kws=line_kws,
                label="Data",
            )
            ax.axline(
                (0, 0),
                slope=1,
                color="black",
                linestyle="--",
                linewidth=0.5,
                label="Unity",
            )

            plt.xlabel(f"Somatic {a_title}", labelpad=15)
            plt.ylabel(f"Dendritic {a_title}", labelpad=15)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            plt.legend()

            count = count + 1

        fig.tight_layout()

        # Save section
        if save:
            if save_path is None:
                save_path = r"C:\Users\Jake\Desktop\Figures"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            p_name = f"{data_type}_{individual}_corr"
            fname = os.path.join(save_path, p_name)
            plt.savefig(fname + ".pdf")

    def plot_within_correlations(
        self, group_type, norm=True, individual=True, save=False, save_path=None
    ):
        """Method to plot different variables within somas and dendrites"""
        COL_NUM = 3
        if group_type == "Soma":
            if norm:
                amplitude = self.soma_amplitudes_norm
                amplitude = [x for i, x in enumerate(amplitude) if (i == 0) or (i == 4)]
                x_title = "Normalized " + "\u0394" + "F/F\u2080"
            else:
                amplitude = self.soma_amplitudes
                amplitude = [x for i, x in enumerate(amplitude) if (i == 0) or (i == 4)]
                x_title = "\u0394" + "F/F\u2080"
            tau = self.soma_tau
            tau = [x for i, x in enumerate(tau) if (i == 0) or (i == 4)]

        elif group_type == "Dendrite":
            if norm:
                amplitude = self.dend_amplitudes_norm
                x_title = "Normalized " + "\u0394" + "F/F\u2080"
            else:
                amplitude = self.dend_amplitudes
                x_title = "\u0394" + "F/F\u2080"
            tau = self.dend_tau

        if individual is not True:
            amplitude = np.concatenate(amplitude)
            amplitude = [amplitude]
            tau = np.concatenate(tau)
            tau = [tau]

        # Set up the figure and subplot
        num = len(amplitude)
        row_num = num // COL_NUM
        row_num += num % COL_NUM
        figsize = (10, 4 * row_num)
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(hspace=0.5)

        # Make each correlation plot
        count = 1
        for i, (a_data, t_data) in enumerate(zip(amplitude, tau)):
            ax = fig.add_subplot(row_num, COL_NUM, count)
            # remove nan values
            a_nan = np.nonzero(~np.isnan(a_data))[0]
            t_nan = np.nonzero(~np.isnan(t_data))[0]
            non_nans = np.intersect1d(a_nan, t_nan)
            a_data = a_data[non_nans]
            t_data = t_data[non_nans]
            # perform correlation of the data
            corr, p = stats.pearsonr(a_data, t_data)
            subtitle = (
                f"{group_type} {i+1}\n r = {np.around(corr, decimals=5)}   p = {p}"
            )
            ax.set_title(subtitle, style="italic")
            scatter_kws = {
                "facecolor": "darkorange",
                "edgecolor": "none",
                "linewidth": 0,
                "alpha": 0.3,
                "s": 15,
            }
            line_kws = {"linewidth": 1, "color": "darkorange"}
            sns.regplot(
                x=a_data,
                y=t_data,
                ci=None,
                scatter_kws=scatter_kws,
                line_kws=line_kws,
                label="Data",
            )
            ax.axline(
                (0, 0),
                slope=1,
                color="black",
                linestyle="--",
                linewidth=0.5,
                label="Unity",
            )

            plt.xlabel(x_title, labelpad=15)
            plt.ylabel(f"Time to {fractions.Fraction(1,2)} amplitude (s)", labelpad=15)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            plt.legend()

            count = count + 1

        fig.tight_layout()

        # Save section
        if save:
            if save_path is None:
                save_path = r"C:\Users\Jake\Desktop\Figures"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            p_name = f"{group_type}_{individual}_corr"
            fname = os.path.join(save_path, p_name)
            plt.savefig(fname + ".pdf")

    def plot_coactive_noncoactive_traces(
        self, norm=True, avg_type="All", save=False, save_path=None
    ):
        """Method to plot the coactive and noncoactive dendritic
            and somatic traces"""
        activity_window = (-1, 4)
        sampling_rate = self.sampling_rate
        # Get relevant data
        if norm:
            dend_coactive = self.coactive_dend_traces_norm
            dend_noncoactive = self.noncoactive_dend_traces_norm
            soma_coactive = self.coactive_soma_traces_norm
            soma_noncoactive = self.noncoactive_soma_traces_norm
            ytitle = "Normalized" + "\u0394" + "F/F\u2080"
        else:
            dend_coactive = self.coactive_dend_traces
            dend_noncoactive = self.noncoactive_dend_traces
            soma_coactive = self.coactive_soma_traces
            soma_noncoactive = self.noncoactive_soma_traces
            ytitle = "\u0394" + "F/F\u2080"

        # Get mean and sems
        if avg_type == "All":
            d_coactive = np.hstack(dend_coactive)
            d_noncoactive = np.hstack(dend_noncoactive)
            s_coactive = np.hstack(soma_coactive)
            s_noncoactive = np.hstack(soma_noncoactive)

        elif avg_type == "Dend":
            d_c_means = []
            d_n_means = []
            s_c_means = []
            s_n_means = []
            for dc, dn, sc, sn in zip(
                dend_coactive, dend_noncoactive, soma_coactive, soma_noncoactive
            ):
                d_c_means.append(np.nanmean(dc, axis=1))
                d_n_means.append(np.nanmean(dn, axis=1))
                s_c_means.append(np.nanmean(sc, axis=1))
                s_n_means.append(np.nanmean(sn, axis=1))
            d_coactive = np.vstack(d_c_means).T
            d_noncoactive = np.vstack(d_n_means).T
            s_coactive = np.vstack(s_c_means).T
            s_noncoactive = np.vstack(s_n_means).T

        dend_co_mean = np.nanmean(d_coactive, axis=1)
        dend_non_mean = np.nanmean(d_noncoactive, axis=1)
        soma_co_mean = np.nanmean(s_coactive, axis=1)
        soma_non_mean = np.nanmean(s_noncoactive, axis=1)
        dend_co_sem = stats.sem(d_coactive, axis=1, nan_policy="omit")
        dend_non_sem = stats.sem(d_noncoactive, axis=1, nan_policy="omit")
        soma_co_sem = stats.sem(s_coactive, axis=1, nan_policy="omit")
        soma_non_sem = stats.sem(s_noncoactive, axis=1, nan_policy="omit")

        # Make the figure
        fig = plt.figure(figsize=(8, 4))
        ## First subplot
        ax1 = fig.add_subplot(1, 2, 1)
        x1 = np.linspace(activity_window[0], activity_window[1], len(dend_co_mean))
        ax1.plot(x1, dend_co_mean, color="darkorange", label="Dendrite")
        ax1.fill_between(
            x1,
            dend_co_mean - dend_co_sem,
            dend_co_mean + dend_co_sem,
            color="darkorange",
            alpha=0.2,
        )
        ax1.plot(x1, soma_co_mean, color="black", label="Soma")
        ax1.fill_between(
            x1,
            soma_co_mean - soma_co_sem,
            soma_co_mean + soma_co_sem,
            color="black",
            alpha=0.2,
        )
        ax1.set_title("Coactive Event Traces")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel(ytitle)
        ax1.set_ylim([-0.05, 0.7])
        plt.xticks(
            ticks=[activity_window[0], 0, activity_window[1]],
            labels=[activity_window[0], 0, activity_window[1]],
        )
        plt.tick_params(axis="both", which="both", direction="in", length=4)
        ax1.legend(loc="upper right")
        ## Second subplot
        ax2 = fig.add_subplot(1, 2, 2)
        x2 = np.linspace(activity_window[0], activity_window[1], len(dend_non_mean))
        ax2.plot(x2, dend_non_mean, color="darkorange", label="Dendrite")
        ax2.fill_between(
            x2,
            dend_non_mean - dend_non_sem,
            dend_non_mean + dend_non_sem,
            color="darkorange",
            alpha=0.2,
        )
        ax2.plot(x2, soma_non_mean, color="black", label="Soma")
        ax2.fill_between(
            x2,
            soma_non_mean - soma_non_sem,
            soma_non_mean + soma_non_sem,
            color="black",
            alpha=0.2,
        )
        ax2.set_title("Non-Coactive Event Traces")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel(ytitle)
        ax2.set_ylim([-0.05, 0.7])
        plt.xticks(
            ticks=[activity_window[0], 0, activity_window[1]],
            labels=[activity_window[0], 0, activity_window[1]],
        )
        plt.tick_params(axis="both", which="both", direction="in", length=4)
        ax2.legend(loc="upper right")

        fig.tight_layout()
        if save:
            if save_path is None:
                save_path = r"C:\Users\Jake\Desktop\Figures"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            fname = os.path.join(save_path, "event_traces")
            plt.savefig(fname + ".pdf")

