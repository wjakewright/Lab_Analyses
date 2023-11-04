import os

import numpy as np
import scipy.signal as sysignal

from Lab_Analyses.Dual_Plane.dual_plane_dataclass import Dual_Plane_Data
from Lab_Analyses.Dual_Plane.preprocess_dual_plane_data import (
    preprocess_dual_plane_data,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.activity_timestamps import get_activity_timestamps
from Lab_Analyses.Utilities.coactivity_functions import calculate_coactivity
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def analyze_dual_plane_data(mouse_list, save=False, save_path=None):
    """Functoin to analyze the dual plane somatic and dendritic imaging datasets
        Combines data across mice and FOVs
        
        INPUT PARAMETERS
            mouse_list - list of str specifying the mice we wish to analyze
            
            save - boolean of whetehr or not to save the output
            
            save_path - str specifying where to save the data
    """
    # Load datasets
    initial_path = r"C:\Users\Jake\Desktop\Analyzed_data\individual"
    datasets = []
    for mouse in mouse_list:
        top_dir = os.path.join(initial_path, mouse, "dual_plane")
        FOVs = next(os.walk(top_dir))[1]
        for FOV in FOVs:
            FOV_path = os.path.join(top_dir, FOV)
            fname = f"{mouse}_{FOV}_dual_soma_dend_data"
            file = os.path.join(FOV_path, fname)
            data = load_pickle([file])[0]
            datasets.append(data)
            dictionary = d_utils.convert_dataclass_to_dict(data)
            save_pickle(f"{mouse}_{FOV}_dual_soma_dend_dict", dictionary, path=FOV_path)

    print(f"Dataset number: {len(datasets)}")

    # Setup outputs
    dendrite_activity = []
    somatic_activity = []
    dendrite_dFoF = []
    somatic_dFoF = []
    dendrite_dFoF_norm = []
    somatic_dFoF_norm = []
    dendrite_noise = []
    somatic_noise = []
    fraction_dendrite_active = []
    fraction_somatic_active = []
    dendrite_amplitudes = []
    somatic_amplitudes = []
    other_dendrite_amplitudes = []
    dendrite_decay = []
    somatic_decay = []
    dendrite_amplitudes_norm = []
    somatic_amplitudes_norm = []
    other_dendrite_amplitudes_norm = []
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
        # Preprocess the data
        processed_data = preprocess_dual_plane_data(dataset)
        dendrite_activity.append(processed_data["dendrite_activity"])
        somatic_activity.append(processed_data["somatic_activity"])
        dendrite_dFoF.append(processed_data["dendrite_dFoF"])
        somatic_dFoF.append(processed_data["somatic_dFoF"])
        dendrite_dFoF_norm.append(processed_data["dendrite_dFoF_norm"])
        somatic_dFoF_norm.append(processed_data["somatic_dFoF_norm"])
        # Analyze each dendrite seperately
        for dend in range(processed_data["dendrite_dFoF"].shape[1]):
            # Pull the relevant data
            d_dFoF = processed_data["dendrite_dFoF"][:, dend]
            d_dFoF_norm = processed_data["dendrite_dFoF_norm"][:, dend]
            d_activity = processed_data["dendrite_activity"][:, dend]
            s_dFoF = processed_data["somatic_dFoF"][:, dend]
            s_dFoF_norm = processed_data["somatic_dFoF_norm"][:, dend]
            s_activity = processed_data["somatic_activity"][:, dend]

            # Estimate the noise of the traces
            d_below_zero = d_dFoF[d_dFoF < 0]
            d_noise = np.nanstd(np.concatenate((d_below_zero, -d_below_zero)))
            s_below_zero = s_dFoF[s_dFoF < 0]
            s_noise = np.nanstd(np.concatenate((s_below_zero, -s_below_zero)))

            # Calculate the fraction of events coactive
            (_, _, frac_dend_active, _, _) = calculate_coactivity(
                d_activity, s_activity, sampling_rate=processed_data["sampling_rate"],
            )
            (_, _, frac_soma_active, _, _) = calculate_coactivity(
                s_activity, d_activity, sampling_rate=processed_data["sampling_rate"],
            )

            # Get dend activity timestamps
            dend_events = get_activity_timestamps(d_activity)
            dend_events = [x[0] for x in dend_events]

            # Get amps and taus of paired events b/w dendrite and soma
            ## Using raw traces
            (dend_amps, soma_amps, dend_tau, soma_tau) = analyze_paired_events(
                dend_events,
                d_dFoF,
                s_dFoF,
                activity_window=(-1, 4),
                sampling_rate=processed_data["sampling_rate"],
            )
            ## Using normalized traces
            (dend_amps_norm, soma_amps_norm, _, _,) = analyze_paired_events(
                dend_events,
                d_dFoF_norm,
                s_dFoF_norm,
                activity_window=(-1, 4),
                sampling_rate=processed_data["sampling_rate"],
            )

            ## Get amps and taus of paired events between dendrite and other dendrites
            other_dend_amps = []
            other_dend_amps_norm = []
            for partner in range(processed_data["dendrite_dFoF"].shape[1]):
                if dend == partner:
                    continue
                (_, other_amps, _, _) = analyze_paired_events(
                    dend_events,
                    d_dFoF,
                    processed_data["dendrite_dFoF"][:, partner],
                    activity_window=(-1, 4),
                    sampling_rate=processed_data["sampling_rate"],
                )
                (_, other_amps_norm, _, _) = analyze_paired_events(
                    dend_events,
                    d_dFoF_norm,
                    processed_data["dendrite_dFoF_norm"][:, partner],
                    activity_window=(-1, 4),
                    sampling_rate=processed_data["sampling_rate"],
                )
                other_dend_amps.append(other_amps)
                other_dend_amps_norm.append(other_amps_norm)

            # Get coactive and noncoactive traces
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
                sampling_rate=processed_data["sampling_rate"],
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
                sampling_rate=processed_data["sampling_rate"],
            )
            dendrite_noise.append(d_noise)
            somatic_noise.append(s_noise)
            fraction_dendrite_active.append(frac_dend_active)
            fraction_somatic_active.append(frac_soma_active)
            dendrite_amplitudes.append(dend_amps)
            somatic_amplitudes.append(soma_amps)
            other_dendrite_amplitudes.append(other_dend_amps)
            dendrite_decay.append(dend_tau)
            somatic_decay.append(soma_tau)
            dendrite_amplitudes_norm.append(dend_amps_norm)
            somatic_amplitudes_norm.append(soma_amps_norm)
            other_dendrite_amplitudes_norm.append(other_dend_amps_norm)
            coactive_dendrite_traces.append(coactive_dend_traces)
            noncoactive_dendrite_traces.append(noncoactive_dend_traces)
            coactive_somatic_traces.append(coactive_soma_traces)
            noncoactive_somatic_traces.append(noncoactive_soma_traces)
            coactive_dendrite_traces_norm.append(coactive_dend_traces_norm)
            noncoactive_dendrite_traces_norm.append(noncoactive_dend_traces_norm)
            coactive_somatic_traces_norm.append(coactive_soma_traces_norm)
            noncoactive_somatic_traces_norm.append(noncoactive_soma_traces_norm)

    dendrite_activity = np.hstack(d_utils.pad_2d_arrays_rows(dendrite_activity))
    somatic_activity = np.hstack(d_utils.pad_2d_arrays_rows(somatic_activity))
    dendrite_dFoF = np.hstack(d_utils.pad_2d_arrays_rows(dendrite_dFoF))
    somatic_dFoF = np.hstack(d_utils.pad_2d_arrays_rows(somatic_dFoF))
    dendrite_dFoF_norm = np.hstack(d_utils.pad_2d_arrays_rows(dendrite_dFoF_norm))
    somatic_dFoF_norm = np.hstack(d_utils.pad_2d_arrays_rows(somatic_dFoF_norm))
    fraction_dendrite_active = np.array(fraction_dendrite_active)
    fraction_somatic_active = np.array(fraction_somatic_active)
    dendrite_noise = np.array(dendrite_noise)
    somatic_noise = np.array(somatic_noise)

    dual_plane_data = Dual_Plane_Data(
        dendrite_activity=dendrite_activity,
        somatic_activity=somatic_activity,
        dendrite_dFoF=dendrite_dFoF,
        somatic_dFoF=somatic_dFoF,
        dendrite_dFoF_norm=dendrite_dFoF_norm,
        somatic_dFoF_norm=somatic_dFoF_norm,
        dendrite_noise=dendrite_noise,
        somatic_noise=somatic_noise,
        fraction_dendrite_active=fraction_dendrite_active,
        fraction_somatic_active=fraction_somatic_active,
        dendrite_amplitudes=dendrite_amplitudes,
        somatic_amplitudes=somatic_amplitudes,
        other_dendrite_amplitudes=other_dendrite_amplitudes,
        dendrite_decay=dendrite_decay,
        somatic_decay=somatic_decay,
        dendrite_amplitudes_norm=dendrite_amplitudes_norm,
        somatic_amplitudes_norm=somatic_amplitudes_norm,
        other_dendrite_amplitudes_norm=other_dendrite_amplitudes_norm,
        coactive_dendrite_traces=coactive_dendrite_traces,
        noncoactive_dendrite_traces=noncoactive_dendrite_traces,
        coactive_somatic_traces=coactive_somatic_traces,
        noncoactive_somatic_traces=noncoactive_somatic_traces,
        coactive_dendrite_traces_norm=coactive_dendrite_traces_norm,
        noncoactive_dendrite_traces_norm=noncoactive_dendrite_traces_norm,
        coactive_somatic_traces_norm=coactive_somatic_traces_norm,
        noncoactive_somatic_traces_norm=noncoactive_somatic_traces_norm,
        sampling_rate=processed_data["sampling_rate"],
    )

    # save sectino
    if save:
        if save_path is None:
            save_path = (
                r"C:\Users\Jake\Desktop\Analyzed_data\grouped\dual_plane_imaging"
            )
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_name = f"analyzed_dual_plane_data"
        save_pickle(save_name, dual_plane_data, save_path)

    return dual_plane_data


###########################################################################################
################################ HELPER FUNCTIONS #########################################
###########################################################################################


def analyze_paired_events(
    timestamps, dend_dFoF, soma_dFoF, activity_window, sampling_rate
):
    """Helper functon to analyze the amplitudes and decay of paired somatic and dendritic
        events
    """
    DISTANCE = 0.5 * sampling_rate
    # Get traces around timestamps
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

    # Set up outputs
    dend_amps = np.zeros(dend_traces.shape[1])
    dend_tau = np.zeros(dend_traces.shape[1])
    soma_amps = np.zeros(dend_traces.shape[1])
    soma_tau = np.zeros(dend_traces.shape[1])

    # Analyze each paired event
    for event in range(dend_traces.shape[1]):
        d_trace = dend_traces[:, event]
        s_trace = soma_traces[:, event]
        # Smooth the traces
        d_smooth = sysignal.savgol_filter(d_trace, 15, 3)
        s_smooth = sysignal.savgol_filter(s_trace, 15, 3)
        # Find peaks
        d_med = np.nanmedian(d_smooth)
        d_std = np.nanstd(d_smooth)
        d_height = d_med + d_std
        s_med = np.nanmedian(s_smooth)
        s_std = np.nanstd(s_smooth)
        s_height = s_med + s_std

        d_peaks, d_props = sysignal.find_peaks(
            d_smooth, height=d_height, distance=DISTANCE,
        )
        s_peaks, s_props = sysignal.find_peaks(
            s_smooth, height=s_height, distance=DISTANCE,
        )

        # Get max amplitudes
        d_amps = d_props["peak_heights"]
        s_amps = s_props["peak_heights"]
        ## Dendrite
        try:
            d_max_amp = np.max(d_amps)
            d_max_peak = d_peaks[np.argmax(d_amps)]
        except ValueError:
            d_max_amp = np.max(d_smooth)
            d_max_peak = np.argmax(d_smooth)
        ## Soma
        try:
            s_max_amp = np.max(s_amps)
            s_max_peak = s_peaks[np.argmax(s_amps)]
        except ValueError:
            s_max_amp = s_smooth[d_max_peak]
            s_max_peak = d_max_peak

        # Get decay kinetics
        ## Get traces after peaks
        d_decay_trace = d_smooth[d_max_peak:]
        s_decay_trace = s_smooth[s_max_peak:]
        ## Find time to half max
        try:
            d_half_max = np.nonzero(d_decay_trace < 0.5 * d_max_amp)[0][0]
        except IndexError:
            d_half_max = len(d_decay_trace)
        try:
            s_half_max = np.nonzero(s_decay_trace < 0.5 * s_max_amp)[0][0]
        except IndexError:
            s_half_max = len(s_decay_trace)

        # Store data
        dend_amps[event] = d_max_amp
        dend_tau[event] = d_half_max
        soma_amps[event] = s_max_amp
        soma_tau[event] = s_half_max

    return dend_amps, soma_amps, dend_tau, soma_tau


def get_coactive_vs_noncoactive_traces(
    dend_activity, soma_activity, dend_dFoF, soma_dFoF, activity_window, sampling_rate,
):
    """Helper function to get traces during periods where soma and dned are determined
        to be coactive vs when they are not"""

    boundary = 2 * sampling_rate
    # Get active epochs
    timestamps = get_activity_timestamps(dend_activity)
    # Refine onsets to remove events within 2 sec of each other
    r_onsets = []
    r_offsets = []
    for i, stamp in enumerate(timestamps):
        if i == len(timestamps) - 1:
            r_onsets.append(stamp[0])
            r_offsets.append(stamp[1])
            continue
        if stamp[0] + boundary <= timestamps[i + 1][0]:
            r_onsets.append(stamp[0])
            r_offsets.append(stamp[1])
    # Compare active epochs to other trace
    coactive_idxs = []
    noncoactive_idxs = []
    for onset, offset in timestamps:
        if np.sum(soma_activity[onset:offset]):
            coactive_idxs.append(onset)
        else:
            noncoactive_idxs.append(onset)

    # Get traces
    coactive_dend_traces, _ = d_utils.get_trace_mean_sem(
        dend_dFoF.reshape(-1, 1),
        ["Dend"],
        coactive_idxs,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    coactive_soma_traces, _, = d_utils.get_trace_mean_sem(
        soma_dFoF.reshape(-1, 1),
        ["Soma"],
        coactive_idxs,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    noncoactive_dend_traces, _ = d_utils.get_trace_mean_sem(
        dend_dFoF.reshape(-1, 1),
        ["Dend"],
        noncoactive_idxs,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    noncoactive_soma_traces, _, = d_utils.get_trace_mean_sem(
        soma_dFoF.reshape(-1, 1),
        ["Soma"],
        noncoactive_idxs,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    coactive_dend_traces = coactive_dend_traces["Dend"]
    noncoactive_dend_traces = noncoactive_dend_traces["Dend"]
    coactive_soma_traces = coactive_soma_traces["Soma"]
    noncoactive_soma_traces = noncoactive_soma_traces["Soma"]

    return (
        coactive_dend_traces,
        noncoactive_dend_traces,
        coactive_soma_traces,
        noncoactive_soma_traces,
    )

