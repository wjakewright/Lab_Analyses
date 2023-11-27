import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy as sy
import scipy.signal as sysignal
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.load_mat_files import load_mat

sns.set()
sns.set_style("ticks")


def matlab_spine_properties(data_dir, smooth=True, save=False):
    """Function to get the spine amplitudes and decay kinetics from Nathan's 
        iGluSnFR2 matlab data files
    """
    DISTANCE = 0.5 * 60

    # Load all of the data from matlab
    ## Get filenames
    fnames = next(os.walk(data_dir))[2]
    fnames = [x.split(".")[0] for x in fnames]
    fnames = [x for x in fnames if "Aligned" in x and "Red" not in x]

    datasets = []
    ## Get all the individual structures within the data
    for file in fnames:
        print(file)
        mat_data = load_mat(file, path=data_dir)
        for data in mat_data:
            if type(data) == sy.io.matlab.mio5_params.mat_struct:
                datasets.append(data)

    # Begin analyzing
    event_traces = []
    event_peak_traces = []
    event_amplitudes = []
    event_taus = []
    avg_taus = []

    for data_i, dataset in enumerate(datasets):
        # Pull relevant data
        spine_dFoF = dataset.ProcessedSpineActivity.T
        spine_activity = dataset.BinarizedOverallSpineData.T.astype(np.int16)

        # Iterate through each spine
        for spine_i, spine in enumerate(range(spine_dFoF.shape[1])):
            dFoF = spine_dFoF[:, spine]
            activity = spine_activity[:, spine]

            if np.isnan(dFoF).all():
                continue

            # Get event onsets
            onsets = t_stamps.get_activity_timestamps(activity)
            onsets = [x[0] for x in onsets]
            onsets = t_stamps.refine_activity_timestamps(
                onsets, (-1, 4), len(activity), 60
            )
            if len(onsets) == 0:
                continue

            # Get traces around timestamps
            traces, _ = d_utils.get_trace_mean_sem(
                dFoF.reshape(-1, 1),
                ["Spine"],
                onsets,
                window=(-1, 4),
                sampling_rate=60,
            )
            traces = traces["Spine"]

            amps = np.zeros(traces.shape[1]) * np.nan
            taus = np.zeros(traces.shape[1]) * np.nan
            peak_times = np.zeros(traces.shape[1]) * np.nan

            # Analyze each event
            for event in range(traces.shape[1]):
                t = traces[:, event]
                # Smooth trace
                if smooth:
                    t_smooth = sysignal.savgol_filter(t, 31, 3)
                else:
                    t_smooth = t

                med = np.nanmedian(t_smooth)
                std = np.nanstd(t_smooth)
                height = med + std
                peaks, props = sysignal.find_peaks(
                    t_smooth, height=height, distance=DISTANCE,
                )
                event_amps = props["peak_heights"]

                # Get max peak
                try:
                    max_amp = np.max(event_amps)
                    max_peak = peaks[np.argmax(event_amps)]
                except ValueError:
                    continue

                # Calculate decay kinetics
                decay_trace = t_smooth[max_peak:]
                # Find event offset

                # offset = np.argmin(decay_trace)
                # decay_trace = decay_trace[:offset]
                try:
                    params, _ = sy.optimize.curve_fit(
                        exp_func,
                        np.arange(len(decay_trace)) / 60,
                        decay_trace,
                        maxfev=1000,
                    )
                except:
                    continue
                _, tau, _ = params
                tau = 1 / tau
                # Remove poorly fitted taus
                if tau < 0:
                    continue
                # remove large taus estimating linear fits
                if tau > 50:
                    continue

                amps[event] = max_amp
                taus[event] = tau
                peak_times[event] = max_peak

            # Get average trace tau
            avg_trace = np.nanmean(traces, axis=1)
            if smooth:
                avg_trace = sysignal.savgol_filter(avg_trace, 31, 3)
            avg_med = np.nanmedian(avg_trace)
            avg_std = np.nanstd(avg_trace)
            avg_height = avg_med + avg_std
            avg_peaks, avg_props = sysignal.find_peaks(
                avg_trace, height=avg_height, distance=DISTANCE,
            )
            try:
                avg_peak = avg_peaks[np.argmax(avg_props["peak_heights"])]
            except:
                avg_peak = np.argmax(avg_trace)

            avg_decay_trace = avg_trace[avg_peak:]
            try:
                avg_params, _ = sy.optimize.curve_fit(
                    exp_func,
                    np.arange(len(avg_decay_trace)) / 60,
                    avg_decay_trace,
                    maxfev=1000,
                )
            except:
                continue
            _, a_tau, _ = avg_params
            a_tau = 1 / a_tau
            # Remove poorly fitted taus
            if a_tau < 0:
                continue

            # Store values
            event_traces.append(traces)
            event_amplitudes.append(amps)
            event_taus.append(taus)
            avg_taus.append(a_tau)

            # Get peak time timestamps
            center_point = int(np.absolute(-1) * 60)
            peak_times = [center_point - a for a in peak_times]
            # corrected_tstamps = [int(x - y) for x, y in zip(onsets, peak_times)]
            corrected_tstamps = []
            for x, y in zip(onsets, peak_times):
                if np.isnan(y):
                    continue
                corrected_tstamps.append(int(x - y))
            if len(corrected_tstamps) == 0:
                continue
            peak_traces, _ = d_utils.get_trace_mean_sem(
                dFoF.reshape(-1, 1),
                ["Spine"],
                corrected_tstamps,
                window=(-1, 2),
                sampling_rate=60,
            )
            peak_traces = peak_traces["Spine"]
            event_peak_traces.append(peak_traces)

    # Prepare data for plotting
    all_event_amplitudes = np.concatenate(event_amplitudes)
    all_event_taus = np.concatenate(event_taus)
    average_taus = np.array(avg_taus)

    ## Select random examples
    example_idxs = random.sample(list(np.arange(len(event_traces))), k=6)
    example_peak_traces = [event_peak_traces[i] for i in example_idxs]

    # Make the plot
    examples_fig, example_axes = plt.subplot_mosaic(
        """
        ABCDEF
        HIJKLM
        """,
        figsize=(11, 5),
    )
    examples_fig.tight_layout()
    examples = [("A", "B"), ("C", "D"), ("E", "F"), ("H", "I"), ("J", "K"), ("L", "M")]
    dist_fig, dist_axes = plt.subplot_mosaic(
        """
        ABC
        """,
        figsize=(8, 2.5),
    )
    dist_fig.tight_layout()

    for i, p_trace in enumerate(example_peak_traces):
        axes = examples[i]
        p_trace = np.array(d_utils.zero_window(p_trace, (0, 1), 60))
        # Amplitude plot
        mean_trace = np.nanmean(p_trace, axis=1)
        x = np.linspace(-1, 2, len(mean_trace))
        for ind in range(p_trace.shape[1]):
            example_axes[axes[0]].plot(
                x, p_trace[:, ind], color="darkgrey", alpha=0.3, linewidth=0.5
            )
        example_axes[axes[0]].plot(x, mean_trace, color="black", linewidth=1)
        example_axes[axes[0]].set_title("Example Events")
        # Peak aligned and scaled plot
        s_x = x[30:110]
        scaler = MinMaxScaler()
        scaler.fit(p_trace[30:60])
        scaled_traces = scaler.transform(p_trace)
        scaler.fit(mean_trace[30:60].reshape(-1, 1))
        scaled_mean = scaler.transform(mean_trace.reshape(-1, 1)).flatten()
        for ind in range(scaled_traces.shape[1]):
            example_axes[axes[1]].plot(
                s_x,
                scaled_traces[30:110, ind],
                color="darkgrey",
                alpha=0.3,
                linewidth=0.5,
            )
        example_axes[axes[1]].plot(s_x, scaled_mean[30:110], color="black", linewidth=1)

    # Plot distributions
    ## Remove some outliers
    non_nan_amp = np.nonzero(~np.isnan(all_event_amplitudes))[0]
    all_event_amplitudes = all_event_amplitudes[non_nan_amp]
    q1 = np.percentile(all_event_amplitudes, 25)
    q3 = np.percentile(all_event_amplitudes, 75)
    iqr = q3 - q1
    threshold = 3 * iqr
    all_event_amplitudes = all_event_amplitudes[all_event_amplitudes < q3 + threshold]
    all_event_amplitudes = all_event_amplitudes[all_event_amplitudes > 0]

    plot_histogram(
        data=all_event_amplitudes,
        bins=80,
        stat="probability",
        avlines=[np.nanmedian(all_event_amplitudes)],
        title=f"Event amplitudes\nN = {len(all_event_amplitudes)} median = {np.nanmedian(all_event_amplitudes):.3}",
        xtitle="dF/F",
        xlim=None,
        figsize=(5, 5),
        color="darkgray",
        alpha=1,
        axis_width=1,
        minor_ticks="x",
        tick_len=4,
        ax=dist_axes["A"],
        save=False,
        save_path=r"C:\Users\Jake\Desktop\Figures\Elimination_Paper",
    )
    non_nan_tau = np.nonzero(~np.isnan(all_event_taus))[0]
    all_event_taus = all_event_taus[non_nan_tau]
    q1 = np.percentile(all_event_taus, 25)
    q3 = np.percentile(all_event_taus, 75)
    iqr = q3 - q1
    threshold = 3 * iqr
    cutoff_idxs = np.nonzero(all_event_taus < q3 + threshold)[0]
    all_event_taus = all_event_taus[cutoff_idxs]

    plot_histogram(
        data=all_event_taus,
        bins=80,
        stat="probability",
        avlines=[np.nanmedian(all_event_taus)],
        title=f"Event taus\nN = {len(all_event_taus)} median = {np.nanmedian(all_event_taus):.3}",
        xtitle="Tau",
        xlim=(0, 2),
        figsize=(5, 5),
        color="darkgray",
        alpha=1,
        axis_width=1,
        minor_ticks="x",
        tick_len=4,
        ax=dist_axes["B"],
        save=False,
        save_path=r"C:\Users\Jake\Desktop\Figures\Elimination_Paper",
    )
    q1 = np.percentile(average_taus, 25)
    q3 = np.percentile(average_taus, 75)
    iqr = q3 - q1
    threshold = 5 * iqr
    cutoff_idxs = np.nonzero(average_taus < q3 + threshold)[0]
    average_taus = average_taus[cutoff_idxs]
    plot_histogram(
        data=average_taus,
        bins=80,
        stat="probability",
        avlines=[np.nanmedian(average_taus)],
        title=f"Average event taus\nN = {len(average_taus)} median = {np.nanmedian(average_taus):.3}",
        xtitle="Tau",
        xlim=(0, 2),
        figsize=(5, 5),
        color="darkgray",
        alpha=1,
        axis_width=1,
        minor_ticks="x",
        tick_len=4,
        ax=dist_axes["C"],
        save=False,
        save_path=r"C:\Users\Jake\Desktop\Figures\Elimination_Paper",
    )

    if save:
        examples_fname = os.path.join(
            r"C:\Users\Jake\Desktop\Figures\Elimination_Paper", "Example_traces"
        )
        examples_fig.savefig(examples_fname + ".pdf")
        dist_fname = os.path.join(
            r"C:\Users\Jake\Desktop\Figures\Elimination_Paper", "Dist_plots"
        )
        dist_fig.savefig(dist_fname + ".pdf")


def exp_func(t, A, K, C):
    return A * np.exp(-K * t) + C

