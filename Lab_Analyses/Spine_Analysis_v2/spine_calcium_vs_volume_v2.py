import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_box_plot import plot_box_plot
from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import load_spine_datasets
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils
from Lab_Analyses.Utilities.coactivity_functions import get_conservative_coactive_binary
from Lab_Analyses.Utilities.mean_trace_functions import analyze_event_activity


def spine_calcium_vs_volume(
    mouse_list,
    fov_type="apical",
    transform=False,
    LTP_id=None,
    LTD_id=None,
    stable_id=None,
    save=False,
    save_path=None,
):
    """Function to compare calcium event amplitudes and volume chagnes

    INPUT PARAMETERS
        mouse_list - list of strs with mouse ids to analyze

       fov_type -  str specifying the type of FOV. All will combine
                    apical and basal

        transform - boolean specifying whether to log transform data
                    for plotting

        save - boolean specifying whether to save the data or not

        save_path - str specifying where to save the data


    """
    # Important outputs
    pre_calcium_amplitudes = []
    post_calcium_amplitudes = []
    pre_glutamate_amplitudes = []
    post_glutamate_amplitudes = []
    pre_calcium_traces = []
    post_calcium_traces = []
    pre_glutamate_traces = []
    post_glutamate_traces = []
    delta_volumes = []
    mouse_ids = []
    fovs = []

    for mouse in mouse_list:
        # Load the data
        if fov_type != "all":
            datasets = load_spine_datasets(
                mouse, ["Early", "Middle"], fov_type=fov_type
            )
        else:
            apical_datasets = load_spine_datasets(
                mouse, ["Early", "Middle"], fov_type="apical"
            )
            basal_datasets = load_spine_datasets(
                mouse, ["Early", "Middle"], fov_type="basal"
            )
            # Combine dictionaries
            datasets = {**apical_datasets, **basal_datasets}

        # Iterate through each dataset
        for FOV, dataset in datasets.items():
            pre_data = dataset["Early"]
            post_data = dataset["Middle"]
            curr_id = pre_data.mouse_id
            sampling_rate = pre_data.imaging_parameters["Sampling Rate"]
            # Pull pre-session data
            pre_activity = pre_data.spine_GluSnFr_activity
            pre_calcium = pre_data.spine_calcium_processed_dFoF
            pre_glutamate = pre_data.spine_GluSnFr_processed_dFoF
            pre_volume = pre_data.corrected_spine_volume
            pix_to_um = pre_data.imaging_parameters["Zoom"] / 2
            pre_volume_um = (np.sqrt(pre_volume) / pix_to_um) ** 2
            pre_flags = pre_data.spine_flags
            pre_dend_activity = pre_data.dendrite_calcium_activity
            # Pull post session data
            post_activity = post_data.spine_GluSnFr_activity
            post_calcium = post_data.spine_calcium_processed_dFoF
            post_glutamate = post_data.spine_GluSnFr_processed_dFoF
            post_volume = post_data.corrected_spine_volume
            pix_to_um = post_data.imaging_parameters["Zoom"] / 2
            post_volume_um = (np.sqrt(post_volume) / pix_to_um) ** 2
            post_flags = post_data.spine_flags
            post_dend_activity = post_data.dendrite_calcium_activity

            # Calculate volume change
            relative_volume, stable_idxs = calculate_volume_change(
                [pre_volume_um, post_volume_um],
                [pre_flags, post_flags],
                norm=False,
                exclude="Shaft Spine",
            )
            # Subselect the stable spine data
            pre_activity = d_utils.subselect_data_by_idxs(pre_activity, stable_idxs)
            pre_calcium = d_utils.subselect_data_by_idxs(
                pre_calcium,
                stable_idxs,
            )
            pre_glutamate = d_utils.subselect_data_by_idxs(pre_glutamate, stable_idxs)
            pre_dend_activity = d_utils.subselect_data_by_idxs(
                pre_dend_activity, stable_idxs
            )
            post_activity = d_utils.subselect_data_by_idxs(
                post_activity,
                stable_idxs,
            )
            post_calcium = d_utils.subselect_data_by_idxs(post_calcium, stable_idxs)
            post_glutamate = d_utils.subselect_data_by_idxs(post_glutamate, stable_idxs)
            post_dend_activity = d_utils.subselect_data_by_idxs(
                post_dend_activity, stable_idxs
            )

            # Get activity onsets
            pre_onsets = []
            post_onsets = []
            for i in range(pre_activity.shape[1]):
                _, pre_act = get_conservative_coactive_binary(
                    pre_activity[:, i], pre_dend_activity[:, i]
                )
                # pre_act = pre_activity[:, i]
                pre_stamps = t_stamps.get_activity_timestamps(pre_act)
                _, post_act = get_conservative_coactive_binary(
                    post_activity[:, i], post_dend_activity[:, i]
                )
                # post_act = post_activity[:, i]
                post_stamps = t_stamps.get_activity_timestamps(post_act)
                pre_stamps = t_stamps.refine_activity_timestamps(
                    pre_stamps,
                    window=(-1, 2),
                    max_len=(len(pre_activity[:, i])),
                    sampling_rate=sampling_rate,
                )
                pre_stamps = [x[0] for x in pre_stamps]
                post_stamps = t_stamps.refine_activity_timestamps(
                    post_stamps,
                    window=(-1, 2),
                    max_len=len(post_activity[:, i]),
                    sampling_rate=sampling_rate,
                )
                post_stamps = [x[0] for x in post_stamps]
                pre_onsets.append(pre_stamps)
                post_onsets.append(post_stamps)

            # Get traces and amplitudes
            (
                pre_cal_traces,
                pre_calcium_amplitude,
                _,
            ) = analyze_event_activity(
                pre_calcium,
                pre_onsets,
                activity_window=(-1, 2),
                center_onset=True,
                smooth=False,
                avg_window=None,
                norm_constant=None,
                sampling_rate=sampling_rate,
            )
            (
                post_cal_traces,
                post_calcium_amplitude,
                _,
            ) = analyze_event_activity(
                post_calcium,
                post_onsets,
                activity_window=(-1, 2),
                center_onset=True,
                smooth=False,
                avg_window=None,
                norm_constant=None,
                sampling_rate=sampling_rate,
            )

            (
                pre_glu_traces,
                pre_glu_amplitude,
                _,
            ) = analyze_event_activity(
                pre_glutamate,
                pre_onsets,
                activity_window=(-1, 2),
                center_onset=True,
                smooth=False,
                avg_window=None,
                norm_constant=None,
                sampling_rate=sampling_rate,
            )
            (
                post_glu_traces,
                post_glu_amplitude,
                _,
            ) = analyze_event_activity(
                post_glutamate,
                post_onsets,
                activity_window=(-1, 2),
                center_onset=True,
                smooth=False,
                avg_window=None,
                norm_constant=None,
                sampling_rate=sampling_rate,
            )

            pre_calcium_amplitudes.append(pre_calcium_amplitude)
            pre_calcium_traces.append(pre_cal_traces)
            post_calcium_amplitudes.append(post_calcium_amplitude)
            post_calcium_traces.append(post_cal_traces)
            delta_volumes.append(relative_volume[-1])
            pre_glutamate_amplitudes.append(pre_glu_amplitude)
            post_glutamate_amplitudes.append(post_glu_amplitude)
            pre_glutamate_traces.append(pre_glu_traces)
            post_glutamate_traces.append(post_glu_traces)
            mouse_ids.append([curr_id for i in range(len(post_cal_traces))])
            fovs.append([FOV for i in range(len(post_cal_traces))])

            # Find example
            if LTP_id is not None:
                ltp_code = LTP_id.split(" ")
                if (ltp_code[0] == curr_id) and (ltp_code[1] == FOV):
                    pre_LTP_cal_example = pre_cal_traces[int(ltp_code[2])]
                    post_LTP_cal_example = post_cal_traces[int(ltp_code[2])]
                    pre_LTP_glu_example = pre_glu_traces[int(ltp_code[2])]
                    post_LTP_glu_example = post_glu_traces[int(ltp_code[2])]
            if LTD_id is not None:
                ltd_code = LTD_id.split(" ")
                if (ltd_code[0] == curr_id) and (ltd_code[1] == FOV):
                    pre_LTD_cal_example = pre_cal_traces[int(ltd_code[2])]
                    post_LTD_cal_example = post_cal_traces[int(ltd_code[2])]
                    pre_LTD_glu_example = pre_glu_traces[int(ltd_code[2])]
                    post_LTD_glu_example = post_glu_traces[int(ltd_code[2])]
            if stable_id is not None:
                stable_code = stable_id.split(" ")
                if (stable_code[0] == curr_id) and (stable_code[1] == FOV):
                    pre_Stable_cal_example = pre_cal_traces[int(stable_code[2])]
                    post_Stable_cal_example = post_cal_traces[int(stable_code[2])]
                    pre_Stable_glu_example = pre_glu_traces[int(stable_code[2])]
                    post_Stable_glu_example = post_glu_traces[int(stable_code[2])]

    # Concatenate values
    delta_volumes = np.concatenate(delta_volumes)
    pre_calcium_amplitudes = np.concatenate(pre_calcium_amplitudes)
    pre_calcium_traces = [y for x in pre_calcium_traces for y in x]
    post_calcium_amplitudes = np.concatenate(post_calcium_amplitudes)
    post_calcium_traces = [y for x in post_calcium_traces for y in x]
    pre_glutamate_amplitudes = np.concatenate(pre_glutamate_amplitudes)
    pre_glutamate_traces = [y for x in pre_glutamate_traces for y in x]
    post_glutamate_amplitudes = np.concatenate(post_glutamate_amplitudes)
    post_glutamate_traces = [y for x in post_glutamate_traces for y in x]

    mouse_ids = [y for x in mouse_ids for y in x]
    fovs = [y for x in fovs for y in x]
    # Calculate calcium amplitude change
    delta_calcium = post_calcium_amplitudes / pre_calcium_amplitudes
    delta_glutamate = post_glutamate_amplitudes / pre_glutamate_amplitudes

    # Classify plasticity
    (
        enlarged,
        shrunken,
        stable,
    ) = classify_plasticity(delta_volumes, (0.25, 0.5), norm=False)

    plastic_groups = {"sLTP": "enlarged", "sLTD": "shrunken", "Stable": "stable"}

    # Subset into plastic groups
    plastic_delta_calcium_amps = {}
    plastic_pre_trace_means = {}
    plastic_pre_trace_sems = {}
    plastic_post_trace_means = {}
    plastic_post_trace_sems = {}

    plastic_delta_glutamate_amps = {}
    plastic_pre_glu_means = {}
    plastic_pre_glu_sems = {}
    plastic_post_glu_means = {}
    plastic_post_glu_sems = {}

    for m, f, d, p in zip(mouse_ids, fovs, delta_calcium, delta_volumes):
        print(f"Mouse: {m}  FOV {f}  Calcium {d}  Volume {p}")

    for key, value in plastic_groups.items():
        spines = eval(value)
        pre_traces = compress(pre_calcium_traces, spines)
        pre_means = [np.nanmean(x, axis=1) for x in pre_traces if type(x) == np.ndarray]
        pre_means = [x - x[0] for x in pre_means]
        pre_means = np.vstack(pre_means)
        plastic_pre_trace_means[key] = np.nanmean(pre_means, axis=0)
        plastic_pre_trace_sems[key] = stats.sem(pre_means, axis=0, nan_policy="omit")

        post_traces = compress(post_calcium_traces, spines)
        post_means = [
            np.nanmean(x, axis=1) for x in post_traces if type(x) == np.ndarray
        ]
        post_means = [x - x[0] for x in post_means]
        post_means = np.vstack(post_means)
        plastic_post_trace_means[key] = np.nanmean(post_means, axis=0)
        plastic_post_trace_sems[key] = stats.sem(post_means, axis=0, nan_policy="omit")
        plastic_delta_calcium_amps[key] = delta_calcium[spines]

        pre_g_traces = compress(pre_glutamate_traces, spines)
        pre_g_means = [
            np.nanmean(x, axis=1) for x in pre_g_traces if type(x) == np.ndarray
        ]
        pre_g_means = [x - x[0] for x in pre_g_means]
        pre_g_means = np.vstack(pre_g_means)
        plastic_pre_glu_means[key] = np.nanmean(pre_g_means, axis=0)
        plastic_pre_glu_sems[key] = stats.sem(pre_g_means, axis=0, nan_policy="omit")

        post_g_traces = compress(post_glutamate_traces, spines)
        post_g_means = [
            np.nanmean(x, axis=1) for x in post_g_traces if type(x) == np.ndarray
        ]
        post_g_means = [x - x[0] for x in post_g_means]
        post_g_means = np.vstack(post_g_means)
        plastic_post_glu_means[key] = np.nanmean(post_g_means, axis=0)
        plastic_post_glu_sems[key] = stats.sem(post_g_means, axis=0, nan_policy="omit")
        plastic_delta_glutamate_amps[key] = delta_glutamate[spines]

    # Generate the plots
    fig, axes = plt.subplot_mosaic(
        """
        ABC.
        EFGH
        IJKL
        MNO.
        PQR.
        """,
        figsize=(10, 15),
    )
    fig.suptitle("Volume and Calcium Plasticity")

    ####################### Plot data onto axes #########################

    # Correlations
    if transform:
        plot_vol = np.log10(delta_volumes)
        plot_cal = np.log10(delta_calcium)
        plot_glu = np.log10(delta_glutamate)
        xlim = (-1.2, 1.2)
        ylim = (-1.2, 1.2)
    else:
        plot_vol = delta_volumes
        plot_cal = delta_calcium
        plot_glu = delta_glutamate
        xlim = (0, 6)
        ylim = (0, 6)
    ## Calcium vs volume
    plot_scatter_correlation(
        x_var=plot_vol,
        y_var=plot_cal,
        CI=95,
        title=None,
        xtitle="\u0394 volume",
        ytitle="\u0394 avg calcium amplitude",
        figsize=(5, 5),
        xlim=xlim,
        ylim=ylim,
        marker_size=15,
        face_color="cmap",
        cmap_color="plasma",
        edge_color="white",
        line_color="black",
        s_alpha=1,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        unity=False,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    ## Calcium vs volume
    plot_scatter_correlation(
        x_var=plot_vol,
        y_var=plot_glu,
        CI=95,
        title=None,
        xtitle="\u0394 volume",
        ytitle="\u0394 avg glutamate amplitude",
        figsize=(5, 5),
        xlim=xlim,
        ylim=None,
        marker_size=15,
        face_color="cmap",
        cmap_color="plasma",
        edge_color="white",
        line_color="black",
        s_alpha=1,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        unity=False,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    ## Calcium vs glutamate
    plot_scatter_correlation(
        x_var=plot_cal,
        y_var=plot_glu,
        CI=95,
        title=None,
        xtitle="\u0394 avg calcium amplitude",
        ytitle="\u0394 avg glutamate amplitude",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=15,
        face_color="cmap",
        cmap_color="plasma",
        edge_color="white",
        line_color="black",
        s_alpha=1,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        unity=False,
        ax=axes["C"],
        save=False,
        save_path=None,
    )

    # Plot traces
    ### sLTP
    plot_mean_activity_traces(
        means=[
            plastic_pre_trace_means["sLTP"],
            plastic_post_trace_means["sLTP"],
        ],
        sems=[plastic_pre_trace_sems["sLTP"], plastic_post_trace_sems["sLTP"]],
        group_names=["Pre", "Post"],
        sampling_rate=sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=["mediumslateblue", "mediumblue"],
        title="sLTP",
        ytitle="dFoF",
        ylim=(-0.01, 0.06),
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ### sLTD
    plot_mean_activity_traces(
        means=[
            plastic_pre_trace_means["sLTD"],
            plastic_post_trace_means["sLTD"],
        ],
        sems=[plastic_pre_trace_sems["sLTD"], plastic_post_trace_sems["sLTD"]],
        group_names=["Pre", "Post"],
        sampling_rate=sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=["tomato", "darkred"],
        title="sLTD",
        ytitle="dFoF",
        ylim=(-0.01, 0.06),
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ### stable
    plot_mean_activity_traces(
        means=[
            plastic_pre_trace_means["Stable"],
            plastic_post_trace_means["Stable"],
        ],
        sems=[
            plastic_pre_trace_sems["Stable"],
            plastic_post_trace_sems["Stable"],
        ],
        group_names=["Pre", "Post"],
        sampling_rate=sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=["silver", "black"],
        title="Stable",
        ytitle="dFoF",
        ylim=(-0.01, 0.06),
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )

    # Box plots
    plot_box_plot(
        plastic_delta_calcium_amps,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Avg. event amplitude (dF/F)",
        ylim=(0, None),
        b_colors=["mediumslateblue", "tomato", "silver"],
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        b_err_alpha=1,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    axes["H"].axhline(y=1, color="black", linestyle="--", linewidth=0.5)

    # Plot traces
    ### sLTP
    plot_mean_activity_traces(
        means=[
            plastic_pre_glu_means["sLTP"],
            plastic_post_glu_means["sLTP"],
        ],
        sems=[plastic_pre_glu_sems["sLTP"], plastic_post_glu_sems["sLTP"]],
        group_names=["Pre", "Post"],
        sampling_rate=sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=["mediumslateblue", "mediumblue"],
        title="sLTP Glut",
        ytitle="dFoF",
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    ### sLTD
    plot_mean_activity_traces(
        means=[
            plastic_pre_glu_means["sLTD"],
            plastic_post_glu_means["sLTD"],
        ],
        sems=[plastic_pre_glu_sems["sLTD"], plastic_post_glu_sems["sLTD"]],
        group_names=["Pre", "Post"],
        sampling_rate=sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=["tomato", "darkred"],
        title="sLTD Glut",
        ytitle="dFoF",
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    ### stable
    plot_mean_activity_traces(
        means=[
            plastic_pre_glu_means["Stable"],
            plastic_post_glu_means["Stable"],
        ],
        sems=[
            plastic_pre_glu_sems["Stable"],
            plastic_post_glu_sems["Stable"],
        ],
        group_names=["Pre", "Post"],
        sampling_rate=sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=["silver", "black"],
        title="Stable Glut",
        ytitle="dFoF",
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["K"],
        save=False,
        save_path=None,
    )

    # Box plots
    plot_box_plot(
        plastic_delta_glutamate_amps,
        figsize=(5, 5),
        title="Glut",
        xtitle=None,
        ytitle="Avg. event amplitude (dF/F)",
        ylim=(0, 3),
        b_colors=["mediumslateblue", "tomato", "silver"],
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        b_err_alpha=1,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    axes["L"].axhline(y=1, color="black", linestyle="--", linewidth=0.5)

    ### sLTP Example calcium
    if LTP_id:
        pre_LTP_cal_example = np.array(pre_LTP_cal_example)
        post_LTP_cal_example = np.array(post_LTP_cal_example)
        plot_mean_activity_traces(
            means=[
                np.nanmean(pre_LTP_cal_example, axis=1).flatten(),
                np.nanmean(post_LTP_cal_example, axis=1).flatten(),
            ],
            sems=[
                stats.sem(pre_LTP_cal_example, axis=1, nan_policy="omit").flatten(),
                stats.sem(post_LTP_cal_example, axis=1, nan_policy="omit").flatten(),
            ],
            group_names=["Pre", "Post"],
            sampling_rate=sampling_rate,
            activity_window=(-1, 2),
            avlines=None,
            ahlines=None,
            figsize=(5, 5),
            colors=["mediumslateblue", "mediumblue"],
            title="sLTP Ind.",
            ytitle="dFoF",
            ylim=None,
            axis_width=1.5,
            minor_ticks="both",
            tick_len=3,
            ax=axes["M"],
            save=False,
            save_path=None,
        )
        pre_LTP_glu_example = np.array(pre_LTP_glu_example)
        post_LTP_glu_example = np.array(post_LTP_glu_example)
        plot_mean_activity_traces(
            means=[
                np.nanmean(pre_LTP_glu_example, axis=1).flatten(),
                np.nanmean(post_LTP_glu_example, axis=1).flatten(),
            ],
            sems=[
                stats.sem(pre_LTP_glu_example, axis=1, nan_policy="omit").flatten(),
                stats.sem(post_LTP_glu_example, axis=1, nan_policy="omit").flatten(),
            ],
            group_names=["Pre", "Post"],
            sampling_rate=sampling_rate,
            activity_window=(-1, 2),
            avlines=None,
            ahlines=None,
            figsize=(5, 5),
            colors=["mediumslateblue", "mediumblue"],
            title="sLTP Ind.",
            ytitle="dFoF",
            ylim=None,
            axis_width=1.5,
            minor_ticks="both",
            tick_len=3,
            ax=axes["P"],
            save=False,
            save_path=None,
        )

    if LTD_id:
        pre_LTD_cal_example = np.array(pre_LTD_cal_example)
        post_LTD_cal_example = np.array(post_LTD_cal_example)
        plot_mean_activity_traces(
            means=[
                np.nanmean(pre_LTD_cal_example, axis=1).flatten(),
                np.nanmean(post_LTD_cal_example, axis=1).flatten(),
            ],
            sems=[
                stats.sem(pre_LTD_cal_example, axis=1, nan_policy="omit").flatten(),
                stats.sem(post_LTD_cal_example, axis=1, nan_policy="omit").flatten(),
            ],
            group_names=["Pre", "Post"],
            sampling_rate=sampling_rate,
            activity_window=(-1, 2),
            avlines=None,
            ahlines=None,
            figsize=(5, 5),
            colors=["tomato", "darkred"],
            title="sLTD Ind.",
            ytitle="dFoF",
            ylim=None,
            axis_width=1.5,
            minor_ticks="both",
            tick_len=3,
            ax=axes["N"],
            save=False,
            save_path=None,
        )
        pre_LTD_glu_example = np.array(pre_LTD_glu_example)
        post_LTD_glu_example = np.array(post_LTD_glu_example)
        plot_mean_activity_traces(
            means=[
                np.nanmean(pre_LTD_glu_example, axis=1).flatten(),
                np.nanmean(post_LTD_glu_example, axis=1).flatten(),
            ],
            sems=[
                stats.sem(pre_LTD_glu_example, axis=1, nan_policy="omit").flatten(),
                stats.sem(post_LTD_glu_example, axis=1, nan_policy="omit").flatten(),
            ],
            group_names=["Pre", "Post"],
            sampling_rate=sampling_rate,
            activity_window=(-1, 2),
            avlines=None,
            ahlines=None,
            figsize=(5, 5),
            colors=["tomato", "darkred"],
            title="sLTD Ind.",
            ytitle="dFoF",
            ylim=None,
            axis_width=1.5,
            minor_ticks="both",
            tick_len=3,
            ax=axes["Q"],
            save=False,
            save_path=None,
        )
    if stable_id:
        pre_Stable_cal_example = np.array(pre_Stable_cal_example)
        post_Stable_cal_example = np.array(post_Stable_cal_example)
        plot_mean_activity_traces(
            means=[
                np.nanmean(pre_Stable_cal_example, axis=1).flatten(),
                np.nanmean(post_Stable_cal_example, axis=1).flatten(),
            ],
            sems=[
                stats.sem(pre_Stable_cal_example, axis=1, nan_policy="omit").flatten(),
                stats.sem(post_Stable_cal_example, axis=1, nan_policy="omit").flatten(),
            ],
            group_names=["Pre", "Post"],
            sampling_rate=sampling_rate,
            activity_window=(-1, 2),
            avlines=None,
            ahlines=None,
            figsize=(5, 5),
            colors=["silver", "black"],
            title="Stable Ind.",
            ytitle="dFoF",
            ylim=None,
            axis_width=1.5,
            minor_ticks="both",
            tick_len=3,
            ax=axes["O"],
            save=False,
            save_path=None,
        )
        pre_Stable_glu_example = np.array(pre_Stable_glu_example)
        post_Stable_glu_example = np.array(post_Stable_glu_example)
        plot_mean_activity_traces(
            means=[
                np.nanmean(pre_Stable_glu_example, axis=1).flatten(),
                np.nanmean(post_Stable_glu_example, axis=1).flatten(),
            ],
            sems=[
                stats.sem(pre_Stable_glu_example, axis=1, nan_policy="omit").flatten(),
                stats.sem(post_Stable_glu_example, axis=1, nan_policy="omit").flatten(),
            ],
            group_names=["Pre", "Post"],
            sampling_rate=sampling_rate,
            activity_window=(-1, 2),
            avlines=None,
            ahlines=None,
            figsize=(5, 5),
            colors=["silver", "black"],
            title="Stable Ind.",
            ytitle="dFoF",
            ylim=None,
            axis_width=1.5,
            minor_ticks="both",
            tick_len=3,
            ax=axes["R"],
            save=False,
            save_path=None,
        )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Volume_vs_Calcium_Plasticity")
        fig.savefig(fname + ".pdf")

    # Statistics
    cal_f, cal_p, cal_df = t_utils.kruskal_wallis_test(
        plastic_delta_calcium_amps,
        "Conover",
        "fdr_tsbh",
    )
    print(f"Calcium amp: f = {cal_f}  p = {cal_p}")
    print("-------------Post Tests----------------")
    print(cal_df)
    print("+++++++++++++++++++++++++++++++++++++++")
    print("One-sample tests against no change")
    LTP_diff = plastic_delta_calcium_amps["sLTP"] - 1
    sLTP_t, sLTP_p = stats.wilcoxon(
        x=LTP_diff[~np.isnan(LTP_diff)],
        zero_method="zsplit",
    )
    print(f"sLTP: t = {sLTP_t}  p = {sLTP_p}")
    LTD_diff = plastic_delta_calcium_amps["sLTD"] - 1
    sLTD_t, sLTD_p = stats.wilcoxon(
        x=LTD_diff[~np.isnan(LTD_diff)],
        zero_method="zsplit",
    )
    print(f"sLTD: t = {sLTD_t}  p = {sLTD_p}")
    stable_diff = plastic_delta_calcium_amps["Stable"] - 1
    stable_t, stable_p = stats.wilcoxon(
        x=stable_diff[~np.isnan(stable_diff)],
        zero_method="zsplit",
    )
    print(f"Stable: t = {stable_t}  p = {stable_p}")

    print("#########################################")

    glu_f, glu_p, glu_df = t_utils.kruskal_wallis_test(
        plastic_delta_glutamate_amps,
        "Conover",
        "fdr_tsbh",
    )
    print(f"Glutamate amp: f = {glu_f}  p = {glu_p}")
    print("-------------Post Tests----------------")
    print(glu_df)
    print("+++++++++++++++++++++++++++++++++++++++")
    print("One-sample tests against no change")
    LTP_g_diff = plastic_delta_glutamate_amps["sLTP"] - 1
    sLTP_g_t, sLTP_g_p = stats.wilcoxon(
        x=LTP_g_diff[~np.isnan(LTP_g_diff)],
        zero_method="zsplit",
    )
    print(f"sLTP: t = {sLTP_g_t}  p = {sLTP_g_p}")
    LTD_g_diff = plastic_delta_glutamate_amps["sLTD"] - 1
    sLTD_g_t, sLTD_g_p = stats.wilcoxon(
        x=LTD_g_diff[~np.isnan(LTD_g_diff)],
        zero_method="zsplit",
    )
    print(f"sLTD: t = {sLTD_g_t}  p = {sLTD_g_p}")
    stable_g_diff = plastic_delta_glutamate_amps["Stable"] - 1
    stable_g_t, stable_g_p = stats.wilcoxon(
        x=stable_g_diff[~np.isnan(stable_g_diff)],
        zero_method="zsplit",
    )
    print(f"Stable: t = {stable_g_t}  p = {stable_g_p}")
