import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from Lab_Analyses.Plotting.plot_box_plot import plot_box_plot
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import load_spine_datasets
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.coactivity_functions import get_conservative_coactive_binary
from Lab_Analyses.Utilities.mean_trace_functions import analyze_event_activity
from Lab_Analyses.Utilities.save_load_pickle import load_pickle, save_pickle


def spine_calcium_vs_volume(
    mouse_list, fov_type="apical", plot=False, save=False, save_path=None
):
    """ "Function to compare calcium event and volume changes"""

    pre_calcium_amplitudes = []
    post_calcium_amplitudes = []
    pre_calcium_traces = []
    post_calcium_traces = []
    pre_volumes = []
    post_volumes = []
    delta_volumes = []

    for mouse in mouse_list:
        datasets = load_spine_datasets(
            mouse,
            ["Early", "Middle"],
            fov_type=fov_type,
        )
        for FOV, dataset in datasets.items():
            pre_data = dataset["Early"]
            post_data = dataset["Middle"]
            sampling_rate = pre_data.imaging_parameters["Sampling Rate"]
            # Pull pre-session data
            pre_activity = pre_data.spine_GluSnFr_activity
            pre_calcium = pre_data.spine_calcium_processed_dFoF
            pre_volume = pre_data.spine_volume
            pix_to_um = pre_data.imaging_parameters["Zoom"] / 2
            pre_volume_um = (np.sqrt(pre_volume) / pix_to_um) ** 2
            pre_flags = pre_data.spine_flags
            pre_dend_activity = pre_data.dendrite_calcium_activity
            # Pull post-session data
            post_activity = post_data.spine_GluSnFr_activity
            post_calcium = post_data.spine_calcium_processed_dFoF
            post_volume = post_data.spine_volume
            pix_to_um = post_data.imaging_parameters["Zoom"] / 2
            post_volume_um = (np.sqrt(post_volume) / pix_to_um) ** 2
            post_flags = post_data.spine_flags
            post_dend_activity = post_data.dendrite_calcium_activity

            # Calulate volume change
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
            pre_dend_activity = d_utils.subselect_data_by_idxs(
                pre_dend_activity, stable_idxs
            )
            post_activity = d_utils.subselect_data_by_idxs(
                post_activity,
                stable_idxs,
            )
            post_calcium = d_utils.subselect_data_by_idxs(post_calcium, stable_idxs)
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

            # Get the traces and amplitudes
            (
                pre_cal_traces,
                pre_calcium_amplitude,
                _,
            ) = analyze_event_activity(
                pre_calcium,
                pre_onsets,
                activity_window=(-1, 2),
                center_onset=True,
                smooth=True,
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
                smooth=True,
                avg_window=None,
                norm_constant=None,
                sampling_rate=sampling_rate,
            )

            pre_calcium_amplitudes.append(pre_calcium_amplitude)
            post_calcium_amplitudes.append(post_calcium_amplitude)
            pre_volumes.append(pre_volume_um)
            post_volumes.append(post_volume_um)
            delta_volumes.append(relative_volume[-1])
            pre_calcium_traces.append(pre_cal_traces)
            post_calcium_traces.append(post_cal_traces)

    # Concatenate values
    pre_calcium_amplitudes = np.concatenate(pre_calcium_amplitudes)
    post_calcium_amplitudes = np.concatenate(post_calcium_amplitudes)
    post_volumes = np.concatenate(post_volumes)
    pre_volumes = np.concatenate(pre_volumes)
    delta_volumes = np.concatenate(delta_volumes)
    pre_calcium_traces = [y for x in pre_calcium_traces for y in x]
    post_calcium_traces = [y for x in post_calcium_traces for y in x]

    # caluculate calcium amp change
    delta_calcium = post_calcium_amplitudes / pre_calcium_amplitudes

    (
        enlarged,
        shrunken,
        stable,
    ) = classify_plasticity(delta_volumes, (0.25, 0.5), norm=False)

    plastic_groups = {"sLTP": "enlarged", "sLTD": "shrunken", "Stable": "stable"}

    plastic_delta_calcium_amps = {}
    plastic_pre_trace_means = {}
    plastic_pre_trace_sems = {}
    plastic_post_trace_means = {}
    plastic_post_trace_sems = {}

    for key, value in plastic_groups.items():
        spines = eval(value)
        pre_traces = compress(pre_calcium_traces, spines)
        pre_means = [np.nanmean(x, axis=1) for x in pre_traces if type(x) == np.ndarray]
        pre_means = np.vstack(pre_means)
        plastic_pre_trace_means[key] = np.nanmean(pre_means, axis=0)
        plastic_pre_trace_sems[key] = stats.sem(pre_means, axis=0, nan_policy="omit")

        post_traces = compress(post_calcium_traces, spines)
        post_means = [
            np.nanmean(x, axis=1) for x in post_traces if type(x) == np.ndarray
        ]
        post_means = np.vstack(post_means)
        plastic_post_trace_means[key] = np.nanmean(post_means, axis=0)
        plastic_post_trace_sems[key] = stats.sem(post_means, axis=0, nan_policy="omit")
        plastic_delta_calcium_amps[key] = delta_calcium[spines]

    if not plot:
        return
    # Generate some plots
    fig, axes = plt.subplot_mosaic(
        """
        AB
        """,
        figsize=(5, 4),
    )
    fig.suptitle("Volume and Calcium Relationship")
    plot_scatter_correlation(
        x_var=delta_volumes,
        y_var=delta_calcium,
        CI=95,
        title=None,
        xtitle="\u0394 volume",
        ytitle="\u0394 calcium amplitude",
        figsize=(5, 5),
        xlim=(0, 6),
        ylim=(0, 6),
        marker_size=15,
        face_color="cmap",
        edge_color="white",
        line_color="black",
        s_alpha=1,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        unity=True,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    plot_box_plot(
        data_dict=plastic_delta_calcium_amps,
        figsize=(5, 5),
        title=None,
        ytitle="\u0394 calcium amplitude",
        ylim=(
            0,
            None,
        ),
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
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    fig.tight_layout()
