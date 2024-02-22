import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_box_plot import plot_box_plot
from Lab_Analyses.Plotting.plot_cummulative_distribution import (
    plot_cummulative_distribution,
)
from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Plotting.plot_multi_line_plot import plot_multi_line_plot
from Lab_Analyses.Plotting.plot_pie_chart import plot_pie_chart
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    find_present_spines,
    load_spine_datasets,
)
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_spine_dynamics,
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils

sns.set()
sns.set_style("ticks")


def plot_kir_vs_ctl_basic_props(
    kir_activity_dataset,
    ctl_activity_dataset,
    kir_coactivity_dataset,
    ctl_coactivity_dataset,
    fov_type="apical",
    figsize=(7, 7),
    threshold=0.3,
    showmeans=False,
    hist_bins=30,
    save=False,
    save_path=None,
):
    """Function to plot the basic properties of kir and ctl spines

    INPUT PARAMETERS
        kir_activity_dataset - Kir_Activity_Data object

        ctl_activity_dataset - Spine_Activity_Data object

        kir_coactivity_dataset - Kir_Coactivity_data object

        ctl_coactivity_dataset - Local_Coactivity_Data object

        figsize - tuple specifying the save of the figure

        threshold - float or int specifying the thresholds for plasticity

        showmeans - boolean specifying whether to show means on box plots

        hist_bins - int specifying the bins for the histogram

        save - boolean specifying whether to save the figure or not

        save_path - str specifying where to save the figure

    """

    COLORS = ["grey", "darkviolet"]

    # pull relvant data
    ## Find present spines
    kir_present = find_present_spines(kir_activity_dataset.spine_flags)
    ctl_present = find_present_spines(ctl_activity_dataset.spine_flags)

    print(f"Kir spines: {np.sum(kir_present)}")
    print(f"Ctl spines: {np.sum(ctl_present)}")

    kir_volume = kir_activity_dataset.spine_volumes
    ctl_volume = ctl_activity_dataset.spine_volumes

    kir_event_rate = d_utils.subselect_data_by_idxs(
        kir_activity_dataset.spine_activity_rate, kir_present
    )
    ctl_event_rate = d_utils.subselect_data_by_idxs(
        ctl_activity_dataset.spine_activity_rate, ctl_present
    )

    kir_distance_coactivity = d_utils.subselect_data_by_idxs(
        kir_coactivity_dataset.distance_coactivity_rate, kir_present
    )
    ctl_distance_coactivity = d_utils.subselect_data_by_idxs(
        ctl_coactivity_dataset.distance_coactivity_rate, ctl_present
    )
    kir_distance_coactivity_norm = d_utils.subselect_data_by_idxs(
        kir_coactivity_dataset.distance_coactivity_rate_norm, kir_present
    )
    ctl_distance_coactivity_norm = d_utils.subselect_data_by_idxs(
        ctl_coactivity_dataset.distance_coactivity_rate_norm, ctl_present
    )
    distance_bins = kir_coactivity_dataset.parameters["position bins"][1:]

    # Calculate volume change
    kir_volume_list = [kir_volume, kir_activity_dataset.followup_volumes]
    kir_flag_list = [
        kir_activity_dataset.spine_flags,
        kir_activity_dataset.followup_flags,
    ]
    ctl_volume_list = [ctl_volume, ctl_activity_dataset.followup_volumes]
    ctl_flag_list = [
        ctl_activity_dataset.spine_flags,
        ctl_activity_dataset.followup_flags,
    ]

    kir_delta_volume, _ = calculate_volume_change(
        kir_volume_list, kir_flag_list, norm=False, exclude="Shaft Spine"
    )
    kir_delta_volume = kir_delta_volume[-1]
    ctl_delta_volume, _ = calculate_volume_change(
        ctl_volume_list, ctl_flag_list, norm=False, exclude="Shaft Spine"
    )
    ctl_delta_volume = ctl_delta_volume[-1]

    kir_enlarged_spines, kir_shrunken_spines, kir_stable_spines = classify_plasticity(
        kir_delta_volume, threshold=threshold, norm=False
    )
    ctl_enlarged_spines, ctl_shrunken_spines, ctl_stable_spines = classify_plasticity(
        ctl_delta_volume, threshold=threshold, norm=False
    )

    # Analyze spine dynamics
    kir_density = []
    kir_new_spines = []
    kir_elim_spines = []
    ctl_density = []
    ctl_new_spines = []
    ctl_elim_spines = []

    kir_mice = list(set(kir_activity_dataset.mouse_id))
    ctl_mice = list(set(ctl_activity_dataset.mouse_id))

    for mouse in kir_mice:
        temp_data = load_spine_datasets(mouse, ["Early", "Middle"], fov_type=fov_type)
        for FOV, dataset in temp_data.items():
            t_data = dataset["Early"]
            t_data_2 = dataset["Middle"]
            temp_groupings = [t_data.spine_groupings, t_data_2.spine_groupings]
            temp_flags = [t_data.spine_flags, t_data_2.spine_flags]
            temp_positions = [t_data.spine_positions, t_data_2.spine_positions]
            temp_density, temp_new, temp_elim = calculate_spine_dynamics(
                temp_flags,
                temp_positions,
                temp_groupings,
            )
            kir_density.append(temp_density[0])
            kir_new_spines.append(temp_new[-1])
            kir_elim_spines.append(temp_elim[-1])
    for mouse in ctl_mice:
        temp_data = load_spine_datasets(mouse, ["Early", "Middle"], fov_type=fov_type)
        for FOV, dataset in temp_data.items():
            t_data = dataset["Early"]
            t_data_2 = dataset["Middle"]
            temp_groupings = [t_data.spine_groupings, t_data_2.spine_groupings]
            temp_flags = [t_data.spine_flags, t_data_2.spine_flags]
            temp_positions = [t_data.spine_positions, t_data_2.spine_positions]
            temp_density, temp_new, temp_elim = calculate_spine_dynamics(
                temp_flags,
                temp_positions,
                temp_groupings,
            )
            ctl_density.append(temp_density[0])
            ctl_new_spines.append(temp_new[-1])
            ctl_elim_spines.append(temp_elim[-1])

    # construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE
        FGHIJ
        """,
        figsize=figsize,
        width_ratios=[1, 1, 1, 2, 2],
    )
    fig.suptitle(f"{fov_type} Kir vs Control")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ############################# Plot data onto axes ###############################
    # Spine density
    plot_swarm_bar_plot(
        data_dict={
            "Ctl": np.concatenate(ctl_density),
            "Kir": np.concatenate(kir_density),
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Spine density (per um)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.9,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # Fraction new spines
    plot_swarm_bar_plot(
        data_dict={
            "Ctl": np.concatenate(ctl_new_spines),
            "Kir": np.concatenate(kir_new_spines),
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction new spines",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.9,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Fraction eliminated spines
    plot_swarm_bar_plot(
        data_dict={
            "Ctl": np.concatenate(ctl_elim_spines),
            "Kir": np.concatenate(kir_elim_spines),
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction eliminated spines",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.9,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )

    # Initial spine volume
    plot_histogram(
        data=list(
            (
                d_utils.subselect_data_by_idxs(ctl_volume, ctl_present),
                d_utils.subselect_data_by_idxs(kir_volume, kir_present),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Initial Spine Volume",
        xtitle="Spine Volume (um)",
        xlim=(0, None),
        figsize=(5, 5),
        color=COLORS,
        alpha=0.4,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    ## Inset box plot
    ax_D_inset = axes["D"].inset_axes([0.7, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_D_inset)
    plot_box_plot(
        data_dict={
            "Ctl": d_utils.subselect_data_by_idxs(ctl_volume, ctl_present),
            "Kir": d_utils.subselect_data_by_idxs(kir_volume, kir_present),
        },
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Volume",
        ylim=(0, None),
        b_colors=COLORS,
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
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_D_inset,
        save=False,
        save_path=None,
    )
    # delta spine volume
    plot_histogram(
        data=list(
            (
                ctl_delta_volume,
                kir_delta_volume,
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Volume Change",
        xtitle="Volume Change",
        xlim=(0, None),
        figsize=(5, 5),
        color=COLORS,
        alpha=0.4,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Inset box plot
    ax_E_inset = axes["E"].inset_axes([0.7, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_E_inset)
    plot_box_plot(
        data_dict={
            "Ctl": ctl_delta_volume,
            "Kir": kir_delta_volume,
        },
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Volume Change",
        ylim=(0, None),
        b_colors=COLORS,
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
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_E_inset,
        save=False,
        save_path=None,
    )
    # Ctl fraction plastic
    plot_pie_chart(
        data_dict={
            "sLTP": np.sum(ctl_enlarged_spines),
            "sLTD": np.sum(ctl_shrunken_spines),
            "Stable": np.sum(ctl_stable_spines),
        },
        title="Ctl",
        figsize=(5, 5),
        colors=["mediumslateblue", "tomato", "silver"],
        alpha=0.9,
        edgecolor="white",
        txt_color="black",
        txt_size=10,
        legend=None,
        donut=0.6,
        linewidth=1.5,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    # Kir fraction plastic
    plot_pie_chart(
        data_dict={
            "sLTP": np.sum(kir_enlarged_spines),
            "sLTD": np.sum(kir_shrunken_spines),
            "Stable": np.sum(kir_stable_spines),
        },
        title="Kir",
        figsize=(5, 5),
        colors=["mediumslateblue", "tomato", "silver"],
        alpha=0.9,
        edgecolor="white",
        txt_color="black",
        txt_size=10,
        legend=None,
        donut=0.6,
        linewidth=1.5,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    # Activity rate
    plot_box_plot(
        data_dict={
            "Ctl": ctl_event_rate,
            "Kir": kir_event_rate,
        },
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Event rate (events/min)",
        ylim=(0, None),
        b_colors=COLORS,
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
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    # Distance coactivity rate
    plot_multi_line_plot(
        data_dict={
            "Ctl": ctl_distance_coactivity,
            "Kir": kir_distance_coactivity,
        },
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Raw Coactivity",
        ytitle="Coactivity rate (events/min)",
        xtitle="Distance (\u03BCm)",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Distance coactivity rate
    plot_multi_line_plot(
        data_dict={
            "Ctl": ctl_distance_coactivity_norm,
            "Kir": kir_distance_coactivity_norm,
        },
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Norm. Coactivity",
        ytitle="Norm. Coactivity",
        xtitle="Distance (\u03BCm)",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["J"],
        legend=True,
        save=False,
        save_path=None,
    )
