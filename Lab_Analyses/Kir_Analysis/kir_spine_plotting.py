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
from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    find_present_spines,
    load_spine_datasets,
)
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_fraction_plastic,
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
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
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

        test_type - str specifying whetehr to perform parametric or nonparameteric stats

        test_methods - str specifying the type of posthoc test to perform

        display_stats - boolean specifying whether to display stat results

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

    kir_delta_volume, kir_idxs = calculate_volume_change(
        kir_volume_list, kir_flag_list, norm=False, exclude="Shaft Spine"
    )
    kir_delta_volume = kir_delta_volume[-1]
    ctl_delta_volume, ctl_idxs = calculate_volume_change(
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
        temp_data = load_spine_datasets(mouse, ["Early"], fov_type=fov_type)
        for FOV, dataset in temp_data.items():
            t_data = dataset["Early"]
            temp_groupings = [t_data.spine_groupings, t_data.followup_groupings]
            temp_flags = [t_data.spine_flags, t_data.followup_flags]
            temp_positions = [t_data.spine_positions, t_data.followup_positions]
            temp_density, temp_new, temp_elim = calculate_spine_dynamics(
                temp_flags,
                temp_positions,
                temp_groupings,
            )
            kir_density.append(temp_density[0])
            kir_new_spines.append(temp_new[-1])
            kir_elim_spines.append(temp_elim[-1])
    for mouse in ctl_mice:
        temp_data = load_spine_datasets(mouse, ["Early"], fov_type=fov_type)
        for FOV, dataset in temp_data.items():
            t_data = dataset["Early"]
            temp_groupings = [t_data.spine_groupings, t_data.followup_groupings]
            temp_flags = [t_data.spine_flags, t_data.followup_flags]
            temp_positions = [t_data.spine_positions, t_data.followup_positions]
            temp_density, temp_new, temp_elim = calculate_spine_dynamics(
                temp_flags,
                temp_positions,
                temp_groupings,
            )
            ctl_density.append(temp_density[0])
            ctl_new_spines.append(temp_new[-1])
            ctl_elim_spines.append(temp_elim[-1])

    # Organize plasticity per dendrite
    kir_dendrites = kir_activity_dataset.dendrite_number
    kir_dendrites_u = np.unique(kir_dendrites)
    ctl_dendrites = ctl_activity_dataset.dendrite_number
    ctl_dendrites_u = np.unique(ctl_dendrites)

    kir_ltp = []
    kir_ltd = []
    ctl_ltp = []
    ctl_ltd = []

    for i, k_dend in enumerate(kir_dendrites_u):
        ## Deal with plasticity first
        if np.isnan(k_dend):
            continue
        dend_idxs = np.nonzero(kir_dendrites == k_dend)[0]
        temp_flag_list = [
            d_utils.subselect_data_by_idxs(kir_flag_list[0], dend_idxs),
            d_utils.subselect_data_by_idxs(kir_flag_list[1], dend_idxs),
        ]
        temp_vol_list = [
            d_utils.subselect_data_by_idxs(kir_volume_list[0], dend_idxs),
            d_utils.subselect_data_by_idxs(kir_volume_list[1], dend_idxs),
        ]
        k_ltp, k_ltd, _ = calculate_fraction_plastic(
            temp_vol_list, temp_flag_list, threshold=threshold, exclude="Shaft Spine"
        )

        kir_ltp.append(k_ltp)
        kir_ltd.append(k_ltd)

    for c_dend in ctl_dendrites_u:
        ## Deal with plasticity first
        if np.isnan(c_dend):
            continue
        dend_idxs = np.nonzero(ctl_dendrites == c_dend)[0]
        temp_flag_list = [
            d_utils.subselect_data_by_idxs(ctl_flag_list[0], dend_idxs),
            d_utils.subselect_data_by_idxs(ctl_flag_list[1], dend_idxs),
        ]
        temp_vol_list = [
            d_utils.subselect_data_by_idxs(ctl_volume_list[0], dend_idxs),
            d_utils.subselect_data_by_idxs(ctl_volume_list[1], dend_idxs),
        ]
        c_ltp, c_ltd, c_stable = calculate_fraction_plastic(
            temp_vol_list, temp_flag_list, threshold=threshold, exclude="Shaft Spine"
        )

        ctl_ltp.append(c_ltp)
        ctl_ltd.append(c_ltd)

    kir_ltp = np.array(kir_ltp)
    kir_ltd = np.array(kir_ltd)
    ctl_ltp = np.array(ctl_ltp)
    ctl_ltd = np.array(ctl_ltd)

    # construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDEK
        FGHIJL
        """,
        figsize=figsize,
        width_ratios=[1, 1, 1, 2, 2, 1],
    )
    fig.suptitle(f"{fov_type} Kir vs Control")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ############################# Plot data onto axes ###############################
    # Spine density
    ctl_density = np.concatenate(ctl_density)
    kir_density = np.concatenate(kir_density)
    plot_swarm_bar_plot(
        data_dict={
            "Ctl": ctl_density,
            "Kir": kir_density,
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
    ctl_new_spines = np.concatenate(ctl_new_spines)
    kir_new_spines = np.concatenate(kir_new_spines)
    plot_swarm_bar_plot(
        data_dict={
            "Ctl": ctl_new_spines,
            "Kir": kir_new_spines,
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
    ctl_elim_spines = np.concatenate(ctl_elim_spines)
    kir_elim_spines = np.concatenate(kir_elim_spines)
    plot_swarm_bar_plot(
        data_dict={
            "Ctl": ctl_elim_spines,
            "Kir": kir_elim_spines,
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
    ctl_volume = d_utils.subselect_data_by_idxs(ctl_volume, ctl_present)
    print(f"ctl spines: {len(ctl_volume)}")
    kir_volume = d_utils.subselect_data_by_idxs(kir_volume, kir_present)
    print(f"kir spines: {len(kir_volume)}")
    plot_histogram(
        data=[ctl_volume, kir_volume],
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
            "Ctl": ctl_volume,
            "Kir": kir_volume,
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
        xlim=(0, 4),
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
    print(
        f"Ctl: sLTP {np.sum(ctl_enlarged_spines)}; sLTD {np.sum(ctl_shrunken_spines)}; Stable {np.sum(ctl_stable_spines)}"
    )
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
    print(
        f"KIR: sLTP {np.sum(kir_enlarged_spines)}; sLTD {np.sum(kir_shrunken_spines)}; Stable {np.sum(kir_stable_spines)}"
    )
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
        mean_type="mean",
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
        mean_type="median",
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

    # Fraction LTP per dendrite
    plot_swarm_bar_plot(
        data_dict={
            "Ctl": np.array(ctl_ltp),
            "Kir": np.array(kir_ltp),
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction LTP / dendrite",
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Fraction LTD per dendrite
    plot_swarm_bar_plot(
        data_dict={
            "Ctl": np.array(ctl_ltd),
            "Kir": np.array(kir_ltd),
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction LTD / dendrite",
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
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    ctl_ltp = np.array(ctl_ltp)
    ctl_ltd = np.array(ctl_ltd)
    kir_ltp = np.array(kir_ltp)
    kir_ltd = np.array(kir_ltd)

    print(f"ctl dend: {len(ctl_ltp)}")
    print(f"kir dend: {len(kir_ltp)}")

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, f"Kir_Basic_Properties")
        fig.savefig(fname + ".pdf")

    ####################### Statistics Section ############################
    if display_stats == False:
        return

    # Perform the statistics
    if test_type == "parametric":
        density_t, density_p = stats.ttest_ind(
            ctl_density, kir_density, nan_policy="omit"
        )
        new_t, new_p = stats.ttest_ind(
            ctl_new_spines, kir_new_spines, nan_policy="omit"
        )
        elim_t, elim_p = stats.ttest_ind(
            ctl_elim_spines, kir_elim_spines, nan_policy="omit"
        )
        vol_t, vol_p = stats.ttest_ind(ctl_volume, kir_volume, nan_policy="omit")
        rel_t, rel_p = stats.ttest_ind(
            ctl_delta_volume, kir_delta_volume, nan_policy="omit"
        )
        ltp_t, ltp_p = stats.ttest_ind(ctl_ltp, kir_ltp, nan_policy="omit")
        ltd_t, ltd_p = stats.ttest_ind(ctl_ltd, kir_ltd, nan_policy="omit")
        rate_t, rate_p = stats.ttest_ind(
            ctl_event_rate, kir_event_rate, nan_policy="omit"
        )
        test_title = "T-test"
    elif test_type == "nonparametric":
        density_t, density_p = stats.mannwhitneyu(
            ctl_density[~np.isnan(ctl_density)], kir_density[~np.isnan(kir_density)]
        )
        new_t, new_p = stats.mannwhitneyu(
            ctl_new_spines[~np.isnan(ctl_new_spines)],
            kir_new_spines[~np.isnan(kir_new_spines)],
        )
        elim_t, elim_p = stats.mannwhitneyu(
            ctl_elim_spines[~np.isnan(ctl_elim_spines)],
            kir_elim_spines[~np.isnan(kir_elim_spines)],
        )
        vol_t, vol_p = stats.mannwhitneyu(
            ctl_volume[~np.isnan(ctl_volume)], kir_volume[~np.isnan(kir_volume)]
        )
        rel_t, rel_p = stats.mannwhitneyu(
            ctl_delta_volume[~np.isnan(ctl_delta_volume)],
            kir_delta_volume[~np.isnan(kir_delta_volume)],
        )
        ltp_t, ltp_p = stats.mannwhitneyu(
            ctl_ltp[~np.isnan(ctl_ltp)], kir_ltp[~np.isnan(kir_ltp)]
        )
        ltd_t, ltd_p = stats.mannwhitneyu(
            ctl_ltd[~np.isnan(ctl_ltd)], kir_ltd[~np.isnan(kir_ltd)]
        )
        rate_t, rate_p = stats.mannwhitneyu(
            ctl_event_rate[~np.isnan(ctl_event_rate)],
            kir_event_rate[~np.isnan(kir_event_rate)],
        )
        test_title = "Mann-Whitney U"

    # Organize the results
    results_dict = {
        "Comparison": [
            "Density",
            "New Spines",
            "Elim Spines",
            "Volume",
            "Rel Vol.",
            "Frac. LTP",
            "Frac. LTD",
            "Event rate",
        ],
        "stat": [
            density_t,
            new_t,
            elim_t,
            vol_t,
            rel_t,
            ltp_t,
            ltd_t,
            rate_t,
        ],
        "p-val": [
            density_p,
            new_p,
            elim_p,
            vol_p,
            rel_p,
            ltp_p,
            ltd_p,
            rate_p,
        ],
    }

    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.update(results_df[["stat"]].applymap("{:.3}".format))
    results_df.update(results_df[["p-val"]].applymap("{:.3E}".format))

    fig2, axes2 = plt.subplot_mosaic("""A""", figsize=(6, 6))

    ## Format the table
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(f"{test_title} result")
    A_table = axes2["A"].table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, f"Kir_Basic_Properties_Stats")
        fig2.savefig(fname + ".pdf")


def plot_kir_spine_plasticity(
    dataset,
    fov_type="apical",
    exclude="Shaft Spine",
    MRSs=None,
    threshold=0.3,
    figsize=(8, 5),
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to plot basic spine properties associated with plasticity

    INPUT PARAMETERS
        dataset - Kir_Activity_Data object

        fov_type - str specifying the type of fov being analyzed

        exclude - str specifying type of spine to be excluded

        MRSs - str specifying if you wish to examine only MRSs or nonMRSs. Accepts
                "MRS" and "nMRS". Default is None to examine all spines

        threshold - float or tuple of floats specifying the threshold cutoffs for
                    classifying plasticity

        figsize - tuple specifying the size of the figure to plot

        showmeans - boolean specifying wheteher to show the mean on box plots

        test_type - str specifying whetehr to perform parametric or nonparametric stats

        test_method - str specifying the type of posthoc test to perform

        display_stats - boolean specifying whether to display stat results

        save - boolean specifying whether to save the figure or not

        save_path - str specifying where to save the path
    """
    COLORS = ["mediumslateblue", "tomato", "silver"]
    spine_groups = {
        "sLTP": "enlarged_spines",
        "sLTD": "shrunken_spines",
        "Stable": "stable_spines",
    }
    # Pull the relevant data
    initial_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    spine_activity_rate = dataset.spine_activity_rate

    # Calculate spine volumes
    ## Get followup volumes
    followup_volumes = dataset.followup_volumes
    followup_flags = dataset.followup_flags

    ## Setup input lists
    volumes = [initial_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    ## Calculate
    delta_volume, spine_idxs = calculate_volume_change(
        volumes,
        flags,
        norm=False,
        exclude=exclude,
    )
    delta_volume = delta_volume[-1]

    # Classify plasticity
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume,
        threshold=threshold,
        norm=False,
    )

    # Subselect data
    initial_volumes = d_utils.subselect_data_by_idxs(initial_volumes, spine_idxs)
    spine_activity_rate = d_utils.subselect_data_by_idxs(
        spine_activity_rate, spine_idxs
    )
    mvmt_spines = d_utils.subselect_data_by_idxs(dataset.movement_spines, spine_idxs)
    nonmvmt_spines = d_utils.subselect_data_by_idxs(
        dataset.nonmovement_spines, spine_idxs
    )

    # Organize data into dictionaries
    initial_vol_dict = {}
    activity_dict = {}
    count_dict = {}
    for key, value in spine_groups.items():
        spines = eval(value)
        if MRSs == "MRS":
            spines = spines * mvmt_spines
        elif MRSs == "nMRS":
            spines = spines * nonmvmt_spines
        vol = initial_volumes[spines]
        activity = spine_activity_rate[spines]
        initial_vol_dict[key] = vol[~np.isnan(vol)]
        activity_dict[key] = activity[~np.isnan(activity)]
        count_dict[key] = np.sum(spines)

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        AB
        CD
        """,
        figsize=figsize,
    )
    fig.suptitle(f"{fov_type} Basic Spine Plasticity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ##################### Plot data onto the axes #######################
    print(f"Spine groups: {count_dict}")
    # Initial volume vs relative volume correlation
    plot_scatter_correlation(
        x_var=np.log10(initial_volumes),
        y_var=delta_volume,
        CI=95,
        title="Initial vs \u0394 Volume",
        xtitle="Log(initial area \u03BCm)",
        ytitle="\u0394 Volume",
        figsize=(5, 5),
        xlim=(None, None),
        ylim=(0, None),
        marker_size=25,
        face_color="cmap",
        edge_color="white",
        line_color="black",
        s_alpha=1,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )

    # Initial volumes for spine types
    plot_box_plot(
        data_dict=initial_vol_dict,
        figsize=(5, 5),
        title="Initial Volumes",
        xtitle=None,
        ytitle="Initial volume \u03BCm",
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
        ax=axes["B"],
        save=False,
        save_path=None,
    )

    # Spine activity rate vs relative volume
    plot_scatter_correlation(
        x_var=spine_activity_rate,
        y_var=delta_volume,
        CI=95,
        title="Event Rate vs \u0394 Volume",
        xtitle="Event rate (events/min)",
        ytitle="\u0394 Volume",
        figsize=(5, 5),
        xlim=(0, None),
        ylim=(0, None),
        marker_size=25,
        face_color="cmap",
        edge_color="white",
        line_color="black",
        s_alpha=1,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )

    # Activity rates for spine types
    plot_box_plot(
        data_dict=activity_dict,
        figsize=(5, 5),
        title="Event rates",
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
        ax=axes["D"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section

    ########################## Statistics Section ############################


def plot_kir_coactivity_plasticity(
    dataset,
    MRSs=None,
    norm=False,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 8),
    showmeans=False,
    mean_type="median",
    err_type="CI",
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to compare coactivity rates across plasticity groups

    INPUT PARAMETERS
        dataset - Local_Coactivity_Data object analyzed over all periods

        MRSs - str specifying if you wish to examine only MRSs or nonMRSs. Accepts
                "MRS" and "nMRS". Defualt is None to examine all spines

        norm - boolean term specifying whether to use the normalized coactivity rate
                or not

    `   exclude - str specifying spine type to exclude from analysis

        threshold - float or tuple of floats specifying the threshold cutoff for
                    classifying plasticity

        figsize - tuple specifying the size of the figure

        showmeans - boolean specifying whether to plot means on box plots

        mean_type - str specifying the mean type for bar plots

        err_type - str specifying the error type for the bar plots

        test_type - str specifying whether to perform parametric or nonparametric tests

        test_method - str specifying the typ of posthoc test to perform

        display_stats - boolean specifying whether to display stats

        save - boolean specifying whether to save the figures or not

        save_path - str specifying where to save the figures
    """
    COLORS = ["mediumslateblue", "tomato", "silver"]
    plastic_groups = {
        "sLTP": "enlarged_spines",
        "sLTD": "shrunken_spines",
        "Stable": "stable_spines",
    }

    # Pull relevant data
    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    followup_volumes = dataset.followup_volumes
    followup_flags = dataset.followup_flags

    ## Coactivity-related variables
    if norm == False:
        nname = "Raw"
        coactivity_title = "Coactivity rate (event/min)"
        distance_coactivity_rate = dataset.distance_coactivity_rate
        avg_local_coactivity_rate = dataset.avg_local_coactivity_rate
        shuff_local_coactivity_rate = dataset.shuff_local_coactivity_rate
        near_vs_dist = dataset.near_vs_dist_coactivity
    else:
        nname = "Norm."
        coactivity_title = "Norm. coactivity rate"
        distance_coactivity_rate = dataset.distance_coactivity_rate_norm
        avg_local_coactivity_rate = dataset.avg_local_coactivity_rate_norm
        shuff_local_coactivity_rate = dataset.shuff_local_coactivity_rate_norm
        near_vs_dist = dataset.near_vs_dist_coactivity_norm

    # Calculate relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=False, exclude=exclude
    )
    delta_volume = delta_volume[-1]

    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume,
        threshold=threshold,
        norm=False,
    )

    # Organize data
    ## Subselect present spines
    distance_coactivity_rate = d_utils.subselect_data_by_idxs(
        distance_coactivity_rate, spine_idxs
    )
    avg_local_coactivity_rate = d_utils.subselect_data_by_idxs(
        avg_local_coactivity_rate, spine_idxs
    )
    shuff_local_coactivity_rate = d_utils.subselect_data_by_idxs(
        shuff_local_coactivity_rate, spine_idxs
    )
    near_vs_dist = d_utils.subselect_data_by_idxs(near_vs_dist, spine_idxs)
    mvmt_spines = d_utils.subselect_data_by_idxs(dataset.movement_spines, spine_idxs)
    nonmvmt_spines = d_utils.subselect_data_by_idxs(
        dataset.nonmovement_spines, spine_idxs
    )

    # Seperate into groups
    plastic_distance_rates = {}
    plastic_local_rates = {}
    plastic_shuff_rates = {}
    plastic_shuff_medians = {}
    plastic_diffs = {}
    distance_bins = dataset.parameters["position bins"][1:]

    for key, value in plastic_groups.items():
        # Get spine types
        spines = eval(value)
        # Further subselect MRSs and nMRSs if specified
        if MRSs == "MRS":
            spines = spines * mvmt_spines
        elif MRSs == "nMRS":
            spines = spines * nonmvmt_spines
        # Subselect data
        plastic_distance_rates[key] = distance_coactivity_rate[:, spines]
        plastic_local_rates[key] = avg_local_coactivity_rate[spines]
        plastic_diffs[key] = near_vs_dist[spines]
        ## Process shuff data
        shuff_rates = shuff_local_coactivity_rate[:, spines]
        plastic_shuff_rates[key] = shuff_rates
        plastic_shuff_medians[key] = np.nanmedian(shuff_rates, axis=1)

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFG.
        """,
        figsize=figsize,
    )
    fig.suptitle(f"Kir Plastic {nname} Coactivity Rates")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ######################## Plot data onto axes ##########################
    # Distance activity rates
    plot_multi_line_plot(
        data_dict=plastic_distance_rates,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="All periods",
        ytitle=coactivity_title,
        xtitle="Distance (\u03BCm)",
        mean_type="median",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["A"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Local coactivity vs delta volume
    plot_scatter_correlation(
        x_var=avg_local_coactivity_rate,
        y_var=delta_volume,
        CI=95,
        title="All periods",
        xtitle=f"Local {coactivity_title}",
        ytitle="\u0394 volume",
        figsize=(5, 5),
        xlim=(0, None),
        ylim=(0, None),
        marker_size=25,
        face_color="cmap",
        edge_color="white",
        line_color="black",
        s_alpha=1,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Local coactivity rate bar plot
    plot_box_plot(
        plastic_local_rates,
        figsize=(5, 5),
        title="All periods",
        xtitle=None,
        ytitle=f"Local {coactivity_title}",
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
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # Near vs distant relative difference
    plot_box_plot(
        plastic_diffs,
        figsize=(5, 5),
        title="All periods",
        xtitle=None,
        ytitle=f"Near - distant coactivity",
        ylim=None,
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
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # Enlarged local vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list(
            (
                plastic_local_rates["sLTP"],
                plastic_shuff_rates["sLTP"],
            )
        ),
        plot_ind=True,
        title="Enlarged",
        xtitle=f"Local {coactivity_title}",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_e_inset = axes["E"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_e_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_local_rates["sLTP"],
            "shuff": plastic_shuff_rates["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        ytitle="Coactivity rate",
        ylim=None,
        b_colors=[COLORS[0], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_e_inset,
        save=False,
        save_path=None,
    )
    # Shrunken local vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list(
            (
                plastic_local_rates["sLTD"],
                plastic_shuff_rates["sLTD"],
            )
        ),
        plot_ind=True,
        title="Shrunken",
        xtitle=f"Local {coactivity_title}",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_f_inset = axes["F"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_f_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_local_rates["sLTD"],
            "shuff": plastic_shuff_rates["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        ytitle="Coactivity rate",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_f_inset,
        save=False,
        save_path=None,
    )
    # Stable local vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list(
            (
                plastic_local_rates["Stable"],
                plastic_shuff_rates["Stable"],
            )
        ),
        plot_ind=True,
        title="Stable",
        xtitle=f"Local {coactivity_title}",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_g_inset = axes["G"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_g_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_local_rates["Stable"],
            "shuff": plastic_shuff_rates["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        ytitle="Coactivity rate",
        ylim=None,
        b_colors=[COLORS[2], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_g_inset,
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section

    ########################## Statistics Section ############################


def plot_nearby_spine_properties(
    dataset,
    followup_dataset=None,
    MRSs=None,
    exclude="Shaft",
    threshold=0.3,
    figsize=(10, 12),
    mean_type="median",
    err_type="CI",
    showmeans=False,
    hist_bins=25,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """Function to plot nearby spine properties

    INPUT PARAMETERS
        dataset - Local_Coactivity_Data object to be analyzed

        followup_dataset - optional Local_Coactivity_Data object of the
                            subsequent session to be used for volume comparision.
                            Default is None to use the followup volumes in the
                            dataset.

        MRSs - str specifying if you wish to examine only MRSs or nonMRSs. Accepts
                "MRS" and "nMRS". Default is None to examine all spines

        exclude - str specifying spine type to exclude from analysis

        threshold - float or tuple of floats specifying the threshold cutoff for
                    classifying plasticity

        figsize - tuple specifying the size of the figure

        mean_type - str specifying the mean type for the bar plots

        err_type - str specifyiung the error type for the bar plots

        showmeans - boolean specifying whether to show means on boxplots

        test_type - str specifying whether to perform parametric or nonparametric stats

        test_method - str specifying the type of posthoc test to perform

        display_stats - boolean specifying whether to display stats

        save - boolean specifying whetehr to save the figures or not

        save_path - str specifying where to save the figures

    """
    COLORS = ["mediumslateblue", "tomato", "silver"]
    plastic_groups = {
        "sLTP": "enlarged_spines",
        "sLTD": "shrunken_spines",
        "Stable": "stable_spines",
    }

    # Pull relevant data
    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset is None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags

    spine_activity_rate_distribution = dataset.spine_activity_rate_distribution
    avg_nearby_spine_rate = dataset.avg_nearby_spine_rate
    shuff_nearby_spine_rate = dataset.shuff_nearby_spine_rate
    near_vs_dist_activity_rate = dataset.near_vs_dist_activity_rate
    local_coactivity_rate_distribution = dataset.local_coactivity_rate_distribution
    avg_nearby_coactivity_rate = dataset.avg_nearby_coactivity_rate
    shuff_nearby_coactivity_rate = dataset.shuff_nearby_coactivity_rate
    near_vs_dist_nearby_coactivity_rate = dataset.near_vs_dist_nearby_coactivity_rate
    MRS_density_distribution = dataset.MRS_density_distribution
    avg_local_MRS_density = dataset.avg_local_MRS_density
    shuff_local_MRS_density = dataset.shuff_local_MRS_density
    rMRS_density_distribution = dataset.rMRS_density_distribution
    avg_local_rMRS_density = dataset.avg_local_rMRS_density
    shuff_local_rMRS_density = dataset.shuff_local_rMRS_density
    local_nn_enlarged = dataset.local_nn_enlarged
    shuff_nn_enlarged = dataset.shuff_nn_enlarged
    local_nn_shrunken = dataset.local_nn_shrunken
    shuff_nn_shrunken = dataset.shuff_nn_shrunken
    avg_nearby_spine_volume = dataset.avg_nearby_spine_volume
    shuff_nearby_spine_volume = dataset.shuff_nearby_spine_volume
    nearby_spine_volume_distribution = dataset.nearby_spine_volume_distribution
    near_vs_dist_volume = dataset.near_vs_dist_volume
    local_relative_vol = dataset.local_relative_vol
    shuff_relative_vol = dataset.shuff_relative_vol
    relative_vol_distribution = dataset.relative_vol_distribution
    near_vs_dist_relative_volume = dataset.near_vs_dist_relative_volume

    # Calculate spine volume
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=vol_norm, exclude=exclude
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume,
        threshold=threshold,
        norm=vol_norm,
    )

    # Subselect for stable spines
    spine_activity_rate_distribution = d_utils.subselect_data_by_idxs(
        spine_activity_rate_distribution, spine_idxs
    )
    avg_nearby_spine_rate = d_utils.subselect_data_by_idxs(
        avg_nearby_spine_rate,
        spine_idxs,
    )
    shuff_nearby_spine_rate = d_utils.subselect_data_by_idxs(
        shuff_nearby_spine_rate, spine_idxs
    )
    near_vs_dist_activity_rate = d_utils.subselect_data_by_idxs(
        near_vs_dist_activity_rate, spine_idxs
    )
    local_coactivity_rate_distribution = d_utils.subselect_data_by_idxs(
        local_coactivity_rate_distribution,
        spine_idxs,
    )
    avg_nearby_coactivity_rate = d_utils.subselect_data_by_idxs(
        avg_nearby_coactivity_rate,
        spine_idxs,
    )
    shuff_nearby_coactivity_rate = d_utils.subselect_data_by_idxs(
        shuff_nearby_coactivity_rate, spine_idxs
    )
    near_vs_dist_nearby_coactivity_rate = d_utils.subselect_data_by_idxs(
        near_vs_dist_nearby_coactivity_rate,
        spine_idxs,
    )
    MRS_density_distribution = d_utils.subselect_data_by_idxs(
        MRS_density_distribution, spine_idxs
    )
    avg_local_MRS_density = d_utils.subselect_data_by_idxs(
        avg_local_MRS_density, spine_idxs
    )
    shuff_local_MRS_density = d_utils.subselect_data_by_idxs(
        shuff_local_MRS_density, spine_idxs
    )
    rMRS_density_distribution = d_utils.subselect_data_by_idxs(
        rMRS_density_distribution, spine_idxs
    )
    avg_local_rMRS_density = d_utils.subselect_data_by_idxs(
        avg_local_rMRS_density, spine_idxs
    )
    shuff_local_rMRS_density = d_utils.subselect_data_by_idxs(
        shuff_local_rMRS_density, spine_idxs
    )
    local_nn_enlarged = d_utils.subselect_data_by_idxs(
        local_nn_enlarged,
        spine_idxs,
    )
    shuff_nn_enlarged = d_utils.subselect_data_by_idxs(
        shuff_nn_enlarged,
        spine_idxs,
    )
    local_nn_shrunken = d_utils.subselect_data_by_idxs(
        local_nn_shrunken,
        spine_idxs,
    )
    shuff_nn_shrunken = d_utils.subselect_data_by_idxs(
        shuff_nn_shrunken,
        spine_idxs,
    )
    avg_nearby_spine_volume = d_utils.subselect_data_by_idxs(
        avg_nearby_spine_volume, spine_idxs
    )
    shuff_nearby_spine_volume = d_utils.subselect_data_by_idxs(
        shuff_nearby_spine_volume, spine_idxs
    )
    nearby_spine_volume_distribution = d_utils.subselect_data_by_idxs(
        nearby_spine_volume_distribution, spine_idxs
    )
    near_vs_dist_volume = d_utils.subselect_data_by_idxs(
        near_vs_dist_volume,
        spine_idxs,
    )
    local_relative_vol = d_utils.subselect_data_by_idxs(
        local_relative_vol,
        spine_idxs,
    )
    shuff_relative_vol = d_utils.subselect_data_by_idxs(
        shuff_relative_vol,
        spine_idxs,
    )
    relative_vol_distribution = d_utils.subselect_data_by_idxs(
        relative_vol_distribution,
        spine_idxs,
    )
    near_vs_dist_relative_volume = d_utils.subselect_data_by_idxs(
        near_vs_dist_relative_volume,
        spine_idxs,
    )
    mvmt_spines = d_utils.subselect_data_by_idxs(dataset.movement_spines, spine_idxs)
    nonmvmt_spines = d_utils.subselect_data_by_idxs(
        dataset.nonmovement_spines, spine_idxs
    )

    # Seperate into groups
    plastic_rate_dist = {}
    plastic_avg_rates = {}
    plastic_shuff_rates = {}
    plastic_shuff_rate_medians = {}
    plastic_near_dist_rates = {}
    plastic_coactivity_dist = {}
    plastic_avg_coactivity = {}
    plastic_shuff_coactivity = {}
    plastic_shuff_coactivity_medians = {}
    plastic_near_dist_coactivity = {}
    plastic_MRS_dist = {}
    plastic_MRS_density = {}
    plastic_MRS_shuff_density = {}
    plastic_MRS_shuff_density_medians = {}
    plastic_rMRS_dist = {}
    plastic_rMRS_density = {}
    plastic_rMRS_shuff_density = {}
    plastic_rMRS_shuff_density_medians = {}
    plastic_nn_enlarged = {}
    plastic_shuff_enlarged = {}
    plastic_shuff_enlarged_medians = {}
    plastic_nn_shrunken = {}
    plastic_shuff_shrunken = {}
    plastic_shuff_shrunken_medians = {}
    plastic_nearby_volumes = {}
    plastic_shuff_volumes = {}
    plastic_volume_dist = {}
    plastic_near_dist_vol = {}
    plastic_rel_vols = {}
    plastic_shuff_rel_vols = {}
    plastic_rel_vol_dist = {}
    plastic_near_dist_rel_vol = {}
    distance_bins = dataset.parameters["position bins"][1:]

    for key, value in plastic_groups.items():
        spines = eval(value)
        if MRSs == "MRS":
            spines = spines * mvmt_spines
        elif MRSs == "nMRS":
            spines = spines * nonmvmt_spines
        plastic_rate_dist[key] = spine_activity_rate_distribution[:, spines]
        plastic_avg_rates[key] = avg_nearby_spine_rate[spines]
        shuff_rates = shuff_nearby_spine_rate[:, spines]
        plastic_shuff_rates[key] = shuff_rates
        plastic_shuff_rate_medians[key] = np.nanmedian(shuff_rates, axis=1)
        plastic_near_dist_rates[key] = near_vs_dist_activity_rate[spines]
        plastic_coactivity_dist[key] = local_coactivity_rate_distribution[:, spines]
        plastic_avg_coactivity[key] = avg_nearby_coactivity_rate[spines]
        shuff_coactivity = shuff_nearby_coactivity_rate[:, spines]
        plastic_shuff_coactivity[key] = shuff_coactivity
        plastic_shuff_coactivity_medians[key] = np.nanmedian(shuff_coactivity, axis=1)
        plastic_near_dist_coactivity[key] = near_vs_dist_nearby_coactivity_rate[spines]
        plastic_MRS_dist[key] = MRS_density_distribution[:, spines]
        plastic_MRS_density[key] = avg_local_MRS_density[spines]
        shuff_MRS = shuff_local_MRS_density[:, spines]
        plastic_MRS_shuff_density[key] = shuff_MRS
        plastic_MRS_shuff_density_medians[key] = np.nanmedian(shuff_MRS, axis=1)
        plastic_rMRS_dist[key] = rMRS_density_distribution[:, spines]
        plastic_rMRS_density[key] = avg_local_rMRS_density[spines]
        shuff_rMRS = shuff_local_rMRS_density[:, spines]
        plastic_rMRS_shuff_density[key] = shuff_rMRS
        plastic_rMRS_shuff_density_medians[key] = np.nanmedian(shuff_rMRS, axis=1)
        plastic_nn_enlarged[key] = local_nn_enlarged[spines]
        shuff_enlarged = shuff_nn_enlarged[:, spines]
        plastic_shuff_enlarged[key] = shuff_enlarged
        plastic_shuff_enlarged_medians[key] = np.nanmedian(shuff_enlarged, axis=1)
        plastic_nn_shrunken[key] = local_nn_shrunken[spines]
        shuff_shrunken = shuff_nn_shrunken[:, spines]
        plastic_shuff_shrunken[key] = shuff_shrunken
        plastic_shuff_shrunken_medians[key] = np.nanmedian(shuff_shrunken, axis=1)
        plastic_volume_dist[key] = nearby_spine_volume_distribution[:, spines]
        plastic_nearby_volumes[key] = avg_nearby_spine_volume[spines]
        plastic_shuff_volumes[key] = shuff_nearby_spine_volume[:, spines]
        plastic_near_dist_vol[key] = near_vs_dist_volume[spines]
        plastic_rel_vol_dist[key] = relative_vol_distribution[:, spines]
        plastic_rel_vols[key] = local_relative_vol[spines]
        plastic_shuff_rel_vols[key] = shuff_relative_vol[:, spines]
        plastic_near_dist_rel_vol[key] = near_vs_dist_relative_volume[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE
        FGHIJ
        KLMNO
        PQRST
        UVWXY
        Zabc.
        defg.
        hijkl
        mnop.
        """,
        figsize=figsize,
    )

    fig.suptitle(f"Nearby Spine Properties")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ########################### Plot data onto axes #############################
    # Activity rate distribution
    plot_multi_line_plot(
        data_dict=plastic_rate_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Activity Rate",
        ytitle="Activity rate (events/min)",
        xtitle="Distance (\u03BCm)",
        mean_type="mean",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["A"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Coactivity rate distribution
    plot_multi_line_plot(
        data_dict=plastic_coactivity_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Coactivity Rate",
        ytitle="Coactivity rate (events/min)",
        xtitle="Distance (\u03BCm)",
        mean_type="mean",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["F"],
        legend=True,
        save=False,
        save_path=None,
    )
    # MRS density distribution
    plot_multi_line_plot(
        data_dict=plastic_MRS_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="MRS density",
        ytitle="MRS density (spines/\u03BCm)",
        xtitle="Distance (\u03BCm)",
        mean_type="mean",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["K"],
        legend=True,
        save=False,
        save_path=None,
    )
    # MRS density distribution
    plot_multi_line_plot(
        data_dict=plastic_rMRS_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="rMRS density",
        ytitle="rMRS density (spines/\u03BCm)",
        xtitle="Distance (\u03BCm)",
        mean_type="mean",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["P"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Spine volume distribution
    plot_multi_line_plot(
        data_dict=plastic_volume_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Initial area",
        ytitle="Spine area (\u03BCm)",
        xtitle="Distance (\u03BCm)",
        mean_type="mean",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["U"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Relative volume distribution
    plot_multi_line_plot(
        data_dict=plastic_rel_vol_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Relative volume",
        ytitle="\u0394 Volume",
        xtitle="Distance (\u03BCm)",
        mean_type="mean",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["h"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Avg local spine activity rate
    plot_box_plot(
        plastic_avg_rates,
        figsize=(5, 5),
        title="Nearby Activity Rate",
        xtitle=None,
        ytitle="Activity rate (events/min)",
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
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Enlarged rate vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_avg_rates["sLTP"],
                plastic_shuff_rates["sLTP"],
            )
        ),
        plot_ind=True,
        title="Enlarged",
        xtitle="Nearby activity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_c_inset = axes["C"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_c_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_rates["sLTP"],
            "shuff": plastic_shuff_rates["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Activity rate",
        ylim=None,
        b_colors=[COLORS[0], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_c_inset,
        save=False,
        save_path=None,
    )
    # Shrunken rate vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_avg_rates["sLTD"],
                plastic_shuff_rates["sLTD"],
            )
        ),
        plot_ind=True,
        title="Shrunken",
        xtitle="Nearby activity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_d_inset = axes["D"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_d_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_rates["sLTD"],
            "shuff": plastic_shuff_rates["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Activity rate",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_d_inset,
        save=False,
        save_path=None,
    )
    # Stable rate vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_avg_rates["Stable"],
                plastic_shuff_rates["Stable"],
            )
        ),
        plot_ind=True,
        title="Stable",
        xtitle="Nearby activity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_e_inset = axes["E"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_e_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_rates["Stable"],
            "shuff": plastic_shuff_rates["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Activity rate",
        ylim=None,
        b_colors=[COLORS[2], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_e_inset,
        save=False,
        save_path=None,
    )
    # Avg local coactivity rate
    plot_box_plot(
        plastic_avg_coactivity,
        figsize=(5, 5),
        title="Avg Local Coactivity Rate",
        xtitle=None,
        ytitle="Coactivity rate (events/min)",
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
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    # Enlarged coactivity rate vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_avg_coactivity["sLTP"],
                plastic_shuff_coactivity["sLTP"],
            )
        ),
        plot_ind=True,
        title="Enlarged",
        xtitle="Avg local coactivity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_h_inset = axes["H"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_h_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_coactivity["sLTP"],
            "shuff": plastic_shuff_coactivity["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Coactivity rate",
        ylim=None,
        b_colors=[COLORS[0], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_h_inset,
        save=False,
        save_path=None,
    )
    # Shrunken coactivity rate vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_avg_coactivity["sLTD"],
                plastic_shuff_coactivity["sLTD"],
            )
        ),
        plot_ind=True,
        title="Shrunken",
        xtitle="Avg local coactivity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_I_inset = axes["I"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_I_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_coactivity["sLTD"],
            "shuff": plastic_shuff_coactivity["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Coactivity rate",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_I_inset,
        save=False,
        save_path=None,
    )
    # Stable coactivity rate vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_avg_coactivity["Stable"],
                plastic_shuff_coactivity["Stable"],
            )
        ),
        plot_ind=True,
        title="Stable",
        xtitle="Avg local coactivity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_j_inset = axes["J"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_j_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_coactivity["Stable"],
            "shuff": plastic_shuff_coactivity["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Coactivity rate",
        ylim=None,
        b_colors=[COLORS[2], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_j_inset,
        save=False,
        save_path=None,
    )
    # Avg local MRS rate
    plot_box_plot(
        plastic_MRS_density,
        figsize=(5, 5),
        title="Local MRS Density",
        xtitle=None,
        ytitle="MRS density (spines/\u03BCm)",
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
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    # Enlarged MRS density vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_MRS_density["sLTP"],
                plastic_MRS_shuff_density["sLTP"],
            )
        ),
        plot_ind=True,
        title="Enlarged",
        xtitle="MRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_m_inset = axes["M"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_m_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_MRS_density["sLTP"],
            "shuff": plastic_MRS_shuff_density["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="MRS density (spines/\u03BCm)",
        ylim=None,
        b_colors=[COLORS[0], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_m_inset,
        save=False,
        save_path=None,
    )
    # Shrunken MRS density vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_MRS_density["sLTD"],
                plastic_MRS_shuff_density["sLTD"],
            )
        ),
        plot_ind=True,
        title="Shrunken",
        xtitle="MRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["N"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_n_inset = axes["N"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_n_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_MRS_density["sLTD"],
            "shuff": plastic_MRS_shuff_density["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="MRS density (spines/\u03BCm)",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_n_inset,
        save=False,
        save_path=None,
    )
    # Enlarged MRS density vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_MRS_density["Stable"],
                plastic_MRS_shuff_density["Stable"],
            )
        ),
        plot_ind=True,
        title="Stable",
        xtitle="MRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_o_inset = axes["O"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_o_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_MRS_density["Stable"],
            "shuff": plastic_MRS_shuff_density["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=(0, None),
        ytitle="MRS density (spines/\u03BCm)",
        ylim=None,
        b_colors=[COLORS[2], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_o_inset,
        save=False,
        save_path=None,
    )
    # Avg local rMRS rate
    plot_box_plot(
        plastic_rMRS_density,
        figsize=(5, 5),
        title="Local rMRS Density",
        xtitle=None,
        ytitle="rMRS density (spines/\u03BCm)",
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
        ax=axes["Q"],
        save=False,
        save_path=None,
    )
    # Enlarged rMRS density vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_rMRS_density["sLTP"],
                plastic_rMRS_shuff_density["sLTP"],
            )
        ),
        plot_ind=True,
        title="Enlarged",
        xtitle="rMRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["R"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_r_inset = axes["R"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_r_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_rMRS_density["sLTP"],
            "shuff": plastic_rMRS_shuff_density["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="rMRS density (spines/\u03BCm)",
        ylim=None,
        b_colors=[COLORS[0], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_r_inset,
        save=False,
        save_path=None,
    )
    # Shrunken rMRS density vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_rMRS_density["sLTD"],
                plastic_rMRS_shuff_density["sLTD"],
            )
        ),
        plot_ind=True,
        title="Shrunken",
        xtitle="rMRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["S"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_s_inset = axes["S"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_s_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_rMRS_density["sLTD"],
            "shuff": plastic_rMRS_shuff_density["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="rMRS density (spines/\u03BCm)",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_s_inset,
        save=False,
        save_path=None,
    )
    # Enlarged rMRS density vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_rMRS_density["Stable"],
                plastic_rMRS_shuff_density["Stable"],
            )
        ),
        plot_ind=True,
        title="Stable",
        xtitle="rMRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["T"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_t_inset = axes["T"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_t_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_rMRS_density["Stable"],
            "shuff": plastic_rMRS_shuff_density["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="rMRS density (spines/\u03BCm)",
        ylim=None,
        b_colors=[COLORS[2], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_t_inset,
        save=False,
        save_path=None,
    )
    # Avg local spine volume
    plot_box_plot(
        plastic_nearby_volumes,
        figsize=(5, 5),
        title="Avg Nearby Volume",
        xtitle=None,
        ytitle="Spine Volume (um)",
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
        ax=axes["V"],
        save=False,
        save_path=None,
    )
    # Enlarged spine vol vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_nearby_volumes["sLTP"],
                plastic_shuff_volumes["sLTP"],
            )
        ),
        plot_ind=True,
        title="Enlarged",
        xtitle="Volume (\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["W"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_w_inset = axes["W"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_w_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nearby_volumes["sLTP"],
            "shuff": plastic_shuff_volumes["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Volume (\u03BCm)",
        ylim=None,
        b_colors=[COLORS[0], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_w_inset,
        save=False,
        save_path=None,
    )
    # Shrunken spine vol vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_nearby_volumes["sLTD"],
                plastic_shuff_volumes["sLTD"],
            )
        ),
        plot_ind=True,
        title="Shrunken",
        xtitle="Volume (\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["X"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_x_inset = axes["X"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_x_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nearby_volumes["sLTD"],
            "shuff": plastic_shuff_volumes["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Volume (\u03BCm)",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_x_inset,
        save=False,
        save_path=None,
    )
    # Stable spine vol vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_nearby_volumes["Stable"],
                plastic_shuff_volumes["Stable"],
            )
        ),
        plot_ind=True,
        title="Stable",
        xtitle="Volume (\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["Y"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_y_inset = axes["Y"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_y_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nearby_volumes["Stable"],
            "shuff": plastic_shuff_volumes["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Volume (\u03BCm)",
        ylim=None,
        b_colors=[COLORS[2], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_y_inset,
        save=False,
        save_path=None,
    )
    # Enlarged nearest neigbhr
    plot_box_plot(
        plastic_nn_enlarged,
        figsize=(5, 5),
        title="Enlarged",
        xtitle=None,
        ytitle="Enlarged nearest neighbor (\u03BCm)",
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
        ax=axes["Z"],
        save=False,
        save_path=None,
    )
    # Enlarged Enlarged nearest neighbor vs chance
    plot_histogram(
        data=list(
            (
                plastic_nn_enlarged["sLTP"],
                plastic_shuff_enlarged["sLTP"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Enlarged",
        xtitle="Enlarged nearest neighbor (\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "grey"],
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["a"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_A_inset = axes["a"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_A_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_enlarged["sLTP"],
            "shuff": plastic_shuff_enlarged["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Enlarged NN (\u03BCm)",
        ylim=None,
        b_colors=[COLORS[0], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_A_inset,
        save=False,
        save_path=None,
    )
    # Shrunken Enlarged nearest neighbor vs chance
    plot_histogram(
        data=list(
            (
                plastic_nn_enlarged["sLTD"],
                plastic_shuff_enlarged["sLTD"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Shrunken",
        xtitle="Enlarged nearest neighbor (\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["b"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_B_inset = axes["b"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_B_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_enlarged["sLTD"],
            "shuff": plastic_shuff_enlarged["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Enlarged NN (\u03BCm)",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_B_inset,
        save=False,
        save_path=None,
    )
    # Stable Enlarged nearest neighbor vs chance
    plot_histogram(
        data=list(
            (
                plastic_nn_enlarged["Stable"],
                plastic_shuff_enlarged["Stable"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Stable",
        xtitle="Enlarged nearest neighbor (\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "grey"],
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["c"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_C_inset = axes["c"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_C_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_enlarged["Stable"],
            "shuff": plastic_shuff_enlarged["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Enlarged NN (\u03BCm)",
        ylim=None,
        b_colors=[COLORS[2], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_C_inset,
        save=False,
        save_path=None,
    )
    # Shrunken nearest neigbhr
    plot_box_plot(
        plastic_nn_shrunken,
        figsize=(5, 5),
        title="Shrunken",
        xtitle=None,
        ytitle="Shrunken nearest neighbor (\u03BCm)",
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
        ax=axes["d"],
        save=False,
        save_path=None,
    )
    # Enlarged Shrunken nearest neighbor vs chance
    plot_histogram(
        data=list(
            (
                plastic_nn_shrunken["sLTP"],
                plastic_shuff_shrunken["sLTP"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Enlarged",
        xtitle="Shrunken nearest neighbor (\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "grey"],
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["e"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_E_inset = axes["e"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_E_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_shrunken["sLTP"],
            "shuff": plastic_shuff_shrunken["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Shrunken NN (\u03BCm)",
        ylim=None,
        b_colors=[COLORS[0], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_E_inset,
        save=False,
        save_path=None,
    )
    # Shrunken Shrunken nearest neighbor vs chance
    plot_histogram(
        data=list(
            (
                plastic_nn_shrunken["sLTD"],
                plastic_shuff_shrunken["sLTD"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Shrunken",
        xtitle="Shrunken nearest neighbor (\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["f"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_F_inset = axes["f"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_F_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_shrunken["sLTD"],
            "shuff": plastic_shuff_shrunken["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Shrunken NN (\u03BCm)",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_F_inset,
        save=False,
        save_path=None,
    )
    # Stable Shrunken nearest neighbor vs chance
    plot_histogram(
        data=list(
            (
                plastic_nn_shrunken["Stable"],
                plastic_shuff_shrunken["Stable"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Stable",
        xtitle="Shrunken nearest neighbor (\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "grey"],
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["g"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_G_inset = axes["g"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_G_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_shrunken["Stable"],
            "shuff": plastic_shuff_shrunken["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Shrunken NN (\u03BCm)",
        ylim=None,
        b_colors=[COLORS[2], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_G_inset,
        save=False,
        save_path=None,
    )
    # Local Relative volume
    plot_box_plot(
        plastic_rel_vols,
        figsize=(5, 5),
        title="Relative volumes",
        xtitle=None,
        ytitle="Nearby \u0394 Volume",
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
        ax=axes["i"],
        save=False,
        save_path=None,
    )
    # Enlarged rel vol vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_rel_vols["sLTP"],
                plastic_shuff_rel_vols["sLTP"],
            )
        ),
        plot_ind=True,
        title="Enlarged",
        xtitle="Nearby \u0394 Volume",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["j"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_J_inset = axes["j"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_J_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_rel_vols["sLTP"],
            "shuff": plastic_shuff_rel_vols["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Nearby \u0394 Volume",
        ylim=None,
        b_colors=[COLORS[0], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_J_inset,
        save=False,
        save_path=None,
    )
    # Shrunken rel vol vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_rel_vols["sLTD"],
                plastic_shuff_rel_vols["sLTD"],
            )
        ),
        plot_ind=True,
        title="Shrunken",
        xtitle="Nearby \u0394 Volume",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["k"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_K_inset = axes["k"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_K_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_rel_vols["sLTD"],
            "shuff": plastic_shuff_rel_vols["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Nearby \u0394 Volume",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_K_inset,
        save=False,
        save_path=None,
    )
    # Stable rate vs chance
    plot_cummulative_distribution(
        data=list(
            (
                plastic_rel_vols["Stable"],
                plastic_shuff_rel_vols["Stable"],
            )
        ),
        plot_ind=True,
        title="Stable",
        xtitle="Nearby \u0394 Volume",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["l"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_L_inset = axes["l"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_L_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_rel_vols["Stable"],
            "shuff": plastic_shuff_rel_vols["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Nearby \u0394 Volume",
        ylim=None,
        b_colors=[COLORS[2], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.9,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_L_inset,
        save=False,
        save_path=None,
    )
    # Near vs distant activity rates
    plot_box_plot(
        plastic_near_dist_rates,
        figsize=(5, 5),
        title="Near - distant activity rates",
        xtitle=None,
        ytitle="Relative activity rate (events/min)",
        ylim=None,
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
        ax=axes["m"],
        save=False,
        save_path=None,
    )
    # Near vs distant coactivity rates
    plot_box_plot(
        plastic_near_dist_coactivity,
        figsize=(5, 5),
        title="Near - distant coactivity rates",
        xtitle=None,
        ytitle="Relative coactivity rate (events/min)",
        ylim=None,
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
        ax=axes["n"],
        save=False,
        save_path=None,
    )
    # Near vs distant spine volumes
    plot_box_plot(
        plastic_near_dist_vol,
        figsize=(5, 5),
        title="Near - distant spine volumes",
        xtitle=None,
        ytitle="Relative spine volume (um)",
        ylim=None,
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
        ax=axes["o"],
        save=False,
        save_path=None,
    )
    # Near vs distant spine relative volumes
    plot_box_plot(
        plastic_near_dist_rel_vol,
        figsize=(5, 5),
        title="Near - distant relative volumes",
        xtitle=None,
        ytitle="Relative relative spine volume (um)",
        ylim=None,
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
        ax=axes["p"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()
