import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_box_plot import plot_box_plot
from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Plotting.plot_multi_line_plot import plot_multi_line_plot
from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils

sns.set()
sns.set_style("ticks")


def plot_coactive_vs_noncoactive_events(
    dataset,
    figsize=(8, 5),
    showmeans=False,
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to compare basic properties of coactive and non coactive events

        INPUT PARAMETERS
            dataset - Local_Coactivity_Data object

            figsize - tuple specifying the figure size

            showmeans - boolean specifying whether to show means on box plots

            test_type - str specifying whether to perform parametric or nonparametric tests

            display_stats - boolean specifying whether to display the stat results

            save - boolean specifying whether to save the data or not

            save_path - str specifying where to save the data
    """
    COLORS = ["forestgreen", "black"]

    # Pull relevant data
    sampling_rate = dataset.parameters["Sampling Rate"]
    activity_window = dataset.parameters["Activity Window"]
    if dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"

    # Activity data
    spine_coactive_traces = dataset.spine_coactive_traces
    spine_noncoactive_traces = dataset.spine_noncoactive_traces
    spine_coactive_calcium_traces = dataset.spine_coactive_calcium_traces
    spine_noncoactive_calcium_traces = dataset.spine_noncoactive_calcium_traces
    spine_coactive_amplitude = dataset.spine_coactive_amplitude
    spine_noncoactive_amplitude = dataset.spine_noncoactive_amplitude
    spine_coactive_calcium_amplitude = dataset.spine_coactive_calcium_amplitude
    spine_noncoactive_calcium_amplitude = dataset.spine_noncoactive_calcium_amplitude
    coactive_local_dend_traces = dataset.coactive_local_dend_traces
    coactive_local_dend_amplitude = dataset.coactive_local_dend_amplitude
    noncoactive_local_dend_traces = dataset.noncoactive_local_dend_traces
    noncoactive_local_dend_amplitude = dataset.noncoactive_local_dend_amplitude

    # Organize the traces for plotting
    coactive_traces = [
        np.nanmean(x, axis=1) for x in spine_coactive_traces if type(x) == np.ndarray
    ]
    coactive_traces = np.vstack(coactive_traces)
    coactive_means = np.nanmean(coactive_traces, axis=0)
    coactive_sems = stats.sem(coactive_traces, axis=0, nan_policy="omit")
    noncoactive_traces = [
        np.nanmean(x, axis=1) for x in spine_noncoactive_traces if type(x) == np.ndarray
    ]
    noncoactive_traces = np.vstack(noncoactive_traces)
    noncoactive_means = np.nanmean(noncoactive_traces, axis=0)
    noncoactive_sems = stats.sem(noncoactive_traces, axis=0, nan_policy="omit")
    coactive_ca_traces = [
        np.nanmean(x, axis=1)
        for x in spine_coactive_calcium_traces
        if type(x) == np.ndarray
    ]
    coactive_ca_traces = np.vstack(coactive_ca_traces)
    coactive_ca_means = np.nanmean(coactive_ca_traces, axis=0)
    coactive_ca_sems = stats.sem(coactive_ca_traces, axis=0, nan_policy="omit")
    noncoactive_ca_traces = [
        np.nanmean(x, axis=1)
        for x in spine_noncoactive_calcium_traces
        if type(x) == np.ndarray
    ]
    noncoactive_ca_traces = np.vstack(noncoactive_ca_traces)
    noncoactive_ca_means = np.nanmean(noncoactive_ca_traces, axis=0)
    noncoactive_ca_sems = stats.sem(noncoactive_ca_traces, axis=0, nan_policy="omit")
    coactive_dend_traces = [
        np.nanmean(x, axis=1)
        for x in coactive_local_dend_traces
        if type(x) == np.ndarray
    ]
    coactive_dend_traces = np.vstack(coactive_dend_traces)
    coactive_dend_means = np.nanmean(coactive_dend_traces, axis=0)
    coactive_dend_sems = stats.sem(coactive_dend_traces, axis=0, nan_policy="omit")
    noncoactive_dend_traces = [
        np.nanmean(x, axis=1)
        for x in noncoactive_local_dend_traces
        if type(x) == np.ndarray
    ]
    noncoactive_dend_traces = np.vstack(noncoactive_dend_traces)
    noncoactive_dend_means = np.nanmean(noncoactive_dend_traces, axis=0)
    noncoactive_dend_sems = stats.sem(
        noncoactive_dend_traces, axis=0, nan_policy="omit"
    )

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EF..
        """,
        figsize=figsize,
    )
    fig.suptitle("Coactive vs Noncoactive Activity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ########################### Plot data onto the axes ######################
    # coactive vs noncoactive GluSnFr traces
    plot_mean_activity_traces(
        means=[coactive_means, noncoactive_means],
        sems=[coactive_sems, noncoactive_sems],
        group_names=["Coactive", "Noncoactive"],
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="GluSnFr traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # coactive vs noncoactive Calcium traces
    plot_mean_activity_traces(
        means=[coactive_ca_means, noncoactive_ca_means],
        sems=[coactive_ca_sems, noncoactive_ca_sems],
        group_names=["Coactive", "Noncoactive"],
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Calcium traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # coactive vs noncoactive local dend traces
    plot_mean_activity_traces(
        means=[coactive_dend_means, noncoactive_dend_means],
        sems=[coactive_dend_sems, noncoactive_dend_sems],
        group_names=["Coactive", "Noncoactive"],
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Local dendrite traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Coactive vs Noncoactive GluSnFr amplitude
    plot_box_plot(
        data_dict={
            "Coactive": spine_coactive_amplitude,
            "Noncoactive": spine_noncoactive_amplitude,
        },
        figsize=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Coactive vs Noncoactive Calcium amplitude
    plot_box_plot(
        data_dict={
            "Coactive": spine_coactive_calcium_amplitude,
            "Noncoactive": spine_noncoactive_calcium_amplitude,
        },
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Coactive vs Noncoactive local amplitude
    plot_box_plot(
        data_dict={
            "Coactive": coactive_local_dend_amplitude,
            "Noncoactive": noncoactive_local_dend_amplitude,
        },
        figsize=(5, 5),
        title="Local Dendrite",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Local_Coactivity_Figure_1")
        fig.savefig(fname + ".pdf")

    ############################# Statistics Section #################################
    if display_stats == False:
        return

    # Perform the statistics
    if test_type == "parametric":
        amp_t, amp_p = stats.ttest_ind(
            spine_coactive_amplitude, spine_noncoactive_amplitude, nan_policy="omit"
        )
        ca_amp_t, ca_amp_p = stats.ttest_ind(
            spine_coactive_calcium_amplitude,
            spine_noncoactive_calcium_amplitude,
            nan_policy="omit",
        )
        dend_amp_t, dend_amp_p = stats.ttest_ind(
            coactive_local_dend_amplitude,
            noncoactive_local_dend_amplitude,
            nan_policy="omit",
        )
        test_title = "T-Test"
    elif test_type == "nonparametric":
        amp_t, amp_p = stats.mannwhitneyu(
            spine_coactive_amplitude[~np.isnan(spine_coactive_amplitude)],
            spine_noncoactive_amplitude[~np.isnan(spine_noncoactive_amplitude)],
        )
        ca_amp_t, ca_amp_p = stats.mannwhitneyu(
            spine_coactive_calcium_amplitude[
                ~np.isnan(spine_coactive_calcium_amplitude)
            ],
            spine_noncoactive_calcium_amplitude[
                ~np.isnan(spine_noncoactive_calcium_amplitude)
            ],
        )
        dend_amp_t, dend_amp_p = stats.mannwhitneyu(
            coactive_local_dend_amplitude[~np.isnan(coactive_local_dend_amplitude)],
            noncoactive_local_dend_amplitude[
                ~np.isnan(noncoactive_local_dend_amplitude)
            ],
        )
        test_title = "Mann-Whitney U"

    # Organize the results
    results_dict = {
        "test": ["GluSnFr Amp", "Calcium Amp", "Local Dend"],
        "stat": [amp_t, ca_amp_t, dend_amp_t],
        "p-val": [amp_p, ca_amp_p, dend_amp_p],
    }
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.update(results_df[["p-val"]].applymap("{:.4E}".format))

    # Display the stats
    fig2, axes2 = plt.subplot_mosaic("""A""", figsize=(4, 4))
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
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Local_Coactivity_Figure_1_Stats")
        fig2.savefig(fname + ".pdf")


def plot_comparative_mvmt_coactivity(
    mvmt_dataset,
    nonmvmt_dataset,
    rwd_mvmts=False,
    figsize=(10, 6),
    showmeans=False,
    mean_type="median",
    err_type="CI",
    hist_bins=30,
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to compare coactivity during movement and non movement periods

        INPUT PARAMETERS
            mvmt_dataset - Local_Coactivity_Data that was constrained to only mvmt periods

            nonmvmt_dataset - Local_Coactivity_Data that was constrained to nonmvmt periods

            rwd_mvmts - boolean of whether or not the datasets are rwd mvmts

            figsize - tuple specifying the size of the figure

            showmeans - boolean specifying whether to plot means on box plots

            hist_bins - int specifying how many bins for the histograms

            test_type - str specifying whether to perform parametric or nonparametric tests

            display_stats - boolean specifying whether to display stat results

            save - boolean specifying whether to save the figure or not

            save_path - str specifying where to save the figure
    
    """
    COLORS = ["mediumblue", "firebrick"]
    if rwd_mvmts:
        mvmt_key = "Rwd mvmt"
        nonmvmt_key = "Nonrwd mvmt"
    else:
        mvmt_key = "Mvmt"
        nonmvmt_key = "Nonmvmt"

    # Pull relevant data
    sampling_rate = mvmt_dataset.parameters["Sampling Rate"]
    activity_window = mvmt_dataset.parameters["Activity Window"]
    if mvmt_dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"

    ## Coactivity related variables
    distance_bins = mvmt_dataset.parameters["position bins"][1:]
    distance_coactivity_rate = {
        mvmt_key: mvmt_dataset.distance_coactivity_rate,
        nonmvmt_key: nonmvmt_dataset.distance_coactivity_rate,
    }
    distance_coactivity_rate_norm = {
        mvmt_key: mvmt_dataset.distance_coactivity_rate_norm,
        nonmvmt_key: nonmvmt_dataset.distance_coactivity_rate_norm,
    }
    avg_local_coactivity_rate = {
        mvmt_key: mvmt_dataset.avg_local_coactivity_rate,
        nonmvmt_key: nonmvmt_dataset.avg_local_coactivity_rate,
    }

    shuff_local_coactivity_rate = {
        mvmt_key: mvmt_dataset.shuff_local_coactivity_rate.flatten().astype(np.float32),
        nonmvmt_key: nonmvmt_dataset.shuff_local_coactivity_rate.flatten().astype(
            np.float32
        ),
    }
    shuff_local_coactivity_medians = {
        mvmt_key: np.nanmedian(mvmt_dataset.shuff_local_coactivity_rate, axis=1),
        nonmvmt_key: np.nanmedian(nonmvmt_dataset.shuff_local_coactivity_rate, axis=1),
    }
    real_vs_shuff_diff = {
        mvmt_key: mvmt_dataset.real_vs_shuff_coactivity_diff,
        nonmvmt_key: nonmvmt_dataset.real_vs_shuff_coactivity_diff,
    }
    avg_local_coactivity_rate_norm = {
        mvmt_key: mvmt_dataset.avg_local_coactivity_rate_norm,
        nonmvmt_key: nonmvmt_dataset.avg_local_coactivity_rate_norm,
    }

    shuff_local_coactivity_rate_norm = {
        mvmt_key: mvmt_dataset.shuff_local_coactivity_rate_norm.flatten().astype(
            np.float32
        ),
        nonmvmt_key: nonmvmt_dataset.shuff_local_coactivity_rate_norm.flatten().astype(
            np.float32
        ),
    }
    shuff_local_coactivity_medians_norm = {
        mvmt_key: np.nanmedian(mvmt_dataset.shuff_local_coactivity_rate_norm, axis=1),
        nonmvmt_key: np.nanmedian(
            nonmvmt_dataset.shuff_local_coactivity_rate_norm, axis=1
        ),
    }
    real_vs_shuff_diff_norm = {
        mvmt_key: mvmt_dataset.real_vs_shuff_coactivity_diff_norm,
        nonmvmt_key: nonmvmt_dataset.real_vs_shuff_coactivity_diff_norm,
    }
    ### Traces
    mvmt_coactive_traces = mvmt_dataset.spine_coactive_traces
    mvmt_coactive_means = [
        np.nanmean(x, axis=1) for x in mvmt_coactive_traces if type(x) == np.ndarray
    ]
    mvmt_coactive_means = np.vstack(mvmt_coactive_means)
    nonmvmt_coactive_traces = nonmvmt_dataset.spine_coactive_traces
    nonmvmt_coactive_means = [
        np.nanmean(x, axis=1) for x in nonmvmt_coactive_traces if type(x) == np.ndarray
    ]
    nonmvmt_coactive_means = np.vstack(nonmvmt_coactive_means)
    coactive_trace_means = {
        mvmt_key: np.nanmean(mvmt_coactive_means, axis=0),
        nonmvmt_key: np.nanmean(nonmvmt_coactive_means, axis=0),
    }
    coactive_trace_sems = {
        mvmt_key: stats.sem(mvmt_coactive_means, axis=0, nan_policy="omit"),
        nonmvmt_key: stats.sem(nonmvmt_coactive_means, axis=0, nan_policy="omit"),
    }
    ### Calcium Traces
    mvmt_coactive_ca_traces = mvmt_dataset.spine_coactive_calcium_traces
    mvmt_coactive_ca_means = [
        np.nanmean(x, axis=1) for x in mvmt_coactive_ca_traces if type(x) == np.ndarray
    ]
    mvmt_coactive_ca_means = np.vstack(mvmt_coactive_ca_means)
    nonmvmt_coactive_ca_traces = nonmvmt_dataset.spine_coactive_calcium_traces
    nonmvmt_coactive_ca_means = [
        np.nanmean(x, axis=1)
        for x in nonmvmt_coactive_ca_traces
        if type(x) == np.ndarray
    ]
    nonmvmt_coactive_ca_means = np.vstack(nonmvmt_coactive_ca_means)
    coactive_trace_ca_means = {
        mvmt_key: np.nanmean(mvmt_coactive_ca_means, axis=0),
        nonmvmt_key: np.nanmean(nonmvmt_coactive_ca_means, axis=0),
    }
    coactive_trace_ca_sems = {
        mvmt_key: stats.sem(mvmt_coactive_ca_means, axis=0, nan_policy="omit"),
        nonmvmt_key: stats.sem(nonmvmt_coactive_ca_means, axis=0, nan_policy="omit"),
    }
    coactive_amplitude = {
        mvmt_key: mvmt_dataset.spine_coactive_amplitude,
        nonmvmt_key: nonmvmt_dataset.spine_coactive_amplitude,
    }
    coactive_calcium_amplitude = {
        mvmt_key: mvmt_dataset.spine_coactive_calcium_amplitude,
        nonmvmt_key: nonmvmt_dataset.spine_coactive_calcium_amplitude,
    }

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE
        FGHIJ
        KL.MN
        """,
        width_ratios=[2, 1, 2, 2, 1],
        figsize=figsize,
    )
    fig.suptitle(f"{mvmt_key} vs {nonmvmt_key} Coactivity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ######################## Plot data onto the axes #######################
    # Mvmt vs nonmvmt distance coactivity
    plot_multi_line_plot(
        data_dict=distance_coactivity_rate,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Coactivity rate (events/min)",
        xtitle="Distance (\u03BCm)",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=5,
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
    # Mvmt vs nonmvmt distance coactivity normalized
    plot_multi_line_plot(
        data_dict=distance_coactivity_rate_norm,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Norm. coactivity rate",
        xtitle="Distance (\u03BCm)",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=5,
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
    # Mvmt vs nonmvmt local coactivity rate
    plot_box_plot(
        avg_local_coactivity_rate,
        figsize=(5, 5),
        title="Local Coactivity",
        xtitle=None,
        ytitle=f"Coactivity rate (events/min)",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Mvmt vs nonmvmt local coactivity rate norm
    plot_box_plot(
        avg_local_coactivity_rate_norm,
        figsize=(5, 5),
        title="Local Coactivity",
        xtitle=None,
        ytitle=f"Norm. coactivity rate",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Movement local coactivity vs chance
    ## Histogram
    plot_histogram(
        data=list(
            (
                avg_local_coactivity_rate[mvmt_key],
                shuff_local_coactivity_rate[mvmt_key],
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title=mvmt_key,
        xtitle="Coactivity rate (events/min)",
        xlim=(0, 10),
        figsize=(5, 5),
        color=[COLORS[0], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    ## Inset bar plot
    ax_c_inset = axes["C"].inset_axes([0.8, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_c_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": avg_local_coactivity_rate[mvmt_key],
            "shuff": shuff_local_coactivity_medians[mvmt_key],
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
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_c_inset,
        save=False,
        save_path=None,
    )
    # Non movement local coactivity vs chance
    ## Histogram
    plot_histogram(
        data=list(
            (
                avg_local_coactivity_rate[nonmvmt_key],
                shuff_local_coactivity_rate[nonmvmt_key],
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title=nonmvmt_key,
        xtitle="Coactivity rate (events/min)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    ## Inset bar plot
    ax_d_inset = axes["D"].inset_axes([0.8, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_d_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": avg_local_coactivity_rate[nonmvmt_key],
            "shuff": shuff_local_coactivity_medians[nonmvmt_key],
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
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_d_inset,
        save=False,
        save_path=None,
    )
    # Movement local norm coactivity vs chance
    ## Histogram
    plot_histogram(
        data=list(
            (
                avg_local_coactivity_rate_norm[mvmt_key],
                shuff_local_coactivity_rate_norm[mvmt_key],
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title=mvmt_key,
        xtitle="Norm. coactivity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    ## Inset bar plot
    ax_h_inset = axes["H"].inset_axes([0.8, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_h_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": avg_local_coactivity_rate_norm[mvmt_key],
            "shuff": shuff_local_coactivity_medians_norm[mvmt_key],
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Norm.\ncoactivity rate",
        ylim=None,
        b_colors=[COLORS[0], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_h_inset,
        save=False,
        save_path=None,
    )
    # Non movement local norm coactivity vs chance
    ## Histogram
    plot_histogram(
        data=list(
            (
                avg_local_coactivity_rate_norm[nonmvmt_key],
                shuff_local_coactivity_rate_norm[nonmvmt_key],
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title=nonmvmt_key,
        xtitle="Norm. coactivity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    ## Inset bar plot
    ax_i_inset = axes["I"].inset_axes([0.8, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_i_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": avg_local_coactivity_rate_norm[nonmvmt_key],
            "shuff": shuff_local_coactivity_medians_norm[nonmvmt_key],
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Norm\ncoactivity rate",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_i_inset,
        save=False,
        save_path=None,
    )
    # Real vs shuff relative difference
    plot_box_plot(
        real_vs_shuff_diff,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle=f"Above chance difference",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Real vs shuff relative difference norm
    plot_box_plot(
        real_vs_shuff_diff_norm,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle=f"Above chance difference (norm)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    # GluSnFr activity traces
    plot_mean_activity_traces(
        means=list(coactive_trace_means.values()),
        sems=list(coactive_trace_sems.values()),
        group_names=list(coactive_trace_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="GluSnFr traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Calcium activity traces
    plot_mean_activity_traces(
        means=list(coactive_trace_ca_means.values()),
        sems=list(coactive_trace_ca_sems.values()),
        group_names=list(coactive_trace_ca_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Calcium traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    # GluSnFr amplitudes
    plot_box_plot(
        coactive_amplitude,
        figsize=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=(0.1, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Calcium amplitudes
    plot_box_plot(
        coactive_calcium_amplitude,
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["N"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if rwd_mvmts:
            fname = os.path.join(save_path, "Local_Coactivity_Figure_3")
        else:
            fname = os.path.join(save_path, "Local_Coactivity_Figure_2")
        fig.savefig(fname + ".pdf")

    ########################### Statistics Section ###########################
    if display_stats == False:
        return

    # Perform t-tests / mann-whitney u tests
    if test_type == "parametric":
        coactivity_t, coactivity_p = stats.ttest_ind(
            avg_local_coactivity_rate[mvmt_key],
            avg_local_coactivity_rate[nonmvmt_key],
            nan_policy="omit",
        )
        coactivity_norm_t, coactivity_norm_p = stats.ttest_ind(
            avg_local_coactivity_rate_norm[mvmt_key],
            avg_local_coactivity_rate_norm[nonmvmt_key],
            nan_policy="omit",
        )
        rel_diff_t, rel_diff_p = stats.ttest_ind(
            real_vs_shuff_diff[mvmt_key],
            real_vs_shuff_diff[nonmvmt_key],
            nan_policy="omit",
        )
        rel_diff_norm_t, rel_diff_norm_p = stats.ttest_ind(
            real_vs_shuff_diff_norm[mvmt_key],
            real_vs_shuff_diff_norm[nonmvmt_key],
            nan_policy="omit",
        )
        amp_t, amp_p = stats.ttest_ind(
            coactive_amplitude[mvmt_key],
            coactive_amplitude[nonmvmt_key],
            nan_policy="omit",
        )
        amp_ca_t, amp_ca_p = stats.ttest_ind(
            coactive_calcium_amplitude[mvmt_key],
            coactive_calcium_amplitude[nonmvmt_key],
            nan_policy="omit",
        )
        test_title = "T-Test"
    elif test_type == "nonparametric":
        coactivity_t, coactivity_p = stats.mannwhitneyu(
            avg_local_coactivity_rate[mvmt_key][
                ~np.isnan(avg_local_coactivity_rate[mvmt_key])
            ],
            avg_local_coactivity_rate[nonmvmt_key][
                ~np.isnan(avg_local_coactivity_rate[nonmvmt_key])
            ],
        )
        coactivity_norm_t, coactivity_norm_p = stats.mannwhitneyu(
            avg_local_coactivity_rate_norm[mvmt_key][
                ~np.isnan(avg_local_coactivity_rate_norm[mvmt_key])
            ],
            avg_local_coactivity_rate_norm[nonmvmt_key][
                ~np.isnan(avg_local_coactivity_rate_norm[nonmvmt_key])
            ],
        )
        rel_diff_t, rel_diff_p = stats.mannwhitneyu(
            real_vs_shuff_diff[mvmt_key][~np.isnan(real_vs_shuff_diff[mvmt_key])],
            real_vs_shuff_diff[nonmvmt_key][~np.isnan(real_vs_shuff_diff[nonmvmt_key])],
        )
        rel_diff_norm_t, rel_diff_norm_p = stats.mannwhitneyu(
            real_vs_shuff_diff_norm[mvmt_key][
                ~np.isnan(real_vs_shuff_diff_norm[mvmt_key])
            ],
            real_vs_shuff_diff_norm[nonmvmt_key][
                ~np.isnan(real_vs_shuff_diff_norm[nonmvmt_key])
            ],
        )
        amp_t, amp_p = stats.mannwhitneyu(
            coactive_amplitude[mvmt_key][~np.isnan(coactive_amplitude[mvmt_key])],
            coactive_amplitude[nonmvmt_key][~np.isnan(coactive_amplitude[nonmvmt_key])],
        )
        amp_ca_t, amp_ca_p = stats.mannwhitneyu(
            coactive_calcium_amplitude[mvmt_key][
                ~np.isnan(coactive_calcium_amplitude[mvmt_key])
            ],
            coactive_calcium_amplitude[nonmvmt_key][
                ~np.isnan(coactive_calcium_amplitude[nonmvmt_key])
            ],
        )
        test_title = "Mann-Whitney U"

    # Oranize these results
    result_dict = {
        "Comparison": [
            "Local coactivity",
            "Local norm. coactivity",
            "Chance diff.",
            "Chance diff. norm.",
            "Amplitude",
            "Ca Amplitude",
        ],
        "stat": [
            coactivity_t,
            coactivity_norm_t,
            rel_diff_t,
            rel_diff_norm_t,
            amp_t,
            amp_ca_t,
        ],
        "p-val": [
            coactivity_p,
            coactivity_norm_p,
            rel_diff_p,
            rel_diff_norm_p,
            amp_p,
            amp_ca_p,
        ],
    }
    result_df = pd.DataFrame.from_dict(result_dict)
    result_df.update(result_df[["p-val"]].applymap("{:.4E}".format))

    # Perform correlations
    _, distance_corr_df = t_utils.correlate_grouped_data(
        distance_coactivity_rate, distance_bins
    )
    _, distance_corr_df_norm = t_utils.correlate_grouped_data(
        distance_coactivity_rate_norm, distance_bins
    )

    ## Comparisons to chance
    mvmt_above, mvmt_below = t_utils.test_against_chance(
        avg_local_coactivity_rate[mvmt_key], mvmt_dataset.shuff_local_coactivity_rate
    )
    nonmvmt_above, nonmvmt_below = t_utils.test_against_chance(
        avg_local_coactivity_rate[nonmvmt_key],
        nonmvmt_dataset.shuff_local_coactivity_rate,
    )
    mvmt_above_norm, mvmt_below_norm = t_utils.test_against_chance(
        avg_local_coactivity_rate_norm[mvmt_key],
        mvmt_dataset.shuff_local_coactivity_rate_norm,
    )
    nonmvmt_above_norm, nonmvmt_below_norm = t_utils.test_against_chance(
        avg_local_coactivity_rate_norm[nonmvmt_key],
        nonmvmt_dataset.shuff_local_coactivity_rate_norm,
    )
    chance_dict = {
        "Comparison": [
            mvmt_key,
            nonmvmt_key,
            f"{mvmt_key} norm",
            f"{nonmvmt_key} norm",
        ],
        "p-val above": [mvmt_above, nonmvmt_above, mvmt_above_norm, nonmvmt_above_norm],
        "p-val below": [mvmt_below, nonmvmt_below, mvmt_below_norm, nonmvmt_below_norm],
    }
    chance_df = pd.DataFrame.from_dict(chance_dict)
    chance_df.update(chance_df[["p-val above"]].applymap("{:.4E}".format))
    chance_df.update(chance_df[["p-val below"]].applymap("{:.4E}".format))

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        """,
        figsize=(10, 6),
    )
    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title("Distance Coactivity Rate")
    A_table = axes2["A"].table(
        cellText=distance_corr_df.values,
        colLabels=distance_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title("Distance Norm. Coactivity Rate")
    B_table = axes2["B"].table(
        cellText=distance_corr_df_norm.values,
        colLabels=distance_corr_df_norm.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title("Chance Coactivity")
    C_table = axes2["C"].table(
        cellText=chance_df.values,
        colLabels=chance_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(f"{test_title} results")
    D_table = axes2["D"].table(
        cellText=result_df.values,
        colLabels=result_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if rwd_mvmts:
            fname = os.path.join(save_path, "Local_Coactivity_Figure_3_Stats")
        else:
            fname = os.path.join(save_path, "Local_Coactivity_Figure_2_Stats")
        fig2.savefig(fname + ".pdf")


def plot_plasticity_coactivity_rates(
    dataset,
    mvmt_dataset,
    nonmvmt_dataset,
    followup_dataset=None,
    norm=False,
    rwd_mvmt=False,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 8),
    hist_bins=30,
    showmeans=False,
    mean_type="median",
    err_type="CI",
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """ Function to compare coactivity rates across plasticity groups

        INPUT PARAMETERS
            dataset - Local_Coactivity_Data object analyzed over all periods

            mvmt_dataset - Local_Coactivity_Data object constrained to mvmt periods

            nonmvmt_dataset - Local_Coactivity_Data object constrained to nonmvmt periods

            followup_dataset - optional Local_Coactivity_Data object of the subsequent
                                session to used for volume comparision. Default is None
                                to use the followup_volumes in the dataset
                
            norm - boolean term specifying whether to use the normalized coactivity rate
                    or not

            rwd_mvmt - boolean term of whether or not we are comparing rwd mvmts
            
        `   exclude - str specifying spine type to exclude from analysis

            threshold - float or tuple of floats specifying the threshold cutoff for 
                        classifying plasticity
                
            figsize - tuple specifying the size of the figure

            hist_bins - int specifying how many bins for the histograms

            showmeans - boolean specifying whether to plot means on box plots

            mean_type - str specifying the mean type for bar plots

            err_type - str specifying the error type for the bar plots

            test_type - str specifying whether to perform parametric or nonparametric tests

            test_method - str specifying the typ of posthoc test to perform

            display_stats - boolean specifying whether to display stats

            save - boolean specifying whether to save the figures or not

            save_path - str specifying where to save the figures
    
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    plastic_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }
    if rwd_mvmt:
        mvmt_key = "Rwd mvmt"
        nonmvmt_key = "Nonrwd mvmt"
    else:
        mvmt_key = "Mvmt"
        nonmvmt_key = "Nonmvmt"

    # Pull relevant data
    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset is None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags

    ## Coactivity-related variables
    if norm == False:
        nname = "Raw"
        coactivity_title = "Coactivity rate (event/min)"
        distance_coactivity_rate = dataset.distance_coactivity_rate
        avg_local_coactivity_rate = dataset.avg_local_coactivity_rate
        shuff_local_coactivity_rate = dataset.shuff_local_coactivity_rate
        coactive_spines = dataset.coactive_spines
        real_vs_shuff_diff = dataset.real_vs_shuff_coactivity_diff
        mvmt_distance_coactivity_rate = mvmt_dataset.distance_coactivity_rate
        mvmt_avg_local_coactivity_rate = mvmt_dataset.avg_local_coactivity_rate
        nonmvmt_distance_coactivity_rate = nonmvmt_dataset.distance_coactivity_rate
        nonmvmt_avg_local_coactivity_rate = nonmvmt_dataset.avg_local_coactivity_rate
        coactive_event_num = dataset.spine_coactive_event_num
        mvmt_coactive_event_num = mvmt_dataset.spine_coactive_event_num
    else:
        nname = "Norm."
        coactivity_title = "Norm. coactivity rate"
        distance_coactivity_rate = dataset.distance_coactivity_rate_norm
        avg_local_coactivity_rate = dataset.avg_local_coactivity_rate_norm
        shuff_local_coactivity_rate = dataset.shuff_local_coactivity_rate_norm
        coactive_spines = dataset.coactive_norm_spines
        real_vs_shuff_diff = dataset.real_vs_shuff_coactivity_diff_norm
        mvmt_distance_coactivity_rate = mvmt_dataset.distance_coactivity_rate_norm
        mvmt_avg_local_coactivity_rate = mvmt_dataset.avg_local_coactivity_rate_norm
        nonmvmt_distance_coactivity_rate = nonmvmt_dataset.distance_coactivity_rate_norm
        nonmvmt_avg_local_coactivity_rate = (
            nonmvmt_dataset.avg_local_coactivity_rate_norm
        )
        coactive_event_num = dataset.spine_coactive_event_num
        mvmt_coactive_event_num = mvmt_dataset.spine_coactive_event_num

    # Calculate relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=vol_norm, exclude=exclude
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=vol_norm
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
    coactive_spines = d_utils.subselect_data_by_idxs(coactive_spines, spine_idxs)
    real_vs_shuff_diff = d_utils.subselect_data_by_idxs(real_vs_shuff_diff, spine_idxs)
    mvmt_distance_coactivity_rate = d_utils.subselect_data_by_idxs(
        mvmt_distance_coactivity_rate, spine_idxs
    )
    mvmt_avg_local_coactivity_rate = d_utils.subselect_data_by_idxs(
        mvmt_avg_local_coactivity_rate, spine_idxs
    )
    nonmvmt_distance_coactivity_rate = d_utils.subselect_data_by_idxs(
        nonmvmt_distance_coactivity_rate, spine_idxs
    )
    nonmvmt_avg_local_coactivity_rate = d_utils.subselect_data_by_idxs(
        nonmvmt_avg_local_coactivity_rate, spine_idxs
    )
    coactive_event_num = d_utils.subselect_data_by_idxs(coactive_event_num, spine_idxs)
    mvmt_coactive_event_num = d_utils.subselect_data_by_idxs(
        mvmt_coactive_event_num, spine_idxs
    )

    # Seperate into groups
    plastic_distance_rates = {}
    plastic_local_rates = {}
    plastic_shuff_rates = {}
    plastic_shuff_medians = {}
    plastic_diffs = {}
    mvmt_distance_rates = {}
    mvmt_local_rates = {}
    nonmvmt_distance_rates = {}
    nonmvmt_local_rates = {}
    fraction_coactive = {}
    fraction_mvmt = {}
    distance_bins = dataset.parameters["position bins"][1:]

    for key, value in plastic_groups.items():
        # Get spine types
        spines = eval(value)
        # Subselect data
        plastic_distance_rates[key] = distance_coactivity_rate[:, spines]
        plastic_local_rates[key] = avg_local_coactivity_rate[spines]
        plastic_diffs[key] = real_vs_shuff_diff[spines]
        mvmt_distance_rates[key] = mvmt_distance_coactivity_rate[:, spines]
        mvmt_local_rates[key] = mvmt_avg_local_coactivity_rate[spines]
        nonmvmt_distance_rates[key] = nonmvmt_distance_coactivity_rate[:, spines]
        nonmvmt_local_rates[key] = nonmvmt_avg_local_coactivity_rate[spines]
        ## Process shuff data
        shuff_rates = shuff_local_coactivity_rate[:, spines]
        plastic_shuff_rates[key] = shuff_rates
        plastic_shuff_medians[key] = np.nanmedian(shuff_rates, axis=1)
        ## Calculate fractions
        co_spines = coactive_spines[spines]
        event_num = coactive_event_num[spines]
        mvmt_event_num = mvmt_coactive_event_num[spines]
        fraction_coactive[key] = np.nansum(co_spines) / len(co_spines)
        fraction_mvmt[key] = mvmt_event_num / event_num

    print(fraction_coactive)

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFGH
        IJKL
        MNO.
        """,
        figsize=figsize,
    )
    fig.suptitle(f"Plastic {nname} Coactivity Rates")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ########################### Plot data onto axes ######################
    # Distance coactivity rates
    plot_multi_line_plot(
        data_dict=plastic_distance_rates,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="All periods",
        ytitle=coactivity_title,
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
        ax=axes["A"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Movement distance coactivity rates
    plot_multi_line_plot(
        data_dict=mvmt_distance_rates,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title=mvmt_key,
        ytitle=coactivity_title,
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
    # Nonmvmt distance coactivity rates
    plot_multi_line_plot(
        data_dict=nonmvmt_distance_rates,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title=nonmvmt_key,
        ytitle=coactivity_title,
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
        ax=axes["M"],
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
    # Mvmt Local coactivity vs delta volume
    plot_scatter_correlation(
        x_var=mvmt_avg_local_coactivity_rate,
        y_var=delta_volume,
        CI=95,
        title=mvmt_key,
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
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    # Nonmvmt Local coactivity vs delta volume
    plot_scatter_correlation(
        x_var=nonmvmt_avg_local_coactivity_rate,
        y_var=delta_volume,
        CI=95,
        title=nonmvmt_key,
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
        ax=axes["N"],
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
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Mvmt local coactivity rate bar plot
    plot_box_plot(
        mvmt_local_rates,
        figsize=(5, 5),
        title=mvmt_key,
        xtitle=None,
        ytitle=f"Local {coactivity_title}",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Nonmvt local coactivity rate bar plot
    plot_box_plot(
        nonmvmt_local_rates,
        figsize=(5, 5),
        title=nonmvmt_key,
        xtitle=None,
        ytitle=f"Local {coactivity_title}",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    # Fraction coactive spines
    plot_swarm_bar_plot(
        fraction_coactive,
        mean_type=mean_type,
        err_type="std",
        figsize=(5, 5),
        title="Coactive spines",
        xtitle=None,
        ytitle=f"Fraction of spines",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.8,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # Fraction mvmt events
    plot_box_plot(
        fraction_mvmt,
        figsize=(5, 5),
        title=mvmt_key,
        xtitle=None,
        ytitle=f"Fraction of events",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Real vs shuff relative difference
    plot_box_plot(
        plastic_diffs,
        figsize=(5, 5),
        title="All periods",
        xtitle=None,
        ytitle=f"Above chance difference",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Enlarged local vs chance
    ## histogram
    plot_histogram(
        data=list(
            (
                plastic_local_rates["Enlarged"],
                plastic_shuff_rates["Enlarged"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Enlarged",
        xtitle=f"Local {coactivity_title}",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_e_inset = axes["E"].inset_axes([0.8, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_e_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_local_rates["Enlarged"],
            "shuff": plastic_shuff_medians["Enlarged"],
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
        b_linewidth=0,
        b_alpha=0.3,
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
    plot_histogram(
        data=list(
            (
                plastic_local_rates["Shrunken"],
                plastic_shuff_rates["Shrunken"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Shrunken",
        xtitle=f"Local {coactivity_title}",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_f_inset = axes["F"].inset_axes([0.8, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_f_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_local_rates["Shrunken"],
            "shuff": plastic_shuff_medians["Shrunken"],
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
        b_linewidth=0,
        b_alpha=0.3,
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
    plot_histogram(
        data=list(
            (
                plastic_local_rates["Stable"],
                plastic_shuff_rates["Stable"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Stable",
        xtitle=f"Local {coactivity_title}",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_g_inset = axes["G"].inset_axes([0.8, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_g_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_local_rates["Stable"],
            "shuff": plastic_shuff_medians["Stable"],
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
        b_linewidth=0,
        b_alpha=0.3,
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
    # Save secton
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if norm == False:
            if not rwd_mvmt:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_4")
            else:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_5")
        else:
            if not rwd_mvmt:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_6")
            else:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_7")
        fig.savefig(fname + ".pdf")

    #################### Statistics Section ##################3
    if display_stats == False:
        return

    # Perform f-tests
    if test_type == "parametric":
        coactivity_f, coactivity_p, _, coactivity_df = t_utils.ANOVA_1way_posthoc(
            plastic_local_rates, method=test_method,
        )
        diff_f, diff_p, _, diff_df = t_utils.ANOVA_1way_posthoc(
            plastic_diffs, test_method,
        )
        (
            mvmt_coactivity_f,
            mvmt_coactivity_p,
            _,
            mvmt_coactivity_df,
        ) = t_utils.ANOVA_1way_posthoc(mvmt_local_rates, test_method,)
        (
            non_coactivity_f,
            non_coactivity_p,
            _,
            non_coactivity_df,
        ) = t_utils.ANOVA_1way_posthoc(nonmvmt_local_rates, test_method,)
        frac_mvmt_f, frac_mvmt_p, _, frac_mvmt_df = t_utils.ANOVA_1way_posthoc(
            fraction_mvmt, test_method,
        )
        test_title = f"One-way ANOVA {test_method}"
    elif test_type == "nonparametric":
        coactivity_f, coactivity_p, coactivity_df = t_utils.kruskal_wallis_test(
            plastic_local_rates, "Conover", test_method,
        )
        diff_f, diff_p, diff_df = t_utils.kruskal_wallis_test(
            plastic_diffs, "Conover", test_method,
        )
        (
            mvmt_coactivity_f,
            mvmt_coactivity_p,
            mvmt_coactivity_df,
        ) = t_utils.kruskal_wallis_test(mvmt_local_rates, "Conover", test_method,)
        (
            non_coactivity_f,
            non_coactivity_p,
            non_coactivity_df,
        ) = t_utils.kruskal_wallis_test(nonmvmt_local_rates, "Conover", test_method,)
        frac_mvmt_f, frac_mvmt_p, frac_mvmt_df = t_utils.kruskal_wallis_test(
            fraction_mvmt, "Conover", test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"

    # Perform correlations
    _, distance_corr_df = t_utils.correlate_grouped_data(
        plastic_distance_rates, distance_bins,
    )
    _, mvmt_distance_corr_df = t_utils.correlate_grouped_data(
        mvmt_distance_rates, distance_bins,
    )
    _, nonmvmt_distance_corr_df = t_utils.correlate_grouped_data(
        nonmvmt_distance_rates, distance_bins,
    )

    # Comparisons to chance
    enlarged_above, enlarged_below = t_utils.test_against_chance(
        plastic_local_rates["Enlarged"], plastic_shuff_rates["Enlarged"]
    )
    shrunken_above, shrunken_below = t_utils.test_against_chance(
        plastic_local_rates["Shrunken"], plastic_shuff_rates["Shrunken"]
    )
    stable_above, stable_below = t_utils.test_against_chance(
        plastic_local_rates["Stable"], plastic_shuff_rates["Stable"]
    )
    chance_dict = {
        "Comparison": ["Enlarged", "Shrunken", "Stable"],
        "p-val above": [enlarged_above, shrunken_above, stable_above],
        "p-val below": [enlarged_below, shrunken_below, stable_below],
    }
    chance_df = pd.DataFrame.from_dict(chance_dict)

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        ABC
        DEF
        GHI
        """,
        figsize=(10, 8),
    )
    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title("Distance Coactivity Rate")
    A_table = axes2["A"].table(
        cellText=distance_corr_df.values,
        colLabels=distance_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(f"{mvmt_key} Distance Coactivity Rate")
    B_table = axes2["B"].table(
        cellText=mvmt_distance_corr_df.values,
        colLabels=mvmt_distance_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(f"{nonmvmt_key} Distance Coactivity Rate")
    C_table = axes2["C"].table(
        cellText=nonmvmt_distance_corr_df.values,
        colLabels=nonmvmt_distance_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(
        f"Local Rate {test_title}\nF = {coactivity_f:.4} p = {coactivity_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=coactivity_df.values,
        colLabels=coactivity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(
        f"{mvmt_key} Local Rate {test_title}\nF = {mvmt_coactivity_f:.4} p = {mvmt_coactivity_p:.3E}"
    )
    E_table = axes2["E"].table(
        cellText=mvmt_coactivity_df.values,
        colLabels=mvmt_coactivity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title(
        f"{nonmvmt_key} Local Rate {test_title}\nF = {non_coactivity_f:.4} p = {non_coactivity_p:.3E}"
    )
    F_table = axes2["F"].table(
        cellText=non_coactivity_df.values,
        colLabels=non_coactivity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)
    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title(f"Rel Diff {test_title}\nF = {diff_f:.4} p = {diff_p:.3E}")
    G_table = axes2["G"].table(
        cellText=diff_df.values,
        colLabels=diff_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title(
        f"Fraction {mvmt_key} {test_title}\nF = {frac_mvmt_f:.4} p = {frac_mvmt_p:.3E}"
    )
    H_table = axes2["H"].table(
        cellText=frac_mvmt_df.values,
        colLabels=frac_mvmt_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)
    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title("Chance Coactivity")
    I_table = axes2["I"].table(
        cellText=chance_df.values,
        colLabels=chance_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if norm == False:
            if not rwd_mvmt:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_4_Stats")
            else:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_5_Stats")
        else:
            if not rwd_mvmt:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_6_Stats")
            else:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_7_Stats")
        fig2.savefig(fname + ".pdf")


def plot_coactive_event_properties(
    dataset,
    mvmt_dataset,
    nonmvmt_dataset,
    followup_dataset=None,
    rwd_mvmt=False,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 8),
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """Function to compare the properties of coactive events across plastic spines
    
        INPUT PARAMETERS
            dataset - Local_Coactivity_Data object analyzed over all periods

            mvmt_dataset - Local_Coactivity_Data object constrained to mvmt periods

            nonmvmt_dataset - Local_Coactivity_Data object constrained to nonmvmt periods

            followup_dataset - optional Local_Coactivity_Data object of the subsequent
                                session to be used for volume comparison. Default is None
                                to use the followup volumes in the dataset

            rwd_mvmt - boolean term of whetehr or not we are comparing rwd mvmts

            exclude - str specifying spine type to exclude from analysis

            threshold - float or tuple of floats specifying the threshold cutoff for 
                        classifying plasticity
            
            figsize - tuple specifying the size of the figures

            showmeans - boolean specifying whether to plot means on boxplots

            test_type - str specifying whether to perform parametric or nonparametric tests

            test_method - str specifying the typ of posthoc test to perform

            display_stats - boolean specifying whether to display stats

            save - boolean specifying whether to save the figures or not

            save_path - str specifying where to save the figures
            
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    plastic_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }
    if rwd_mvmt:
        mvmt_key = "Rwd movement"
        nonmvmt_key = "Nonrwd movement"
    else:
        mvmt_key = "Movement"
        nonmvmt_key = "Non-movement"

    # Pull relevant data
    sampling_rate = dataset.parameters["Sampling Rate"]
    activity_window = dataset.parameters["Activity Window"]
    if dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"

    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset is None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags

    # Amplitudes, fraction coactive, fraction particpating
    spine_coactive_amplitude = dataset.spine_coactive_amplitude
    mvmt_spine_coactive_amplitude = mvmt_dataset.spine_coactive_amplitude
    nonmvmt_spine_coactive_amplitude = nonmvmt_dataset.spine_coactive_amplitude
    spine_coactive_ca_amplitude = dataset.spine_coactive_calcium_amplitude
    mvmt_spine_coactive_ca_amplitude = mvmt_dataset.spine_coactive_calcium_amplitude
    nonmvmt_spine_coactive_ca_amplitude = (
        nonmvmt_dataset.spine_coactive_calcium_amplitude
    )
    fraction_spine_coactive = dataset.fraction_spine_coactive
    mvmt_fraction_spine_coactive = mvmt_dataset.fraction_spine_coactive
    nonmvmt_fraction_spine_coactive = nonmvmt_dataset.fraction_spine_coactive
    fraction_participating = dataset.fraction_coactivity_participation
    mvmt_fraction_participating = mvmt_dataset.fraction_coactivity_participation
    nonmvmt_fraction_participating = nonmvmt_dataset.fraction_coactivity_participation
    coactive_spine_num = dataset.coactive_spine_num
    mvmt_coactive_spine_num = mvmt_dataset.coactive_spine_num
    nonmvmt_coactive_spine_num = nonmvmt_dataset.coactive_spine_num
    # Traces
    spine_coactive_traces = dataset.spine_coactive_traces

    # spine_coactive_means = np.vstack(spine_coactive_means)
    mvmt_spine_coactive_traces = mvmt_dataset.spine_coactive_traces

    # mvmt_spine_coactive_means = np.vstack(mvmt_spine_coactive_means)
    nonmvmt_spine_coactive_traces = nonmvmt_dataset.spine_coactive_traces

    # nonmvmt_spine_coactive_means = np.vstack(nonmvmt_spine_coactive_means)
    # Calcium traces
    spine_coactive_ca_traces = dataset.spine_coactive_calcium_traces

    # spine_coactive_ca_means = np.vstack(spine_coactive_ca_means)
    mvmt_spine_coactive_ca_traces = mvmt_dataset.spine_coactive_calcium_traces

    # mvmt_spine_coactive_ca_means = np.vstack(mvmt_spine_coactive_ca_means)
    nonmvmt_spine_coactive_ca_traces = nonmvmt_dataset.spine_coactive_calcium_traces

    # nonmvmt_spine_coactive_ca_means = np.vstack(nonmvmt_spine_coactive_ca_means)

    # Calculate relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=vol_norm, exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=vol_norm,
    )

    # Subselect present spines
    spine_coactive_traces = d_utils.subselect_data_by_idxs(
        spine_coactive_traces, spine_idxs
    )
    spine_coactive_ca_traces = d_utils.subselect_data_by_idxs(
        spine_coactive_ca_traces, spine_idxs
    )
    spine_coactive_amplitude = d_utils.subselect_data_by_idxs(
        spine_coactive_amplitude, spine_idxs
    )
    spine_coactive_ca_amplitude = d_utils.subselect_data_by_idxs(
        spine_coactive_ca_amplitude, spine_idxs
    )
    fraction_spine_coactive = d_utils.subselect_data_by_idxs(
        fraction_spine_coactive, spine_idxs
    )
    fraction_participating = d_utils.subselect_data_by_idxs(
        fraction_participating, spine_idxs
    )
    coactive_spine_num = d_utils.subselect_data_by_idxs(coactive_spine_num, spine_idxs)

    mvmt_spine_coactive_traces = d_utils.subselect_data_by_idxs(
        mvmt_spine_coactive_traces, spine_idxs
    )
    mvmt_spine_coactive_ca_traces = d_utils.subselect_data_by_idxs(
        mvmt_spine_coactive_ca_traces, spine_idxs
    )
    mvmt_spine_coactive_amplitude = d_utils.subselect_data_by_idxs(
        mvmt_spine_coactive_amplitude, spine_idxs
    )
    mvmt_spine_coactive_ca_amplitude = d_utils.subselect_data_by_idxs(
        mvmt_spine_coactive_ca_amplitude, spine_idxs
    )
    mvmt_fraction_spine_coactive = d_utils.subselect_data_by_idxs(
        mvmt_fraction_spine_coactive, spine_idxs
    )
    mvmt_fraction_participating = d_utils.subselect_data_by_idxs(
        mvmt_fraction_participating, spine_idxs
    )
    mvmt_coactive_spine_num = d_utils.subselect_data_by_idxs(
        mvmt_coactive_spine_num, spine_idxs
    )

    nonmvmt_spine_coactive_traces = d_utils.subselect_data_by_idxs(
        nonmvmt_spine_coactive_traces, spine_idxs
    )
    nonmvmt_spine_coactive_ca_traces = d_utils.subselect_data_by_idxs(
        nonmvmt_spine_coactive_ca_traces, spine_idxs
    )
    nonmvmt_spine_coactive_amplitude = d_utils.subselect_data_by_idxs(
        nonmvmt_spine_coactive_amplitude, spine_idxs
    )
    nonmvmt_spine_coactive_ca_amplitude = d_utils.subselect_data_by_idxs(
        nonmvmt_spine_coactive_ca_amplitude, spine_idxs
    )
    nonmvmt_fraction_spine_coactive = d_utils.subselect_data_by_idxs(
        nonmvmt_fraction_spine_coactive, spine_idxs
    )
    nonmvmt_fraction_participating = d_utils.subselect_data_by_idxs(
        nonmvmt_fraction_participating, spine_idxs
    )
    nonmvmt_coactive_spine_num = d_utils.subselect_data_by_idxs(
        nonmvmt_coactive_spine_num, spine_idxs
    )

    # Seperate into dicts for plotting
    plastic_trace_means = {}
    plastic_trace_sems = {}
    plastic_ca_trace_means = {}
    plastic_ca_trace_sems = {}
    plastic_amps = {}
    plastic_ca_amps = {}
    plastic_frac_coactive = {}
    plastic_frac_participating = {}
    plastic_coactive_num = {}
    mvmt_plastic_trace_means = {}
    mvmt_plastic_trace_sems = {}
    mvmt_plastic_ca_trace_means = {}
    mvmt_plastic_ca_trace_sems = {}
    mvmt_plastic_amps = {}
    mvmt_plastic_ca_amps = {}
    mvmt_plastic_frac_coactive = {}
    mvmt_plastic_frac_participating = {}
    mvmt_plastic_coactive_num = {}
    nonmvmt_plastic_trace_means = {}
    nonmvmt_plastic_trace_sems = {}
    nonmvmt_plastic_ca_trace_means = {}
    nonmvmt_plastic_ca_trace_sems = {}
    nonmvmt_plastic_amps = {}
    nonmvmt_plastic_ca_amps = {}
    nonmvmt_plastic_frac_coactive = {}
    nonmvmt_plastic_frac_participating = {}
    nonmvmt_plastic_coactive_num = {}

    for key, value in plastic_groups.items():
        spines = eval(value)
        trace_means = compress(spine_coactive_traces, spines)
        trace_means = [
            np.nanmean(x, axis=1) for x in trace_means if type(x) == np.ndarray
        ]
        trace_means = np.vstack(trace_means)
        plastic_trace_means[key] = np.nanmean(trace_means, axis=0)
        plastic_trace_sems[key] = stats.sem(trace_means, axis=0, nan_policy="omit")
        ca_trace_means = compress(spine_coactive_ca_traces, spines)
        ca_trace_means = [
            np.nanmean(x, axis=1) for x in ca_trace_means if type(x) == np.ndarray
        ]
        ca_trace_means = np.vstack(ca_trace_means)
        plastic_ca_trace_means[key] = np.nanmean(ca_trace_means, axis=0)
        plastic_ca_trace_sems[key] = stats.sem(
            ca_trace_means, axis=0, nan_policy="omit"
        )
        plastic_amps[key] = spine_coactive_amplitude[spines]
        plastic_ca_amps[key] = spine_coactive_ca_amplitude[spines]
        plastic_frac_coactive[key] = fraction_spine_coactive[spines]
        plastic_frac_participating[key] = fraction_participating[spines]
        plastic_coactive_num[key] = coactive_spine_num[spines]

        mvmt_trace_means = compress(mvmt_spine_coactive_traces, spines)
        mvmt_trace_means = [
            np.nanmean(x, axis=1) for x in mvmt_trace_means if type(x) == np.ndarray
        ]
        mvmt_trace_means = np.vstack(mvmt_trace_means)
        mvmt_plastic_trace_means[key] = np.nanmean(mvmt_trace_means, axis=0)
        mvmt_plastic_trace_sems[key] = stats.sem(
            mvmt_trace_means, axis=0, nan_policy="omit"
        )
        mvmt_ca_trace_means = compress(mvmt_spine_coactive_ca_traces, spines)
        mvmt_ca_trace_means = [
            np.nanmean(x, axis=1) for x in mvmt_ca_trace_means if type(x) == np.ndarray
        ]
        mvmt_ca_trace_means = np.vstack(mvmt_ca_trace_means)
        mvmt_plastic_ca_trace_means[key] = np.nanmean(mvmt_ca_trace_means, axis=0)
        mvmt_plastic_ca_trace_sems[key] = stats.sem(
            mvmt_ca_trace_means, axis=0, nan_policy="omit"
        )
        mvmt_plastic_amps[key] = mvmt_spine_coactive_amplitude[spines]
        mvmt_plastic_ca_amps[key] = mvmt_spine_coactive_ca_amplitude[spines]
        mvmt_plastic_frac_coactive[key] = mvmt_fraction_spine_coactive[spines]
        mvmt_plastic_frac_participating[key] = mvmt_fraction_participating[spines]
        mvmt_plastic_coactive_num[key] = mvmt_coactive_spine_num[spines]

        nonmvmt_trace_means = compress(nonmvmt_spine_coactive_traces, spines)
        nonmvmt_trace_means = [
            np.nanmean(x, axis=1) for x in nonmvmt_trace_means if type(x) == np.ndarray
        ]
        nonmvmt_trace_means = np.vstack(nonmvmt_trace_means)
        nonmvmt_plastic_trace_means[key] = np.nanmean(nonmvmt_trace_means, axis=0)
        nonmvmt_plastic_trace_sems[key] = stats.sem(
            nonmvmt_trace_means, axis=0, nan_policy="omit"
        )
        nonmvmt_ca_trace_means = compress(nonmvmt_spine_coactive_ca_traces, spines)
        nonmvmt_ca_trace_means = [
            np.nanmean(x, axis=1)
            for x in nonmvmt_ca_trace_means
            if type(x) == np.ndarray
        ]
        nonmvmt_ca_trace_means = np.vstack(nonmvmt_ca_trace_means)
        nonmvmt_plastic_ca_trace_means[key] = np.nanmean(nonmvmt_ca_trace_means, axis=0)
        nonmvmt_plastic_ca_trace_sems[key] = stats.sem(
            nonmvmt_ca_trace_means, axis=0, nan_policy="omit"
        )
        nonmvmt_plastic_amps[key] = nonmvmt_spine_coactive_amplitude[spines]
        nonmvmt_plastic_ca_amps[key] = nonmvmt_spine_coactive_ca_amplitude[spines]
        nonmvmt_plastic_frac_coactive[key] = nonmvmt_fraction_spine_coactive[spines]
        nonmvmt_plastic_frac_participating[key] = nonmvmt_fraction_participating[spines]
        nonmvmt_plastic_coactive_num[key] = nonmvmt_coactive_spine_num[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFG.
        HIJK
        LMN.
        OPQR
        STU.
        """,
        figsize=figsize,
    )
    fig.suptitle("Coactive Event Properties")
    ############################ Plot data onto axes ################################
    # All period GluSnFr traces
    plot_mean_activity_traces(
        means=list(plastic_trace_means.values()),
        sems=list(plastic_trace_sems.values()),
        group_names=list(plastic_trace_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="All period GluSnFr",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # All period Calcium traces
    plot_mean_activity_traces(
        means=list(plastic_ca_trace_means.values()),
        sems=list(plastic_ca_trace_sems.values()),
        group_names=list(plastic_ca_trace_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="All period Calcium",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # mvmt period GluSnFr traces
    plot_mean_activity_traces(
        means=list(mvmt_plastic_trace_means.values()),
        sems=list(mvmt_plastic_trace_sems.values()),
        group_names=list(mvmt_plastic_trace_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title=f"{mvmt_key} GluSnFr",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    # mvmt period Calcium traces
    plot_mean_activity_traces(
        means=list(mvmt_plastic_ca_trace_means.values()),
        sems=list(mvmt_plastic_ca_trace_sems.values()),
        group_names=list(mvmt_plastic_ca_trace_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title=f"{mvmt_key} Calcium",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    # Nonmvmt period GluSnFr traces
    plot_mean_activity_traces(
        means=list(nonmvmt_plastic_trace_means.values()),
        sems=list(nonmvmt_plastic_trace_sems.values()),
        group_names=list(nonmvmt_plastic_trace_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title=f"{nonmvmt_key} GluSnFr",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    # Nonmvmt period Calcium traces
    plot_mean_activity_traces(
        means=list(nonmvmt_plastic_ca_trace_means.values()),
        sems=list(nonmvmt_plastic_ca_trace_sems.values()),
        group_names=list(nonmvmt_plastic_trace_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title=f"{nonmvmt_key} Calcium",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["Q"],
        save=False,
        save_path=None,
    )
    # All period GluSnFr amplitude
    plot_box_plot(
        plastic_amps,
        figsize=(5, 5),
        title="All period GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # All period Calcium amplitude
    plot_box_plot(
        plastic_ca_amps,
        figsize=(5, 5),
        title="All period Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Mvmt GluSnFr amplitude
    plot_box_plot(
        mvmt_plastic_amps,
        figsize=(5, 5),
        title=f"{mvmt_key} GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    # mvmt Calcium amplitude
    plot_box_plot(
        mvmt_plastic_ca_amps,
        figsize=(5, 5),
        title=f"{mvmt_key} Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Nonmvmt GluSnFr amplitude
    plot_box_plot(
        nonmvmt_plastic_amps,
        figsize=(5, 5),
        title=f"{nonmvmt_key} GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["P"],
        save=False,
        save_path=None,
    )
    # Nonmvmt Calcium amplitude
    plot_box_plot(
        nonmvmt_plastic_ca_amps,
        figsize=(5, 5),
        title=f"{nonmvmt_key} Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["R"],
        save=False,
        save_path=None,
    )
    # All period fraction coactive
    plot_box_plot(
        plastic_frac_coactive,
        figsize=(5, 5),
        title=f"All periods",
        xtitle=None,
        ytitle=f"Fraction Coactive",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Mvmt fraction coactive
    plot_box_plot(
        mvmt_plastic_frac_coactive,
        figsize=(5, 5),
        title=f"{mvmt_key}",
        xtitle=None,
        ytitle=f"Fraction Coactive",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Nonmvmt fraction coactive
    plot_box_plot(
        nonmvmt_plastic_frac_coactive,
        figsize=(5, 5),
        title=f"{nonmvmt_key}",
        xtitle=None,
        ytitle=f"Fraction Coactive",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["S"],
        save=False,
        save_path=None,
    )
    # All period fraction participating
    plot_box_plot(
        plastic_frac_participating,
        figsize=(5, 5),
        title=f"All periods",
        xtitle=None,
        ytitle=f"Fraction Participating",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    # Mvmt fraction participating
    plot_box_plot(
        mvmt_plastic_frac_participating,
        figsize=(5, 5),
        title=f"{mvmt_key}",
        xtitle=None,
        ytitle=f"Fraction Participating",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    # Nonmvmt fraction participating
    plot_box_plot(
        nonmvmt_plastic_frac_participating,
        figsize=(5, 5),
        title=f"{nonmvmt_key}",
        xtitle=None,
        ytitle=f"Fraction Participating",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["T"],
        save=False,
        save_path=None,
    )
    # All period coactive number
    plot_box_plot(
        plastic_coactive_num,
        figsize=(5, 5),
        title=f"All periods",
        xtitle=None,
        ytitle=f"Coactive spine number",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    # Mvmt coactive number
    plot_box_plot(
        mvmt_plastic_coactive_num,
        figsize=(5, 5),
        title=f"{mvmt_key}",
        xtitle=None,
        ytitle=f"Coactive spine number",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["N"],
        save=False,
        save_path=None,
    )
    # Nonmvmt coactive number
    plot_box_plot(
        nonmvmt_plastic_coactive_num,
        figsize=(5, 5),
        title=f"{nonmvmt_key}",
        xtitle=None,
        ytitle=f"Coactive spine number",
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["U"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if not rwd_mvmt:
            fname = os.path.join(save_path, "Local_Coactivity_Figure_8")
        else:
            fname = os.path.join(save_path, "Local_Coactivity_Figure_9")
        fig.savefig(fname + ".pdf")

    ####################### Statistics Section ###########################
    if display_stats == False:
        return

    # Perform the f-tests
    if test_type == "parametric":
        amp_f, amp_p, _, amp_df = t_utils.ANOVA_1way_posthoc(
            plastic_amps, method=test_method,
        )
        mvmt_amp_f, mvmt_amp_p, _, mvmt_amp_df = t_utils.ANOVA_1way_posthoc(
            mvmt_plastic_amps, test_method=test_method,
        )
        nonmvmt_amp_f, nonmvmt_amp_p, _, nonmvmt_amp_df = t_utils.ANOVA_1way_posthoc(
            nonmvmt_plastic_amps, test_method=test_method,
        )
        ca_amp_f, ca_amp_p, _, ca_amp_df = t_utils.ANOVA_1way_posthoc(
            plastic_ca_amps, method=test_method,
        )
        mvmt_ca_amp_f, mvmt_ca_amp_p, _, mvmt_ca_amp_df = t_utils.ANOVA_1way_posthoc(
            mvmt_plastic_ca_amps, test_method=test_method,
        )
        (
            nonmvmt_ca_amp_f,
            nonmvmt_ca_amp_p,
            _,
            nonmvmt_ca_amp_df,
        ) = t_utils.ANOVA_1way_posthoc(
            nonmvmt_plastic_ca_amps, test_method=test_method,
        )
        frac_co_f, frac_co_p, _, frac_co_df = t_utils.ANOVA_1way_posthoc(
            plastic_frac_coactive, test_method=test_method,
        )
        mvmt_frac_co_f, mvmt_frac_co_p, _, mvmt_frac_co_df = t_utils.ANOVA_1way_posthoc(
            mvmt_plastic_frac_coactive, test_method=test_method,
        )
        (
            nonmvmt_frac_co_f,
            nonmvmt_frac_co_p,
            _,
            nonmvmt_frac_co_df,
        ) = t_utils.ANOVA_1way_posthoc(
            nonmvmt_plastic_frac_coactive, test_method=test_method,
        )

        frac_pa_f, frac_pa_p, _, frac_pa_df = t_utils.ANOVA_1way_posthoc(
            plastic_frac_participating, test_method=test_method,
        )
        mvmt_frac_pa_f, mvmt_frac_pa_p, _, mvmt_frac_pa_df = t_utils.ANOVA_1way_posthoc(
            mvmt_plastic_frac_participating, test_method=test_method,
        )
        (
            nonmvmt_frac_pa_f,
            nonmvmt_frac_pa_p,
            _,
            nonmvmt_frac_pa_df,
        ) = t_utils.ANOVA_1way_posthoc(
            nonmvmt_plastic_frac_participating, test_method=test_method,
        )
        num_f, num_p, _, num_df = t_utils.ANOVA_1way_posthoc(
            plastic_coactive_num, test_method=test_method,
        )
        mvmt_num_f, mvmt_num_p, _, mvmt_num_df = t_utils.ANOVA_1way_posthoc(
            mvmt_plastic_coactive_num, test_method=test_method,
        )
        nonmvmt_num_f, nonmvmt_num_p, _, nonmvmt_num_df = t_utils.ANOVA_1way_posthoc(
            nonmvmt_plastic_coactive_num, test_method=test_method,
        )
        test_title = f"One-way ANOVA {test_method}"
    elif test_type == "nonparametric":
        amp_f, amp_p, amp_df = t_utils.kruskal_wallis_test(
            plastic_amps, "Conover", test_method,
        )
        mvmt_amp_f, mvmt_amp_p, mvmt_amp_df = t_utils.kruskal_wallis_test(
            mvmt_plastic_amps, "Conover", test_method,
        )
        nonmvmt_amp_f, nonmvmt_amp_p, nonmvmt_amp_df = t_utils.kruskal_wallis_test(
            nonmvmt_plastic_amps, "Conover", test_method,
        )
        ca_amp_f, ca_amp_p, ca_amp_df = t_utils.kruskal_wallis_test(
            plastic_ca_amps, "Conover", test_method,
        )
        mvmt_ca_amp_f, mvmt_ca_amp_p, mvmt_ca_amp_df = t_utils.kruskal_wallis_test(
            mvmt_plastic_ca_amps, "Conover", test_method,
        )
        (
            nonmvmt_ca_amp_f,
            nonmvmt_ca_amp_p,
            nonmvmt_ca_amp_df,
        ) = t_utils.kruskal_wallis_test(
            nonmvmt_plastic_ca_amps, "Conover", test_method,
        )
        frac_co_f, frac_co_p, frac_co_df = t_utils.kruskal_wallis_test(
            plastic_frac_coactive, "Conover", test_method,
        )
        mvmt_frac_co_f, mvmt_frac_co_p, mvmt_frac_co_df = t_utils.kruskal_wallis_test(
            mvmt_plastic_frac_coactive, "Conover", test_method,
        )
        (
            nonmvmt_frac_co_f,
            nonmvmt_frac_co_p,
            nonmvmt_frac_co_df,
        ) = t_utils.kruskal_wallis_test(
            nonmvmt_plastic_frac_coactive, "Conover", test_method,
        )

        frac_pa_f, frac_pa_p, frac_pa_df = t_utils.kruskal_wallis_test(
            plastic_frac_participating, "Conover", test_method,
        )
        mvmt_frac_pa_f, mvmt_frac_pa_p, mvmt_frac_pa_df = t_utils.kruskal_wallis_test(
            mvmt_plastic_frac_participating, "Conover", test_method,
        )
        (
            nonmvmt_frac_pa_f,
            nonmvmt_frac_pa_p,
            nonmvmt_frac_pa_df,
        ) = t_utils.kruskal_wallis_test(
            nonmvmt_plastic_frac_participating, "Conover", test_method,
        )
        num_f, num_p, num_df = t_utils.kruskal_wallis_test(
            plastic_coactive_num, "Conover", test_method,
        )
        mvmt_num_f, mvmt_num_p, mvmt_num_df = t_utils.kruskal_wallis_test(
            mvmt_plastic_coactive_num, "Conover", test_method,
        )
        nonmvmt_num_f, nonmvmt_num_p, nonmvmt_num_df = t_utils.kruskal_wallis_test(
            nonmvmt_plastic_coactive_num, "Conover", test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        IJ
        KL
        MN
        O.
        """,
        figsize=(10, 15),
    )
    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(f"All Amplitude\n{test_title}\nF = {amp_f:.4} p = {amp_p:.3E}")
    A_table = axes2["A"].table(
        cellText=amp_df.values,
        colLabels=amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(
        f"All Calcium Amplitude\n{test_title}\nF = {ca_amp_f:.4} p = {ca_amp_p:.3E}"
    )
    B_table = axes2["B"].table(
        cellText=ca_amp_df.values,
        colLabels=ca_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(
        f"All Fraction Coactive\n{test_title}\nF = {frac_co_f:.4} p = {frac_co_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=frac_co_df.values,
        colLabels=frac_co_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(
        f"All Fraction Participating\n{test_title}\nF = {frac_pa_f:.4} p = {frac_pa_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=frac_pa_df.values,
        colLabels=frac_pa_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(
        f"All Coactive Num\n{test_title}\nF = {num_f:.4} p = {num_p:.3E}"
    )
    E_table = axes2["E"].table(
        cellText=num_df.values,
        colLabels=num_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title(
        f"{mvmt_key} Amplitude\n{test_title}\nF = {mvmt_amp_f:.4} p = {mvmt_amp_p:.3E}"
    )
    F_table = axes2["F"].table(
        cellText=mvmt_amp_df.values,
        colLabels=mvmt_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)
    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title(
        f"{mvmt_key} Calcium Amplitude\n{test_title}\nF = {mvmt_ca_amp_f:.4} p = {mvmt_ca_amp_p:.3E}"
    )
    G_table = axes2["G"].table(
        cellText=mvmt_ca_amp_df.values,
        colLabels=mvmt_ca_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title(
        f"{mvmt_key} Fraction Coactive\n{test_title}\nF = {mvmt_frac_co_f:.4} p = {mvmt_frac_co_p:.3E}"
    )
    H_table = axes2["H"].table(
        cellText=mvmt_frac_co_df.values,
        colLabels=mvmt_frac_co_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)
    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title(
        f"{mvmt_key} Fraction Participating\n{test_title}\nF = {mvmt_frac_pa_f:.4} p = {mvmt_frac_pa_p:.3E}"
    )
    I_table = axes2["I"].table(
        cellText=mvmt_frac_pa_df.values,
        colLabels=mvmt_frac_pa_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)
    axes2["J"].axis("off")
    axes2["J"].axis("tight")
    axes2["J"].set_title(
        f"{mvmt_key} Coactive Num\n{test_title}\nF = {mvmt_num_f:.4} p = {mvmt_num_p:.3E}"
    )
    J_table = axes2["J"].table(
        cellText=mvmt_num_df.values,
        colLabels=mvmt_num_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    J_table.auto_set_font_size(False)
    J_table.set_fontsize(8)
    axes2["K"].axis("off")
    axes2["K"].axis("tight")
    axes2["K"].set_title(
        f"{nonmvmt_key} Amplitude\n{test_title}\nF = {nonmvmt_amp_f:.4} p = {nonmvmt_amp_p:.3E}"
    )
    K_table = axes2["K"].table(
        cellText=nonmvmt_amp_df.values,
        colLabels=nonmvmt_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    K_table.auto_set_font_size(False)
    K_table.set_fontsize(8)
    axes2["L"].axis("off")
    axes2["L"].axis("tight")
    axes2["L"].set_title(
        f"{nonmvmt_key} Calcium Amplitude\n{test_title}\nF = {nonmvmt_ca_amp_f:.4} p = {nonmvmt_ca_amp_p:.3E}"
    )
    L_table = axes2["L"].table(
        cellText=nonmvmt_ca_amp_df.values,
        colLabels=nonmvmt_ca_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    L_table.auto_set_font_size(False)
    L_table.set_fontsize(8)
    axes2["M"].axis("off")
    axes2["M"].axis("tight")
    axes2["M"].set_title(
        f"{nonmvmt_key} Fraction Coactive\n{test_title}\nF = {nonmvmt_frac_co_f:.4} p = {nonmvmt_frac_co_p:.3E}"
    )
    M_table = axes2["M"].table(
        cellText=nonmvmt_frac_co_df.values,
        colLabels=nonmvmt_frac_co_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    M_table.auto_set_font_size(False)
    M_table.set_fontsize(8)
    axes2["N"].axis("off")
    axes2["N"].axis("tight")
    axes2["N"].set_title(
        f"{nonmvmt_key} Fraction Participating\n{test_title}\nF = {nonmvmt_frac_pa_f:.4} p = {nonmvmt_frac_pa_p:.3E}"
    )
    N_table = axes2["N"].table(
        cellText=nonmvmt_frac_pa_df.values,
        colLabels=nonmvmt_frac_pa_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    N_table.auto_set_font_size(False)
    N_table.set_fontsize(8)
    axes2["O"].axis("off")
    axes2["O"].axis("tight")
    axes2["O"].set_title(
        f"{nonmvmt_key} Coactive Num\n{test_title}\nF = {nonmvmt_num_f:.4} p = {nonmvmt_num_p:.3E}"
    )
    O_table = axes2["O"].table(
        cellText=nonmvmt_num_df.values,
        colLabels=nonmvmt_num_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    O_table.auto_set_font_size(False)
    O_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if not rwd_mvmt:
            fname = os.path.join(save_path, "Local_Coactivity_Figure_8_Stats")
        else:
            fname = os.path.join(save_path, "Local_Coactivity_Figure_9_Stats")
        fig2.savefig(fname + ".pdf")


def plot_nearby_spine_properties(
    dataset,
    followup_dataset=None,
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
    COLORS = ["darkorange", "darkviolet", "silver"]
    plastic_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
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
    local_coactivity_rate_distribution = dataset.local_coactivity_rate_distribution
    avg_nearby_coactivity_rate = dataset.avg_nearby_coactivity_rate
    shuff_nearby_coactivity_rate = dataset.shuff_nearby_coactivity_rate
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

    # Calculate spine volume
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=vol_norm, exclude=exclude
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=vol_norm,
    )

    # Subselect for stable spines
    spine_activity_rate_distribution = d_utils.subselect_data_by_idxs(
        spine_activity_rate_distribution, spine_idxs
    )
    avg_nearby_spine_rate = d_utils.subselect_data_by_idxs(
        avg_nearby_spine_rate, spine_idxs,
    )
    shuff_nearby_spine_rate = d_utils.subselect_data_by_idxs(
        shuff_nearby_spine_rate, spine_idxs
    )
    local_coactivity_rate_distribution = d_utils.subselect_data_by_idxs(
        local_coactivity_rate_distribution, spine_idxs,
    )
    avg_nearby_coactivity_rate = d_utils.subselect_data_by_idxs(
        avg_nearby_coactivity_rate, spine_idxs,
    )
    shuff_nearby_coactivity_rate = d_utils.subselect_data_by_idxs(
        shuff_nearby_coactivity_rate, spine_idxs
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
    local_nn_enlarged = d_utils.subselect_data_by_idxs(local_nn_enlarged, spine_idxs,)
    shuff_nn_enlarged = d_utils.subselect_data_by_idxs(shuff_nn_enlarged, spine_idxs,)
    local_nn_shrunken = d_utils.subselect_data_by_idxs(local_nn_shrunken, spine_idxs,)
    shuff_nn_shrunken = d_utils.subselect_data_by_idxs(shuff_nn_shrunken, spine_idxs,)

    # Seperate into groups
    plastic_rate_dist = {}
    plastic_avg_rates = {}
    plastic_shuff_rates = {}
    plastic_shuff_rate_medians = {}
    plastic_coactivity_dist = {}
    plastic_avg_coactivity = {}
    plastic_shuff_coactivity = {}
    plastic_shuff_coactivity_medians = {}
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
    distance_bins = dataset.parameters["position bins"][1:]

    for key, value in plastic_groups.items():
        spines = eval(value)
        plastic_rate_dist[key] = spine_activity_rate_distribution[:, spines]
        plastic_avg_rates[key] = avg_nearby_spine_rate[spines]
        shuff_rates = shuff_nearby_spine_rate[:, spines]
        plastic_shuff_rates[key] = shuff_rates
        plastic_shuff_rate_medians[key] = np.nanmedian(shuff_rates, axis=1)
        plastic_coactivity_dist[key] = local_coactivity_rate_distribution[:, spines]
        plastic_avg_coactivity[key] = avg_nearby_coactivity_rate[spines]
        shuff_coactivity = shuff_nearby_coactivity_rate[:, spines]
        plastic_shuff_coactivity[key] = shuff_coactivity
        plastic_shuff_coactivity_medians[key] = np.nanmedian(shuff_coactivity, axis=1)
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

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE
        FGHIJ
        KLMNO
        PQRST
        UVWX.
        YZab.
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
    # Avg local spine activity rate
    plot_box_plot(
        plastic_avg_rates,
        figsize=(5, 5),
        title="Nearby Activity Rate",
        xtitle=None,
        ytitle="Activity rate (events/min)",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    plot_histogram(
        data=list(
            (
                plastic_avg_rates["Enlarged"],
                plastic_shuff_rates["Enlarged"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Enlarged",
        xtitle="Nearby activity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_c_inset = axes["C"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_c_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_rates["Enlarged"],
            "shuff": plastic_shuff_rate_medians["Enlarged"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
    plot_histogram(
        data=list(
            (
                plastic_avg_rates["Shrunken"],
                plastic_shuff_rates["Shrunken"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Shrunken",
        xtitle="Nearby activity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_d_inset = axes["D"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_d_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_rates["Shrunken"],
            "shuff": plastic_shuff_rate_medians["Shrunken"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
    plot_histogram(
        data=list(
            (
                plastic_avg_rates["Stable"],
                plastic_shuff_rates["Stable"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Stable",
        xtitle="Nearby activity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_e_inset = axes["E"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_e_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_rates["Stable"],
            "shuff": plastic_shuff_rate_medians["Stable"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    plot_histogram(
        data=list(
            (
                plastic_avg_coactivity["Enlarged"],
                plastic_shuff_coactivity["Enlarged"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Enlarged",
        xtitle="Avg local coactivity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_h_inset = axes["H"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_h_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_coactivity["Enlarged"],
            "shuff": plastic_shuff_coactivity_medians["Enlarged"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
    plot_histogram(
        data=list(
            (
                plastic_avg_coactivity["Shrunken"],
                plastic_shuff_coactivity["Shrunken"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Shrunken",
        xtitle="Avg local coactivity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_I_inset = axes["I"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_I_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_coactivity["Shrunken"],
            "shuff": plastic_shuff_coactivity_medians["Shrunken"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
    plot_histogram(
        data=list(
            (
                plastic_avg_coactivity["Stable"],
                plastic_shuff_coactivity["Stable"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Stable",
        xtitle="Avg local coactivity rate",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_j_inset = axes["J"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_j_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_coactivity["Stable"],
            "shuff": plastic_shuff_coactivity_medians["Stable"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    plot_histogram(
        data=list(
            (
                plastic_MRS_density["Enlarged"],
                plastic_MRS_shuff_density["Enlarged"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Enlarged",
        xtitle="MRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_m_inset = axes["M"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_m_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_MRS_density["Enlarged"],
            "shuff": plastic_MRS_shuff_density_medians["Enlarged"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
    plot_histogram(
        data=list(
            (
                plastic_MRS_density["Shrunken"],
                plastic_MRS_shuff_density["Shrunken"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Shrunken",
        xtitle="MRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["N"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_n_inset = axes["N"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_n_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_MRS_density["Shrunken"],
            "shuff": plastic_MRS_shuff_density_medians["Shrunken"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
    plot_histogram(
        data=list(
            (
                plastic_MRS_density["Stable"],
                plastic_MRS_shuff_density["Stable"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Stable",
        xtitle="MRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_o_inset = axes["O"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_o_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_MRS_density["Stable"],
            "shuff": plastic_MRS_shuff_density_medians["Stable"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    plot_histogram(
        data=list(
            (
                plastic_rMRS_density["Enlarged"],
                plastic_rMRS_shuff_density["Enlarged"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Enlarged",
        xtitle="rMRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["R"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_r_inset = axes["R"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_r_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_rMRS_density["Enlarged"],
            "shuff": plastic_rMRS_shuff_density_medians["Enlarged"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
    plot_histogram(
        data=list(
            (
                plastic_rMRS_density["Shrunken"],
                plastic_rMRS_shuff_density["Shrunken"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Shrunken",
        xtitle="rMRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["S"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_s_inset = axes["S"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_s_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_rMRS_density["Shrunken"],
            "shuff": plastic_rMRS_shuff_density_medians["Shrunken"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
    plot_histogram(
        data=list(
            (
                plastic_rMRS_density["Stable"],
                plastic_rMRS_shuff_density["Stable"].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Stable",
        xtitle="rMRS density (spines/\u03BCm)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "grey"],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["T"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_t_inset = axes["T"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_t_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_rMRS_density["Stable"],
            "shuff": plastic_rMRS_shuff_density_medians["Stable"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
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
    # Enlarged nearest neigbhr
    plot_box_plot(
        plastic_nn_enlarged,
        figsize=(5, 5),
        title="Enlarged",
        xtitle=None,
        ytitle="Enlarged nearest neighbor (\u03BCm)",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["U"],
        save=False,
        save_path=None,
    )
    # Enlarged Enlarged nearest neighbor vs chance
    plot_histogram(
        data=list(
            (
                plastic_nn_enlarged["Enlarged"],
                plastic_shuff_enlarged["Enlarged"].flatten().astype(np.float32),
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
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["V"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_v_inset = axes["V"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_v_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_enlarged["Enlarged"],
            "shuff": plastic_shuff_enlarged_medians["Enlarged"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_v_inset,
        save=False,
        save_path=None,
    )
    # Shrunken Enlarged nearest neighbor vs chance
    plot_histogram(
        data=list(
            (
                plastic_nn_enlarged["Shrunken"],
                plastic_shuff_enlarged["Shrunken"].flatten().astype(np.float32),
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
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["W"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_w_inset = axes["W"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_w_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_enlarged["Shrunken"],
            "shuff": plastic_shuff_enlarged_medians["Shrunken"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_w_inset,
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
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["X"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_x_inset = axes["X"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_x_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_enlarged["Stable"],
            "shuff": plastic_shuff_enlarged_medians["Stable"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_x_inset,
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
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["Y"],
        save=False,
        save_path=None,
    )
    # Enlarged Shrunken nearest neighbor vs chance
    plot_histogram(
        data=list(
            (
                plastic_nn_shrunken["Enlarged"],
                plastic_shuff_shrunken["Enlarged"].flatten().astype(np.float32),
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
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["Z"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_z_inset = axes["Z"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_z_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_shrunken["Enlarged"],
            "shuff": plastic_shuff_shrunken_medians["Enlarged"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
        s_colors=[COLORS[0], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_z_inset,
        save=False,
        save_path=None,
    )
    # Shrunken Shrunken nearest neighbor vs chance
    plot_histogram(
        data=list(
            (
                plastic_nn_shrunken["Shrunken"],
                plastic_shuff_shrunken["Shrunken"].flatten().astype(np.float32),
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
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["a"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_a_inset = axes["a"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_a_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_shrunken["Shrunken"],
            "shuff": plastic_shuff_shrunken_medians["Shrunken"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_a_inset,
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
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["b"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_b_inset = axes["b"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_b_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_nn_shrunken["Stable"],
            "shuff": plastic_shuff_shrunken_medians["Stable"],
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
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.7,
        s_colors=[COLORS[2], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_b_inset,
        save=False,
        save_path=None,
    )

    fig.tight_layout()
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Local_Coactivity_Figure_10")
        fig.savefig(fname + ".pdf")

    ###################### Statistics Section ##########################
    if display_stats == False:
        return

    # Perform the F-tests
    if test_type == "parametric":
        activity_f, activity_p, _, activity_df = t_utils.ANOVA_1way_posthoc(
            plastic_avg_rates, method=test_method,
        )
        coactivity_f, coactivity_p, _, coactivity_df = t_utils.ANOVA_1way_posthoc(
            plastic_avg_coactivity, method=test_method,
        )
        MRS_f, MRS_p, _, MRS_df = t_utils.ANOVA_1way_posthoc(
            plastic_MRS_density, method=test_method,
        )
        rMRS_f, rMRS_p, _, rMRS_df = t_utils.ANOVA_1way_posthoc(
            plastic_rMRS_density, method=test_method,
        )
        enlarged_f, enlarged_p, _, enlarged_df = t_utils.ANOVA_1way_posthoc(
            plastic_nn_enlarged, method=test_method,
        )
        shrunken_f, shrunken_p, _, shrunken_df = t_utils.ANOVA_1way_posthoc(
            plastic_nn_shrunken, method=test_method,
        )
        test_title = f"One-way ANOVA {test_method}"
    elif test_type == "nonparametric":
        activity_f, activity_p, activity_df = t_utils.kruskal_wallis_test(
            plastic_avg_rates, "Conover", test_method,
        )
        coactivity_f, coactivity_p, coactivity_df = t_utils.kruskal_wallis_test(
            plastic_avg_coactivity, "Conover", test_method,
        )
        MRS_f, MRS_p, MRS_df = t_utils.kruskal_wallis_test(
            plastic_MRS_density, "Conover", test_method,
        )
        rMRS_f, rMRS_p, rMRS_df = t_utils.kruskal_wallis_test(
            plastic_rMRS_density, "Conover", test_method,
        )
        enlarged_f, enlarged_p, enlarged_df = t_utils.kruskal_wallis_test(
            plastic_nn_enlarged, "Conover", test_method,
        )
        shrunken_f, shrunken_p, shrunken_df = t_utils.kruskal_wallis_test(
            plastic_nn_shrunken, "Conover", test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"

    # Perform correlations
    _, activity_corr_df = t_utils.correlate_grouped_data(
        plastic_rate_dist, distance_bins,
    )
    _, coactivity_corr_df = t_utils.correlate_grouped_data(
        plastic_coactivity_dist, distance_bins,
    )
    _, MRS_corr_df = t_utils.correlate_grouped_data(plastic_MRS_dist, distance_bins,)
    _, rMRS_corr_df = t_utils.correlate_grouped_data(plastic_rMRS_dist, distance_bins,)

    # Comparisions to chance
    ## Activity rate
    e_rate_above, e_rate_below = t_utils.test_against_chance(
        plastic_avg_rates["Enlarged"], plastic_shuff_rates["Enlarged"]
    )
    s_rate_above, s_rate_below = t_utils.test_against_chance(
        plastic_avg_rates["Shrunken"], plastic_shuff_rates["Shrunken"]
    )
    st_rate_above, st_rate_below = t_utils.test_against_chance(
        plastic_avg_rates["Stable"], plastic_shuff_rates["Stable"]
    )
    chance_activity = {
        "Comparison": ["Enlarged", "Shrunken", "Stable"],
        "p-val above": [e_rate_above, s_rate_above, st_rate_above],
        "p-val below": [e_rate_below, s_rate_below, st_rate_below],
    }
    chance_activity_df = pd.DataFrame.from_dict(chance_activity)
    ## Coactivity rate
    e_coactivity_above, e_coactivity_below = t_utils.test_against_chance(
        plastic_avg_coactivity["Enlarged"], plastic_shuff_coactivity["Enlarged"]
    )
    s_coactivity_above, s_coactivity_below = t_utils.test_against_chance(
        plastic_avg_coactivity["Shrunken"], plastic_shuff_coactivity["Shrunken"]
    )
    st_coactivity_above, st_coactivity_below = t_utils.test_against_chance(
        plastic_avg_coactivity["Stable"], plastic_shuff_coactivity["Stable"]
    )
    chance_coactivity = {
        "Comparison": ["Enlarged", "Shrunken", "Stable"],
        "p-val above": [e_coactivity_above, s_coactivity_above, st_coactivity_above],
        "p-val below": [e_coactivity_below, s_coactivity_below, st_coactivity_below],
    }
    chance_coactivity_df = pd.DataFrame.from_dict(chance_coactivity)
    ## MRS density
    e_MRS_above, e_MRS_below = t_utils.test_against_chance(
        plastic_MRS_density["Enlarged"], plastic_MRS_shuff_density["Enlarged"]
    )
    s_MRS_above, s_MRS_below = t_utils.test_against_chance(
        plastic_MRS_density["Shrunken"], plastic_MRS_shuff_density["Shrunken"]
    )
    st_MRS_above, st_MRS_below = t_utils.test_against_chance(
        plastic_MRS_density["Stable"], plastic_MRS_shuff_density["Stable"]
    )
    chance_MRS = {
        "Comparison": ["Enlarged", "Shrunken", "Stable"],
        "p-val above": [e_MRS_above, s_MRS_above, st_MRS_above],
        "p-val below": [e_MRS_below, s_MRS_below, st_MRS_below],
    }
    chance_MRS_df = pd.DataFrame.from_dict(chance_MRS)
    ## rMRS density
    e_rMRS_above, e_rMRS_below = t_utils.test_against_chance(
        plastic_rMRS_density["Enlarged"], plastic_rMRS_shuff_density["Enlarged"]
    )
    s_rMRS_above, s_rMRS_below = t_utils.test_against_chance(
        plastic_rMRS_density["Shrunken"], plastic_rMRS_shuff_density["Shrunken"]
    )
    st_rMRS_above, st_rMRS_below = t_utils.test_against_chance(
        plastic_rMRS_density["Stable"], plastic_rMRS_shuff_density["Stable"]
    )
    chance_rMRS = {
        "Comparison": ["Enlarged", "Shrunken", "Stable"],
        "p-val above": [e_rMRS_above, s_rMRS_above, st_rMRS_above],
        "p-val below": [e_rMRS_below, s_rMRS_below, st_rMRS_below],
    }
    chance_rMRS_df = pd.DataFrame.from_dict(chance_rMRS)
    ## Enlarged NN
    e_enlarged_above, e_enlarged_below = t_utils.test_against_chance(
        plastic_nn_enlarged["Enlarged"], plastic_shuff_enlarged["Enlarged"]
    )
    s_enlarged_above, s_enlarged_below = t_utils.test_against_chance(
        plastic_nn_enlarged["Shrunken"], plastic_shuff_enlarged["Shrunken"]
    )
    st_enlarged_above, st_enlarged_below = t_utils.test_against_chance(
        plastic_nn_enlarged["Stable"], plastic_shuff_enlarged["Stable"]
    )
    chance_enlarged = {
        "Comparison": ["Enlarged", "Shrunken", "Stable"],
        "p-val above": [e_enlarged_above, s_enlarged_above, st_enlarged_above],
        "p-val below": [e_enlarged_below, s_enlarged_below, st_enlarged_below],
    }
    chance_enlarged_df = pd.DataFrame.from_dict(chance_enlarged)
    ## Shrunken NN
    e_shrunken_above, e_shrunken_below = t_utils.test_against_chance(
        plastic_nn_shrunken["Enlarged"], plastic_shuff_shrunken["Enlarged"]
    )
    s_shrunken_above, s_shrunken_below = t_utils.test_against_chance(
        plastic_nn_shrunken["Shrunken"], plastic_shuff_shrunken["Shrunken"]
    )
    st_shrunken_above, st_shrunken_below = t_utils.test_against_chance(
        plastic_nn_shrunken["Stable"], plastic_shuff_shrunken["Stable"]
    )
    chance_shrunken = {
        "Comparison": ["Enlarged", "Shrunken", "Stable"],
        "p-val above": [e_shrunken_above, s_shrunken_above, st_shrunken_above],
        "p-val below": [e_shrunken_below, s_shrunken_below, st_shrunken_below],
    }
    chance_shrunken_df = pd.DataFrame.from_dict(chance_shrunken)

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        IJ
        KL
        MN
        OP
        """,
        figsize=(10, 15),
    )
    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title("Average Activity Rate")
    A_table = axes2["A"].table(
        cellText=activity_corr_df.values,
        colLabels=activity_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title("Average Local Coactivity Rate")
    B_table = axes2["B"].table(
        cellText=coactivity_corr_df.values,
        colLabels=coactivity_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title("MRS density")
    C_table = axes2["C"].table(
        cellText=MRS_corr_df.values,
        colLabels=MRS_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title("rMRS density")
    D_table = axes2["D"].table(
        cellText=rMRS_corr_df.values,
        colLabels=rMRS_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(
        f"Local Activity rate\n{test_title}\nF = {activity_f:.4} p ={activity_p:.3E}"
    )
    E_table = axes2["E"].table(
        cellText=activity_df.values,
        colLabels=activity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title(
        f"Local Coactivity rate\n{test_title}\nF = {coactivity_f:.4} p ={coactivity_p:.3E}"
    )
    F_table = axes2["F"].table(
        cellText=coactivity_df.values,
        colLabels=coactivity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)
    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title(
        f"Local MRS density\n{test_title}\nF = {MRS_f:.4} p ={MRS_p:.3E}"
    )
    G_table = axes2["G"].table(
        cellText=MRS_df.values,
        colLabels=MRS_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title(
        f"Local rMRS density\n{test_title}\nF = {rMRS_f:.4} p ={rMRS_p:.3E}"
    )
    H_table = axes2["H"].table(
        cellText=rMRS_df.values,
        colLabels=rMRS_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)
    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title(
        f"Enlarged Nearest Neighbor\n{test_title}\nF = {enlarged_f:.4} p ={enlarged_p:.3E}"
    )
    I_table = axes2["I"].table(
        cellText=enlarged_df.values,
        colLabels=enlarged_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)
    axes2["J"].axis("off")
    axes2["J"].axis("tight")
    axes2["J"].set_title(
        f"Shrunken Nearest Neighbor\n{test_title}\nF = {shrunken_f:.4} p ={shrunken_p:.3E}"
    )
    J_table = axes2["J"].table(
        cellText=shrunken_df.values,
        colLabels=shrunken_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    J_table.auto_set_font_size(False)
    J_table.set_fontsize(8)
    axes2["K"].axis("off")
    axes2["K"].axis("tight")
    axes2["K"].set_title(f"Chance activity")
    K_table = axes2["K"].table(
        cellText=chance_activity_df.values,
        colLabels=chance_activity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    K_table.auto_set_font_size(False)
    K_table.set_fontsize(8)
    axes2["L"].axis("off")
    axes2["L"].axis("tight")
    axes2["L"].set_title(f"Chance Coactivity")
    L_table = axes2["L"].table(
        cellText=chance_coactivity_df.values,
        colLabels=chance_coactivity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    L_table.auto_set_font_size(False)
    L_table.set_fontsize(8)
    axes2["M"].axis("off")
    axes2["M"].axis("tight")
    axes2["M"].set_title(f"Chance MRS density")
    M_table = axes2["M"].table(
        cellText=chance_MRS_df.values,
        colLabels=chance_MRS_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    M_table.auto_set_font_size(False)
    M_table.set_fontsize(8)
    axes2["N"].axis("off")
    axes2["N"].axis("tight")
    axes2["N"].set_title(f"Chance rMRS density")
    N_table = axes2["N"].table(
        cellText=chance_rMRS_df.values,
        colLabels=chance_rMRS_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    N_table.auto_set_font_size(False)
    N_table.set_fontsize(8)
    axes2["O"].axis("off")
    axes2["O"].axis("tight")
    axes2["O"].set_title(f"Chance Enlarged NN")
    O_table = axes2["O"].table(
        cellText=chance_enlarged_df.values,
        colLabels=chance_enlarged_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    O_table.auto_set_font_size(False)
    O_table.set_fontsize(8)
    axes2["P"].axis("off")
    axes2["P"].axis("tight")
    axes2["P"].set_title(f"Chance Shrunken NN")
    P_table = axes2["P"].table(
        cellText=chance_shrunken_df.values,
        colLabels=chance_shrunken_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    P_table.auto_set_font_size(False)
    P_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Local_Coactivity_Figure_10_Stats")
        fig2.savefit(fname, ".pdf")


def plot_nearby_spine_coactivity(
    dataset,
    followup_dataset=None,
    mvmt_type="All periods",
    exclude="Shaft",
    threshold=0.3,
    figsize=(10, 6),
    hist_bins=30,
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to plot and compare the activity of nearby spines during coactivity
        events
        
        INPUT PARAMETERS
            dataset - Local_Coactivity_Data object analyzed over all periods

            followup_dataset - optional Local_Coactivity_Data object of the 
                                subsequent session to use for volume comparision.
                                Default is None to use the followup_volumes in the
                                dataset
                        
            mvmt_type - str specifying what mvmt period the data is constrained to.
                        Accepts "All periods", "movement", "nonmovement", "rewarded movement"

            exclude - str specifying the types of spines to exclude from volume assessment

            threshold - float or tuple of floats specifying the threshold cutoff for 
                        classifying plasticity

            figsize - tuple specifying the size of the figure

            showmeans - boolean specifying whether to show means on box plots or not

            mean_type - str specifying thge mean type for bar plots

            err_type - str specifying the err type for bar plots

            test_type - str specifying whether to perform parametric or nonparametric tests

            test_method - str specifying the type of posthoc test to perform

            display_stats - boolean specifying whetehr to display stats

            save - boolean specifying whether to save the figures or not

            save_path - str specifying where to save the figures
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    plastic_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }

    # PUll relevant data
    sampling_rate = dataset.parameters["Sampling Rate"]
    activity_window = dataset.parameters["Activity Window"]
    if dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"

    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset is None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags

    # Amplitude and onset-related variables
    nearby_coactive_amplitude = dataset.nearby_coactive_amplitude
    nearby_coactive_calcium_amplitude = dataset.nearby_coactive_calcium_amplitude
    nearby_spine_onset = dataset.nearby_spine_onset
    nearby_spine_onset_jitter = dataset.nearby_spine_onset_jitter
    # Traces
    nearby_coactive_traces = dataset.nearby_coactive_traces

    nearby_coactive_calcium_traces = dataset.nearby_coactive_calcium_traces

    # Calculate relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=False, exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=False,
    )

    # Subselect present spines
    nearby_coactive_amplitude = d_utils.subselect_data_by_idxs(
        nearby_coactive_amplitude, spine_idxs
    )
    nearby_coactive_calcium_amplitude = d_utils.subselect_data_by_idxs(
        nearby_coactive_calcium_amplitude, spine_idxs
    )
    nearby_spine_onset = d_utils.subselect_data_by_idxs(nearby_spine_onset, spine_idxs)
    nearby_spine_onset_jitter = d_utils.subselect_data_by_idxs(
        nearby_spine_onset_jitter, spine_idxs
    )
    nearby_coactive_means = d_utils.subselect_data_by_idxs(
        nearby_coactive_traces, spine_idxs
    )
    nearby_coactive_calcium_means = d_utils.subselect_data_by_idxs(
        nearby_coactive_calcium_traces, spine_idxs
    )

    # Seperate into dicts for plotting
    hmap_traces = {}
    plastic_trace_means = {}
    plastic_trace_sems = {}
    plastic_ca_trace_means = {}
    plastic_ca_trace_sems = {}
    plastic_amps = {}
    plastic_ca_amps = {}
    plastic_onsets = {}
    plastic_onset_jitter = {}

    for key, value in plastic_groups.items():
        spines = eval(value)
        traces = compress(nearby_coactive_means, spines)
        traces = [np.nanmean(x, axis=1) for x in traces if type(x) == np.ndarray]
        traces = np.vstack(traces)
        hmap_traces[key] = traces.T
        plastic_trace_means[key] = np.nanmean(traces, axis=0)
        plastic_trace_sems[key] = stats.sem(traces, axis=0, nan_policy="omit")
        ca_traces = compress(nearby_coactive_calcium_means, spines)
        ca_traces = [np.nanmean(x, axis=1) for x in ca_traces if type(x) == np.ndarray]
        ca_traces = np.vstack(ca_traces)
        plastic_ca_trace_means[key] = np.nanmean(ca_traces, axis=0)
        plastic_ca_trace_sems[key] = stats.sem(ca_traces, axis=0, nan_policy="omit")
        plastic_amps[key] = nearby_coactive_amplitude[spines]
        plastic_ca_amps[key] = nearby_coactive_calcium_amplitude[spines]
        plastic_onsets[key] = nearby_spine_onset[spines]
        plastic_onset_jitter[key] = nearby_spine_onset_jitter[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABDCE
        FGHIJ
        """,
        figsize=figsize,
    )
    fig.suptitle(f"{mvmt_type} Nearby Spine Coactivity Properties")

    ####################### Plot data onto axes ############################
    # Enlarged heatmap
    plot_activity_heatmap(
        hmap_traces["Enlarged"],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Enlarged",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="Oranges",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # Shrunken heatmap
    plot_activity_heatmap(
        hmap_traces["Shrunken"],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Enlarged",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="Purples",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Enlarged heatmap
    plot_activity_heatmap(
        hmap_traces["Stable"],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Enlarged",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="Greys",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # GluSnFr traces
    plot_mean_activity_traces(
        means=list(plastic_trace_means.values()),
        sems=list(plastic_trace_sems.values()),
        group_names=list(plastic_trace_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="GluSnFr",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # Calcium traces
    plot_mean_activity_traces(
        means=list(plastic_ca_trace_means.values()),
        sems=list(plastic_ca_trace_sems.values()),
        group_names=list(plastic_ca_trace_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Calcium",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    # GluSnFr amp box plot
    plot_box_plot(
        plastic_amps,
        figsize=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=2,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Calcium amp box plot
    plot_box_plot(
        plastic_ca_amps,
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=2,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    # Onset histogram
    plot_histogram(
        data=list(plastic_onsets.values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="GluSnFr",
        xtitle="Relative onset (s)",
        xlim=activity_window,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    # Relative onset box plot
    plot_box_plot(
        plastic_onsets,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle=f"Relative onset (s)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=2,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    # Relative onset jitter box plot
    plot_box_plot(
        plastic_onset_jitter,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle=f"Onset jitter (s)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=2,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["J"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if mvmt_type == "All periods":
            fname = os.path.join(save_path, "Local_Coactivity_Figure_11")
        if mvmt_type == "movement":
            fname = os.path.jion(save_path, "Local_Coactivity_Figure_12")
        if mvmt_type == "nonmovement":
            fname = os.path.join(save_path, "Local_Coactivity_Figure_13")

        fig.savefig(fname + ".pdf")

    ########################### Statistics Section ###############################
    if display_stats is False:
        return

    # Perform the f-tests
    if test_type == "parametric":
        amp_f, amp_p, _, amp_df = t_utils.ANOVA_1way_posthoc(
            plastic_amps, method=test_method,
        )
        ca_amp_f, ca_amp_p, _, ca_amp_df = t_utils.ANOVA_1way_posthoc(
            plastic_ca_amps, method=test_method,
        )
        onset_f, onset_p, _, onset_df = t_utils.ANOVA_1way_posthoc(
            plastic_onsets, method=test_method,
        )
        jitter_f, jitter_p, _, jitter_df = t_utils.ANOVA_1way_posthoc(
            plastic_onset_jitter, method=test_method,
        )
        test_title = f"One-way ANOVA {test_method}"
    elif test_type == "nonparametric":
        amp_f, amp_p, amp_df = t_utils.kruskal_wallis_test(
            plastic_amps, "Conover", test_method,
        )
        ca_amp_f, ca_amp_p, ca_amp_df = t_utils.kruskal_wallis_test(
            plastic_ca_amps, "Conover", test_method,
        )
        onset_f, onset_p, onset_df = t_utils.kruskal_wallis_test(
            plastic_onsets, "Conover", test_method,
        )
        jitter_f, jitter_p, jitter_df = t_utils.kruskal_wallis_test(
            plastic_onset_jitter, "Conover", test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        """,
        figsize=(8, 8),
    )
    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(
        f"GluSnFr Amplitude\n{test_title}\nF = {amp_f:.4} p = {amp_p:.3E}"
    )
    A_table = axes2["A"].table(
        cellText=amp_df.values,
        colLabels=amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(
        f"Ca Amplitdue\n{test_title}\nF = {ca_amp_f:.4} p = {ca_amp_p:.3E}"
    )
    B_table = axes2["B"].table(
        cellText=ca_amp_df.values,
        colLabels=ca_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(
        f"GluSnFr Onsets\n{test_title}\nF = {onset_f:.4} p = {onset_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=onset_df.values,
        colLabels=onset_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(
        f"Onset Jitter\n{test_title}\nF = {jitter_f:.4} p = {jitter_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=jitter_df.values,
        colLabels=jitter_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if mvmt_type == "All periods":
            fname = os.path.join(save_path, "Local_Coactivity_Figure_11_Stats")
        if mvmt_type == "movement":
            fname = os.path.jion(save_path, "Local_Coactivity_Figure_12_Stats")
        if mvmt_type == "nonmovement":
            fname = os.path.join(save_path, "Local_Coactivity_Figure_13_Stats")

        fig2.savefig(fname + ".pdf")


def plot_nearby_spine_movement_encoding(
    dataset,
    followup_dataset=None,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 6),
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """Function to plot the movement encoding related variables for nearby spines
    
        INPUT PARAMETERS
            dataset - Local_Coactivity_Data object
            
            followup_dataset - optional Local_Coactivity_Data object of the subsequent
                                session to use for volume comparision. Default is None,
                                to use the followup volumes in the dataset
                                
            exclude - str specifying the type of spines to exlcude from analysis
            
            threshold - float or tuple of floats specifying the threshold cutoffs for
                        classifying plasticity
                        
            figsize - tuple specifying the figure size
            
            showmeans - boolean specifying whether to display mean values on box plots
            
            test_type - str specifying whether to perform parameteric or nonparametric tests
            
            test_method - str specifying the type of posthoc test to perform
            
            display_stats - boolean specifying whether to display the statistics
            
            vol_norm - boolean specifying whether to use normalized relative volume
            
            save - boolean specifying whether to save the figure or not
            
            save_path - str specifying where to save the figures
            
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    plastic_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }

    # Pull the relevant data
    ## Volume related information
    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset == None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags

    # Movement encoding variables
    nearby_movement_correlation = dataset.nearby_movement_correlation
    nearby_movement_stereotypy = dataset.nearby_movement_stereotypy
    nearby_movement_reliability = dataset.nearby_movement_reliability
    nearby_movement_specificity = dataset.nearby_movement_specificity
    nearby_rwd_movement_correlation = dataset.nearby_rwd_movement_correlation
    nearby_rwd_movement_stereotypy = dataset.nearby_rwd_movement_stereotypy
    nearby_rwd_movement_reliability = dataset.nearby_rwd_movement_reliability
    nearby_rwd_movement_specificity = dataset.nearby_rwd_movement_specificity

    # Calculate the relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=vol_norm, exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=vol_norm,
    )

    # Organize data
    ## Subselect present spines
    nearby_movement_correlation = d_utils.subselect_data_by_idxs(
        nearby_movement_correlation, spine_idxs,
    )
    nearby_movement_stereotypy = d_utils.subselect_data_by_idxs(
        nearby_movement_stereotypy, spine_idxs,
    )
    nearby_movement_reliability = d_utils.subselect_data_by_idxs(
        nearby_movement_reliability, spine_idxs,
    )
    nearby_movement_specificity = d_utils.subselect_data_by_idxs(
        nearby_movement_specificity, spine_idxs,
    )
    nearby_rwd_movement_correlation = d_utils.subselect_data_by_idxs(
        nearby_rwd_movement_correlation, spine_idxs
    )
    nearby_rwd_movement_stereotypy = d_utils.subselect_data_by_idxs(
        nearby_rwd_movement_stereotypy, spine_idxs,
    )
    nearby_rwd_movement_reliability = d_utils.subselect_data_by_idxs(
        nearby_rwd_movement_reliability, spine_idxs,
    )
    nearby_rwd_movement_specificity = d_utils.subselect_data_by_idxs(
        nearby_rwd_movement_specificity, spine_idxs,
    )

    ## Seperate into plasticity groups
    plastic_mvmt_corr = {}
    plastic_mvmt_stereo = {}
    plastic_mvmt_reli = {}
    plastic_mvmt_speci = {}
    plastic_rwd_corr = {}
    plastic_rwd_stereo = {}
    plastic_rwd_reli = {}
    plastic_rwd_speci = {}
    for key, value in plastic_groups.items():
        spines = eval(value)
        plastic_mvmt_corr[key] = nearby_movement_correlation[spines]
        plastic_mvmt_stereo[key] = nearby_movement_stereotypy[spines]
        plastic_mvmt_reli[key] = nearby_movement_reliability[spines]
        plastic_mvmt_speci[key] = nearby_movement_specificity[spines]
        plastic_rwd_corr[key] = nearby_rwd_movement_correlation[spines]
        plastic_rwd_stereo[key] = nearby_rwd_movement_stereotypy[spines]
        plastic_rwd_reli[key] = nearby_rwd_movement_specificity[spines]
        plastic_rwd_speci[key] = nearby_rwd_movement_specificity[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFGH
        """,
        figsize=figsize,
    )
    fig.suptitle("Nearby Spine Movement Encoding")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ########################## Plot data onto the axes ############################
    ## Nearby movement correlation
    plot_box_plot(
        plastic_mvmt_corr,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="LMP Correlation (r)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    ## Nearby movement stereotypy
    plot_box_plot(
        plastic_mvmt_stereo,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="Movement stereotypy",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    ## Nearby movement reliability
    plot_box_plot(
        plastic_mvmt_reli,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="Movement reliability",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    ## Nearby movement specificity
    plot_box_plot(
        plastic_mvmt_speci,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="Movement specificity",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    ## Nearby rwd movement correlation
    plot_box_plot(
        plastic_rwd_corr,
        figsize=(5, 5),
        title="Rwd Mvmts",
        xtitle=None,
        ytitle="LMP correlation (r)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Nearby rwd movement stereotypy
    plot_box_plot(
        plastic_rwd_stereo,
        figsize=(5, 5),
        title="Rwd Mvmts",
        xtitle=None,
        ytitle="Movement stereotypy",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ## Nearby rwd movement reliability
    plot_box_plot(
        plastic_rwd_reli,
        figsize=(5, 5),
        title="Rwd Mvmts",
        xtitle=None,
        ytitle="Movement reliability",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    ## Nearby rwd movement specificity
    plot_box_plot(
        plastic_rwd_speci,
        figsize=(5, 5),
        title="Rwd Mvmts",
        xtitle=None,
        ytitle="Movement specificity",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Local_Coactivity_Figure_14")
        fig.savefig(fname + ".pdf")

    ########################## Statistics Section #############################
    if display_stats == False:
        return

    # perform the f-tests
    if test_type == "parametric":
        corr_f, corr_p, _, corr_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_corr, test_method,
        )
        stereo_f, stereo_p, _, stereo_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_stereo, test_method
        )
        reli_f, reli_p, _, reli_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_reli, test_method,
        )
        speci_f, speci_p, _, speci_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_speci, test_method,
        )
        r_corr_f, r_corr_p, _, r_corr_df = t_utils.ANOVA_1way_posthoc(
            plastic_rwd_corr, test_method,
        )
        r_stereo_f, r_stereo_p, _, r_stereo_df = t_utils.ANOVA_1way_posthoc(
            plastic_rwd_stereo, test_method,
        )
        r_reli_f, r_reli_p, _, r_reli_df = t_utils.ANOVA_1way_posthoc(
            plastic_rwd_reli, test_method,
        )
        r_speci_f, r_speci_p, _, r_speci_df = t_utils.ANOVA_1way_posthoc(
            plastic_rwd_speci, test_method,
        )
        test_title = f"One-Way ANOVA {test_method}"
    elif test_type == "nonparametric":
        corr_f, corr_p, corr_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_corr, "Conover", test_method,
        )
        stereo_f, stereo_p, stereo_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_stereo, "Conover", test_method
        )
        reli_f, reli_p, reli_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_reli, "Conover", test_method,
        )
        speci_f, speci_p, speci_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_speci, "Conover", test_method,
        )
        r_corr_f, r_corr_p, r_corr_df = t_utils.kruskal_wallis_test(
            plastic_rwd_corr, "Conover", test_method,
        )
        r_stereo_f, r_stereo_p, r_stereo_df = t_utils.kruskal_wallis_test(
            plastic_rwd_stereo, "Conover", test_method,
        )
        r_reli_f, r_reli_p, r_reli_df = t_utils.kruskal_wallis_test(
            plastic_rwd_reli, "Conover", test_method,
        )
        r_speci_f, r_speci_p, r_speci_df = t_utils.kruskal_wallis_test(
            plastic_rwd_speci, "Conover", test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        """,
        figsize=(8, 10),
    )
    ## Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(
        f"Movement Correlation\n{test_title}\nF = {corr_f:.4} p = {corr_p:.3E}"
    )
    A_table = axes2["A"].table(
        cellText=corr_df.values,
        colLabels=corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(
        f"Movement Stereotypy\n{test_title}\nF = {stereo_f:.4} p = {stereo_p:.3E}"
    )
    B_table = axes2["B"].table(
        cellText=stereo_df.values,
        colLabels=stereo_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(
        f"Movement Reliability\n{test_title}\nF = {reli_f:.4} p = {reli_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=reli_df.values,
        colLabels=reli_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(
        f"Movement Specificity\n{test_title}\nF = {speci_f:.4} p = {speci_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=speci_df.values,
        colLabels=speci_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(
        f"Rwd Mvmt Correlation\n{test_title}\nF = {r_corr_f:.4} p = {r_corr_p:.3E}"
    )
    E_table = axes2["E"].table(
        cellText=r_corr_df.values,
        colLabels=r_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title(
        f"Rwd Mvmt Stereotypy\n{test_title}\nF = {r_stereo_f:.4} p = {r_stereo_p:.3E}"
    )
    F_table = axes2["F"].table(
        cellText=r_stereo_df.values,
        colLabels=r_stereo_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)
    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title(
        f"Rwd Mvmt Reliability\n{test_title}\nF = {r_reli_f:.4} p = {r_reli_p:.3E}"
    )
    G_table = axes2["G"].table(
        cellText=r_reli_df.values,
        colLabels=r_reli_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title(
        f"Rwd Mvmt Specificity\n{test_title}\nF = {r_speci_f:.4} p = {r_speci_p:.3E}"
    )
    H_table = axes2["H"].table(
        cellText=r_speci_df.values,
        colLabels=r_speci_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Local_Coactivity_Figure_14_Stats")
        fig.savefig(fname + ".pdf")


def plot_local_movement_encoding(
    dataset,
    followup_dataset=None,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 6),
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """Function to plot the movement encoding when spines are locally coactive

        INPUT PARAMETERS
            dataset - Local_Coactivity_Data object

            followup_dataset - optional Local_Coactivity_Data object of the subsequent
                                session to use for volume comparision. Default is None,
                                to use the followup volumes in the dataset

            exclude - str specifying the type of spines to exclude from analysis

            threshold - float or tuple of floats specifying the threshold cutoffs for
                        classifying plasticity

            figsize - tuple specifying the figure size

            showmeans - boolean specifying whether to display mean values on box plots

            test_type - str specifying whether to perfor parametric or nonparametric tests

            test_method - str specifying the type of posthoc test to perform

            display_stats - boolean specifying whether to display the statistics

            vol_norm - boolean specifying whether to use normalized relative volume values

            save - boolean specifyiung whether to save the figure or not

            save_path - str specifying where to save the figures
    
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    plastic_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }

    # Pull the relevant data
    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset == None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags

    ## Movement encoding variables
    coactive_movement_correlation = dataset.coactive_movement_correlation
    coactive_movement_stereotypy = dataset.coactive_movement_stereotypy
    coactive_movement_reliability = dataset.coactive_movement_reliability
    coactive_movement_specificity = dataset.coactive_movement_specificity
    coactive_rwd_movement_correlation = dataset.coactive_rwd_movement_correlation
    coactive_rwd_movement_stereotypy = dataset.coactive_rwd_movement_stereotypy
    coactive_rwd_movement_reliability = dataset.coactive_rwd_movement_reliability
    coactive_rwd_movement_specificity = dataset.coactive_rwd_movement_specificity
    coactive_fraction_rwd_mvmts = dataset.coactive_fraction_rwd_mvmts

    # Calculate the relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=vol_norm, exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=vol_norm,
    )

    # Organize data
    ## Subselect present spines
    coactive_movement_correlation = d_utils.subselect_data_by_idxs(
        coactive_movement_correlation, spine_idxs,
    )
    coactive_movement_stereotypy = d_utils.subselect_data_by_idxs(
        coactive_movement_stereotypy, spine_idxs,
    )
    coactive_movement_reliability = d_utils.subselect_data_by_idxs(
        coactive_movement_reliability, spine_idxs,
    )
    coactive_movement_specificity = d_utils.subselect_data_by_idxs(
        coactive_movement_specificity, spine_idxs,
    )
    coactive_rwd_movement_correlation = d_utils.subselect_data_by_idxs(
        coactive_rwd_movement_correlation, spine_idxs,
    )
    coactive_rwd_movement_stereotypy = d_utils.subselect_data_by_idxs(
        coactive_rwd_movement_stereotypy, spine_idxs,
    )
    coactive_rwd_movement_reliability = d_utils.subselect_data_by_idxs(
        coactive_rwd_movement_reliability, spine_idxs,
    )
    coactive_rwd_movement_specificity = d_utils.subselect_data_by_idxs(
        coactive_rwd_movement_specificity, spine_idxs,
    )
    coactive_fraction_rwd_mvmts = d_utils.subselect_data_by_idxs(
        coactive_fraction_rwd_mvmts, spine_idxs,
    )

    ## Seperate into plasticity groups
    plastic_mvmt_corr = {}
    plastic_mvmt_stereo = {}
    plastic_mvmt_reli = {}
    plastic_mvmt_speci = {}
    plastic_rwd_corr = {}
    plastic_rwd_stereo = {}
    plastic_rwd_reli = {}
    plastic_rwd_speci = {}
    plastic_frac_rwd = {}
    for key, value, in plastic_groups.items():
        spines = eval(value)
        plastic_mvmt_corr[key] = coactive_movement_correlation[spines]
        plastic_mvmt_stereo[key] = coactive_movement_stereotypy[spines]
        plastic_mvmt_reli[key] = coactive_movement_reliability[spines]
        plastic_mvmt_speci[key] = coactive_movement_specificity[spines]
        plastic_rwd_corr[key] = coactive_rwd_movement_correlation[spines]
        plastic_rwd_stereo[key] = coactive_rwd_movement_stereotypy[spines]
        plastic_rwd_reli[key] = coactive_rwd_movement_reliability[spines]
        plastic_rwd_speci[key] = coactive_rwd_movement_specificity[spines]
        plastic_frac_rwd[key] = coactive_fraction_rwd_mvmts[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFGH
        I...
        """,
        figsize=figsize,
    )
    fig.suptitle("Local Coactivity Movement Encoding")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ####################### Plot data onto the axes #############################
    ## Coactive movement correlation
    plot_box_plot(
        plastic_mvmt_corr,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="LMP Correlation (r)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    ## Coactive movement stereotypy
    plot_box_plot(
        plastic_mvmt_stereo,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="Movement stereotypy",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    ## Coactive movement reliability
    plot_box_plot(
        plastic_mvmt_reli,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="Movement reliability",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    ## Coactive movement specificity
    plot_box_plot(
        plastic_mvmt_speci,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="Movement specificity",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    ## Coactive rwd movement correlation
    plot_box_plot(
        plastic_rwd_corr,
        figsize=(5, 5),
        title="Rwd Mvmts",
        xtitle=None,
        ytitle="LMP correlation (r)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Coactive rwd movement stereotypy
    plot_box_plot(
        plastic_rwd_stereo,
        figsize=(5, 5),
        title="Rwd Mvmts",
        xtitle=None,
        ytitle="Movement stereotypy",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ## Coactive rwd movement reliability
    plot_box_plot(
        plastic_rwd_reli,
        figsize=(5, 5),
        title="Rwd Mvmts",
        xtitle=None,
        ytitle="Movement reliability",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    ## Coactive rwd movement specificity
    plot_box_plot(
        plastic_rwd_speci,
        figsize=(5, 5),
        title="Rwd Mvmts",
        xtitle=None,
        ytitle="Movement specificity",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
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
    ## Coactive fraction rwd mvmts
    plot_box_plot(
        plastic_frac_rwd,
        figsize=(5, 5),
        title="",
        xtitle=None,
        ytitle="Fraction of rwd mvmts",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.8,
        b_err_alpha=0.8,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Local_Coactivity_Figure_15")
        fig.savefig(fname + ".pdf")

    ########################## Statistics Section ##############################
    if display_stats == False:
        return

    # Perform the f-tests
    # perform the f-tests
    if test_type == "parametric":
        corr_f, corr_p, _, corr_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_corr, test_method,
        )
        stereo_f, stereo_p, _, stereo_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_stereo, test_method
        )
        reli_f, reli_p, _, reli_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_reli, test_method,
        )
        speci_f, speci_p, _, speci_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_speci, test_method,
        )
        r_corr_f, r_corr_p, _, r_corr_df = t_utils.ANOVA_1way_posthoc(
            plastic_rwd_corr, test_method,
        )
        r_stereo_f, r_stereo_p, _, r_stereo_df = t_utils.ANOVA_1way_posthoc(
            plastic_rwd_stereo, test_method,
        )
        r_reli_f, r_reli_p, _, r_reli_df = t_utils.ANOVA_1way_posthoc(
            plastic_rwd_reli, test_method,
        )
        r_speci_f, r_speci_p, _, r_speci_df = t_utils.ANOVA_1way_posthoc(
            plastic_rwd_speci, test_method,
        )
        frac_f, frac_p, _, frac_df = t_utils.ANOVA_1way_posthoc(
            plastic_frac_rwd, test_method,
        )
        test_title = f"One-Way ANOVA {test_method}"
    elif test_type == "nonparametric":
        corr_f, corr_p, corr_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_corr, "Conover", test_method,
        )
        stereo_f, stereo_p, stereo_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_stereo, "Conover", test_method
        )
        reli_f, reli_p, reli_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_reli, "Conover", test_method,
        )
        speci_f, speci_p, speci_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_speci, "Conover", test_method,
        )
        r_corr_f, r_corr_p, r_corr_df = t_utils.kruskal_wallis_test(
            plastic_rwd_corr, "Conover", test_method,
        )
        r_stereo_f, r_stereo_p, r_stereo_df = t_utils.kruskal_wallis_test(
            plastic_rwd_stereo, "Conover", test_method,
        )
        r_reli_f, r_reli_p, r_reli_df = t_utils.kruskal_wallis_test(
            plastic_rwd_reli, "Conover", test_method,
        )
        r_speci_f, r_speci_p, r_speci_df = t_utils.kruskal_wallis_test(
            plastic_rwd_speci, "Conover", test_method,
        )
        frac_f, frac_p, frac_df = t_utils.kruskal_wallis_test(
            plastic_frac_rwd, "Conover", test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        I.
        """,
        figsize=(8, 11),
    )

    ## Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(
        f"Movement Correlation\n{test_title}\nF = {corr_f:.4} p = {corr_p:.3E}"
    )
    A_table = axes2["A"].table(
        cellText=corr_df.values,
        colLabels=corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(
        f"Movement Stereotypy\n{test_title}\nF = {stereo_f:.4} p = {stereo_p:.3E}"
    )
    B_table = axes2["B"].table(
        cellText=stereo_df.values,
        colLabels=stereo_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(
        f"Movement Reliability\n{test_title}\nF = {reli_f:.4} p = {reli_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=reli_df.values,
        colLabels=reli_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(
        f"Movement Specificity\n{test_title}\nF = {speci_f:.4} p = {speci_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=speci_df.values,
        colLabels=speci_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(
        f"Rwd Mvmt Correlation\n{test_title}\nF = {r_corr_f:.4} p = {r_corr_p:.3E}"
    )
    E_table = axes2["E"].table(
        cellText=r_corr_df.values,
        colLabels=r_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title(
        f"Rwd Mvmt Stereotypy\n{test_title}\nF = {r_stereo_f:.4} p = {r_stereo_p:.3E}"
    )
    F_table = axes2["F"].table(
        cellText=r_stereo_df.values,
        colLabels=r_stereo_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)
    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title(
        f"Rwd Mvmt Reliability\n{test_title}\nF = {r_reli_f:.4} p = {r_reli_p:.3E}"
    )
    G_table = axes2["G"].table(
        cellText=r_reli_df.values,
        colLabels=r_reli_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title(
        f"Rwd Mvmt Specificity\n{test_title}\nF = {r_speci_f:.4} p = {r_speci_p:.3E}"
    )
    H_table = axes2["H"].table(
        cellText=r_speci_df.values,
        colLabels=r_speci_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)
    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title(
        f"Frac Rwd Movements\n{test_title}\nF = {frac_f:.4} p = {frac_p:.3E}"
    )
    I_table = axes2["I"].table(
        cellText=frac_df.values,
        colLabels=frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Local_Coactivity_Figure_15_Stats")
        fig.savefig(fname + ".pdf")
