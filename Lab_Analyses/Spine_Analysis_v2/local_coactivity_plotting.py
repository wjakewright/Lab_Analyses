import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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
    mean_type="median",
    err_type="CI",
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to compare basic properties of coactive and non coactive events

        INPUT PARAMETERS
            dataset - Local_Coactivity_Data object

            figsize - tuple specifying the figure size

            mean_type - str specifying the mean type for bar plots

            err_type - str specifying the err type for bar plots

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
        noncoactive_dend_traces, axis=0, nan_policty="omit"
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
    fig.subplot_adjust(hspace=1, wspace=0.5)

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
    plot_swarm_bar_plot(
        data_dict={
            "Coactive": spine_coactive_amplitude,
            "Noncoactive": spine_noncoactive_amplitude,
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.8,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Coactive vs Noncoactive Calcium amplitude
    plot_swarm_bar_plot(
        data_dict={
            "Coactive": spine_coactive_calcium_amplitude,
            "Noncoactive": spine_noncoactive_calcium_amplitude,
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.8,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # Coactive vs Noncoactive local amplitude
    plot_swarm_bar_plot(
        data_dict={
            "Coactive": coactive_local_dend_amplitude,
            "Noncoactive": noncoactive_local_dend_amplitude,
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="Local Dendrite",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.8,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
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
        fname = os.path.join(save, "Local_Coactivity_Figure_1")
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
            spine_coactive_amplitude, spine_noncoactive_amplitude, nan_policy="omit"
        )
        ca_amp_t, ca_amp_p = stats.mannwhitneyu(
            spine_coactive_calcium_amplitude,
            spine_noncoactive_calcium_amplitude,
            nan_policy="omit",
        )
        dend_amp_t, dend_amp_p = stats.mannwhitneyu(
            coactive_local_dend_amplitude,
            noncoactive_local_dend_amplitude,
            nan_policy="omit",
        )
        test_title = "Mann-Whitney U"

    # Organize the results
    results_dict = {
        "test": ["GluSnFr Amp", "Calcium Amp", "Local Dend"],
        "stat": [amp_t, ca_amp_t, dend_amp_t],
        "p-val": [amp_p, ca_amp_p, dend_amp_p],
    }
    results_df = pd.DataFrame.from_dict(results_dict)

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
    A_table.autoset_font_size(False)
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

            mean_type - str specifying the mean type for bar plots

            err_type - str specifying the err type for bar plots

            hist_bins - int specifying how many bins for the histograms

            test_type - str specifying whether to perform parametric or nonparametric tests

            display_stats - boolean specifying whether to display stat results

            save - boolean specifying whether to save the figure or not

            save_path - str specifying where to save the figure
    
    """
    COLORS = ["mediumblue", "firebrick"]
    if rwd_mvmts:
        mvmt_key = "Rwd movement"
        nonmvmt_key = "Nonrwd movement"
    else:
        mvmt_key = "Movement"
        nonmvmt_key = "Non-movement"

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
        mvmt_key: np.nanmedian(shuff_local_coactivity_rate["Movement"], axis=1),
        nonmvmt_key: np.nanmedian(shuff_local_coactivity_rate["Non-movement"], axis=1),
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
        mvmt_key: np.nanmedian(shuff_local_coactivity_rate_norm["Movement"], axis=1),
        nonmvmt_key: np.nanmedian(
            shuff_local_coactivity_rate_norm["Non-movement"], axis=1
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
        AABCCDDE
        FFGHHIIJ
        KKLMMN..
        """,
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
        m_size=7,
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
        m_size=7,
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
    plot_swarm_bar_plot(
        avg_local_coactivity_rate,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="Local Coactivity",
        xtitle=None,
        ytitle=f"Coactivity rate (events/min)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Mvmt vs nonmvmt local coactivity rate norm
    plot_swarm_bar_plot(
        avg_local_coactivity_rate_norm,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="Local Coactivity",
        xtitle=None,
        ytitle=f"Norm. coactivity rate",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
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
            avg_local_coactivity_rate[mvmt_key], shuff_local_coactivity_rate[mvmt_key],
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title=mvmt_key,
        xtitle="Coactivity rate (events/min)",
        xlim=None,
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
    ax_c_inset = axes["C"].inset_axes([0.9, 0.4, 0.4, 0.6])
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
            avg_local_coactivity_rate[nonmvmt_key],
            shuff_local_coactivity_rate[nonmvmt_key],
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title=nonmvmt_key,
        xtitle="Coactivity rate (events/min)",
        xlim=None,
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
    ax_d_inset = axes["D"].inset_axes([0.9, 0.4, 0.4, 0.6])
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
            avg_local_coactivity_rate_norm[mvmt_key],
            shuff_local_coactivity_rate_norm[mvmt_key],
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title=mvmt_key,
        xtitle="Norm. coactivity rate",
        xlim=None,
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
    ax_h_inset = axes["H"].inset_axes([0.9, 0.4, 0.4, 0.6])
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
        ytitle="Norm. coactivity rate",
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
            avg_local_coactivity_rate_norm[nonmvmt_key],
            shuff_local_coactivity_rate_norm[nonmvmt_key],
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title=nonmvmt_key,
        xtitle="Norm. coactivity rate",
        xlim=None,
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
    ax_i_inset = axes["I"].inset_axes([0.9, 0.4, 0.4, 0.6])
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
        ytitle="Norm coactivity rate",
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
    plot_swarm_bar_plot(
        real_vs_shuff_diff,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle=f"Above chance difference",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Real vs shuff relative difference norm
    plot_swarm_bar_plot(
        real_vs_shuff_diff_norm,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle=f"Above chance difference (norm)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # GluSnFr activity traces
    plot_mean_activity_traces(
        means=list(coactive_trace_means.values()),
        sems=list(coactive_trace_sems.value()),
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
        sems=list(coactive_trace_ca_sems.value()),
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
    plot_swarm_bar_plot(
        coactive_amplitude,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    # Calcium amplitudes
    plot_swarm_bar_plot(
        coactive_calcium_amplitude,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
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
            avg_local_coactivity_rate[mvmt_key],
            avg_local_coactivity_rate[nonmvmt_key],
            nan_policy="omit",
        )
        coactivity_norm_t, coactivity_norm_p = stats.mannwhitneyu(
            avg_local_coactivity_rate_norm[mvmt_key],
            avg_local_coactivity_rate_norm[nonmvmt_key],
            nan_policy="omit",
        )
        rel_diff_t, rel_diff_p = stats.mannwhitneyu(
            real_vs_shuff_diff[mvmt_key],
            real_vs_shuff_diff[nonmvmt_key],
            nan_policy="omit",
        )
        rel_diff_norm_t, rel_diff_norm_p = stats.mannwhitneyu(
            real_vs_shuff_diff_norm[mvmt_key],
            real_vs_shuff_diff_norm[nonmvmt_key],
            nan_policy="omit",
        )
        amp_t, amp_p = stats.mannwhitneyu(
            coactive_amplitude[mvmt_key],
            coactive_amplitude[nonmvmt_key],
            nan_policy="omit",
        )
        amp_ca_t, amp_ca_p = stats.mannwhitneyu(
            coactive_calcium_amplitude[mvmt_key],
            coactive_calcium_amplitude[nonmvmt_key],
            nan_policy="omit",
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

    # Perform correlations
    _, distance_corr_df = t_utils.correlate_grouped_data(
        distance_coactivity_rate, distance_bins
    )
    _, distance_corr_df_norm = t_utils.correlate_grouped_data(
        distance_coactivity_rate_norm, distance_bins
    )

    ## Comparisons to chance
    mvmt_above, mvmt_below = t_utils.test_against_chance(
        avg_local_coactivity_rate[mvmt_key], shuff_local_coactivity_rate[mvmt_key]
    )
    nonmvmt_above, nonmvmt_below = t_utils.test_against_chance(
        avg_local_coactivity_rate[nonmvmt_key],
        shuff_local_coactivity_rate[nonmvmt_key],
    )
    mvmt_above_norm, mvmt_below_norm = t_utils.test_against_chance(
        avg_local_coactivity_rate_norm[mvmt_key],
        shuff_local_coactivity_rate_norm[mvmt_key],
    )
    nonmvmt_above_norm, nonmvmt_below_norm = t_utils.test_against_chance(
        avg_local_coactivity_rate_norm[nonmvmt_key],
        shuff_local_coactivity_rate_norm[nonmvmt_key],
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

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        """,
        figsize=(8, 6),
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
    A_table.autoset_font_size(False)
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
    B_table.autoset_font_size(False)
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
    C_table.autoset_font_size(False)
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
    D_table.autoset_font_size(False)
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
        fig2.savefig(fname, ".pdf")


def plot_plasticity_coactivity_rates(
    dataset,
    mvmt_dataset,
    nonmvmt_dataset,
    followup_dataset=None,
    norm=False,
    rwd_mvmt=False,
    exclude="Shaft",
    threshold=0.3,
    figsize=(10, 8),
    hist_bins=30,
    mean_type="median",
    err_type="CI",
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
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
        mvmt_key = "Rwd movement"
        nonmvmt_key = "Nonrwd movement"
    else:
        mvmt_key = "Movement"
        nonmvmt_key = "Non-movemment"

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
        volumes, flags, norm=False, exclude=exclude
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=False
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
    coactive_spines = d_utils.subselect_data_by_idxs(coactive_spines)
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
        plastic_shuff_rates[key] = shuff_rates.flatten().astype(np.float32)
        plastic_shuff_medians[key] = np.nanmedian(shuff_rates, axis=1)
        ## Calculate fractions
        co_spines = coactive_spines[spines]
        event_num = coactive_event_num[spines]
        mvmt_event_num = mvmt_coactive_event_num[spines]
        fraction_coactive[key] = np.nansum(co_spines) / len(co_spines)
        fraction_mvmt[key] = mvmt_event_num / event_num

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
        line_colors=COLORS,
        face_color="white",
        m_size=7,
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
        line_colors=COLORS,
        face_color="white",
        m_size=7,
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
        line_colors=COLORS,
        face_color="white",
        m_size=7,
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
        xlim=None,
        ylim=None,
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
        xlim=None,
        ylim=None,
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
        xlim=None,
        ylim=None,
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
    plot_swarm_bar_plot(
        plastic_local_rates,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="All periods",
        xtitle=None,
        ytitle=f"Local {coactivity_title}",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # Mvmt local coactivity rate bar plot
    plot_swarm_bar_plot(
        mvmt_local_rates,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=mvmt_key,
        xtitle=None,
        ytitle=f"Local {coactivity_title}",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Nonmvt local coactivity rate bar plot
    plot_swarm_bar_plot(
        nonmvmt_local_rates,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=nonmvmt_key,
        xtitle=None,
        ytitle=f"Local {coactivity_title}",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
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
        err_type=err_type,
        figsize=(5, 5),
        title="Coactive spines",
        xtitle=None,
        ytitle=f"Fraction of spines",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # Fraction mvmt events
    plot_swarm_bar_plot(
        fraction_mvmt,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=mvmt_key,
        xtitle=None,
        ytitle=f"Fraction of events",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    # Real vs shuff relative difference
    plot_swarm_bar_plot(
        plastic_diffs,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="All periods",
        xtitle=None,
        ytitle=f"Above chance difference",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
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
        data=list(plastic_local_rates["Enlarged"], plastic_shuff_rates["Enlarged"]),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Enlarged",
        xtitle=f"Local {coactivity_title}",
        xlim=None,
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
    ax_e_inset = axes["E"].inset_axes([0.9, 0.4, 0.4, 0.6])
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
        data=list(plastic_local_rates["Shrunken"], plastic_shuff_rates["Shrunken"]),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Shrunken",
        xtitle=f"Local {coactivity_title}",
        xlim=None,
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
    ax_f_inset = axes["F"].inset_axes([0.9, 0.4, 0.4, 0.6])
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
        data=list(plastic_local_rates["Stable"], plastic_shuff_rates["Stable"]),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Stable",
        xtitle=f"Local {coactivity_title}",
        xlim=None,
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
    ax_g_inset = axes["G"].inset_axes([0.9, 0.4, 0.4, 0.6])
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
                fname = os.path.join(save_path, "Local_Coactivity_Figure_3")
            else:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_4")
        else:
            if not rwd_mvmt:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_5")
            else:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_6")
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
    if test_type == "nonparametric":
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
    A_table.autoset_font_size(False)
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
    B_table.autoset_font_size(False)
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
    C_table.autoset_font_size(False)
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
    D_table.autoset_font_size(False)
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
    E_table.autoset_font_size(False)
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
    F_table.autoset_font_size(False)
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
    G_table.autoset_font_size(False)
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
    H_table.autoset_font_size(False)
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
    I_table.autoset_font_size(False)
    I_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if norm == False:
            if not rwd_mvmt:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_3_Stats")
            else:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_4_Stats")
        else:
            if not rwd_mvmt:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_5_Stats")
            else:
                fname = os.path.join(save_path, "Local_Coactivity_Figure_6_Stats")
        fig2.savefig(fname + ".pdf")

