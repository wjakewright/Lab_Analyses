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
from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Spine_Analysis_v2 import spine_utilities as s_utils
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils

sns.set()
sns.set_style("ticks")


def plot_comparative_mvmt_coactivity(
    mvmt_dataset,
    nonmvmt_dataset,
    coactivity_type="All",
    figsize=(10, 10),
    showmeans=False,
    mean_type="median",
    err_type="CI",
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to compare coactivity during movement and nonmovement periods

        INPUT PARAMETERS
            mvmt_dataset - Dendritic_Coactivity_Data that was constrained to mvmt periods

            nonmvmt_dataset - Dendritic_Coactivity_Data that was constrained to nonmvmt periods

            period - str specifying what type of coactivity to look at. Accepts 
                    "All", "nonconj", and "conj"
            
            figsize - tuple specifying the size of the figure

            showmeans - boolean specifying whether to show means on box plots

            mean_type - str specifying the type of central tendency for bar plots

            err_type - str specifying the type of error for bar plots

            test_type - str specifying whether to perform parametric or nonparametric tests

            display_stats - boolean specifying whether to display stats

            save  -boolean specifying whether to save the figure

            save_path - str specifying where to save the figure
    """
    COLORS = ["mediumblue", "firebrick"]
    if coactivity_type == "All":
        main_title = "All events"
    elif coactivity_type == "nonconj":
        main_title = "Without local coactivity"
    elif coactivity_type == "conj":
        main_title = "With local coactivity"

    # Pull relevant data
    sampling_rate = mvmt_dataset.parameters["Sampling Rate"]
    activity_window = mvmt_dataset.parameters["Activity Window"]
    if mvmt_dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"

    # Coactivity related variables
    if coactivity_type == "All":
        mvmt_dendrite_coactivity_rate = mvmt_dataset.all_dendrite_coactivity_rate
        nonmvmt_dendrite_coactivity_rate = nonmvmt_dataset.all_dendrite_coactivity_rate
        mvmt_shuff_coactivity_rate = mvmt_dataset.all_shuff_dendrite_coactivity_rate
        nonmvmt_shuff_coactivity_rate = (
            nonmvmt_dataset.all_shuff_dendrite_coactivity_rate
        )
        mvmt_above_chance = mvmt_dataset.all_above_chance_coactivity
        nonmvmt_above_chance = nonmvmt_dataset.all_above_chance_coactivity
        mvmt_dendrite_coactivity_rate_norm = (
            mvmt_dataset.all_dendrite_coactivity_rate_norm
        )
        nonmvmt_dendrite_coactivity_rate_norm = (
            nonmvmt_dataset.all_dendrite_coactivity_rate_norm
        )
        mvmt_shuff_coactivity_rate_norm = (
            mvmt_dataset.all_shuff_dendrite_coactivity_rate_norm
        )
        nonmvmt_shuff_coactivity_rate_norm = (
            nonmvmt_dataset.all_shuff_dendrite_coactivity_rate_norm
        )
        mvmt_above_chance_norm = mvmt_dataset.all_above_chance_coactivity_norm
        nonmvmt_above_chance_norm = nonmvmt_dataset.all_above_chance_coactivity_norm
        mvmt_spine_coactive_traces = mvmt_dataset.all_spine_coactive_traces
        nonmvmt_spine_coactive_traces = nonmvmt_dataset.all_spine_coactive_traces
        mvmt_spine_calcium_traces = mvmt_dataset.all_spine_coactive_calcium_traces
        nonmvmt_spine_calcium_traces = nonmvmt_dataset.all_spine_coactive_calcium_traces
        mvmt_dendrite_traces = mvmt_dataset.all_dendrite_coactive_traces
        nonmvmt_dendrite_traces = nonmvmt_dataset.all_dendrite_coactive_traces
        mvmt_spine_coactive_amp = mvmt_dataset.all_spine_coactive_amplitude
        nonmvmt_spine_coactive_amp = nonmvmt_dataset.all_spine_coactive_amplitude
        mvmt_spine_calcium_amp = mvmt_dataset.all_spine_coactive_calcium_amplitude
        nonmvmt_spine_calcium_amp = nonmvmt_dataset.all_spine_coactive_calcium_amplitude
        mvmt_dendrite_coactive_amp = mvmt_dataset.all_dendrite_coactive_amplitude
        nonmvmt_dendrite_coactive_amp = nonmvmt_dataset.all_dendrite_coactive_amplitude
    elif coactivity_type == "conj":
        mvmt_dendrite_coactivity_rate = mvmt_dataset.conj_dendrite_coactivity_rate
        nonmvmt_dendrite_coactivity_rate = nonmvmt_dataset.conj_dendrite_coactivity_rate
        mvmt_shuff_coactivity_rate = mvmt_dataset.conj_shuff_dendrite_coactivity_rate
        nonmvmt_shuff_coactivity_rate = (
            nonmvmt_dataset.conj_shuff_dendrite_coactivity_rate
        )
        mvmt_above_chance = mvmt_dataset.conj_above_chance_coactivity
        nonmvmt_above_chance = nonmvmt_dataset.conj_above_chance_coactivity
        mvmt_dendrite_coactivity_rate_norm = (
            mvmt_dataset.conj_dendrite_coactivity_rate_norm
        )
        nonmvmt_dendrite_coactivity_rate_norm = (
            nonmvmt_dataset.conj_dendrite_coactivity_rate_norm
        )
        mvmt_shuff_coactivity_rate_norm = (
            mvmt_dataset.conj_shuff_dendrite_coactivity_rate_norm
        )
        nonmvmt_shuff_coactivity_rate_norm = (
            nonmvmt_dataset.conj_shuff_dendrite_coactivity_rate_norm
        )
        mvmt_above_chance_norm = mvmt_dataset.conj_above_chance_coactivity_norm
        nonmvmt_above_chance_norm = nonmvmt_dataset.conj_above_chance_coactivity_norm
        mvmt_spine_coactive_traces = mvmt_dataset.conj_spine_coactive_traces
        nonmvmt_spine_coactive_traces = nonmvmt_dataset.conj_spine_coactive_traces
        mvmt_spine_calcium_traces = mvmt_dataset.conj_spine_coactive_calcium_traces
        nonmvmt_spine_calcium_traces = (
            nonmvmt_dataset.conj_spine_coactive_calcium_traces
        )
        mvmt_dendrite_traces = mvmt_dataset.conj_dendrite_coactive_traces
        nonmvmt_dendrite_traces = nonmvmt_dataset.conj_dendrite_coactive_traces
        mvmt_spine_coactive_amp = mvmt_dataset.conj_spine_coactive_amplitude
        nonmvmt_spine_coactive_amp = nonmvmt_dataset.conj_spine_coactive_amplitude
        mvmt_spine_calcium_amp = mvmt_dataset.conj_spine_coactive_calcium_amplitude
        nonmvmt_spine_calcium_amp = (
            nonmvmt_dataset.conj_spine_coactive_calcium_amplitude
        )
        mvmt_dendrite_coactive_amp = mvmt_dataset.conj_dendrite_coactive_amplitude
        nonmvmt_dendrite_coactive_amp = nonmvmt_dataset.conj_dendrite_coactive_amplitude
    elif coactivity_type == "nonconj":
        mvmt_dendrite_coactivity_rate = mvmt_dataset.nonconj_dendrite_coactivity_rate
        nonmvmt_dendrite_coactivity_rate = (
            nonmvmt_dataset.nonconj_dendrite_coactivity_rate
        )
        mvmt_shuff_coactivity_rate = mvmt_dataset.nonconj_shuff_dendrite_coactivity_rate
        nonmvmt_shuff_coactivity_rate = (
            nonmvmt_dataset.nonconj_shuff_dendrite_coactivity_rate
        )
        mvmt_above_chance = mvmt_dataset.nonconj_above_chance_coactivity
        nonmvmt_above_chance = nonmvmt_dataset.nonconj_above_chance_coactivity
        mvmt_dendrite_coactivity_rate_norm = (
            mvmt_dataset.nonconj_dendrite_coactivity_rate_norm
        )
        nonmvmt_dendrite_coactivity_rate_norm = (
            nonmvmt_dataset.nonconj_dendrite_coactivity_rate_norm
        )
        mvmt_shuff_coactivity_rate_norm = (
            mvmt_dataset.nonconj_shuff_dendrite_coactivity_rate_norm
        )
        nonmvmt_shuff_coactivity_rate_norm = (
            nonmvmt_dataset.nonconj_shuff_dendrite_coactivity_rate_norm
        )
        mvmt_above_chance_norm = mvmt_dataset.nonconj_above_chance_coactivity_norm
        nonmvmt_above_chance_norm = nonmvmt_dataset.nonconj_above_chance_coactivity_norm
        mvmt_spine_coactive_traces = mvmt_dataset.nonconj_spine_coactive_traces
        nonmvmt_spine_coactive_traces = nonmvmt_dataset.nonconj_spine_coactive_traces
        mvmt_spine_calcium_traces = mvmt_dataset.nonconj_spine_coactive_calcium_traces
        nonmvmt_spine_calcium_traces = (
            nonmvmt_dataset.nonconj_spine_coactive_calcium_traces
        )
        mvmt_dendrite_traces = mvmt_dataset.nonconj_dendrite_coactive_traces
        nonmvmt_dendrite_traces = nonmvmt_dataset.nonconj_dendrite_coactive_traces
        mvmt_spine_coactive_amp = mvmt_dataset.nonconj_spine_coactive_amplitude
        nonmvmt_spine_coactive_amp = nonmvmt_dataset.nonconj_spine_coactive_amplitude
        mvmt_spine_calcium_amp = mvmt_dataset.nonconj_spine_coactive_calcium_amplitude
        nonmvmt_spine_calcium_amp = (
            nonmvmt_dataset.nonconj_spine_coactive_calcium_amplitude
        )
        mvmt_dendrite_coactive_amp = mvmt_dataset.nonconj_dendrite_coactive_amplitude
        nonmvmt_dendrite_coactive_amp = (
            nonmvmt_dataset.nonconj_dendrite_coactive_amplitude
        )

    # Subselect only present spines
    present_spines = s_utils.find_present_spines(mvmt_dataset.spine_flags)
    mvmt_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
        mvmt_dendrite_coactivity_rate, present_spines
    )
    nonmvmt_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
        nonmvmt_dendrite_coactivity_rate, present_spines
    )
    mvmt_shuff_coactivity_rate = d_utils.subselect_data_by_idxs(
        mvmt_shuff_coactivity_rate, present_spines
    )
    nonmvmt_shuff_coactivity_rate = d_utils.subselect_data_by_idxs(
        nonmvmt_shuff_coactivity_rate, present_spines
    )
    mvmt_above_chance = d_utils.subselect_data_by_idxs(
        mvmt_above_chance, present_spines
    )
    nonmvmt_above_chance = d_utils.subselect_data_by_idxs(
        nonmvmt_above_chance, present_spines
    )
    mvmt_dendrite_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
        mvmt_dendrite_coactivity_rate_norm, present_spines
    )
    nonmvmt_dendrite_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
        nonmvmt_dendrite_coactivity_rate_norm, present_spines
    )
    mvmt_shuff_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
        mvmt_shuff_coactivity_rate_norm, present_spines
    )
    nonmvmt_shuff_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
        nonmvmt_shuff_coactivity_rate_norm, present_spines
    )
    mvmt_above_chance_norm = d_utils.subselect_data_by_idxs(
        mvmt_above_chance_norm, present_spines
    )
    nonmvmt_above_chance_norm = d_utils.subselect_data_by_idxs(
        nonmvmt_above_chance_norm, present_spines
    )
    mvmt_spine_coactive_traces = d_utils.subselect_data_by_idxs(
        mvmt_spine_coactive_traces, present_spines
    )
    nonmvmt_spine_coactive_traces = d_utils.subselect_data_by_idxs(
        nonmvmt_spine_coactive_traces, present_spines,
    )
    mvmt_spine_calcium_traces = d_utils.subselect_data_by_idxs(
        mvmt_spine_calcium_traces, present_spines,
    )
    nonmvmt_spine_calcium_traces = d_utils.subselect_data_by_idxs(
        nonmvmt_spine_calcium_traces, present_spines,
    )
    mvmt_dendrite_traces = d_utils.subselect_data_by_idxs(
        mvmt_dendrite_traces, present_spines,
    )
    nonmvmt_dendrite_traces = d_utils.subselect_data_by_idxs(
        nonmvmt_dendrite_traces, present_spines
    )
    mvmt_spine_coactive_amp = d_utils.subselect_data_by_idxs(
        mvmt_spine_coactive_amp, present_spines,
    )
    nonmvmt_spine_coactive_amp = d_utils.subselect_data_by_idxs(
        nonmvmt_spine_coactive_amp, present_spines,
    )
    mvmt_spine_calcium_amp = d_utils.subselect_data_by_idxs(
        mvmt_spine_calcium_amp, present_spines,
    )
    nonmvmt_spine_calcium_amp = d_utils.subselect_data_by_idxs(
        nonmvmt_spine_calcium_amp, present_spines,
    )
    mvmt_dendrite_coactive_amp = d_utils.subselect_data_by_idxs(
        mvmt_dendrite_coactive_amp, present_spines,
    )
    nonmvmt_dendrite_coactive_amp = d_utils.subselect_data_by_idxs(
        nonmvmt_dendrite_coactive_amp, present_spines,
    )

    # Organize data for plotting
    dendrite_coactivity_rate = {
        "Mvmt": mvmt_dendrite_coactivity_rate,
        "Nonmvmt": nonmvmt_dendrite_coactivity_rate,
    }
    dendrite_coactivity_rate_norm = {
        "Mvmt": mvmt_dendrite_coactivity_rate_norm,
        "Nonmvmt": nonmvmt_dendrite_coactivity_rate_norm,
    }
    above_chance = {
        "Mvmt": mvmt_above_chance,
        "Nonmvmt": nonmvmt_above_chance,
    }
    above_chance_norm = {
        "Mvmt": mvmt_above_chance_norm,
        "Nonmvmt": nonmvmt_above_chance,
    }
    shuff_coactivity_rate = {
        "Mvmt": mvmt_shuff_coactivity_rate,
        "Nonmvmt": nonmvmt_shuff_coactivity_rate,
    }
    shuff_coactivity_median = {
        "Mvmt": np.nanmedian(mvmt_shuff_coactivity_rate, axis=1),
        "Nonmvmt": np.nanmedian(nonmvmt_shuff_coactivity_rate, axis=1),
    }
    shuff_coactivity_rate_norm = {
        "Mvmt": mvmt_shuff_coactivity_rate_norm,
        "Nonmvmt": nonmvmt_shuff_coactivity_rate_norm,
    }
    shuff_coactivity_median_norm = {
        "Mvmt": np.nanmedian(mvmt_shuff_coactivity_rate_norm, axis=1),
        "Nonmvmt": np.nanmedian(nonmvmt_shuff_coactivity_rate_norm),
    }
    mvmt_spine_coactive_traces = [
        np.nanmean(x, axis=1)
        for x in mvmt_spine_coactive_traces
        if type(x) == np.ndarray
    ]
    mvmt_spine_coactive_traces = np.vstack(mvmt_spine_coactive_traces,)
    nonmvmt_spine_coactive_traces = [
        np.nanmean(x, axis=1)
        for x in nonmvmt_spine_coactive_traces
        if type(x) == np.ndarray
    ]
    nonmvmt_spine_coactive_traces = np.vstack(nonmvmt_spine_coactive_traces)
    spine_coactive_means = {
        "Mvmt": np.nanmean(mvmt_spine_coactive_traces, axis=0),
        "Nonmvmt": np.nanmean(nonmvmt_spine_coactive_traces, axis=0),
    }
    spine_coactive_sems = {
        "Mvmt": stats.sem(mvmt_spine_coactive_traces, axis=0, nan_policy="omit"),
        "Nonmvmt": stats.sem(nonmvmt_spine_coactive_traces, axis=0, nan_policy="omit"),
    }
    mvmt_spine_calcium_traces = [
        np.nanmean(x, axis=1)
        for x in mvmt_spine_calcium_traces
        if type(x) == np.ndarray
    ]
    mvmt_spine_calcium_traces = np.vstack(mvmt_spine_calcium_traces)
    nonmvmt_spine_calcium_traces = [
        np.nanmean(x, axis=1)
        for x in nonmvmt_spine_calcium_traces
        if type(x) == np.ndarray
    ]
    nonmvmt_spine_calcium_traces = np.vstack(nonmvmt_spine_calcium_traces)
    spine_calcium_means = {
        "Mvmt": np.nanmean(mvmt_spine_calcium_traces, axis=0),
        "Nonmvmt": np.nanmean(nonmvmt_spine_calcium_traces, axis=0),
    }
    spine_calcium_sems = {
        "Mvmt": stats.sem(mvmt_spine_calcium_traces, axis=0, nan_policy="omit"),
        "Nonmvmt": stats.sem(nonmvmt_spine_calcium_traces, axis=0, nan_policy="omit"),
    }
    mvmt_dendrite_traces = [
        np.nanmean(x, axis=1) for x in mvmt_dendrite_traces if type(x) == np.ndarray
    ]
    mvmt_dendrite_traces = np.vstack(mvmt_dendrite_traces)
    nonmvmt_dendrite_traces = [
        np.nanmean(x, axis=1) for x in nonmvmt_dendrite_traces if type(x) == np.ndarray
    ]
    nonmvmt_dendrite_traces = np.vstack(nonmvmt_dendrite_traces)
    dendrite_coactive_means = {
        "Mvmt": np.nanmean(mvmt_dendrite_traces, axis=0),
        "Nonmvmt": np.nanmean(nonmvmt_dendrite_traces, axis=0),
    }
    dendrite_coactive_sems = {
        "Mvmt": stats.sem(mvmt_dendrite_traces, axis=0, nan_policy="omit"),
        "Nonmvmt": stats.sem(nonmvmt_dendrite_traces, axis=0, nan_policy="omit"),
    }
    spine_coactive_amps = {
        "Mvmt": mvmt_spine_coactive_amp,
        "Nonmvmt": nonmvmt_spine_coactive_amp,
    }
    spine_calcium_amps = {
        "Mvmt": mvmt_spine_calcium_amp,
        "Nonmvmt": nonmvmt_spine_calcium_amp,
    }
    dendrite_coactive_amps = {
        "Mvmt": mvmt_dendrite_coactive_amp,
        "Nonmvmt": nonmvmt_dendrite_coactive_amp,
    }

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFGH
        IJKL
        MN..
        """,
        figsize=figsize,
    )
    fig.suptitle(f"Mvmt vs Nonmvmt Dendritic coactivity {main_title}")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ####################### Plot data onto axes #########################
    # Raw coactivity rates
    plot_box_plot(
        dendrite_coactivity_rate,
        figsize=(5, 5),
        title="Raw",
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
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # Norm coactivity rates
    plot_box_plot(
        dendrite_coactivity_rate_norm,
        figsize=(5, 5),
        title="Norm.",
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
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Mvmt raw coactivity vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list((dendrite_coactivity_rate["Mvmt"], shuff_coactivity_rate["Mvmt"],)),
        plot_ind=True,
        title="Mvmt",
        xtitle="Coactivity rate (events/min)",
        xlime=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.03,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_b_inset = axes["B"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_b_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": dendrite_coactivity_rate["Mvmt"],
            "shuff": shuff_coactivity_median["Mvmt"],
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
        ax=ax_b_inset,
        save=False,
        save_path=None,
    )
    # Nonmvmt raw coactivity vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list(
            (dendrite_coactivity_rate["Nonmvmt"], shuff_coactivity_rate["Nonmvmt"],)
        ),
        plot_ind=True,
        title="Nonmvmt",
        xtitle="Coactivity rate (events/min)",
        xlime=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
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
            "data": dendrite_coactivity_rate["Nonmvmt"],
            "shuff": shuff_coactivity_median["Nonmvmt"],
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
        ax=ax_c_inset,
        save=False,
        save_path=None,
    )
    # Mvmt norm coactivity vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list(
            (dendrite_coactivity_rate_norm["Mvmt"], shuff_coactivity_rate_norm["Mvmt"],)
        ),
        plot_ind=True,
        title="Mvmt",
        xtitle="Norm. coactivity rate",
        xlime=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
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
    ax_f_inset = axes["F"].inset_axes([0.8, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_f_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": dendrite_coactivity_rate_norm["Mvmt"],
            "shuff": shuff_coactivity_median_norm["Mvmt"],
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
        ax=ax_f_inset,
        save=False,
        save_path=None,
    )
    # Nonmvmt norm coactivity vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list(
            (
                dendrite_coactivity_rate_norm["Nonmvmt"],
                shuff_coactivity_rate_norm["Nonmvmt"],
            )
        ),
        plot_ind=True,
        title="Mvmt",
        xtitle="Norm. coactivity rate",
        xlime=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
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
            "data": dendrite_coactivity_rate_norm["Nonmvmt"],
            "shuff": shuff_coactivity_median_norm["Nonmvmt"],
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
        ax=ax_g_inset,
        save=False,
        save_path=None,
    )
    # Raw coactivity above chance
    plot_box_plot(
        above_chance,
        figsize=(5, 5),
        title="Raw",
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
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # Norm coactivity above chance
    plot_box_plot(
        above_chance_norm,
        figsize=(5, 5),
        title="Norm.",
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
    # Spine GluSnFr traces
    plot_mean_activity_traces(
        means=list(spine_coactive_means.values()),
        sems=list(spine_coactive_sems.values()),
        group_names=list(spine_coactive_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Spine GluSnFr",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    # Spine Calcium traces
    plot_mean_activity_traces(
        means=list(spine_calcium_means.values()),
        sems=list(spine_calcium_sems.values()),
        group_names=list(spine_calcium_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Spine calcium",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Dendrite traces
    plot_mean_activity_traces(
        means=list(dendrite_coactive_means.values()),
        sems=list(dendrite_coactive_sems.values()),
        group_names=list(dendrite_coactive_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Dendrite calcium",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    # Spine GluSnFr amplitude
    plot_box_plot(
        spine_coactive_amps,
        figsize=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Event ampltude ({activity_type})",
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
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    # Spine Calcium amplitude
    plot_box_plot(
        spine_calcium_amps,
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Event ampltude ({activity_type})",
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
    # Dendrite calcium amplitude
    plot_box_plot(
        dendrite_coactive_amps,
        figsize=(5, 5),
        title="Dendrite",
        xtitle=None,
        ytitle=f"Event ampltude ({activity_type})",
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
        if coactivity_type == "All":
            fname = os.path.join(save_path, "Dendrite_Coactivity_Figure_1")
        elif coactivity_type == "conj":
            fname = os.path.join(save_path, "Dendrite_Coactivity_Figure_2")
        elif coactivity_type == "nonconj":
            fname = os.path.join(save_path, "Dendrite_Coactivity_Figure_3")
        fig.savefig(fname + ".pdf")

    ########################### Statistics Section ###############################
    if display_stats == False:
        return

    # Perform t-tests / mann-whitney u tests
    if test_type == "parametric":
        coactivity_t, coactivity_p = stats.ttest_ind(
            dendrite_coactivity_rate["Mvmt"],
            dendrite_coactivity_rate["Nonmvmt"],
            nan_policy="omit",
        )
        coactivity_norm_t, coactivity_norm_p = stats.ttest_ind(
            dendrite_coactivity_rate_norm["Mvmt"],
            dendrite_coactivity_rate_norm["Nonmvmt"],
            nan_policy="omit",
        )
        chance_t, chance_p = stats.ttest_ind(
            above_chance["Mvmt"], above_chance["Nonmvmt"], nan_policy="omit",
        )
        chance_norm_t, chance_norm_p = stats.ttest_ind(
            above_chance_norm["Mvmt"], above_chance_norm["Nonmvmt"], nan_policy="omit",
        )
        spine_amp_t, spine_amp_p = stats.ttest_ind(
            spine_coactive_amps["Mvmt"],
            spine_coactive_amps["Nonmvmt"],
            nan_policy="omit",
        )
        spine_ca_t, spine_ca_p = stats.ttest_ind(
            spine_calcium_amps["Mvmt"],
            spine_calcium_amps["Nonmvmt"],
            nan_policy="omit",
        )
        dend_amp_t, dend_amp_p = stats.ttest_ind(
            dendrite_coactive_amps["Mvmt"],
            dendrite_coactive_amps["Nonmvmt"],
            nan_policy="omit",
        )
        test_title = "T-Test"
    elif test_type == "nonparametric":
        coactivity_t, coactivity_p = stats.mannwhitneyu(
            dendrite_coactivity_rate["Mvmt"][
                ~np.isnan(dendrite_coactivity_rate["Mvmt"])
            ],
            dendrite_coactivity_rate["Nonmvmt"][
                ~np.isnan(dendrite_coactivity_rate["Nonmvmt"])
            ],
        )
        coactivity_norm_t, coactivity_norm_p = stats.ttest_ind(
            dendrite_coactivity_rate_norm["Mvmt"][
                ~np.isnan(dendrite_coactivity_rate_norm["Mvmt"])
            ],
            dendrite_coactivity_rate_norm["Nonmvmt"][
                ~np.isnan(dendrite_coactivity_rate_norm["Nonmvmt"])
            ],
        )
        chance_t, chance_p = stats.ttest_ind(
            above_chance["Mvmt"][~np.isnan(above_chance["Mvmt"])],
            above_chance["Nonmvmt"][~np.isnan(above_chance["Nonmvmt"])],
        )
        chance_norm_t, chance_norm_p = stats.ttest_ind(
            above_chance_norm["Mvmt"][~np.isnan(above_chance_norm["Mvmt"])],
            above_chance_norm["Nonmvmt"][~np.isnan(above_chance_norm["Nonmvmt"])],
        )
        spine_amp_t, spine_amp_p = stats.ttest_ind(
            spine_coactive_amps["Mvmt"][~np.isnan(spine_coactive_amps["Mvmt"])],
            spine_coactive_amps["Nonmvmt"][~np.isnan(spine_coactive_amps["Nonmvmt"])],
        )
        spine_ca_t, spine_ca_p = stats.ttest_ind(
            spine_calcium_amps["Mvmt"][~np.isnan(spine_calcium_amps["Mvmt"])],
            spine_calcium_amps["Nonmvmt"][~np.isnan(spine_calcium_amps["Nonmvmt"])],
        )
        dend_amp_t, dend_amp_p = stats.ttest_ind(
            dendrite_coactive_amps["Mvmt"][~np.isnan(dendrite_coactive_amps["Mvmt"])],
            dendrite_coactive_amps["Nonmvmt"][
                ~np.isnan(dendrite_coactive_amps["Nonmvmt"])
            ],
        )
        test_title = "Mann-Whitney U"

    # Organize these results
    results_dict = {
        "Comparision": [
            "Dendrite cocactivity",
            "Dendrite coactivity norm",
            "Chance diff.",
            "Chance diff. norm.",
            "Spine amp.",
            "Spine ca amp.",
            "Dend amp",
        ],
        "stat": [
            coactivity_t,
            coactivity_norm_t,
            chance_t,
            chance_norm_t,
            spine_amp_t,
            spine_ca_t,
            dend_amp_t,
        ],
        "p-val": [
            coactivity_p,
            coactivity_norm_p,
            chance_p,
            chance_norm_p,
            spine_amp_p,
            spine_ca_p,
            dend_amp_p,
        ],
    }
    result_df = pd.DataFrame.from_dict(results_dict)
    result_df.update(result_df[["p-val"]].applymap("{:.4E}".format))

    ## Comparisons to chance
    mvmt_above, mvmt_below = t_utils.test_against_chance(
        dendrite_coactivity_rate["Mvmt"], shuff_coactivity_rate["Mvmt"]
    )
    nonmvmt_above, nonmvmt_below = t_utils.test_against_chance(
        dendrite_coactivity_rate["Nonmvmt"], shuff_coactivity_rate["Nonmvmt"],
    )
    mvmt_above_norm, mvmt_below_norm = t_utils.test_against_chance(
        dendrite_coactivity_rate_norm["Mvmt"], shuff_coactivity_rate_norm["Mvmt"]
    )
    nonmvmt_above_norm, nonmvmt_below_norm = t_utils.test_against_chance(
        dendrite_coactivity_rate_norm["Nonmvmt"], shuff_coactivity_rate_norm["Nonmvmt"],
    )
    chance_dict = {
        "Comparision": ["Mvmt", "Nonmvmt", "Mvmt norm.", "Nonmvmt norm."],
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
        """,
        figsize=(8, 4),
    )
    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(f"{test_title} results")
    A_table = axes2["A"].table(
        cellText=result_df.values,
        colLabels=result_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(f"Chance coactivity")
    B_table = axes2["B"].table(
        cellText=chance_df.values,
        colLabels=chance_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if coactivity_type == "All":
            fname = os.path.join(save_path, "Dendrite_Coactivity_Figure_1_Stats")
        elif coactivity_type == "conj":
            fname = os.path.join(save_path, "Dendrite_Coactivity_Figure_2_Stats")
        elif coactivity_type == "nonconj":
            fname = os.path.join(save_path, "Dendrite_Coactivity_Figure_3_Stats")
        fig2.savefig(fname + ".pdf")


def plot_plasticity_coactivity_rates(
    dataset,
    followup_dataset=None,
    period="All periods",
    norm=False,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 10),
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
    """Function to compare coactivity rates across plasticity groups

        INPUT PARAMETERS
            dataset - Dendritic_Coactivity_Data object 

            period - str specifying if the dataset analysis was constrained to a given
                    period. Currently accepts "All periods", "movement", and "nonmovement"

            norm - boolean specifying whether to look at normalized coactivity 

            exclude - str specifying spine type to exclude form analysis

            threshold - float or tuple of floats specifying the threshold cutoff for
                        classifying plasticity

            figsize - tuple specifying the size of the figure

            showmeans - boolean specifying whether to plot means on box plots

            mean_type - str specifying the mean tyhpe for bar plots

            err_type - str specifying the error type for the bar plots

            test_type - str specifying whether to perform parametric or nonparametric tests

            test_method - str specifying the type of posthoc test to perform

            display_stats - boolean specifying whether to display stats

            vol_norm - boolean specifying whether to use normalized relative volumes

            save - boolean specifying whether to save the figure

            save_path - str specifying where to save the figure
    
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    plastic_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }
    if period == "All periods":
        main_title = "Entire session"
    elif period == "movement":
        main_title = "Movement periods"
    elif period == "nonmovement":
        main_title = "Nonmovement periods"

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

