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


def plot_conj_vs_nonconj_events(
    dataset,
    figsize=(8, 5),
    showmeans=False,
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to compare conj vs nonconj events

    INPUT PARAMETERS
        dataset - Dendritic_Coactivity_Data object

        figsize - tuple specifying the figure size

        showmeans - boolean specifying whether to show means on box plots

        test_type - str specifying whether to perorm parametric or nonparametric tests

        display_stats - boolean specifying whetehr to display the stat results

        save - boolean specifying whether to save the data or not

        save_path - str specifying where to save the data

    """
    COLORS = ["forestgreen", "black"]

    # Pull the relevant data
    sampling_rate = dataset.parameters["Sampling Rate"]
    activity_window = dataset.parameters["Activity Window"]
    if dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"

    # Activity data
    conj_spine_coactive_traces = dataset.conj_spine_coactive_traces
    conj_spine_coactive_calcium_traces = dataset.conj_spine_coactive_calcium_traces
    conj_dendrite_coactive_traces = dataset.conj_dendrite_coactive_traces
    conj_spine_coactive_amplitude = dataset.conj_spine_coactive_amplitude
    conj_spine_coactive_calcium_amplitude = (
        dataset.conj_spine_coactive_calcium_amplitude
    )
    conj_dendrite_coactive_amplitude = dataset.conj_dendrite_coactive_amplitude
    nonconj_spine_coactive_traces = dataset.nonconj_spine_coactive_traces
    nonconj_spine_coactive_calcium_traces = (
        dataset.nonconj_spine_coactive_calcium_traces
    )
    nonconj_dendrite_coactive_traces = dataset.nonconj_dendrite_coactive_traces
    nonconj_spine_coactive_amplitude = dataset.nonconj_spine_coactive_amplitude
    nonconj_spine_coactive_calcium_amplitude = (
        dataset.nonconj_spine_coactive_calcium_amplitude
    )
    nonconj_dendrite_coactive_amplitude = dataset.nonconj_dendrite_coactive_amplitude

    # Organize the data
    spine_coactive_amplitude = {
        "local": conj_spine_coactive_amplitude,
        "no local": nonconj_spine_coactive_amplitude,
    }
    spine_coactive_calcium_amplitude = {
        "local": conj_spine_coactive_calcium_amplitude,
        "no local": nonconj_spine_coactive_calcium_amplitude,
    }
    dendrite_coactive_amplitude = {
        "local": conj_dendrite_coactive_amplitude,
        "no local": nonconj_dendrite_coactive_amplitude,
    }
    conj_spine_traces = [
        np.nanmean(x, axis=1)
        for x in conj_spine_coactive_traces
        if type(x) == np.ndarray
    ]
    conj_spine_traces = np.vstack(conj_spine_traces)
    nonconj_spine_traces = [
        np.nanmean(x, axis=1)
        for x in nonconj_spine_coactive_traces
        if type(x) == np.ndarray
    ]
    nonconj_spine_traces = np.vstack(nonconj_spine_traces)
    spine_coactive_means = {
        "local": np.nanmean(conj_spine_traces, axis=0),
        "no local": np.nanmean(nonconj_spine_traces, axis=0),
    }
    spine_coactive_sems = {
        "local": stats.sem(conj_spine_traces, axis=0, nan_policy="omit"),
        "no local": stats.sem(nonconj_spine_traces, axis=0, nan_policy="omit"),
    }
    conj_ca_traces = [
        np.nanmean(x, axis=1)
        for x in conj_spine_coactive_calcium_traces
        if type(x) == np.ndarray
    ]
    conj_ca_traces = np.vstack(conj_ca_traces)
    nonconj_ca_traces = [
        np.nanmean(x, axis=1)
        for x in nonconj_spine_coactive_calcium_traces
        if type(x) == np.ndarray
    ]
    nonconj_ca_traces = np.vstack(nonconj_ca_traces)
    spine_coactive_ca_means = {
        "local": np.nanmean(conj_ca_traces, axis=0),
        "no local": np.nanmean(nonconj_ca_traces, axis=0),
    }
    spine_coactive_ca_sems = {
        "local": stats.sem(conj_ca_traces, axis=0, nan_policy="omit"),
        "no local": stats.sem(nonconj_ca_traces, axis=0, nan_policy="omit"),
    }
    conj_dend_traces = [
        np.nanmean(x, axis=1)
        for x in conj_dendrite_coactive_traces
        if type(x) == np.ndarray
    ]
    conj_dend_traces = np.vstack(conj_dend_traces)
    nonconj_dend_traces = [
        np.nanmean(x, axis=1)
        for x in nonconj_dendrite_coactive_traces
        if type(x) == np.ndarray
    ]
    nonconj_dend_traces = np.vstack(nonconj_dend_traces)
    dendrite_coactive_means = {
        "local": np.nanmean(conj_dend_traces, axis=0),
        "no local": np.nanmean(nonconj_dend_traces, axis=0),
    }
    dendrite_coactive_sems = {
        "local": stats.sem(conj_dend_traces, axis=0, nan_policy="omit"),
        "no local": stats.sem(nonconj_dend_traces, axis=0, nan_policy="omit"),
    }

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EF..
        """,
        figsize=figsize,
    )
    fig.suptitle(f"Conj vs Nonconj Dendritic Coactivity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ###################### Plot data onto axes #########################
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
    # Spine Calcium traces
    plot_mean_activity_traces(
        means=list(spine_coactive_ca_means.values()),
        sems=list(spine_coactive_ca_sems.values()),
        group_names=list(spine_coactive_ca_means.keys()),
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
    # Spine GluSnFr traces
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
        title="Dendrite traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Spine GluSnFr amplitude box plot
    plot_box_plot(
        spine_coactive_amplitude,
        figsize=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.5,
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
    # Spine Calcium amplitude box plot
    plot_box_plot(
        spine_coactive_calcium_amplitude,
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.5,
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
    # Spine GluSnFr amplitude box plot
    plot_box_plot(
        dendrite_coactive_amplitude,
        figsize=(5, 5),
        title="Dendrite",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.5,
        b_err_alpha=1,
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
        fname = os.path.join(save_path, "Conj_vs_Nonconj_Coactivity_Figure")
        fig.savefig(fname + ".pdf")

    ################################ Statistics Section ################################
    if display_stats == False:
        return

    # Perform statistics
    if test_type == "parametric":
        amp_t, amp_p = stats.ttest_ind(
            spine_coactive_amplitude["local"],
            spine_coactive_amplitude["no local"],
            nan_policy="omit",
        )
        ca_amp_t, ca_amp_p = stats.ttest_ind(
            spine_coactive_calcium_amplitude["local"],
            spine_coactive_calcium_amplitude["no local"],
            nan_policy="omit",
        )
        dend_amp_t, dend_amp_p = stats.ttest_ind(
            dendrite_coactive_amplitude["local"],
            dendrite_coactive_amplitude["no local"],
            nan_policy="omit",
        )
        test_title = "T-Test"
    elif test_type == "nonparametric":
        amp_t, amp_p = stats.mannwhitneyu(
            spine_coactive_amplitude["local"][
                ~np.isnan(spine_coactive_amplitude["local"])
            ],
            spine_coactive_amplitude["no local"][
                ~np.isnan(spine_coactive_amplitude["no local"])
            ],
        )
        ca_amp_t, ca_amp_p = stats.mannwhitneyu(
            spine_coactive_calcium_amplitude["local"][
                ~np.isnan(spine_coactive_calcium_amplitude["local"])
            ],
            spine_coactive_calcium_amplitude["no local"][
                ~np.isnan(spine_coactive_calcium_amplitude["no local"])
            ],
        )
        dend_amp_t, dend_amp_p = stats.mannwhitneyu(
            dendrite_coactive_amplitude["local"][
                ~np.isnan(dendrite_coactive_amplitude["local"])
            ],
            dendrite_coactive_amplitude["no local"][
                ~np.isnan(dendrite_coactive_amplitude["no local"])
            ],
        )
        test_title = "Mann-Whitney U"

        # Organize the results
        result_dict = {
            "test": ["GluSnFr Amp", "Calcium Amp", "Dend Amp"],
            "stat": [amp_t, ca_amp_t, dend_amp_t],
            "p-val": [amp_p, ca_amp_p, dend_amp_p],
        }
        results_df = pd.DataFrame.from_dict(result_dict)
        results_df.update(results_df[["p-val"]].applymap("{:.4E}".format))

        # Display the stats
        fig2, axes2 = plt.subplot_mosaic("""A""", figsize=(4, 4))
        # Format the table
        axes2["A"].axis("off")
        axes2["A"].axis("tight")
        axes2["A"].set_title(f"{test_title} results")
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
            fname = os.path.join(save_path, "Conj_vs_Nonconj_Coactivity_Stats")
            fig2.savefig(fname + ".pdf")


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
    COLORS = ["darkorange", "darkviolet"]
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

    spine_idxs = s_utils.find_present_spines(mvmt_dataset.spine_flags)
    spine_idxs1 = s_utils.find_present_spines(nonmvmt_dataset.spine_flags)

    # Coactivity related variables
    if coactivity_type == "All":
        mvmt_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
            mvmt_dataset.all_dendrite_coactivity_rate, spine_idxs
        )
        nonmvmt_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.all_dendrite_coactivity_rate, spine_idxs1
        )
        mvmt_shuff_coactivity_rate = d_utils.subselect_data_by_idxs(
            mvmt_dataset.all_shuff_dendrite_coactivity_rate, spine_idxs
        )
        nonmvmt_shuff_coactivity_rate = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.all_shuff_dendrite_coactivity_rate, spine_idxs1
        )
        mvmt_dendrite_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            mvmt_dataset.all_dendrite_coactivity_rate_norm, spine_idxs
        )
        nonmvmt_dendrite_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.all_dendrite_coactivity_rate_norm, spine_idxs1
        )
        mvmt_shuff_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            mvmt_dataset.all_shuff_dendrite_coactivity_rate_norm, spine_idxs
        )
        nonmvmt_shuff_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.all_shuff_dendrite_coactivity_rate_norm, spine_idxs1
        )
        mvmt_spine_coactive_traces = d_utils.subselect_data_by_idxs(
            mvmt_dataset.all_spine_coactive_traces, spine_idxs
        )
        nonmvmt_spine_coactive_traces = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.all_spine_coactive_traces, spine_idxs1
        )
        mvmt_spine_calcium_traces = d_utils.subselect_data_by_idxs(
            mvmt_dataset.all_spine_coactive_calcium_traces, spine_idxs
        )
        nonmvmt_spine_calcium_traces = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.all_spine_coactive_calcium_traces, spine_idxs1
        )
        mvmt_dendrite_traces = d_utils.subselect_data_by_idxs(
            mvmt_dataset.all_dendrite_coactive_traces, spine_idxs
        )
        nonmvmt_dendrite_traces = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.all_dendrite_coactive_traces, spine_idxs1
        )
        mvmt_spine_coactive_amp = d_utils.subselect_data_by_idxs(
            mvmt_dataset.all_spine_coactive_amplitude, spine_idxs
        )
        nonmvmt_spine_coactive_amp = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.all_spine_coactive_amplitude, spine_idxs1
        )
        mvmt_spine_calcium_amp = d_utils.subselect_data_by_idxs(
            mvmt_dataset.all_spine_coactive_calcium_amplitude, spine_idxs
        )
        nonmvmt_spine_calcium_amp = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.all_spine_coactive_calcium_amplitude, spine_idxs1
        )
        mvmt_dendrite_coactive_amp = d_utils.subselect_data_by_idxs(
            mvmt_dataset.all_dendrite_coactive_amplitude, spine_idxs
        )
        nonmvmt_dendrite_coactive_amp = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.all_dendrite_coactive_amplitude, spine_idxs1
        )
    elif coactivity_type == "conj":
        mvmt_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
            mvmt_dataset.conj_dendrite_coactivity_rate, spine_idxs
        )
        nonmvmt_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.conj_dendrite_coactivity_rate, spine_idxs1
        )
        mvmt_shuff_coactivity_rate = d_utils.subselect_data_by_idxs(
            mvmt_dataset.conj_shuff_dendrite_coactivity_rate, spine_idxs
        )
        nonmvmt_shuff_coactivity_rate = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.conj_shuff_dendrite_coactivity_rate, spine_idxs1
        )
        mvmt_dendrite_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            mvmt_dataset.conj_dendrite_coactivity_rate_norm, spine_idxs
        )
        nonmvmt_dendrite_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.conj_dendrite_coactivity_rate_norm, spine_idxs1
        )
        mvmt_shuff_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            mvmt_dataset.conj_shuff_dendrite_coactivity_rate_norm, spine_idxs
        )
        nonmvmt_shuff_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.conj_shuff_dendrite_coactivity_rate_norm, spine_idxs1
        )
        mvmt_spine_coactive_traces = d_utils.subselect_data_by_idxs(
            mvmt_dataset.conj_spine_coactive_traces, spine_idxs
        )
        nonmvmt_spine_coactive_traces = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.conj_spine_coactive_traces, spine_idxs1
        )
        mvmt_spine_calcium_traces = d_utils.subselect_data_by_idxs(
            mvmt_dataset.conj_spine_coactive_calcium_traces, spine_idxs
        )
        nonmvmt_spine_calcium_traces = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.conj_spine_coactive_calcium_traces, spine_idxs1
        )
        mvmt_dendrite_traces = d_utils.subselect_data_by_idxs(
            mvmt_dataset.conj_dendrite_coactive_traces, spine_idxs
        )
        nonmvmt_dendrite_traces = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.conj_dendrite_coactive_traces, spine_idxs1
        )
        mvmt_spine_coactive_amp = d_utils.subselect_data_by_idxs(
            mvmt_dataset.conj_spine_coactive_amplitude, spine_idxs
        )
        nonmvmt_spine_coactive_amp = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.conj_spine_coactive_amplitude, spine_idxs1
        )
        mvmt_spine_calcium_amp = d_utils.subselect_data_by_idxs(
            mvmt_dataset.conj_spine_coactive_calcium_amplitude, spine_idxs
        )
        nonmvmt_spine_calcium_amp = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.conj_spine_coactive_calcium_amplitude, spine_idxs1
        )
        mvmt_dendrite_coactive_amp = d_utils.subselect_data_by_idxs(
            mvmt_dataset.conj_dendrite_coactive_amplitude, spine_idxs
        )
        nonmvmt_dendrite_coactive_amp = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.conj_dendrite_coactive_amplitude, spine_idxs1
        )
    elif coactivity_type == "nonconj":
        mvmt_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
            mvmt_dataset.nonconj_dendrite_coactivity_rate, spine_idxs
        )
        nonmvmt_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.nonconj_dendrite_coactivity_rate, spine_idxs1
        )
        mvmt_shuff_coactivity_rate = d_utils.subselect_data_by_idxs(
            mvmt_dataset.nonconj_shuff_dendrite_coactivity_rate, spine_idxs
        )
        nonmvmt_shuff_coactivity_rate = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.nonconj_shuff_dendrite_coactivity_rate, spine_idxs1
        )
        mvmt_dendrite_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            mvmt_dataset.nonconj_dendrite_coactivity_rate_norm, spine_idxs
        )
        nonmvmt_dendrite_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.nonconj_dendrite_coactivity_rate_norm, spine_idxs1
        )
        mvmt_shuff_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            mvmt_dataset.nonconj_shuff_dendrite_coactivity_rate_norm, spine_idxs
        )
        nonmvmt_shuff_coactivity_rate_norm = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.nonconj_shuff_dendrite_coactivity_rate_norm, spine_idxs1
        )
        mvmt_spine_coactive_traces = d_utils.subselect_data_by_idxs(
            mvmt_dataset.nonconj_spine_coactive_traces, spine_idxs
        )
        nonmvmt_spine_coactive_traces = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.nonconj_spine_coactive_traces, spine_idxs1
        )
        mvmt_spine_calcium_traces = d_utils.subselect_data_by_idxs(
            mvmt_dataset.nonconj_spine_coactive_calcium_traces, spine_idxs
        )
        nonmvmt_spine_calcium_traces = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.nonconj_spine_coactive_calcium_traces, spine_idxs1
        )
        mvmt_dendrite_traces = d_utils.subselect_data_by_idxs(
            mvmt_dataset.nonconj_dendrite_coactive_traces, spine_idxs
        )
        nonmvmt_dendrite_traces = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.nonconj_dendrite_coactive_traces, spine_idxs1
        )
        mvmt_spine_coactive_amp = d_utils.subselect_data_by_idxs(
            mvmt_dataset.nonconj_spine_coactive_amplitude, spine_idxs
        )
        nonmvmt_spine_coactive_amp = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.nonconj_spine_coactive_amplitude, spine_idxs1
        )
        mvmt_spine_calcium_amp = d_utils.subselect_data_by_idxs(
            mvmt_dataset.nonconj_spine_coactive_calcium_amplitude, spine_idxs
        )
        nonmvmt_spine_calcium_amp = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.nonconj_spine_coactive_calcium_amplitude, spine_idxs1
        )
        mvmt_dendrite_coactive_amp = d_utils.subselect_data_by_idxs(
            mvmt_dataset.nonconj_dendrite_coactive_amplitude, spine_idxs
        )
        nonmvmt_dendrite_coactive_amp = d_utils.subselect_data_by_idxs(
            nonmvmt_dataset.nonconj_dendrite_coactive_amplitude, spine_idxs1
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
        "Nonmvmt": np.nanmedian(nonmvmt_shuff_coactivity_rate_norm, axis=1),
    }
    mvmt_spine_coactive_traces = [
        np.nanmean(x, axis=1)
        for x in mvmt_spine_coactive_traces
        if type(x) == np.ndarray
    ]
    mvmt_spine_coactive_traces = np.vstack(
        mvmt_spine_coactive_traces,
    )
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
        ABC.
        EFG.
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
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.7,
        b_err_alpha=1,
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
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.7,
        b_err_alpha=1,
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
        data=list(
            (
                dendrite_coactivity_rate["Mvmt"],
                shuff_coactivity_rate["Mvmt"],
            )
        ),
        plot_ind=True,
        title="Mvmt",
        xtitle="Coactivity rate (events/min)",
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
        b_linewidth=1.5,
        b_alpha=0.7,
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
            (
                dendrite_coactivity_rate["Nonmvmt"],
                shuff_coactivity_rate["Nonmvmt"],
            )
        ),
        plot_ind=True,
        title="Nonmvmt",
        xtitle="Coactivity rate (events/min)",
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
        b_linewidth=1.5,
        b_alpha=0.7,
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
            (
                dendrite_coactivity_rate_norm["Mvmt"],
                shuff_coactivity_rate_norm["Mvmt"],
            )
        ),
        plot_ind=True,
        title="Mvmt",
        xtitle="Norm. coactivity rate",
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
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_f_inset = axes["F"].inset_axes([0.8, 0.25, 0.4, 0.6])
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
        b_linewidth=1.5,
        b_alpha=0.7,
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
        b_linewidth=1.5,
        b_alpha=0.7,
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
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.7,
        b_err_alpha=1,
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
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.7,
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
    # Dendrite calcium amplitude
    plot_box_plot(
        dendrite_coactive_amps,
        figsize=(5, 5),
        title="Dendrite",
        xtitle=None,
        ytitle=f"Event ampltude ({activity_type})",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1.5,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.7,
        b_err_alpha=1,
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
            fname = os.path.join(
                save_path, "Mvmt_vs_Nonmvmt_Dendrite_Coactivity_Figure"
            )
        elif coactivity_type == "conj":
            fname = os.path.join(save_path, "Mvmt_vs_Nonmvmt_Dendrite_Conj_Figure")
        elif coactivity_type == "nonconj":
            fname = os.path.join(save_path, "Mvmt_vs_Nonmvmt_Dendrite_Nonconj_Figure")
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
            "Spine amp.",
            "Spine ca amp.",
            "Dend amp",
        ],
        "stat": [
            coactivity_t,
            coactivity_norm_t,
            spine_amp_t,
            spine_ca_t,
            dend_amp_t,
        ],
        "p-val": [
            coactivity_p,
            coactivity_norm_p,
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
        dendrite_coactivity_rate["Nonmvmt"],
        shuff_coactivity_rate["Nonmvmt"],
    )
    mvmt_above_norm, mvmt_below_norm = t_utils.test_against_chance(
        dendrite_coactivity_rate_norm["Mvmt"], shuff_coactivity_rate_norm["Mvmt"]
    )
    nonmvmt_above_norm, nonmvmt_below_norm = t_utils.test_against_chance(
        dendrite_coactivity_rate_norm["Nonmvmt"],
        shuff_coactivity_rate_norm["Nonmvmt"],
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
            fname = os.path.join(save_path, "Mvmt_vs_Nonmvmt_Dendrite_Coactivity_Stats")
        elif coactivity_type == "conj":
            fname = os.path.join(save_path, "Mvmt_vs_Nonmvmt_Dendrite_Conj_Stats")
        elif coactivity_type == "nonconj":
            fname = os.path.join(save_path, "Mvmt_vs_Nonmvmt_Dendrite_Nonconj_Stats")
        fig2.savefig(fname + ".pdf")


def plot_plasticity_coactivity_rates(
    dataset,
    followup_dataset=None,
    period="All periods",
    norm=False,
    exclude="Shaft Spine",
    MRSs=None,
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

         MRSs - str specifying if you wish to examine only MRSs or nonMRSs. Accepts
                "MRS" and "nMRS". Defualt is None to examine all spines

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
    COLORS = ["mediumslateblue", "tomato", "silver"]
    plastic_groups = {
        "sLTP": "enlarged_spines",
        "sLTD": "shrunken_spines",
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
    if norm == False:
        nname = "Raw"
        coactivity_title = "Coactivity rate (event/min)"
        all_dendrite_coactivity_rate = dataset.all_dendrite_coactivity_rate
        all_shuff_dendrite_coactivity_rate = dataset.all_shuff_dendrite_coactivity_rate
        all_fraction_dendrite_coactive = dataset.all_fraction_dendrite_coactive
        all_fraction_spine_coactive = dataset.all_fraction_spine_coactive
        nonconj_dendrite_coactivity_rate = dataset.nonconj_dendrite_coactivity_rate
        nonconj_shuff_dendrite_coactivity_rate = (
            dataset.nonconj_shuff_dendrite_coactivity_rate
        )
        nonconj_fraction_dendrite_coactive = dataset.nonconj_fraction_dendrite_coactive
        nonconj_fraction_spine_coactive = dataset.nonconj_fraction_spine_coactive
        conj_dendrite_coactivity_rate = dataset.conj_dendrite_coactivity_rate
        conj_shuff_dendrite_coactivity_rate = (
            dataset.conj_shuff_dendrite_coactivity_rate
        )
        conj_fraction_dendrite_coactive = dataset.conj_fraction_dendrite_coactive
        conj_fraction_spine_coactive = dataset.conj_fraction_spine_coactive
        fraction_conj_events = dataset.fraction_conj_events
    else:
        nname = "Norm."
        coactivity_title = "Norm. coactivity rate"
        all_dendrite_coactivity_rate = dataset.all_dendrite_coactivity_rate_norm
        all_shuff_dendrite_coactivity_rate = (
            dataset.all_shuff_dendrite_coactivity_rate_norm
        )
        all_fraction_dendrite_coactive = dataset.all_fraction_dendrite_coactive
        all_fraction_spine_coactive = dataset.all_fraction_spine_coactive
        nonconj_dendrite_coactivity_rate = dataset.nonconj_dendrite_coactivity_rate_norm
        nonconj_shuff_dendrite_coactivity_rate = (
            dataset.nonconj_shuff_dendrite_coactivity_rate_norm
        )
        nonconj_fraction_dendrite_coactive = dataset.nonconj_fraction_dendrite_coactive
        nonconj_fraction_spine_coactive = dataset.nonconj_fraction_spine_coactive
        conj_dendrite_coactivity_rate = dataset.conj_dendrite_coactivity_rate_norm
        conj_shuff_dendrite_coactivity_rate = (
            dataset.conj_shuff_dendrite_coactivity_rate_norm
        )
        conj_fraction_dendrite_coactive = dataset.conj_fraction_dendrite_coactive
        conj_fraction_spine_coactive = dataset.conj_fraction_spine_coactive
        fraction_conj_events = dataset.fraction_conj_events

    # Calculate relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes,
        flags,
        norm=vol_norm,
        exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=vol_norm
    )

    # Organize the data
    ## Subselect present spines
    all_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
        all_dendrite_coactivity_rate,
        spine_idxs,
    )
    all_shuff_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
        all_shuff_dendrite_coactivity_rate,
        spine_idxs,
    )
    all_fraction_dendrite_coactive = d_utils.subselect_data_by_idxs(
        all_fraction_dendrite_coactive, spine_idxs
    )
    all_fraction_spine_coactive = d_utils.subselect_data_by_idxs(
        all_fraction_spine_coactive,
        spine_idxs,
    )
    nonconj_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
        nonconj_dendrite_coactivity_rate,
        spine_idxs,
    )
    nonconj_shuff_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
        nonconj_shuff_dendrite_coactivity_rate,
        spine_idxs,
    )
    nonconj_fraction_dendrite_coactive = d_utils.subselect_data_by_idxs(
        nonconj_fraction_dendrite_coactive,
        spine_idxs,
    )
    nonconj_fraction_spine_coactive = d_utils.subselect_data_by_idxs(
        nonconj_fraction_spine_coactive,
        spine_idxs,
    )
    conj_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
        conj_dendrite_coactivity_rate,
        spine_idxs,
    )
    conj_shuff_dendrite_coactivity_rate = d_utils.subselect_data_by_idxs(
        conj_shuff_dendrite_coactivity_rate,
        spine_idxs,
    )
    conj_fraction_dendrite_coactive = d_utils.subselect_data_by_idxs(
        conj_fraction_dendrite_coactive,
        spine_idxs,
    )
    conj_fraction_spine_coactive = d_utils.subselect_data_by_idxs(
        conj_fraction_spine_coactive,
        spine_idxs,
    )
    fraction_conj_events = d_utils.subselect_data_by_idxs(
        fraction_conj_events,
        spine_idxs,
    )
    mvmt_spines = d_utils.subselect_data_by_idxs(dataset.movement_spines, spine_idxs)
    nonmvmt_spines = d_utils.subselect_data_by_idxs(
        dataset.nonmovement_spines, spine_idxs
    )

    ## Seperate into groups
    all_coactivity_rate = {}
    all_shuff_coactivity_rate = {}
    all_dend_fraction = {}
    all_spine_fraction = {}
    nonconj_coactivity_rate = {}
    nonconj_shuff_coactivity_rate = {}
    nonconj_dend_fraction = {}
    nonconj_spine_fraction = {}
    conj_coactivity_rate = {}
    conj_shuff_coactivity_rate = {}
    conj_dend_fraction = {}
    conj_spine_fraction = {}
    frac_conj_events = {}

    for key, value in plastic_groups.items():
        spines = eval(value)
        if MRSs == "MRS":
            spines = spines * mvmt_spines
        elif MRSs == "nMRS":
            spines = spines * nonmvmt_spines
        all_coactivity_rate[key] = all_dendrite_coactivity_rate[spines]
        all_shuff_coactivity_rate[key] = all_shuff_dendrite_coactivity_rate[:, spines]
        all_dend_fraction[key] = all_fraction_dendrite_coactive[spines]
        all_spine_fraction[key] = all_fraction_spine_coactive[spines]
        nonconj_coactivity_rate[key] = nonconj_dendrite_coactivity_rate[spines]
        nonconj_shuff_coactivity_rate[key] = nonconj_shuff_dendrite_coactivity_rate[
            :, spines
        ]
        nonconj_dend_fraction[key] = nonconj_fraction_dendrite_coactive[spines]
        nonconj_spine_fraction[key] = nonconj_fraction_spine_coactive[spines]
        conj_coactivity_rate[key] = conj_dendrite_coactivity_rate[spines]
        conj_shuff_coactivity_rate[key] = conj_shuff_dendrite_coactivity_rate[:, spines]
        conj_dend_fraction[key] = conj_fraction_dendrite_coactive[spines]
        conj_spine_fraction[key] = conj_fraction_spine_coactive[spines]
        frac_conj_events[key] = fraction_conj_events[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABFG
        CDE.
        HIMN
        JKL.
        OPTU
        QRSV
        """,
        figsize=figsize,
    )

    fig.suptitle(f"{main_title} Plastic {nname} Coactivity Rates")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ###################### Plot data onto axes #############################
    # All coactivity scatter
    plot_scatter_correlation(
        x_var=all_dendrite_coactivity_rate,
        y_var=delta_volume,
        CI=95,
        title="All events",
        xtitle=f"{coactivity_title}",
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
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # nonconj coactivity scatter
    plot_scatter_correlation(
        x_var=nonconj_dendrite_coactivity_rate,
        y_var=delta_volume,
        CI=95,
        title="Nonconj",
        xtitle=f"{coactivity_title}",
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
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    # conj coactivity scatter
    plot_scatter_correlation(
        x_var=conj_dendrite_coactivity_rate,
        y_var=delta_volume,
        CI=95,
        title="Conj",
        xtitle=f"{coactivity_title}",
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
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    # All coactivity box plot
    plot_box_plot(
        all_coactivity_rate,
        figsize=(5, 5),
        title="All events",
        xtitle=None,
        ytitle=f"{coactivity_title}",
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
    # Nonconj coactivity box plot
    plot_box_plot(
        nonconj_coactivity_rate,
        figsize=(5, 5),
        title="Nonconj",
        xtitle=None,
        ytitle=f"{coactivity_title}",
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
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    # Conj coactivity box plot
    plot_box_plot(
        conj_coactivity_rate,
        figsize=(5, 5),
        title="Conj",
        xtitle=None,
        ytitle=f"{coactivity_title}",
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
        ax=axes["P"],
        save=False,
        save_path=None,
    )
    # All events spine fraction
    plot_box_plot(
        all_spine_fraction,
        figsize=(5, 5),
        title="All events",
        xtitle=None,
        ytitle=f"Fraction of spine activity",
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
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    # Nonconj events spine fraction
    plot_box_plot(
        nonconj_spine_fraction,
        figsize=(5, 5),
        title="Nonconj",
        xtitle=None,
        ytitle=f"Fraction of spine activity",
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
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    # Conj events spine fraction
    plot_box_plot(
        conj_spine_fraction,
        figsize=(5, 5),
        title="Conj",
        xtitle=None,
        ytitle=f"Fraction of spine activity",
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
        ax=axes["T"],
        save=False,
        save_path=None,
    )
    # All events dendrite fraction
    plot_box_plot(
        all_dend_fraction,
        figsize=(5, 5),
        title="All events",
        xtitle=None,
        ytitle=f"Fraction of dendrite activity",
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
    # Nonconj events dendrite fraction
    plot_box_plot(
        nonconj_dend_fraction,
        figsize=(5, 5),
        title="Nonconj",
        xtitle=None,
        ytitle=f"Fraction of dendrite activity",
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
        ax=axes["N"],
        save=False,
        save_path=None,
    )
    # Conj events dendrite fraction
    plot_box_plot(
        conj_dend_fraction,
        figsize=(5, 5),
        title="Conj events",
        xtitle=None,
        ytitle=f"Fraction of dendrite activity",
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
        ax=axes["U"],
        save=False,
        save_path=None,
    )
    # Fraction conj events
    plot_box_plot(
        frac_conj_events,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle=f"Fraction of events\nwith local coactivity",
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
    # All enlarged local vs chance
    ## cum distribution
    plot_cummulative_distribution(
        data=list((all_coactivity_rate["sLTP"], all_shuff_coactivity_rate["sLTP"])),
        plot_ind=True,
        title="Enlarged",
        xtitle=f"{coactivity_title}",
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
            "data": all_coactivity_rate["sLTP"],
            "shuff": all_shuff_coactivity_rate["sLTP"].flatten().astype(np.float32),
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
        ax=ax_c_inset,
        save=False,
        save_path=None,
    )
    # All shrunken local vs chance
    ## cum distribution
    plot_cummulative_distribution(
        data=list((all_coactivity_rate["sLTD"], all_shuff_coactivity_rate["sLTD"])),
        plot_ind=True,
        title="Shrunken",
        xtitle=f"{coactivity_title}",
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
            "data": all_coactivity_rate["sLTD"],
            "shuff": all_shuff_coactivity_rate["sLTD"].flatten().astype(np.float32),
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
        ax=ax_d_inset,
        save=False,
        save_path=None,
    )
    # All stable local vs chance
    ## cum distribution
    plot_cummulative_distribution(
        data=list((all_coactivity_rate["Stable"], all_shuff_coactivity_rate["Stable"])),
        plot_ind=True,
        title="Stable",
        xtitle=f"{coactivity_title}",
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
            "data": all_coactivity_rate["Stable"],
            "shuff": all_shuff_coactivity_rate["Stable"].flatten().astype(np.float32),
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
        ax=ax_e_inset,
        save=False,
        save_path=None,
    )
    # Nonconj enlarged local vs chance
    ## cum distribution
    plot_cummulative_distribution(
        data=list(
            (
                nonconj_coactivity_rate["sLTP"],
                nonconj_shuff_coactivity_rate["sLTP"],
            )
        ),
        plot_ind=True,
        title="Enlarged",
        xtitle=f"{coactivity_title}",
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
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_j_inset = axes["J"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_j_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": nonconj_coactivity_rate["sLTP"],
            "shuff": nonconj_shuff_coactivity_rate["sLTP"].flatten().astype(np.float32),
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
        ax=ax_j_inset,
        save=False,
        save_path=None,
    )
    # nonconj shrunken local vs chance
    ## cum distribution
    plot_cummulative_distribution(
        data=list(
            (
                nonconj_coactivity_rate["sLTD"],
                nonconj_shuff_coactivity_rate["sLTD"],
            )
        ),
        plot_ind=True,
        title="Shrunken",
        xtitle=f"{coactivity_title}",
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_k_inset = axes["K"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_k_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": nonconj_coactivity_rate["sLTD"],
            "shuff": nonconj_shuff_coactivity_rate["sLTD"].flatten().astype(np.float32),
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
        ax=ax_k_inset,
        save=False,
        save_path=None,
    )
    # Nonconj stable local vs chance
    ## cum distribution
    plot_cummulative_distribution(
        data=list(
            (nonconj_coactivity_rate["Stable"], nonconj_shuff_coactivity_rate["Stable"])
        ),
        plot_ind=True,
        title="Stable",
        xtitle=f"{coactivity_title}",
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
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_l_inset = axes["L"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_l_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": nonconj_coactivity_rate["Stable"],
            "shuff": nonconj_shuff_coactivity_rate["Stable"]
            .flatten()
            .astype(np.float32),
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
        ax=ax_l_inset,
        save=False,
        save_path=None,
    )
    # Conj enlarged local vs chance
    ## cum distribution
    plot_cummulative_distribution(
        data=list((conj_coactivity_rate["sLTP"], conj_shuff_coactivity_rate["sLTP"])),
        plot_ind=True,
        title="Enlarged",
        xtitle=f"{coactivity_title}",
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
        ax=axes["Q"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_q_inset = axes["Q"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_q_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": conj_coactivity_rate["sLTP"],
            "shuff": conj_shuff_coactivity_rate["sLTP"].flatten().astype(np.float32),
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
        ax=ax_q_inset,
        save=False,
        save_path=None,
    )
    # Conj shrunken local vs chance
    ## cum distribution
    plot_cummulative_distribution(
        data=list((conj_coactivity_rate["sLTD"], conj_shuff_coactivity_rate["sLTD"])),
        plot_ind=True,
        title="Shrunken",
        xtitle=f"{coactivity_title}",
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
        ax=axes["R"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_r_inset = axes["R"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_r_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": conj_coactivity_rate["sLTD"],
            "shuff": conj_shuff_coactivity_rate["sLTD"].flatten().astype(np.float32),
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
        ax=ax_r_inset,
        save=False,
        save_path=None,
    )
    # All stable local vs chance
    ## cum distribution
    plot_cummulative_distribution(
        data=list(
            (conj_coactivity_rate["Stable"], conj_shuff_coactivity_rate["Stable"])
        ),
        plot_ind=True,
        title="Stable",
        xtitle=f"{coactivity_title}",
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
        ax=axes["S"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_s_inset = axes["S"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_s_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": conj_coactivity_rate["Stable"],
            "shuff": conj_shuff_coactivity_rate["Stable"].flatten().astype(np.float32),
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
        ax=ax_s_inset,
        save=False,
        save_path=None,
    )

    fig.tight_layout()
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if MRSs is not None:
            mrs_name = f"{MRSs}_"
        else:
            mrs_name = ""
        if norm == False:
            if period == "All periods":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Rate_{mrs_name}Figure"
                )
            elif period == "movement":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Rate_Mvmt_{mrs_name}Figure"
                )
            elif period == "nonmovement":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Rate_Nonmvmt_{mrs_name}Figure"
                )
        else:
            if period == "All periods":
                fname = os.path.join(
                    save_path, f"Dendritic_Norm_Coactivity_Rate_{mrs_name}Figure"
                )
            elif period == "movement":
                fname = os.path.join(
                    save_path, f"Dendritic_Norm_Coactivity_Rate_Mvmt_{mrs_name}Figure"
                )
            elif period == "nonmovement":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Norm_Coactivity_Rate_Nonmvmt_{mrs_name}Figure",
                )
        fig.savefig(fname + ".pdf")

    ########################### Statistics Section ############################
    if display_stats == False:
        return

    # Perform f-tests
    if test_type == "parametric":
        (
            all_coactivity_f,
            all_coactivity_p,
            _,
            all_coactivity_df,
        ) = t_utils.ANOVA_1way_posthoc(
            all_coactivity_rate,
            test_method,
        )
        all_s_frac_f, all_s_frac_p, _, all_s_frac_df = t_utils.ANOVA_1way_posthoc(
            all_spine_fraction, test_method
        )
        all_d_frac_f, all_d_frac_p, _, all_d_frac_df = t_utils.ANOVA_1way_posthoc(
            all_dend_fraction,
            test_method,
        )
        (
            nonconj_coactivity_f,
            nonconj_coactivity_p,
            _,
            nonconj_coactivity_df,
        ) = t_utils.ANOVA_1way_posthoc(
            nonconj_coactivity_rate,
            test_method,
        )
        (
            nonconj_s_frac_f,
            nonconj_s_frac_p,
            _,
            nonconj_s_frac_df,
        ) = t_utils.ANOVA_1way_posthoc(nonconj_spine_fraction, test_method)
        (
            nonconj_d_frac_f,
            nonconj_d_frac_p,
            _,
            nonconj_d_frac_df,
        ) = t_utils.ANOVA_1way_posthoc(
            nonconj_dend_fraction,
            test_method,
        )
        (
            conj_coactivity_f,
            conj_coactivity_p,
            _,
            conj_coactivity_df,
        ) = t_utils.ANOVA_1way_posthoc(
            conj_coactivity_rate,
            test_method,
        )
        conj_s_frac_f, conj_s_frac_p, _, conj_s_frac_df = t_utils.ANOVA_1way_posthoc(
            conj_spine_fraction, test_method
        )
        conj_d_frac_f, conj_d_frac_p, _, conj_d_frac_df = t_utils.ANOVA_1way_posthoc(
            conj_dend_fraction,
            test_method,
        )
        frac_conj_f, frac_conj_p, _, frac_conj_df = t_utils.ANOVA_1way_posthoc(
            frac_conj_events,
            test_method,
        )
        test_title = f"One-way ANOVA {test_method}"
    elif test_type == "nonparametric":
        (
            all_coactivity_f,
            all_coactivity_p,
            all_coactivity_df,
        ) = t_utils.kruskal_wallis_test(
            all_coactivity_rate,
            "Conover",
            test_method,
        )
        all_s_frac_f, all_s_frac_p, all_s_frac_df = t_utils.kruskal_wallis_test(
            all_spine_fraction, "Conover", test_method
        )
        all_d_frac_f, all_d_frac_p, all_d_frac_df = t_utils.kruskal_wallis_test(
            all_dend_fraction,
            "Conover",
            test_method,
        )
        (
            nonconj_coactivity_f,
            nonconj_coactivity_p,
            nonconj_coactivity_df,
        ) = t_utils.kruskal_wallis_test(
            nonconj_coactivity_rate,
            "Conover",
            test_method,
        )
        (
            nonconj_s_frac_f,
            nonconj_s_frac_p,
            nonconj_s_frac_df,
        ) = t_utils.kruskal_wallis_test(nonconj_spine_fraction, "Conover", test_method)
        (
            nonconj_d_frac_f,
            nonconj_d_frac_p,
            nonconj_d_frac_df,
        ) = t_utils.kruskal_wallis_test(
            nonconj_dend_fraction,
            "Conover",
            test_method,
        )
        (
            conj_coactivity_f,
            conj_coactivity_p,
            conj_coactivity_df,
        ) = t_utils.kruskal_wallis_test(
            conj_coactivity_rate,
            "Conover",
            test_method,
        )
        conj_s_frac_f, conj_s_frac_p, conj_s_frac_df = t_utils.kruskal_wallis_test(
            conj_spine_fraction, "Conover", test_method
        )
        conj_d_frac_f, conj_d_frac_p, conj_d_frac_df = t_utils.kruskal_wallis_test(
            conj_dend_fraction,
            "Conover",
            test_method,
        )
        frac_conj_f, frac_conj_p, frac_conj_df = t_utils.kruskal_wallis_test(
            frac_conj_events,
            "Conover",
            test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"

    # Comparisions to chance
    ## All events
    all_enlarged_above, all_enlarged_below = t_utils.test_against_chance(
        all_coactivity_rate["sLTP"], all_shuff_coactivity_rate["sLTP"]
    )
    all_shrunken_above, all_shrunken_below = t_utils.test_against_chance(
        all_coactivity_rate["sLTD"],
        all_shuff_coactivity_rate["sLTD"],
    )
    all_stable_above, all_stable_below = t_utils.test_against_chance(
        all_coactivity_rate["Stable"], all_shuff_coactivity_rate["Stable"]
    )
    all_chance_dict = {
        "Comparison": ["sLTP", "sLTD", "Stable"],
        "p-val above": [all_enlarged_above, all_shrunken_above, all_stable_above],
        "p-val below": [all_enlarged_below, all_shrunken_below, all_stable_below],
    }
    all_chance_df = pd.DataFrame.from_dict(all_chance_dict)
    ## Nonconj events
    nonconj_enlarged_above, nonconj_enlarged_below = t_utils.test_against_chance(
        nonconj_coactivity_rate["sLTP"], nonconj_shuff_coactivity_rate["sLTP"]
    )
    nonconj_shrunken_above, nonconj_shrunken_below = t_utils.test_against_chance(
        nonconj_coactivity_rate["sLTD"],
        nonconj_shuff_coactivity_rate["sLTD"],
    )
    nonconj_stable_above, nonconj_stable_below = t_utils.test_against_chance(
        nonconj_coactivity_rate["Stable"], nonconj_shuff_coactivity_rate["Stable"]
    )
    nonconj_chance_dict = {
        "Comparison": ["sLTP", "sLTD", "Stable"],
        "p-val above": [
            nonconj_enlarged_above,
            nonconj_shrunken_above,
            nonconj_stable_above,
        ],
        "p-val below": [
            nonconj_enlarged_below,
            nonconj_shrunken_below,
            nonconj_stable_below,
        ],
    }
    nonconj_chance_df = pd.DataFrame.from_dict(nonconj_chance_dict)
    ## Conj events
    conj_enlarged_above, conj_enlarged_below = t_utils.test_against_chance(
        conj_coactivity_rate["sLTP"], conj_shuff_coactivity_rate["sLTP"]
    )
    conj_shrunken_above, conj_shrunken_below = t_utils.test_against_chance(
        conj_coactivity_rate["sLTD"],
        conj_shuff_coactivity_rate["sLTD"],
    )
    conj_stable_above, conj_stable_below = t_utils.test_against_chance(
        conj_coactivity_rate["Stable"], conj_shuff_coactivity_rate["Stable"]
    )
    conj_chance_dict = {
        "Comparison": ["sLTP", "sLTD", "Stable"],
        "p-val above": [conj_enlarged_above, conj_shrunken_above, conj_stable_above],
        "p-val below": [conj_enlarged_below, conj_shrunken_below, conj_stable_below],
    }
    conj_chance_df = pd.DataFrame.from_dict(conj_chance_dict)

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        ABC
        DEF
        GHI
        JKL
        M..
        """,
        figsize=(12, 10),
    )
    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(
        f"All events Coactivity\n{test_title}\nF = {all_coactivity_f:.4} p = {all_coactivity_p:.3E}"
    )
    A_table = axes2["A"].table(
        cellText=all_coactivity_df.values,
        colLabels=all_coactivity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(
        f"Nonconj events Coactivity\n{test_title}\nF = {nonconj_coactivity_f:.4} p = {nonconj_coactivity_p:.3E}"
    )
    B_table = axes2["B"].table(
        cellText=nonconj_coactivity_df.values,
        colLabels=nonconj_coactivity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(
        f"Conj events Coactivity\n{test_title}\nF = {conj_coactivity_f:.4} p = {conj_coactivity_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=conj_coactivity_df.values,
        colLabels=conj_coactivity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(f"All coactivity vs chance")
    D_table = axes2["D"].table(
        cellText=all_chance_df.values,
        colLabels=all_chance_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(f"Nonconj coactivity vs chance")
    E_table = axes2["E"].table(
        cellText=nonconj_chance_df.values,
        colLabels=nonconj_chance_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title(f"Conj coactivity vs chance")
    F_table = axes2["F"].table(
        cellText=conj_chance_df.values,
        colLabels=conj_chance_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)
    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title(
        f"All spine fraction\n{test_title}\nF = {all_s_frac_f:.4} p = {all_s_frac_p:.3E}"
    )
    G_table = axes2["G"].table(
        cellText=all_s_frac_df.values,
        colLabels=all_s_frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title(
        f"Nonconj spine fraction\n{test_title}\nF = {nonconj_s_frac_f:.4} p = {nonconj_s_frac_p:.3E}"
    )
    H_table = axes2["H"].table(
        cellText=nonconj_s_frac_df.values,
        colLabels=nonconj_s_frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)
    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title(
        f"Conj spine fraction\n{test_title}\nF = {conj_s_frac_f:.4} p = {conj_s_frac_p:.3E}"
    )
    I_table = axes2["I"].table(
        cellText=conj_s_frac_df.values,
        colLabels=conj_s_frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)
    axes2["J"].axis("off")
    axes2["J"].axis("tight")
    axes2["J"].set_title(
        f"All dendrite fraction\n{test_title}\nF = {all_d_frac_f:.4} p = {all_d_frac_p:.3E}"
    )
    J_table = axes2["J"].table(
        cellText=all_d_frac_df.values,
        colLabels=all_d_frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    J_table.auto_set_font_size(False)
    J_table.set_fontsize(8)
    axes2["K"].axis("off")
    axes2["K"].axis("tight")
    axes2["K"].set_title(
        f"Nonconj dendrite fraction\n{test_title}\nF = {nonconj_d_frac_f:.4} p = {nonconj_d_frac_p:.3E}"
    )
    K_table = axes2["K"].table(
        cellText=nonconj_d_frac_df.values,
        colLabels=nonconj_d_frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    K_table.auto_set_font_size(False)
    K_table.set_fontsize(8)
    axes2["L"].axis("off")
    axes2["L"].axis("tight")
    axes2["L"].set_title(
        f"Conj dendrite fraction\n{test_title}\nF = {conj_d_frac_f:.4} p = {conj_d_frac_p:.3E}"
    )
    L_table = axes2["L"].table(
        cellText=conj_d_frac_df.values,
        colLabels=conj_d_frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    L_table.auto_set_font_size(False)
    L_table.set_fontsize(8)
    axes2["M"].axis("off")
    axes2["M"].axis("tight")
    axes2["M"].set_title(
        f"Fraction conj events\n{test_title}\nF = {frac_conj_f:.4} p = {frac_conj_p:.3E}"
    )
    M_table = axes2["M"].table(
        cellText=frac_conj_df.values,
        colLabels=frac_conj_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    M_table.auto_set_font_size(False)
    M_table.set_fontsize(8)

    fig2.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if MRSs is not None:
            mrs_name = f"{MRSs}_"
        else:
            mrs_name = ""
        if norm == False:
            if period == "All periods":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Rate_{mrs_name}Stats"
                )
            elif period == "movement":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Rate_Mvmt_{mrs_name}Stats"
                )
            elif period == "nonmovement":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Rate_Nonmvmt_{mrs_name}Stats"
                )
        else:
            if period == "All periods":
                fname = os.path.join(
                    save_path, f"Dendritic_Norm_Coactivity_Rate_{mrs_name}Stats"
                )
            elif period == "movement":
                fname = os.path.join(
                    save_path, f"Dendritic_Norm_Coactivity_Rate_Mvmt_{mrs_name}Stats"
                )
            elif period == "nonmovement":
                fname = os.path.join(
                    save_path, f"Dendritic_Norm_Coactivity_Rate_Nonmvmt_{mrs_name}Stats"
                )
        fig2.savefig(fname + ".pdf")


def plot_coactive_event_properties(
    dataset,
    event_type="All",
    period="All periods",
    followup_dataset=None,
    exclude="Shaft Spine",
    MRSs=None,
    threshold=0.3,
    figsize=(10, 10),
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
        dataset - Dendritic_Coactivity_Data object

        event_type - str specifying whether to analyze all events, conj, or nonconj
                    events

        period - str specifying whether the dataset in constrained to a particular
                movement period

        followup_data - optional Dendritic_Coactivity_Data object of the subsequent
                        session to be used for volume comparision. Default is None
                        to use the followup volumes in the dataset

        exclude - str specifying spine type to exclude form analysis

        MRSs - str specifying if you wish to examine only MRSs or nonMRSs. Accepts
                "MRS" and "nMRS". Defualt is None to examine all spines

        threshold - float or tuple of floats specifying the threshold cutoff
                    for classifying plasticity

        showmeans - boolean specifying whether to plot means on boxplots

        test_type - str specifying whetehr to perform parametric or nonparametric tests

        test_method - str specifying the type of posthoc test to perform

        display_stats - boolean specifying whehter to display stats

        save - boolean specifying whether to save the figures or not

        save_path - str specifying where to save the figures
    """
    COLORS = ["mediumslateblue", "tomato", "silver"]
    plastic_groups = {
        "sLTP": "enlarged_spines",
        "sLTD": "shrunken_spines",
        "Stable": "stable_spines",
    }
    if event_type == "All":
        event_title = "All events"
    elif event_type == "nonconj":
        event_title = "Without local coactivity"
    elif event_type == "conj":
        event_title = "With local coactivity"
    if period == "All periods":
        period_title = "Entire session"
    elif period == "movement":
        period_title = "Movement periods"
    elif period == "nonmovement":
        period_title = "Nonmovement periods"

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

    ## Coactivity related variables
    if event_type == "All":
        spine_coactive_traces = dataset.all_spine_coactive_traces
        spine_coactive_calcium_traces = dataset.all_spine_coactive_calcium_traces
        dendrite_coactive_traces = dataset.all_dendrite_coactive_traces
        spine_coactive_amplitude = dataset.all_spine_coactive_amplitude
        spine_coactive_calcium_amplitude = dataset.all_spine_coactive_calcium_amplitude
        dendrite_coactive_amplitude = dataset.all_dendrite_coactive_amplitude
        relative_onsets = dataset.all_relative_onsets
    elif event_type == "nonconj":
        spine_coactive_traces = dataset.nonconj_spine_coactive_traces
        spine_coactive_calcium_traces = dataset.nonconj_spine_coactive_calcium_traces
        dendrite_coactive_traces = dataset.nonconj_dendrite_coactive_traces
        spine_coactive_amplitude = dataset.nonconj_spine_coactive_amplitude
        spine_coactive_calcium_amplitude = (
            dataset.nonconj_spine_coactive_calcium_amplitude
        )
        dendrite_coactive_amplitude = dataset.nonconj_dendrite_coactive_amplitude
        relative_onsets = dataset.nonconj_relative_onsets
    elif event_type == "conj":
        spine_coactive_traces = dataset.conj_spine_coactive_traces
        spine_coactive_calcium_traces = dataset.conj_spine_coactive_calcium_traces
        dendrite_coactive_traces = dataset.conj_dendrite_coactive_traces
        spine_coactive_amplitude = dataset.conj_spine_coactive_amplitude
        spine_coactive_calcium_amplitude = dataset.conj_spine_coactive_calcium_amplitude
        dendrite_coactive_amplitude = dataset.conj_dendrite_coactive_amplitude
        relative_onsets = dataset.conj_relative_onsets

    # Calculate relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes,
        flags,
        norm=vol_norm,
        exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=vol_norm
    )

    # Subselect present spines
    spine_coactive_traces = d_utils.subselect_data_by_idxs(
        spine_coactive_traces,
        spine_idxs,
    )
    spine_coactive_calcium_traces = d_utils.subselect_data_by_idxs(
        spine_coactive_calcium_traces,
        spine_idxs,
    )
    dendrite_coactive_traces = d_utils.subselect_data_by_idxs(
        dendrite_coactive_traces,
        spine_idxs,
    )
    spine_coactive_amplitude = d_utils.subselect_data_by_idxs(
        spine_coactive_amplitude,
        spine_idxs,
    )
    spine_coactive_calcium_amplitude = d_utils.subselect_data_by_idxs(
        spine_coactive_calcium_amplitude,
        spine_idxs,
    )
    dendrite_coactive_amplitude = d_utils.subselect_data_by_idxs(
        dendrite_coactive_amplitude,
        spine_idxs,
    )
    relative_onsets = d_utils.subselect_data_by_idxs(
        relative_onsets,
        spine_idxs,
    )
    mvmt_spines = d_utils.subselect_data_by_idxs(dataset.movement_spines, spine_idxs)
    nonmvmt_spines = d_utils.subselect_data_by_idxs(
        dataset.nonmovement_spines, spine_idxs
    )

    # Seperate into dicts for plotting
    hmap_traces = {}
    plastic_trace_means = {}
    plastic_trace_sems = {}
    plastic_ca_trace_means = {}
    plastic_ca_trace_sems = {}
    plastic_dend_means = {}
    plastic_dend_sems = {}
    plastic_amps = {}
    plastic_ca_amps = {}
    plastic_dend_amps = {}
    plastic_onsets = {}

    for key, value in plastic_groups.items():
        spines = eval(value)
        if MRSs == "MRS":
            spines = spines * mvmt_spines
        elif MRSs == "nMRS":
            spines = spines * nonmvmt_spines
        trace_means = compress(spine_coactive_traces, spines)
        trace_means = [
            np.nanmean(x, axis=1) for x in trace_means if type(x) == np.ndarray
        ]
        trace_means = np.vstack(trace_means)
        hmap_traces[key] = trace_means.T
        plastic_trace_means[key] = np.nanmean(trace_means, axis=0)
        plastic_trace_sems[key] = stats.sem(trace_means, axis=0, nan_policy="omit")
        ca_means = compress(spine_coactive_calcium_traces, spines)
        ca_means = [np.nanmean(x, axis=1) for x in ca_means if type(x) == np.ndarray]
        ca_means = np.vstack(ca_means)
        plastic_ca_trace_means[key] = np.nanmean(ca_means, axis=0)
        plastic_ca_trace_sems[key] = stats.sem(ca_means, axis=0, nan_policy="omit")
        dend_means = compress(dendrite_coactive_traces, spines)
        dend_means = [
            np.nanmean(x, axis=1) for x in dend_means if type(x) == np.ndarray
        ]
        dend_means = np.vstack(dend_means)
        plastic_dend_means[key] = np.nanmean(dend_means, axis=0)
        plastic_dend_sems[key] = stats.sem(dend_means, axis=0, nan_policy="omit")
        plastic_amps[key] = spine_coactive_amplitude[spines]
        plastic_ca_amps[key] = spine_coactive_calcium_amplitude[spines]
        plastic_dend_amps[key] = dendrite_coactive_amplitude[spines]
        plastic_onsets[key] = relative_onsets[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE
        FGHI.
        JK...
        """,
        figsize=figsize,
    )
    fig.suptitle(f"{period_title} {event_title} Coactivity Properties")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ###################### Plot data onto axes ##########################
    # Enlarged heatmap
    plot_activity_heatmap(
        hmap_traces["sLTP"],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Enlarged",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="Blues",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # Shrunken heatmap
    plot_activity_heatmap(
        hmap_traces["sLTD"],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Shrunken",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="Reds",
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
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Stable",
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
    # Dendrite traces
    plot_mean_activity_traces(
        means=list(plastic_dend_means.values()),
        sems=list(plastic_dend_sems.values()),
        group_names=list(plastic_dend_means.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Dendrite",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
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
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Calcum amp box plot
    plot_box_plot(
        plastic_ca_amps,
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
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
    # Dendrite amp box plot
    plot_box_plot(
        plastic_dend_amps,
        figsize=(5, 5),
        title="Dendrite",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
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
        ax=axes["I"],
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )

    # Onset cummulative distribution
    plot_cummulative_distribution(
        data=list(plastic_onsets.values()),
        plot_ind=False,
        title="Relative onset",
        xtitle="Relative onset (s)",
        xlim=None,
        figsize=(5, 5),
        color=COLORS,
        ind_color=COLORS,
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

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if MRSs is not None:
            mrs_name = f"{MRSs}_"
        else:
            mrs_name = ""
        if period == "All periods":
            if event_type == "All":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Event_Props_{mrs_name}Figure"
                )
            elif event_type == "nonconj":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Nonconj_{mrs_name}Figure",
                )
            elif event_type == "conj":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Event_Props_Conj_{mrs_name}Figure"
                )
        elif period == "movement":
            if event_type == "All":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Event_Props_Mvmt_{mrs_name}Figure"
                )
            elif event_type == "nonconj":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Mvmt_Nonconj_{mrs_name}Figure",
                )
            elif event_type == "conj":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Mvmt_Conj_{mrs_name}Figure",
                )
        elif period == "nonmovement":
            if event_type == "All":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Nonmvmt_{mrs_name}Figure",
                )
            elif event_type == "nonconj":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Nonmvmt_Nonconj_{mrs_name}Figure",
                )
            elif event_type == "conj":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Nonmvmt_Conj_{mrs_name}Figure",
                )
        fig.savefig(fname + ".pdf")

    ######################## Statistics Section #########################
    if display_stats == False:
        return

    # Perform f-tests
    if test_type == "parametric":
        amp_f, amp_p, _, amp_df = t_utils.ANOVA_1way_posthoc(
            plastic_amps,
            test_method,
        )
        ca_amp_f, ca_amp_p, _, ca_amp_df = t_utils.ANOVA_1way_posthoc(
            plastic_ca_amps,
            test_method,
        )
        dend_amp_f, dend_amp_p, _, dend_amp_df = t_utils.ANOVA_1way_posthoc(
            plastic_dend_amps,
            test_method,
        )
        onset_f, onset_p, _, onset_df = t_utils.ANOVA_1way_posthoc(
            plastic_onsets,
            test_method,
        )
        test_title = f"One-way ANOVA {test_method}"
    elif test_type == "nonparametric":
        amp_f, amp_p, amp_df = t_utils.kruskal_wallis_test(
            plastic_amps,
            "Conover",
            test_method,
        )
        ca_amp_f, ca_amp_p, ca_amp_df = t_utils.kruskal_wallis_test(
            plastic_ca_amps,
            "Conover",
            test_method,
        )
        dend_amp_f, dend_amp_p, dend_amp_df = t_utils.kruskal_wallis_test(
            plastic_dend_amps,
            "Conover",
            test_method,
        )
        onset_f, onset_p, onset_df = t_utils.kruskal_wallis_test(
            plastic_onsets,
            "Conover",
            test_method,
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
    axes2["A"].set_title(f"GluSnFr amp\n{test_title}\nF = {amp_f:.4} p = {amp_p:.3E}")
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
        f"Calcium amp\n{test_title}\nF = {ca_amp_f:.4} p = {ca_amp_p:.3E}"
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
        f"Dendrite amp\n{test_title}\nF = {dend_amp_f:.4} p = {dend_amp_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=dend_amp_df.values,
        colLabels=dend_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(
        f"Relative onset\n{test_title}\nF = {onset_f:.4} p = {onset_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=onset_df.values,
        colLabels=onset_df.columns,
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
        if MRSs is not None:
            mrs_name = f"{MRSs}_"
        else:
            mrs_name = ""
        if period == "All periods":
            if event_type == "All":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Event_Props_{mrs_name}Stats"
                )
            elif event_type == "nonconj":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Nonconj_{mrs_name}Stats",
                )
            elif event_type == "conj":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Event_Props_Conj_{mrs_name}Stats"
                )
        elif period == "movement":
            if event_type == "All":
                fname = os.path.join(
                    save_path, f"Dendritic_Coactivity_Event_Props_Mvmt_{mrs_name}Stats"
                )
            elif event_type == "nonconj":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Mvmt_Nonconj_{mrs_name}Stats",
                )
            elif event_type == "conj":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Mvmt_Conj_{mrs_name}Stats",
                )
        elif period == "nonmovement":
            if event_type == "All":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Nonmvmt_{mrs_name}Stats",
                )
            elif event_type == "nonconj":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Nonmvmt_Nonconj_{mrs_name}Stats",
                )
            elif event_type == "conj":
                fname = os.path.join(
                    save_path,
                    f"Dendritic_Coactivity_Event_Props_Nonmvmt_Conj_{mrs_name}Stats",
                )
        fig2.savefig(fname + ".pdf")


def plot_nearby_spine_conj_activity(
    dataset,
    followup_dataset=None,
    period="All periods",
    exclude="Shaft",
    MRSs=None,
    threshold=0.3,
    figsize=(10, 6),
    hist_bins=30,
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """Function to plot the activity of nearby spines during conj events

    INPUT PARAMETERS
        dataset - Local_Coactivity_Data object analyzed over all periods

        followup_dataset - optional Local_Coactivity_Data object of the
                            subsequent session to use for volume comparision.
                            Default is None to use the followup_volumes in the
                            dataset

        period - str specifying what mvmt period the data is constrained to.
                Accepts "All periods", "movement", "nonmovement", "rewarded movement"

        exclude - str specifying the types of spines to exclude from volume assessment

        MRSs - str specifying if you wish to examine only MRSs or nonMRSs. Accepts
                "MRS" and "nMRS". Defualt is None to examine all spines

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
    COLORS = ["mediumslateblue", "tomato", "silver"]
    plastic_groups = {
        "sLTP": "enlarged_spines",
        "sLTD": "shrunken_spines",
        "Stable": "stable_spines",
    }

    # pull relevant data
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
        followup_flags = dataset.spine_flags

    # Amplitude and onset-related variables
    conj_coactive_spine_num = dataset.conj_coactive_spine_num
    conj_nearby_coactive_spine_amplitude = dataset.conj_nearby_coactive_spine_amplitude
    conj_nearby_coactive_spine_calcium = dataset.conj_nearby_coactive_spine_calcium
    conj_nearby_spine_onset = dataset.conj_nearby_spine_onset
    conj_nearby_spine_onset_jitter = dataset.conj_nearby_spine_onset_jitter
    conj_nearby_coactive_spine_traces = dataset.conj_nearby_coactive_spine_traces
    conj_nearby_coactive_spine_calcium_traces = (
        dataset.conj_nearby_coactive_spine_calcium_traces
    )

    # Calculate relative volumes
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

    # Subselect present spines
    conj_coactive_spine_num = d_utils.subselect_data_by_idxs(
        conj_coactive_spine_num, spine_idxs
    )
    conj_nearby_coactive_spine_amplitude = d_utils.subselect_data_by_idxs(
        conj_nearby_coactive_spine_amplitude,
        spine_idxs,
    )
    conj_nearby_coactive_spine_calcium = d_utils.subselect_data_by_idxs(
        conj_nearby_coactive_spine_calcium,
        spine_idxs,
    )
    conj_nearby_spine_onset = d_utils.subselect_data_by_idxs(
        conj_nearby_spine_onset,
        spine_idxs,
    )
    conj_nearby_spine_onset_jitter = d_utils.subselect_data_by_idxs(
        conj_nearby_spine_onset_jitter,
        spine_idxs,
    )
    conj_nearby_coactive_spine_traces = d_utils.subselect_data_by_idxs(
        conj_nearby_coactive_spine_traces,
        spine_idxs,
    )
    conj_nearby_coactive_spine_calcium_traces = d_utils.subselect_data_by_idxs(
        conj_nearby_coactive_spine_calcium_traces,
        spine_idxs,
    )
    mvmt_spines = d_utils.subselect_data_by_idxs(dataset.movement_spines, spine_idxs)
    nonmvmt_spines = d_utils.subselect_data_by_idxs(
        dataset.nonmovement_spines, spine_idxs
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
    plastic_num = {}

    for key, value in plastic_groups.items():
        spines = eval(value)
        if MRSs == "MRS":
            spines = spines * mvmt_spines
        elif MRSs == "nMRS":
            spines = spines * nonmvmt_spines
        traces = compress(conj_nearby_coactive_spine_traces, spines)
        traces = [np.nanmean(x, axis=1) for x in traces if type(x) == np.ndarray]
        traces = np.vstack(traces)
        hmap_traces[key] = traces.T
        plastic_trace_means[key] = np.nanmean(traces, axis=0)
        plastic_trace_sems[key] = stats.sem(traces, axis=0, nan_policy="omit")
        ca_traces = compress(conj_nearby_coactive_spine_calcium_traces, spines)
        ca_traces = [np.nanmean(x, axis=1) for x in ca_traces if type(x) == np.ndarray]
        ca_traces = np.vstack(ca_traces)
        plastic_ca_trace_means[key] = np.nanmean(ca_traces, axis=0)
        plastic_ca_trace_sems[key] = stats.sem(ca_traces, axis=0, nan_policy="omit")
        plastic_amps[key] = conj_nearby_coactive_spine_amplitude[spines]
        plastic_ca_amps[key] = conj_nearby_coactive_spine_calcium[spines]
        plastic_onsets[key] = conj_nearby_spine_onset[spines]
        plastic_onset_jitter[key] = conj_nearby_spine_onset_jitter[spines]
        plastic_num[key] = conj_coactive_spine_num[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDEF
        GHIJK.
        """,
        figsize=figsize,
    )
    fig.suptitle(f"{period} Nearby Spine Conj Activity")

    ########################## Plot data onto axes ##########################
    # Enlarged heatmap
    plot_activity_heatmap(
        hmap_traces["sLTP"],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Enlarged",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="Blues",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # Shrunken heatmap
    plot_activity_heatmap(
        hmap_traces["sLTD"],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Shrunken",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="Reds",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Stable heatmap
    plot_activity_heatmap(
        hmap_traces["Stable"],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Stable",
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
    # GluSnFr Traces
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
    # Calcium Traces
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
        ax=axes["G"],
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
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Coactive spine number box plot
    plot_box_plot(
        plastic_num,
        figsize=(5, 5),
        title="",
        xtitle=None,
        ytitle=f"Number of coactive spines",
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
        ax=axes["F"],
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
    # Onset histogram
    plot_histogram(
        data=list(plastic_onsets.values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="GluSnFr",
        xtitle="Relative onset (s)",
        xlim=(-2, 2),
        figsize=(5, 5),
        color=COLORS,
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
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
        ax=axes["J"],
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if MRSs is not None:
            mrs_name = f"{MRSs}_"
        else:
            mrs_name = ""
        if period == "All periods":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Conj_Activity_{mrs_name}Figure"
            )
        elif period == "movement":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Conj_Activity_Mvmt_{mrs_name}Figure"
            )
        elif period == "nonmovement":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Conj_Activity_Nonmvmt_{mrs_name}Figure"
            )

        fig.savefig(fname + ".pdf")

    ####################### Statistics Section ###########################
    if display_stats is False:
        return

    # Perform f-tests
    if test_type == "parametric":
        amp_f, amp_p, _, amp_df = t_utils.ANOVA_1way_posthoc(
            plastic_amps,
            test_method,
        )
        ca_amp_f, ca_amp_p, _, ca_amp_df = t_utils.ANOVA_1way_posthoc(
            plastic_ca_amps,
            test_method,
        )
        onset_f, onset_p, _, onset_df = t_utils.ANOVA_1way_posthoc(
            plastic_onsets,
            test_method,
        )
        jitter_f, jitter_p, _, jitter_df = t_utils.ANOVA_1way_posthoc(
            plastic_onset_jitter,
            test_method,
        )
        num_f, num_p, _, num_df = t_utils.ANOVA_1way_posthoc(
            plastic_num,
            test_method,
        )
        test_title = f"One-way ANOVA {test_method}"
    elif test_type == "nonparametric":
        amp_f, amp_p, amp_df = t_utils.kruskal_wallis_test(
            plastic_amps,
            "Conover",
            test_method,
        )
        ca_amp_f, ca_amp_p, ca_amp_df = t_utils.kruskal_wallis_test(
            plastic_ca_amps,
            "Conover",
            test_method,
        )
        onset_f, onset_p, onset_df = t_utils.kruskal_wallis_test(
            plastic_onsets,
            "Conover",
            test_method,
        )
        jitter_f, jitter_p, jitter_df = t_utils.kruskal_wallis_test(
            plastic_onset_jitter,
            "Conover",
            test_method,
        )
        num_f, num_p, num_df = t_utils.kruskal_wallis_test(
            plastic_num,
            "Conover",
            test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        E.
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
        f"Calcium Amplitude\n{test_title}\nF = {ca_amp_f:.4} p = {ca_amp_p:.3E}"
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
        f"Relative onsets\n{test_title}\nF = {onset_f:.4} p = {onset_p:.3E}"
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
        f"Onset jitter\n{test_title}\nF = {jitter_f:.4} p = {jitter_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=jitter_df.values,
        colLabels=jitter_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(
        f"Coactive number\n{test_title}\nF = {num_f:.4} p = {num_p:.3E}"
    )
    E_table = axes2["E"].table(
        cellText=num_df.values,
        colLabels=num_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if MRSs is not None:
            mrs_name = f"{MRSs}_"
        else:
            mrs_name = ""
        if period == "All periods":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Conj_Activity_{mrs_name}Figure"
            )
        elif period == "movement":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Conj_Activity_Mvmt_{mrs_name}Figure"
            )
        elif period == "nonmovement":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Conj_Activity_Nonmvmt_{mrs_name}Figure"
            )

        fig2.savefig(fname + ".pdf")


def plot_nearby_spine_properties(
    dataset,
    followup_dataset=None,
    period="All periods",
    exclude="Shaft",
    MRSs=None,
    threshold=0.3,
    figsize=(10, 15),
    mean_type="median",
    err_type="CI",
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """Function to plot nearby spine coactivity properties

    INPUT PARAMETERS
        dataset - Dendritic_Coactivity_Data object

        followup_dataset - optional Dendritic_Coactivity_Data object of the
                            following session to make volume comparisions

        period - str specifying if the dataset analyzed was constrained to a given
                period. Currently accepts "All periods", "movement", and "nonmovement"

        exclude - str specifying spine type to exclude from analysis

        MRSs - str specifying if you wish to examine only MRSs or nonMRSs. Accepts
                "MRS" and "nMRS". Default is None to examine all spines

        threshold - float or tuple of floats specifying the threshold cutoff for
                    classifying plasticity

        figsize - tuple specifying the size of the figure

        mean_type - str specifying the central tendency to display on bar plots

        err_type - str specifying the type of error to display on bar plots

        shomeans - boolean specifying whether to show the means on box plots

        test_type - str specifying whether to perform parametric or nonparametric tests

        test_method - str specifying the posthoc test to perform

        display_stats - boolean specifying whether to display stats or not

        vol_norm - boolean specifying whether to use normalized relative volumes

        save - boolean specifying whether to save the figure

        save_path - str specifying where to save the figure

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

    ## nearby spine variables
    coactivity_rate_distribution = dataset.cocativity_rate_distribution
    avg_nearby_spine_coactivity_rate = dataset.avg_nearby_spine_coactivity_rate
    shuff_nearby_spine_coactivity_rate = dataset.shuff_nearby_spine_coactivity_rate
    rel_nearby_spine_coactivity_rate = dataset.rel_nearby_spine_coactivity_rate
    conj_coactivity_rate_distribution = dataset.conj_coactivity_rate_distribution
    avg_nearby_spine_conj_rate = dataset.avg_nearby_spine_conj_rate
    shuff_nearby_spine_conj_rate = dataset.shuff_nearby_spine_conj_rate
    rel_nearby_spine_conj_rate = dataset.rel_nearby_spine_conj_rate
    spine_fraction_coactive_distribution = dataset.spine_fraction_coactive_distribution
    avg_nearby_spine_fraction = dataset.avg_nearby_spine_fraction
    shuff_nearby_spine_fraction = dataset.shuff_nearby_spine_fraction
    rel_spine_fraction = dataset.rel_spine_fraction
    dendrite_fraction_coactive_distribution = (
        dataset.dendrite_fraction_coactive_distribution
    )
    avg_nearby_dendrite_fraction = dataset.avg_nearby_dendrite_fraction
    shuff_nearby_dendrite_fraction = dataset.shuff_nearby_dendrite_fraction
    rel_dendrite_fraction = dataset.rel_dendrite_fraction
    relative_onset_distribution = dataset.relative_onset_distribution
    avg_nearby_relative_onset = dataset.avg_nearby_relative_onset
    shuff_nearby_relative_onset = dataset.shuff_nearby_relative_onset
    rel_nearby_relative_onset = dataset.rel_nearby_relative_onset

    # Calculate spine volume
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes,
        flags,
        norm=vol_norm,
        exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=vol_norm
    )

    # Subselect for present spines
    coactivity_rate_distribution = d_utils.subselect_data_by_idxs(
        coactivity_rate_distribution,
        spine_idxs,
    )
    avg_nearby_spine_coactivity_rate = d_utils.subselect_data_by_idxs(
        avg_nearby_spine_coactivity_rate,
        spine_idxs,
    )
    shuff_nearby_spine_coactivity_rate = d_utils.subselect_data_by_idxs(
        shuff_nearby_spine_coactivity_rate,
        spine_idxs,
    )
    rel_nearby_spine_coactivity_rate = d_utils.subselect_data_by_idxs(
        rel_nearby_spine_coactivity_rate,
        spine_idxs,
    )
    conj_coactivity_rate_distribution = d_utils.subselect_data_by_idxs(
        conj_coactivity_rate_distribution,
        spine_idxs,
    )
    avg_nearby_spine_conj_rate = d_utils.subselect_data_by_idxs(
        avg_nearby_spine_conj_rate,
        spine_idxs,
    )
    shuff_nearby_spine_conj_rate = d_utils.subselect_data_by_idxs(
        shuff_nearby_spine_conj_rate,
        spine_idxs,
    )
    rel_nearby_spine_conj_rate = d_utils.subselect_data_by_idxs(
        rel_nearby_spine_conj_rate,
        spine_idxs,
    )
    spine_fraction_coactive_distribution = d_utils.subselect_data_by_idxs(
        spine_fraction_coactive_distribution,
        spine_idxs,
    )
    avg_nearby_spine_fraction = d_utils.subselect_data_by_idxs(
        avg_nearby_spine_fraction,
        spine_idxs,
    )
    shuff_nearby_spine_fraction = d_utils.subselect_data_by_idxs(
        shuff_nearby_spine_fraction,
        spine_idxs,
    )
    rel_spine_fraction = d_utils.subselect_data_by_idxs(
        rel_spine_fraction,
        spine_idxs,
    )
    dendrite_fraction_coactive_distribution = d_utils.subselect_data_by_idxs(
        dendrite_fraction_coactive_distribution,
        spine_idxs,
    )
    avg_nearby_dendrite_fraction = d_utils.subselect_data_by_idxs(
        avg_nearby_dendrite_fraction,
        spine_idxs,
    )
    shuff_nearby_dendrite_fraction = d_utils.subselect_data_by_idxs(
        shuff_nearby_dendrite_fraction,
        spine_idxs,
    )
    rel_dendrite_fraction = d_utils.subselect_data_by_idxs(
        rel_dendrite_fraction,
        spine_idxs,
    )
    relative_onset_distribution = d_utils.subselect_data_by_idxs(
        relative_onset_distribution,
        spine_idxs,
    )
    avg_nearby_relative_onset = d_utils.subselect_data_by_idxs(
        avg_nearby_relative_onset,
        spine_idxs,
    )
    shuff_nearby_relative_onset = d_utils.subselect_data_by_idxs(
        shuff_nearby_relative_onset,
        spine_idxs,
    )
    rel_nearby_relative_onset = d_utils.subselect_data_by_idxs(
        rel_nearby_relative_onset,
        spine_idxs,
    )
    mvmt_spines = d_utils.subselect_data_by_idxs(dataset.movement_spines, spine_idxs)
    nonmvmt_spines = d_utils.subselect_data_by_idxs(dataset.movement_spines, spine_idxs)

    ## Seperate into plastic groups
    plastic_rate_dist = {}
    plastic_avg_rates = {}
    plastic_shuff_rates = {}
    plastic_rel_rates = {}
    plastic_conj_dist = {}
    plastic_avg_conj = {}
    plastic_shuff_conj = {}
    plastic_rel_conj = {}
    plastic_spine_frac_dist = {}
    plastic_avg_spine_frac = {}
    plastic_shuff_spine_frac = {}
    plastic_rel_spine_frac = {}
    plastic_dend_frac_dist = {}
    plastic_avg_dend_frac = {}
    plastic_shuff_dend_frac = {}
    plastic_rel_dend_frac = {}
    plastic_onset_dist = {}
    plastic_avg_onset = {}
    plastic_shuff_onset = {}
    plastic_rel_onset = {}
    distance_bins = dataset.parameters["position bins"][1:]

    for key, value in plastic_groups.items():
        spines = eval(value)
        if MRSs == "MRS":
            spines = spines * mvmt_spines
        elif MRSs == "nMRS":
            spines = spines * nonmvmt_spines
        plastic_rate_dist[key] = coactivity_rate_distribution[:, spines]
        plastic_avg_rates[key] = avg_nearby_spine_coactivity_rate[spines]
        plastic_shuff_rates[key] = shuff_nearby_spine_coactivity_rate[:, spines]
        plastic_rel_rates[key] = rel_nearby_spine_coactivity_rate[spines]
        plastic_conj_dist[key] = conj_coactivity_rate_distribution[:, spines]
        plastic_avg_conj[key] = avg_nearby_spine_conj_rate[spines]
        plastic_shuff_conj[key] = shuff_nearby_spine_conj_rate[:, spines]
        plastic_rel_conj[key] = rel_nearby_spine_conj_rate[spines]
        plastic_spine_frac_dist[key] = spine_fraction_coactive_distribution[:, spines]
        plastic_avg_spine_frac[key] = avg_nearby_spine_fraction[spines]
        plastic_shuff_spine_frac[key] = shuff_nearby_spine_fraction[:, spines]
        plastic_rel_spine_frac[key] = rel_spine_fraction[spines]
        plastic_dend_frac_dist[key] = dendrite_fraction_coactive_distribution[:, spines]
        plastic_avg_dend_frac[key] = avg_nearby_dendrite_fraction[spines]
        plastic_shuff_dend_frac[key] = shuff_nearby_dendrite_fraction[:, spines]
        plastic_rel_dend_frac[key] = rel_dendrite_fraction[spines]
        plastic_onset_dist[key] = relative_onset_distribution[:, spines]
        plastic_avg_onset[key] = avg_nearby_relative_onset[spines]
        plastic_shuff_onset[key] = shuff_nearby_relative_onset[:, spines]
        plastic_rel_onset[key] = rel_nearby_relative_onset[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDEF
        GHIJKL
        MNOPQR
        STUVWX
        YZabcd
        """,
        figsize=figsize,
    )
    fig.suptitle(f"{period} Nearby Spine Coactivity Properties")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ##################### Plot data onto axes ########################
    # Coactivity rate distribution
    plot_multi_line_plot(
        data_dict=plastic_rate_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Dendritic Coactivity Rate",
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
        ax=axes["A"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## Conj rate distribution
    plot_multi_line_plot(
        data_dict=plastic_conj_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Dendritic Conj Rate",
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
        ax=axes["G"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## Spine fraction distribution
    plot_multi_line_plot(
        data_dict=plastic_spine_frac_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Spine Fraction Coactive",
        ytitle="Fraction of spine events",
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
    ## Dendrite fraction distribution
    plot_multi_line_plot(
        data_dict=plastic_dend_frac_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Dendrite Fraction Coactive",
        ytitle="Fraction of dendrite events",
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
        ax=axes["S"],
        legend=True,
        save=False,
        save_path=None,
    )
    ## relative onset distribution
    plot_multi_line_plot(
        data_dict=plastic_onset_dist,
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        title="Relative onset",
        ytitle="Relative onset (s)",
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
        ax=axes["Y"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Ave local dendritic coactivity rate
    plot_box_plot(
        plastic_avg_rates,
        figsize=(5, 5),
        title="Nearby Coactivity Rate",
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
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Enlarged coactivity vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_rates["sLTP"], plastic_shuff_rates["sLTP"])),
        plot_ind=True,
        title="Enlarged",
        xtitle="Nearby Dendritic Coactivity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
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
        ax=ax_c_inset,
        save=False,
        save_path=None,
    )
    # Shrunken coactivity vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_rates["sLTD"], plastic_shuff_rates["sLTD"])),
        plot_ind=True,
        title="Shrunken",
        xtitle="Nearby Dendritic Coactivity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
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
        ax=ax_d_inset,
        save=False,
        save_path=None,
    )
    # Stable coactivity vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_rates["Stable"], plastic_shuff_rates["Stable"])),
        plot_ind=True,
        title="Stable",
        xtitle="Nearby Dendritic Coactivity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_e_inset = axes["E"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_c_inset)
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
        ax=ax_e_inset,
        save=False,
        save_path=None,
    )
    # Relative local dendritic coactivity rate
    plot_box_plot(
        plastic_rel_rates,
        figsize=(5, 5),
        title="Relative Coactivity Rate",
        xtitle=None,
        ytitle="Realative coactivity rate (events/min)",
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
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    # Ave local dendritic conj coactivity rate
    plot_box_plot(
        plastic_avg_conj,
        figsize=(5, 5),
        title="Nearby Conj Rate",
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
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    # Enlarged conj coactivity vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_conj["sLTP"], plastic_shuff_conj["sLTP"])),
        plot_ind=True,
        title="Enlarged",
        xtitle="Nearby Conj Coactivity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_i_inset = axes["I"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_i_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_conj["sLTP"],
            "shuff": plastic_shuff_conj["sLTP"].flatten().astype(np.float32),
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
        ax=ax_i_inset,
        save=False,
        save_path=None,
    )
    # Shrunken conj coactivity vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_conj["sLTD"], plastic_shuff_conj["sLTD"])),
        plot_ind=True,
        title="Shrunken",
        xtitle="Nearby Conj Coactivity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
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
            "data": plastic_avg_conj["sLTD"],
            "shuff": plastic_shuff_conj["sLTD"].flatten().astype(np.float32),
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
        ax=ax_j_inset,
        save=False,
        save_path=None,
    )
    # Stable conj coactivity vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_conj["Stable"], plastic_shuff_conj["Stable"])),
        plot_ind=True,
        title="Stable",
        xtitle="Nearby Conj Coactivity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_k_inset = axes["K"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_k_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_conj["Stable"],
            "shuff": plastic_shuff_conj["Stable"].flatten().astype(np.float32),
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
        ax=ax_k_inset,
        save=False,
        save_path=None,
    )
    # Relative local dendritic conj coactivity rate
    plot_box_plot(
        plastic_rel_conj,
        figsize=(5, 5),
        title="Relative Conj Rate",
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
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    # Ave local spine fraction coactive
    plot_box_plot(
        plastic_avg_spine_frac,
        figsize=(5, 5),
        title="nearby Spine Fraction",
        xtitle=None,
        ytitle="Fraction of spine activity",
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
        ax=axes["N"],
        save=False,
        save_path=None,
    )
    # Enlarged spine fraction vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_spine_frac["sLTP"], plastic_shuff_spine_frac["sLTP"])),
        plot_ind=True,
        title="Enlarged",
        xtitle="Fraction of spine activity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
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
            "data": plastic_avg_spine_frac["sLTP"],
            "shuff": plastic_shuff_spine_frac["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction coactive",
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
        ax=ax_o_inset,
        save=False,
        save_path=None,
    )
    # Shrunken spine fraction vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_spine_frac["sLTD"], plastic_shuff_spine_frac["sLTD"])),
        plot_ind=True,
        title="Shrunken",
        xtitle="Fraction of spine activity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["P"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_p_inset = axes["P"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_p_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_spine_frac["sLTD"],
            "shuff": plastic_shuff_spine_frac["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction coactive",
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
        ax=ax_p_inset,
        save=False,
        save_path=None,
    )
    # Stable spine fraction vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list(
            (plastic_avg_spine_frac["Stable"], plastic_shuff_spine_frac["Stable"])
        ),
        plot_ind=True,
        title="Stable",
        xtitle="Fraction of spine activity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["Q"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_q_inset = axes["Q"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_q_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_spine_frac["Stable"],
            "shuff": plastic_shuff_spine_frac["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction coactive",
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
        ax=ax_q_inset,
        save=False,
        save_path=None,
    )
    # Relative local spine fraction coactive
    plot_box_plot(
        plastic_rel_spine_frac,
        figsize=(5, 5),
        title="Relative Spine Fraction",
        xtitle=None,
        ytitle="Relative fraction of spine activity",
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
        ax=axes["R"],
        save=False,
        save_path=None,
    )
    # Ave local dendrite fraction coactive
    plot_box_plot(
        plastic_avg_dend_frac,
        figsize=(5, 5),
        title="Nearby Dendrite Fraction",
        xtitle=None,
        ytitle="Fraction of dendrite activity",
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
        ax=axes["T"],
        save=False,
        save_path=None,
    )
    # Enlarged dendritic fraction vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_dend_frac["sLTP"], plastic_shuff_dend_frac["sLTP"])),
        plot_ind=True,
        title="Enlarged",
        xtitle="Fraction of dendrite activity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["U"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_u_inset = axes["U"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_u_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_dend_frac["sLTP"],
            "shuff": plastic_shuff_dend_frac["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction coactive",
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
        ax=ax_u_inset,
        save=False,
        save_path=None,
    )
    # Shrunken dendrite fraction vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_dend_frac["sLTD"], plastic_shuff_dend_frac["sLTD"])),
        plot_ind=True,
        title="Shrunken",
        xtitle="Fraction of dendrite activity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["V"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_v_inset = axes["V"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_v_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_dend_frac["sLTD"],
            "shuff": plastic_shuff_dend_frac["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction coactive",
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
        ax=ax_v_inset,
        save=False,
        save_path=None,
    )
    # Stable dendrite fraction vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_dend_frac["Stable"], plastic_shuff_dend_frac["Stable"])),
        plot_ind=True,
        title="Stable",
        xtitle="Fraction of dendrite activity",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
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
            "data": plastic_avg_dend_frac["Stable"],
            "shuff": plastic_shuff_dend_frac["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction coactive",
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
        ax=ax_w_inset,
        save=False,
        save_path=None,
    )
    # Relative local dendrite fraction coactive
    plot_box_plot(
        plastic_rel_dend_frac,
        figsize=(5, 5),
        title="Relative nearby Dendrite Fraction",
        xtitle=None,
        ytitle="Relative fraction of dendrite activity",
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
        ax=axes["X"],
        save=False,
        save_path=None,
    )
    # Ave local relative onset
    plot_box_plot(
        plastic_avg_onset,
        figsize=(5, 5),
        title="Nearby Relative Onset",
        xtitle=None,
        ytitle="Nearby relative onset (s)",
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
    # Enlarged relative onset vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_onset["sLTP"], plastic_shuff_onset["sLTP"])),
        plot_ind=True,
        title="Enlarged",
        xtitle="Nearby relative onset (s)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[0], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["a"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_A_inset = axes["a"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_A_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_onset["sLTP"],
            "shuff": plastic_shuff_onset["sLTP"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Relative onset (s)",
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
    # Shrunken relative onset vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_onset["sLTD"], plastic_shuff_onset["sLTD"])),
        plot_ind=True,
        title="Shrunken",
        xtitle="Nearby relative onset (s)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["b"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_B_inset = axes["b"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_B_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_onset["sLTD"],
            "shuff": plastic_shuff_onset["sLTD"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Relative onset (s)",
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
    # Stable relative onset vs chance
    ## Cummulative distribution
    plot_cummulative_distribution(
        data=list((plastic_avg_onset["Stable"], plastic_shuff_onset["Stable"])),
        plot_ind=True,
        title="Stable",
        xtitle="Nearby relative onset (s)",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[2], "black"],
        ind_color=[None, "black"],
        line_width=1.5,
        ind_line_width=0.5,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["c"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_C_inset = axes["c"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_C_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": plastic_avg_onset["Stable"],
            "shuff": plastic_shuff_onset["Stable"].flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Relative onset (s)",
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
    # relative local relative onset
    plot_box_plot(
        plastic_rel_onset,
        figsize=(5, 5),
        title="Nearby Relative Relative Onset",
        xtitle=None,
        ytitle="Nearby relative onset (s)",
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
        ax=axes["d"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if MRSs is not None:
            mrs_name = f"{MRSs}_"
        else:
            mrs_name = ""
        if period == "All periods":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Dend_Properties_{mrs_name}Figure"
            )
        elif period == "movement":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Dend_Properties_Mvmt_{mrs_name}Figure"
            )
        elif period == "nonmovement":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Dend_Properties_Nonmvmt_{mrs_name}Figure"
            )

        fig.savefig(fname + ".pdf")

    ############################ Statistics Section ############################
    if display_stats == False:
        return

    # Perform the F-tests
    if test_type == "parametric":
        coactivity_f, coactivity_p, _, coactivity_df = t_utils.ANOVA_1way_posthoc(
            plastic_avg_rates,
            test_method,
        )
        (
            rel_coactivity_f,
            rel_coactivity_p,
            _,
            rel_coactivity_df,
        ) = t_utils.ANOVA_1way_posthoc(
            plastic_rel_rates,
            test_method,
        )
        conj_f, conj_p, _, conj_df = t_utils.ANOVA_1way_posthoc(
            plastic_avg_conj,
            test_method,
        )
        rel_conj_f, rel_conj_p, _, rel_conj_df = t_utils.ANOVA_1way_posthoc(
            plastic_rel_conj,
            test_method,
        )
        spine_frac_f, spine_frac_p, _, spine_frac_df = t_utils.ANOVA_1way_posthoc(
            plastic_avg_spine_frac,
            test_method,
        )
        (
            rel_spine_frac_f,
            rel_spine_frac_p,
            _,
            rel_spine_frac_df,
        ) = t_utils.ANOVA_1way_posthoc(
            plastic_rel_spine_frac,
            test_method,
        )
        dend_frac_f, dend_frac_p, _, dend_frac_df = t_utils.ANOVA_1way_posthoc(
            plastic_avg_dend_frac,
            test_method,
        )
        (
            rel_dend_frac_f,
            rel_dend_frac_p,
            _,
            rel_dend_frac_df,
        ) = t_utils.ANOVA_1way_posthoc(
            plastic_rel_dend_frac,
            test_method,
        )
        onset_f, onset_p, _, onset_df = t_utils.ANOVA_1way_posthoc(
            plastic_avg_onset,
            test_method,
        )
        rel_onset_f, rel_onset_p, _, rel_onset_df = t_utils.ANOVA_1way_posthoc(
            plastic_rel_onset,
            test_method,
        )
        test_title = f"One-Way ANOVA {test_method}"
    elif test_type == "nonparametric":
        coactivity_f, coactivity_p, coactivity_df = t_utils.kruskal_wallis_test(
            plastic_avg_rates,
            "Conover",
            test_method,
        )
        (
            rel_coactivity_f,
            rel_coactivity_p,
            rel_coactivity_df,
        ) = t_utils.kruskal_wallis_test(
            plastic_rel_rates,
            "Conover",
            test_method,
        )
        conj_f, conj_p, conj_df = t_utils.kruskal_wallis_test(
            plastic_avg_conj,
            "Conover",
            test_method,
        )
        rel_conj_f, rel_conj_p, rel_conj_df = t_utils.kruskal_wallis_test(
            plastic_rel_conj,
            "Conover",
            test_method,
        )
        spine_frac_f, spine_frac_p, spine_frac_df = t_utils.kruskal_wallis_test(
            plastic_avg_spine_frac,
            "Conover",
            test_method,
        )
        (
            rel_spine_frac_f,
            rel_spine_frac_p,
            rel_spine_frac_df,
        ) = t_utils.kruskal_wallis_test(
            plastic_rel_spine_frac,
            "Conover",
            test_method,
        )
        dend_frac_f, dend_frac_p, dend_frac_df = t_utils.kruskal_wallis_test(
            plastic_avg_dend_frac,
            "Conover",
            test_method,
        )
        (
            rel_dend_frac_f,
            rel_dend_frac_p,
            rel_dend_frac_df,
        ) = t_utils.kruskal_wallis_test(
            plastic_rel_dend_frac,
            "Conover",
            test_method,
        )
        onset_f, onset_p, onset_df = t_utils.kruskal_wallis_test(
            plastic_avg_onset,
            "Conover",
            test_method,
        )
        rel_onset_f, rel_onset_p, rel_onset_df = t_utils.kruskal_wallis_test(
            plastic_rel_onset,
            "Conover",
            test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"

    # Perform correlations
    _, coactivity_corr_df = t_utils.correlate_grouped_data(
        plastic_rate_dist,
        distance_bins,
    )
    _, conj_corr_df = t_utils.correlate_grouped_data(
        plastic_conj_dist,
        distance_bins,
    )
    _, spine_frac_corr_df = t_utils.correlate_grouped_data(
        plastic_spine_frac_dist,
        distance_bins,
    )
    _, dend_frac_corr_df = t_utils.correlate_grouped_data(
        plastic_dend_frac_dist,
        distance_bins,
    )
    _, onset_corr_df = t_utils.correlate_grouped_data(
        plastic_onset_dist,
        distance_bins,
    )
    # Comparisions to chance
    ## Coactivity
    e_rate_above, e_rate_below = t_utils.test_against_chance(
        plastic_avg_rates["sLTP"], plastic_shuff_rates["sLTP"]
    )
    s_rate_above, s_rate_below = t_utils.test_against_chance(
        plastic_avg_rates["sLTD"], plastic_shuff_rates["sLTD"]
    )
    st_rate_above, st_rate_below = t_utils.test_against_chance(
        plastic_avg_rates["Stable"], plastic_shuff_rates["Stable"]
    )
    chance_coactivity = {
        "Comparison": ["sLTP", "sLTD", "Stable"],
        "p-val above": [e_rate_above, s_rate_above, st_rate_above],
        "p-val below": [e_rate_below, s_rate_below, st_rate_below],
    }
    chance_coactivity_df = pd.DataFrame.from_dict(chance_coactivity)
    ## Conj coactivity
    e_conj_above, e_conj_below = t_utils.test_against_chance(
        plastic_avg_conj["sLTP"], plastic_shuff_conj["sLTP"]
    )
    s_conj_above, s_conj_below = t_utils.test_against_chance(
        plastic_avg_conj["sLTD"], plastic_shuff_conj["sLTD"]
    )
    st_conj_above, st_conj_below = t_utils.test_against_chance(
        plastic_avg_conj["Stable"], plastic_shuff_conj["Stable"]
    )
    chance_conj = {
        "Comparison": ["sLTP", "sLTD", "Stable"],
        "p-val above": [e_conj_above, s_conj_above, st_conj_above],
        "p-val below": [e_conj_below, s_conj_below, st_conj_below],
    }
    chance_conj_df = pd.DataFrame.from_dict(chance_conj)
    ## Spine fraction
    e_sfrac_above, e_sfrac_below = t_utils.test_against_chance(
        plastic_avg_spine_frac["sLTP"], plastic_shuff_spine_frac["sLTP"]
    )
    s_sfrac_above, s_sfrac_below = t_utils.test_against_chance(
        plastic_avg_spine_frac["sLTD"], plastic_shuff_spine_frac["sLTD"]
    )
    st_sfrac_above, st_sfrac_below = t_utils.test_against_chance(
        plastic_avg_spine_frac["Stable"], plastic_shuff_spine_frac["Stable"]
    )
    chance_sfrac = {
        "Comparison": ["sLTP", "sLTD", "Stable"],
        "p-val above": [e_sfrac_above, s_sfrac_above, st_sfrac_above],
        "p-val below": [e_sfrac_below, s_sfrac_below, st_sfrac_below],
    }
    chance_sfrac_df = pd.DataFrame.from_dict(chance_sfrac)
    ## Dendrite fraction
    e_dfrac_above, e_dfrac_below = t_utils.test_against_chance(
        plastic_avg_dend_frac["sLTP"], plastic_shuff_dend_frac["sLTP"]
    )
    s_dfrac_above, s_dfrac_below = t_utils.test_against_chance(
        plastic_avg_dend_frac["sLTD"], plastic_shuff_dend_frac["sLTD"]
    )
    st_dfrac_above, st_dfrac_below = t_utils.test_against_chance(
        plastic_avg_dend_frac["Stable"], plastic_shuff_dend_frac["Stable"]
    )
    chance_dfrac = {
        "Comparison": ["sLTP", "sLTD", "Stable"],
        "p-val above": [e_dfrac_above, s_dfrac_above, st_dfrac_above],
        "p-val below": [e_dfrac_below, s_dfrac_below, st_dfrac_below],
    }
    chance_dfrac_df = pd.DataFrame.from_dict(chance_dfrac)
    ## Relative onset
    e_onset_above, e_onset_below = t_utils.test_against_chance(
        plastic_avg_onset["sLTP"], plastic_shuff_onset["sLTP"]
    )
    s_onset_above, s_onset_below = t_utils.test_against_chance(
        plastic_avg_onset["sLTD"], plastic_shuff_onset["sLTD"]
    )
    st_onset_above, st_onset_below = t_utils.test_against_chance(
        plastic_avg_onset["Stable"], plastic_shuff_onset["Stable"]
    )
    chance_onset = {
        "Comparison": ["sLTP", "sLTD", "Stable"],
        "p-val above": [e_onset_above, s_onset_above, st_onset_above],
        "p-val below": [e_onset_below, s_onset_below, st_onset_below],
    }
    chance_onset_df = pd.DataFrame.from_dict(chance_onset)

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
        QR
        ST
        """,
        figsize=(10, 22),
    )
    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(f"Avg Local Coactivity Rate")
    A_table = axes2["A"].table(
        cellText=coactivity_corr_df.values,
        colLabels=coactivity_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(f"Avg Local Conj Rate")
    B_table = axes2["B"].table(
        cellText=conj_corr_df.values,
        colLabels=conj_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(f"Avg Local Spine Frac")
    C_table = axes2["C"].table(
        cellText=spine_frac_corr_df.values,
        colLabels=spine_frac_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(f"Avg Local Dendrite frac")
    D_table = axes2["D"].table(
        cellText=dend_frac_corr_df.values,
        colLabels=dend_frac_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(f"Avg Local Rel Onset Rate")
    E_table = axes2["E"].table(
        cellText=onset_corr_df.values,
        colLabels=onset_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title(
        f"Local Coactivity Rate\n{test_title}\nF = {coactivity_f:.4} p = {coactivity_p:.3E}"
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
        f"Relative Coactivity Rate\n{test_title}\nF = {rel_coactivity_f:.4} p = {rel_coactivity_p:.3E}"
    )
    G_table = axes2["G"].table(
        cellText=rel_coactivity_df.values,
        colLabels=rel_coactivity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title(
        f"Local Conj Rate\n{test_title}\nF = {conj_f:.4} p = {conj_p:.3E}"
    )
    H_table = axes2["H"].table(
        cellText=conj_df.values,
        colLabels=conj_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)
    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title(
        f"Relative Conj Rate\n{test_title}\nF = {rel_conj_f:.4} p = {rel_conj_p:.3E}"
    )
    I_table = axes2["I"].table(
        cellText=rel_conj_df.values,
        colLabels=rel_conj_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)
    axes2["J"].axis("off")
    axes2["J"].axis("tight")
    axes2["J"].set_title(
        f"Spine Fraction\n{test_title}\nF = {spine_frac_f:.4} p = {spine_frac_p:.3E}"
    )
    J_table = axes2["J"].table(
        cellText=spine_frac_df.values,
        colLabels=spine_frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    J_table.auto_set_font_size(False)
    J_table.set_fontsize(8)
    axes2["K"].axis("off")
    axes2["K"].axis("tight")
    axes2["K"].set_title(
        f"Relative Spine Fraction\n{test_title}\nF = {rel_spine_frac_f:.4} p = {rel_spine_frac_p:.3E}"
    )
    K_table = axes2["K"].table(
        cellText=rel_spine_frac_df.values,
        colLabels=rel_spine_frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    K_table.auto_set_font_size(False)
    K_table.set_fontsize(8)
    axes2["L"].axis("off")
    axes2["L"].axis("tight")
    axes2["L"].set_title(
        f"Dendrite Fraction\n{test_title}\nF = {dend_frac_f:.4} p = {dend_frac_p:.3E}"
    )
    L_table = axes2["L"].table(
        cellText=dend_frac_df.values,
        colLabels=dend_frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    L_table.auto_set_font_size(False)
    L_table.set_fontsize(8)
    axes2["M"].axis("off")
    axes2["M"].axis("tight")
    axes2["M"].set_title(
        f"Relative Dendrite Fraction\n{test_title}\nF = {rel_dend_frac_f:.4} p = {rel_dend_frac_p:.3E}"
    )
    M_table = axes2["M"].table(
        cellText=rel_dend_frac_df.values,
        colLabels=rel_dend_frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    M_table.auto_set_font_size(False)
    M_table.set_fontsize(8)
    axes2["N"].axis("off")
    axes2["N"].axis("tight")
    axes2["N"].set_title(f"Onsets\n{test_title}\nF = {onset_f:.4} p = {onset_p:.3E}")
    N_table = axes2["N"].table(
        cellText=onset_df.values,
        colLabels=onset_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    N_table.auto_set_font_size(False)
    N_table.set_fontsize(8)
    axes2["O"].axis("off")
    axes2["O"].axis("tight")
    axes2["O"].set_title(
        f"Relative Onsets\n{test_title}\nF = {rel_onset_f:.4} p = {rel_onset_p:.3E}"
    )
    O_table = axes2["O"].table(
        cellText=rel_onset_df.values,
        colLabels=rel_onset_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    O_table.auto_set_font_size(False)
    O_table.set_fontsize(8)
    axes2["P"].axis("off")
    axes2["P"].axis("tight")
    axes2["P"].set_title(f"Chance Coactivity")
    P_table = axes2["P"].table(
        cellText=chance_coactivity_df.values,
        colLabels=chance_coactivity_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    P_table.auto_set_font_size(False)
    P_table.set_fontsize(8)
    axes2["Q"].axis("off")
    axes2["Q"].axis("tight")
    axes2["Q"].set_title(f"Chance Conj Coactivity")
    Q_table = axes2["P"].table(
        cellText=chance_conj_df.values,
        colLabels=chance_conj_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    Q_table.auto_set_font_size(False)
    Q_table.set_fontsize(8)
    axes2["R"].axis("off")
    axes2["R"].axis("tight")
    axes2["R"].set_title(f"Chance Spine Fraction")
    R_table = axes2["R"].table(
        cellText=chance_sfrac_df.values,
        colLabels=chance_sfrac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    R_table.auto_set_font_size(False)
    R_table.set_fontsize(8)
    axes2["S"].axis("off")
    axes2["S"].axis("tight")
    axes2["S"].set_title(f"Chance Dendrite Fraction")
    S_table = axes2["S"].table(
        cellText=chance_dfrac_df.values,
        colLabels=chance_dfrac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    S_table.auto_set_font_size(False)
    S_table.set_fontsize(8)
    axes2["T"].axis("off")
    axes2["T"].axis("tight")
    axes2["T"].set_title(f"Chance Relative onsets")
    T_table = axes2["T"].table(
        cellText=chance_onset_df.values,
        colLabels=chance_onset_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    T_table.auto_set_font_size(False)
    T_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if MRSs is not None:
            mrs_name = f"{MRSs}_"
        else:
            mrs_name = ""
        if period == "All periods":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Dend_Properties_{mrs_name}Stats"
            )
        elif period == "movement":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Dend_Properties_Mvmt_{mrs_name}Stats"
            )
        elif period == "nonmovement":
            fname = os.path.join(
                save_path, f"Nearby_Spine_Dend_Properties_Nonmvmt_{mrs_name}Stats"
            )

        fig.savefig(fname + ".pdf")


def plot_global_movement_encoding(
    dataset,
    followup_dataset=None,
    exclude="Shaft",
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
    """Function to plot the movement encoding when spines are coactive with dendrite

    INPUT PARAMETERS
        dataset - Global_Coactivity_Data object

        followup_dataset - optional Global_Coactivity_Data object of the subsequent
                           session to use for volume comparision. Default is None,
                           to use the followup volumes in the datset

        exclude - str specifying the type of spines to exclude from analysis

        threshold - float or tuple of floats specifying the threshold cutoffs for
                    classifying plasticity

        figsize - tuple specifying the size of the figure

        showmeans - boolean specifying whether to display mean values on box plots

        test_type - str specifying whether to perform parametric or nonparametric
                    tests

        test_method - str specifying the type of posthoct test to perform

        display_stats - boolean specifying whether to display the statistics

        vol_norm - boolean specifying whether to use normalized relative volume values

        save - boolean specifying whether to save the figure or not

        save_path - str specifying where to save the figures

    """
    COLORS = ["mediumslateblue", "tomato", "silver"]
    plastic_groups = {
        "sLTP": "enlarged_spines",
        "sLTD": "shrunken_spines",
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

    # Movement encoding variables
    all_coactive_movement_correlation = dataset.all_coactive_movement_correlation
    all_coactive_movement_stereotypy = dataset.all_coactive_movement_stereotypy
    all_coactive_movement_reliability = dataset.all_coactive_movement_reliability
    all_coactive_movement_specificity = dataset.all_coactive_movement_specificity
    conj_coactive_movement_correlation = dataset.conj_movement_correlation
    conj_coactive_movement_stereotypy = dataset.conj_movement_stereotypy
    conj_coactive_movement_reliability = dataset.conj_movement_reliability
    conj_coactive_movement_specificity = dataset.conj_movement_specificity
    nonconj_coactive_movement_correlation = dataset.nonconj_movement_correlation
    nonconj_coactive_movement_stereotypy = dataset.nonconj_movement_stereotypy
    nonconj_coactive_movement_reliability = dataset.nonconj_movement_reliability
    nonconj_coactive_movement_specificity = dataset.nonconj_movement_specificity

    # Calculate the relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes,
        flags,
        norm=vol_norm,
        exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume,
        threshold=threshold,
        norm=vol_norm,
    )

    # Organize data
    ## Subselect present spines
    all_coactive_movement_correlation = d_utils.subselect_data_by_idxs(
        all_coactive_movement_correlation,
        spine_idxs,
    )
    all_coactive_movement_stereotypy = d_utils.subselect_data_by_idxs(
        all_coactive_movement_stereotypy,
        spine_idxs,
    )
    all_coactive_movement_reliability = d_utils.subselect_data_by_idxs(
        all_coactive_movement_reliability,
        spine_idxs,
    )
    all_coactive_movement_specificity = d_utils.subselect_data_by_idxs(
        all_coactive_movement_specificity,
        spine_idxs,
    )
    conj_coactive_movement_correlation = d_utils.subselect_data_by_idxs(
        conj_coactive_movement_correlation,
        spine_idxs,
    )
    conj_coactive_movement_stereotypy = d_utils.subselect_data_by_idxs(
        conj_coactive_movement_stereotypy,
        spine_idxs,
    )
    conj_coactive_movement_reliability = d_utils.subselect_data_by_idxs(
        conj_coactive_movement_reliability,
        spine_idxs,
    )
    conj_coactive_movement_specificity = d_utils.subselect_data_by_idxs(
        conj_coactive_movement_specificity,
        spine_idxs,
    )
    nonconj_coactive_movement_correlation = d_utils.subselect_data_by_idxs(
        nonconj_coactive_movement_correlation,
        spine_idxs,
    )
    nonconj_coactive_movement_stereotypy = d_utils.subselect_data_by_idxs(
        nonconj_coactive_movement_stereotypy,
        spine_idxs,
    )
    nonconj_coactive_movement_reliability = d_utils.subselect_data_by_idxs(
        nonconj_coactive_movement_reliability,
        spine_idxs,
    )
    nonconj_coactive_movement_specificity = d_utils.subselect_data_by_idxs(
        nonconj_coactive_movement_specificity,
        spine_idxs,
    )

    ## Seperate into plasticity groups
    plastic_mvmt_corr = {}
    plastic_mvmt_stereo = {}
    plastic_mvmt_reli = {}
    plastic_mvmt_speci = {}
    plastic_conj_corr = {}
    plastic_conj_stereo = {}
    plastic_conj_reli = {}
    plastic_conj_speci = {}
    plastic_nonconj_corr = {}
    plastic_nonconj_stereo = {}
    plastic_nonconj_reli = {}
    plastic_nonconj_speci = {}
    for (
        key,
        value,
    ) in plastic_groups.items():
        spines = eval(value)
        plastic_mvmt_corr[key] = all_coactive_movement_correlation[spines]
        plastic_mvmt_stereo[key] = all_coactive_movement_stereotypy[spines]
        plastic_mvmt_reli[key] = all_coactive_movement_reliability[spines]
        plastic_mvmt_speci[key] = all_coactive_movement_specificity[spines]
        plastic_conj_corr[key] = conj_coactive_movement_correlation[spines]
        plastic_conj_stereo[key] = conj_coactive_movement_stereotypy[spines]
        plastic_conj_reli[key] = conj_coactive_movement_reliability[spines]
        plastic_conj_speci[key] = conj_coactive_movement_specificity[spines]
        plastic_nonconj_corr[key] = nonconj_coactive_movement_correlation[spines]
        plastic_nonconj_stereo[key] = nonconj_coactive_movement_stereotypy[spines]
        plastic_nonconj_reli[key] = nonconj_coactive_movement_reliability[spines]
        plastic_nonconj_speci[key] = nonconj_coactive_movement_specificity[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFGH
        IJKL
        """,
        figsize=figsize,
    )
    fig.suptitle("Global Coactivity Movement Encoding")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ############################# Plot data onto axes ################################
    ## All coactive movement correlation
    plot_box_plot(
        plastic_mvmt_corr,
        figsize=(5, 5),
        title="All events",
        xtitle=None,
        ytitle="LMP Correlation (r)",
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
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    ## All coactive movement stereotypy
    plot_box_plot(
        plastic_mvmt_stereo,
        figsize=(5, 5),
        title="All events",
        xtitle=None,
        ytitle="Movement stereotypy",
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
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    ## All coactive movement reliability
    plot_box_plot(
        plastic_mvmt_reli,
        figsize=(5, 5),
        title="All events",
        xtitle=None,
        ytitle="Movement reliability",
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
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    ## All coactive movement specificity
    plot_box_plot(
        plastic_mvmt_speci,
        figsize=(5, 5),
        title="All events",
        xtitle=None,
        ytitle="Movement specificity",
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
    ## Conj coactive movement correlation
    plot_box_plot(
        plastic_conj_corr,
        figsize=(5, 5),
        title="Conj events",
        xtitle=None,
        ytitle="LMP correlation (r)",
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
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Conj coactive movement stereotypy
    plot_box_plot(
        plastic_conj_stereo,
        figsize=(5, 5),
        title="Conj events",
        xtitle=None,
        ytitle="Movement stereotypy",
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
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ## Conj coactive movement reliability
    plot_box_plot(
        plastic_conj_reli,
        figsize=(5, 5),
        title="Conj events",
        xtitle=None,
        ytitle="Movement reliability",
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
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    ## Conj coactive movement specificity
    plot_box_plot(
        plastic_conj_speci,
        figsize=(5, 5),
        title="Conj events",
        xtitle=None,
        ytitle="Movement specificity",
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
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    ## Nononj coactive movement correlation
    plot_box_plot(
        plastic_nonconj_corr,
        figsize=(5, 5),
        title="Nonconj events",
        xtitle=None,
        ytitle="LMP correlation (r)",
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
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    ## Nononj coactive movement stereotypy
    plot_box_plot(
        plastic_nonconj_stereo,
        figsize=(5, 5),
        title="Nonconj events",
        xtitle=None,
        ytitle="Movement stereotypy",
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
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    ## Nononj coactive movement reliability
    plot_box_plot(
        plastic_nonconj_reli,
        figsize=(5, 5),
        title="Nonconj events",
        xtitle=None,
        ytitle="Movement reliability",
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    ## Nononj coactive movement specificity
    plot_box_plot(
        plastic_nonconj_speci,
        figsize=(5, 5),
        title="Nonconj events",
        xtitle=None,
        ytitle="Movement specificity",
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
        ax=axes["L"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Dendritic_Coactivity_Figure_25")

        fig.savefig(fname + ".pdf")

    ########################### Statistics Section ############################
    if display_stats == False:
        return

    # Perform the f-tests
    if test_type == "parametric":
        all_corr_f, all_corr_p, _, all_corr_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_corr,
            test_method,
        )
        all_stereo_f, all_stereo_p, _, all_stereo_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_stereo,
            test_method,
        )
        all_reli_f, all_reli_p, _, all_reli_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_reli,
            test_method,
        )
        all_speci_f, all_speci_p, _, all_speci_df = t_utils.ANOVA_1way_posthoc(
            plastic_mvmt_speci,
            test_method,
        )
        conj_corr_f, conj_corr_p, _, conj_corr_df = t_utils.ANOVA_1way_posthoc(
            plastic_conj_corr,
            test_method,
        )
        conj_stereo_f, conj_stereo_p, _, conj_stereo_df = t_utils.ANOVA_1way_posthoc(
            plastic_conj_stereo,
            test_method,
        )
        conj_reli_f, conj_reli_p, _, conj_reli_df = t_utils.ANOVA_1way_posthoc(
            plastic_conj_reli,
            test_method,
        )
        conj_speci_f, conj_speci_p, _, conj_speci_df = t_utils.ANOVA_1way_posthoc(
            plastic_conj_speci,
            test_method,
        )
        nonconj_corr_f, nonconj_corr_p, _, nonconj_corr_df = t_utils.ANOVA_1way_posthoc(
            plastic_nonconj_corr,
            test_method,
        )
        (
            nonconj_stereo_f,
            nonconj_stereo_p,
            _,
            nonconj_stereo_df,
        ) = t_utils.ANOVA_1way_posthoc(
            plastic_nonconj_stereo,
            test_method,
        )
        nonconj_reli_f, nonconj_reli_p, _, nonconj_reli_df = t_utils.ANOVA_1way_posthoc(
            plastic_nonconj_reli,
            test_method,
        )
        (
            nonconj_speci_f,
            nonconj_speci_p,
            _,
            nonconj_speci_df,
        ) = t_utils.ANOVA_1way_posthoc(
            plastic_nonconj_speci,
            test_method,
        )
        test_title = f"One-way ANOVA {test_method}"
    elif test_type == "nonparametric":
        all_corr_f, all_corr_p, all_corr_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_corr,
            "Conover",
            test_method,
        )
        all_stereo_f, all_stereo_p, all_stereo_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_stereo,
            "Conover",
            test_method,
        )
        all_reli_f, all_reli_p, all_reli_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_reli,
            "Conover",
            test_method,
        )
        all_speci_f, all_speci_p, all_speci_df = t_utils.kruskal_wallis_test(
            plastic_mvmt_speci,
            "Conover",
            test_method,
        )
        conj_corr_f, conj_corr_p, conj_corr_df = t_utils.kruskal_wallis_test(
            plastic_conj_corr,
            "Conover",
            test_method,
        )
        conj_stereo_f, conj_stereo_p, conj_stereo_df = t_utils.kruskal_wallis_test(
            plastic_conj_stereo,
            "Conover",
            test_method,
        )
        conj_reli_f, conj_reli_p, conj_reli_df = t_utils.kruskal_wallis_test(
            plastic_conj_reli,
            "Conover",
            test_method,
        )
        conj_speci_f, conj_speci_p, conj_speci_df = t_utils.kruskal_wallis_test(
            plastic_conj_speci,
            "Conover",
            test_method,
        )
        nonconj_corr_f, nonconj_corr_p, nonconj_corr_df = t_utils.kruskal_wallis_test(
            plastic_nonconj_corr,
            "Conover",
            test_method,
        )
        (
            nonconj_stereo_f,
            nonconj_stereo_p,
            nonconj_stereo_df,
        ) = t_utils.kruskal_wallis_test(
            plastic_nonconj_stereo,
            "Conover",
            test_method,
        )
        nonconj_reli_f, nonconj_reli_p, nonconj_reli_df = t_utils.kruskal_wallis_test(
            plastic_nonconj_reli,
            "Conover",
            test_method,
        )
        (
            nonconj_speci_f,
            nonconj_speci_p,
            nonconj_speci_df,
        ) = t_utils.kruskal_wallis_test(
            plastic_nonconj_speci,
            "Conover",
            test_method,
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
        """,
        figsize=(8, 13),
    )

    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(
        f"All Mvmt Correlation\n{test_title}\nF = {all_corr_f:.4} p = {all_corr_p:.3E}"
    )
    A_table = axes2["A"].table(
        cellText=all_corr_df.values,
        colLabels=all_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(
        f"All Mvmt Stereotypy\n{test_title}\nF = {all_stereo_f:.4} p = {all_stereo_p:.3E}"
    )
    B_table = axes2["B"].table(
        cellText=all_stereo_df.values,
        colLabels=all_stereo_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(
        f"All Mvmt Reliability\n{test_title}\nF = {all_reli_f:.4} p = {all_reli_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=all_reli_df.values,
        colLabels=all_reli_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(
        f"All Mvmt Specificity\n{test_title}\nF = {all_speci_f:.4} p = {all_speci_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=all_speci_df.values,
        colLabels=all_speci_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(
        f"Conj Mvmt Correlation\n{test_title}\nF = {conj_corr_f:.4} p = {conj_corr_p:.3E}"
    )
    E_table = axes2["E"].table(
        cellText=conj_corr_df.values,
        colLabels=conj_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title(
        f"Conj Mvmt Stereotypy\n{test_title}\nF = {conj_stereo_f:.4} p = {conj_stereo_p:.3E}"
    )
    F_table = axes2["F"].table(
        cellText=conj_stereo_df.values,
        colLabels=conj_stereo_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)
    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title(
        f"Conj Mvmt Reliability\n{test_title}\nF = {conj_reli_f:.4} p = {conj_reli_p:.3E}"
    )
    G_table = axes2["G"].table(
        cellText=conj_reli_df.values,
        colLabels=conj_reli_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title(
        f"Conj Mvmt Specificity\n{test_title}\nF = {conj_speci_f:.4} p = {conj_speci_p:.3E}"
    )
    H_table = axes2["H"].table(
        cellText=conj_speci_df.values,
        colLabels=conj_speci_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)
    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title(
        f"Nonconj Mvmt Correlation\n{test_title}\nF = {nonconj_corr_f:.4} p = {nonconj_corr_p:.3E}"
    )
    I_table = axes2["I"].table(
        cellText=nonconj_corr_df.values,
        colLabels=nonconj_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)
    axes2["J"].axis("off")
    axes2["J"].axis("tight")
    axes2["J"].set_title(
        f"Nonconj Mvmt Stereotypy\n{test_title}\nF = {nonconj_stereo_f:.4} p = {nonconj_stereo_p:.3E}"
    )
    J_table = axes2["J"].table(
        cellText=nonconj_stereo_df.values,
        colLabels=nonconj_stereo_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    J_table.auto_set_font_size(False)
    J_table.set_fontsize(8)
    axes2["K"].axis("off")
    axes2["K"].axis("tight")
    axes2["K"].set_title(
        f"Nonconj Mvmt Reliability\n{test_title}\nF = {nonconj_reli_f:.4} p = {nonconj_reli_p:.3E}"
    )
    K_table = axes2["K"].table(
        cellText=nonconj_reli_df.values,
        colLabels=nonconj_reli_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    K_table.auto_set_font_size(False)
    K_table.set_fontsize(8)
    axes2["L"].axis("off")
    axes2["L"].axis("tight")
    axes2["L"].set_title(
        f"Nonconj Mvmt Specificity\n{test_title}\nF = {nonconj_speci_f:.4} p = {nonconj_speci_p:.3E}"
    )
    L_table = axes2["L"].table(
        cellText=nonconj_speci_df.values,
        colLabels=nonconj_speci_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    L_table.auto_set_font_size(False)
    L_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Dendritic_Coactivity_Figure_25_Stats")

        fig2.savefig(fname + ".pdf")


def plot_noncoactive_calcium(
    dataset,
    period="All periods",
    followup_dataset=None,
    exclude="Shaft Spine",
    MRSs=None,
    threshold=0.3,
    figsize=(10, 10),
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """Function to plot spine calcium when the dendrite and neighbors
        are active, but the spine isn't

    INPUT PARAMETERS
        dataset - Dendritic_Coactivity_Data object

        period - str specifying whether the dataset is constrained to a particular
                movement period

        followup_data - optional Dendritic_Coactivity_Data object of the subsequent
                        session to be used for volume comparisoin. Default is None
                        to use the followup volumes in the dataset

        exclude - str specifying spine type to exclude from data analysis

        MRSs - str specifying if you wish to examine only MRSs or nonMRSs. Accepts
                "MRS" or "nMRS". Default is None to examine all spines

        threshold - float or tuple of floats specifying the threshold cutoff
                    for classifying plasticity

        showmeans - boolean specifying whether to plot means on box plots

        test_type - str specifying whether to perform parametric or nonparametric tests

        test_method - str specifying the type of posthoc tests to perform

        display_stats - boolean specifying whether to display stats

        save - boolean specifying whether to save the figure or not

        save_path - str specifying where to save the figure


    """
    COLORS = ["mediumslateblue", "tomato", "silver"]
    plastic_groups = {
        "sLTP": "enlarged_spines",
        "sLTD": "shrunken_spines",
        "Stable": "stable_spines",
    }

    if period == "All periods":
        period_title = "Entire session"
    elif period == "movement":
        period_title = "Movement periods"
    elif period == "nonmovement":
        period_title = "Nonmovement periods"

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

    ## Calcium related variables
    noncoactive_spine_calcium_amplitude = dataset.noncoactive_spine_calcium_amplitude
    noncoactive_spine_calcium_traces = dataset.noncoactive_spine_calcium_traces
    conj_fraction_participating = dataset.conj_fraction_participating
    nonparticipating_spine_calcium_amplitude = (
        dataset.nonparticipating_spine_calcium_amplitude
    )
    nonparticipating_spine_calcium_traces = (
        dataset.nonparticipating_spine_calcium_traces
    )

    # Calculate the relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes,
        flags,
        norm=vol_norm,
        exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume,
        threshold=threshold,
        norm=vol_norm,
    )

    # Subselect present spines
    noncoactive_spine_calcium_amplitude = d_utils.subselect_data_by_idxs(
        noncoactive_spine_calcium_amplitude,
        spine_idxs,
    )
    noncoactive_spine_calcium_traces = d_utils.subselect_data_by_idxs(
        noncoactive_spine_calcium_traces,
        spine_idxs,
    )
    conj_fraction_participating = d_utils.subselect_data_by_idxs(
        conj_fraction_participating,
        spine_idxs,
    )
    nonparticipating_spine_calcium_amplitude = d_utils.subselect_data_by_idxs(
        nonparticipating_spine_calcium_amplitude,
        spine_idxs,
    )
    nonparticipating_spine_calcium_traces = d_utils.subselect_data_by_idxs(
        nonparticipating_spine_calcium_traces, spine_idxs
    )

    mvmt_spines = d_utils.subselect_data_by_idxs(dataset.movement_spines, spine_idxs)
    nonmvmt_spines = d_utils.subselect_data_by_idxs(
        dataset.nonmovement_spines, spine_idxs
    )

    # Seperate into dicts for plotting
    plastic_noncoactive_amp = {}
    plastic_noncoactive_traces = {}
    plastic_noncoactive_sems = {}
    plastic_frac_participating = {}
    plastic_nonparticipating_amp = {}
    plastic_nonparticipating_traces = {}
    plastic_nonparticipating_sems = {}

    for key, value in plastic_groups.items():
        spines = eval(value)
        if MRSs == "MRS":
            spines = spines * mvmt_spines
        elif MRSs == "nMRS":
            spines = spines * nonmvmt_spines
        nonco_traces = compress(noncoactive_spine_calcium_traces, spines)
        nonco_traces = [
            np.nanmean(x, axis=1) for x in nonco_traces if type(x) == np.ndarray
        ]
        nonco_traces = np.vstack(nonco_traces)
        plastic_noncoactive_traces[key] = np.nanmean(nonco_traces, axis=0)
        plastic_noncoactive_sems[key] = stats.sem(
            nonco_traces, axis=0, nan_policy="omit"
        )
        nonpart_traces = compress(nonparticipating_spine_calcium_traces, spines)
        nonpart_traces = [
            np.nanmean(x, axis=1) for x in nonpart_traces if type(x) == np.ndarray
        ]
        nonpart_traces = np.vstack(nonpart_traces)
        plastic_nonparticipating_traces[key] = np.nanmean(nonpart_traces, axis=0)
        plastic_nonparticipating_sems[key] = stats.sem(
            nonpart_traces, axis=0, nan_policy="omit"
        )
        plastic_noncoactive_amp[key] = noncoactive_spine_calcium_amplitude[spines]
        plastic_frac_participating[key] = conj_fraction_participating[spines]
        plastic_nonparticipating_amp[key] = nonparticipating_spine_calcium_amplitude[
            spines
        ]

    # Construct the figure
    fig, axes = plt.subplot_mosaic("""ABCDE""", figsize=figsize)
    fig.suptitle(f"{period_title} Noncoactive Calcium")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ################## Plot data onto axes ######################
    # noncoactive calcium traces
    plot_mean_activity_traces(
        means=list(plastic_noncoactive_traces.values()),
        sems=list(plastic_noncoactive_sems.values()),
        group_names=list(plastic_noncoactive_traces.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Noncoactive",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # nonparticipating calcium traces
    plot_mean_activity_traces(
        means=list(plastic_nonparticipating_traces.values()),
        sems=list(plastic_nonparticipating_sems.values()),
        group_names=list(plastic_nonparticipating_traces.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Nonparticipating",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # noncoactive calcium amplitude box plot
    plot_box_plot(
        plastic_noncoactive_amp,
        figsize=(5, 5),
        title="Noncoactive",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
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
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # fraction nonparticpating box plot
    plot_box_plot(
        plastic_frac_participating,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle=f"Frac. participating",
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
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # nonparticipating calcium amplitude box plot
    plot_box_plot(
        plastic_nonparticipating_amp,
        figsize=(5, 5),
        title="Nonparticipating",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
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
        ax=axes["E"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if MRSs is not None:
            mrs_name = f"{MRSs}_"
        else:
            mrs_name = ""
        if period == "All periods":
            fname = os.path.join(
                save_path, f"Dendritic_Noncoactive_Event_Props_{mrs_name}Figure"
            )
        elif period == "movement":
            fname = os.path.join(
                save_path, f"Dendritic_Noncoactive_Event_Props_Mvmt_{mrs_name}Figure"
            )
        elif period == "nonmovement":
            fname = os.path.join(
                save_path, f"Dendritic_Noncoactive_Event_Props_Nonmvmt_{mrs_name}Figure"
            )
        fig.savefig(fname + ".pdf")

    ######################### Statistics Section #########################3
