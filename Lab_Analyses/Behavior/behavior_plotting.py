"""Module to perform the plotting of behavioral data

    CREATOR
        William (Jake) Wright - 02/08/2022

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_general_heatmap import plot_general_heatmap
from Lab_Analyses.Plotting.plot_multi_line_plot import plot_multi_line_plot
from Lab_Analyses.Utilities import data_utilities as utils

sns.set()
sns.set_style("ticks")


def plot_multi_group_lever_behavior(
    data_dict,
    figsize=(7.5, 9),
    main_title=None,
    colors=["black", "mediumslateblue"],
    ylims=None,
    save=False,
    save_path=None,
):
    """figure to plot lever press behavior from multiple groups"""

    # Pull relevant data
    success_rate = {}
    reaction_time = {}
    cue_to_reward = {}
    within_corr = {}
    across_corr = {}
    correlation_matrix = {}

    for key, value in data_dict.items():
        success_rate[key] = value.ind_success_rate
        reaction_time[key] = value.ind_reaction_time
        cue_to_reward[key] = value.ind_cue_to_reward
        within_corr[key] = value.ind_within_sess_corr
        across_corr[key] = value.ind_across_sess_corr
        correlation_matrix[key] = value.avg_corr_matrix
        sessions = value.success_rate["session"]
        across_sessions = value.across_sess_corr["session"]

    fig, axes = plt.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        """,
        figsize=figsize,
    )

    fig.suptitle(main_title)
    fig.subplots_adjust(hspace=1, wspace=0.5)

    # Success rate
    plot_multi_line_plot(
        data_dict=success_rate,
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=f"Success Rate",
        ytitle="Successful trials (%)",
        xtitle="Session",
        ylim=ylims["success"],
        line_color=colors,
        face_color="white",
        m_size=7,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["A"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Reaction time
    plot_multi_line_plot(
        data_dict=reaction_time,
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=f"Movement reaction time",
        ytitle="Reaction time (s)",
        xtitle="Session",
        ylim=ylims["reaction_time"],
        line_color=colors,
        face_color="white",
        m_size=7,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["B"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Cue to reward
    plot_multi_line_plot(
        data_dict=cue_to_reward,
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=f"Cue to reward time",
        ytitle="Cue to reward time (s)",
        xtitle="Session",
        ylim=ylims["cue_to_reward"],
        line_color=colors,
        face_color="white",
        m_size=7,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["C"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Within Session correlation
    plot_multi_line_plot(
        data_dict=within_corr,
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=f"Within Session Correlation",
        ytitle="Move. correlation (r)",
        xtitle="Session",
        ylim=ylims["within"],
        line_color=colors,
        face_color="white",
        m_size=7,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["D"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Across Session correlation
    plot_multi_line_plot(
        data_dict=across_corr,
        x_vals=across_sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=f"Across Session Correlation",
        ytitle="Move. correlation (r)",
        xtitle="Session",
        ylim=ylims["across"],
        line_color=colors,
        face_color="white",
        m_size=7,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["E"],
        legend=True,
        save=False,
        save_path=None,
    )

    # Movement correlation heatmap
    axes["F"].set_aspect(aspect="equal", adjustable="box")
    plot_general_heatmap(
        data=list(correlation_matrix.values())[0],
        figsize=(5, 5),
        title=list(correlation_matrix.keys())[0],
        xtitle="Session",
        ytitle="Session",
        cbar_label="Correlation (r)",
        hmap_range=ylims["cmap"],
        center=None,
        cmap="plasma",
        axis_width=2.5,
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    axes["G"].set_aspect(aspect="equal", adjustable="box")
    plot_general_heatmap(
        data=list(correlation_matrix.values())[1],
        figsize=(5, 5),
        title=list(correlation_matrix.keys())[1],
        xtitle="Session",
        ytitle="Session",
        cbar_label="Correlation (r)",
        hmap_range=ylims["cmap"],
        center=None,
        cmap="plasma",
        axis_width=2.5,
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    # Difference heatmap
    diff_matrix = (
        list(correlation_matrix.values())[0] - list(correlation_matrix.values())[1]
    )
    axes["H"].set_aspect(aspect="equal", adjustable="box")
    plot_general_heatmap(
        data=diff_matrix,
        figsize=(5, 5),
        title="Difference matrix",
        xtitle="Session",
        ytitle="Session",
        cbar_label=f"Diff. ({list(correlation_matrix.keys())[0]} - {list(correlation_matrix.keys())[1]})",
        hmap_range=(-0.2, 0.2),
        center=0,
        cmap="bwr",
        axis_width=2.5,
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
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, f"{main_title}_Summarized_Behavior")
        fig.savefig(fname + ".pdf")
