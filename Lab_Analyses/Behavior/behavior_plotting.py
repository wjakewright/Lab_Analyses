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
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Utilities import data_utilities as utils
from Lab_Analyses.Utilities import test_utilities as t_utils

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
        cmap="seismic",
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


def plot_multi_group_bar_plots(
    data_dict,
    figsize=(7, 9),
    main_title=None,
    colors=["silver", "lightskyblue", "black", "mediumslateblue"],
    display_stats=False,
    test_method="fdr_tsbh",
    save=False,
    save_path=None,
):
    """Figure to plot lever behavior against multiple groups for
    beginner and expert sessions"""

    # Organize the relevant data
    main_keys = list(data_dict.keys())
    success_rate = {}
    reaction_time = {}
    cue_to_reward = {}
    within_corr = {}
    across_corr = {}

    for key, value in data_dict.items():
        # setup new keys
        e_key = f"{key}_early"
        l_key = f"{key}_late"
        success_rate[e_key] = value.ind_success_rate[:3, :].flatten()
        success_rate[l_key] = value.ind_success_rate[-2:, :].flatten()
        reaction_time[e_key] = value.ind_reaction_time[:3, :].flatten()
        reaction_time[l_key] = value.ind_reaction_time[-2:, :].flatten()
        cue_to_reward[e_key] = value.ind_cue_to_reward[:3, :].flatten()
        cue_to_reward[l_key] = value.ind_cue_to_reward[-2:, :].flatten()
        within_corr[e_key] = value.ind_within_sess_corr[:3, :].flatten()
        within_corr[l_key] = value.ind_within_sess_corr[-2:, :].flatten()
        across_corr[e_key] = value.ind_across_sess_corr[:3, :].flatten()
        across_corr[l_key] = value.ind_across_sess_corr[-2:, :].flatten()

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABC
        DE.
        """,
        figsize=figsize,
    )

    fig.suptitle(main_title)
    fig.subplots_adjust(hspace=1, wspace=0.5)

    # Success rate
    plot_swarm_bar_plot(
        data_dict=success_rate,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Success rate (%)",
        ylim=None,
        b_colors=colors,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=colors,
        s_size=4,
        s_alpha=1,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["A"],
        save=False,
        save_path=None,
    )

    # Reaction time
    plot_swarm_bar_plot(
        data_dict=reaction_time,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Reaction time (s)",
        ylim=None,
        b_colors=colors,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=colors,
        s_size=4,
        s_alpha=1,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["B"],
        save=False,
        save_path=None,
    )

    # Cue to reward time
    plot_swarm_bar_plot(
        data_dict=cue_to_reward,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Cue to reward (s)",
        ylim=None,
        b_colors=colors,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=colors,
        s_size=4,
        s_alpha=1,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["C"],
        save=False,
        save_path=None,
    )

    # Within correlation
    plot_swarm_bar_plot(
        data_dict=within_corr,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Correlation (r)",
        ylim=(0, None),
        b_colors=colors,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=colors,
        s_size=4,
        s_alpha=1,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["D"],
        save=False,
        save_path=None,
    )

    # Across correlation
    plot_swarm_bar_plot(
        data_dict=across_corr,
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Correlation (r)",
        ylim=(0, None),
        b_colors=colors,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.6,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=colors,
        s_size=4,
        s_alpha=1,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        x_rotation=45,
        ax=axes["E"],
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
        fname = os.path.jon(save_path, f"{main_title}_bar_plots")
        fig.savefig(fname + ".pdf")

    # Stats section
    if display_stats is False:
        return

    # Organize the data for 2-Way ANOVAs
    success_stats = {
        "Early": {
            main_keys[0]: success_rate[f"{main_keys[0]}_early"],
            main_keys[1]: success_rate[f"{main_keys[1]}_early"],
        },
        "Late": {
            main_keys[0]: success_rate[f"{main_keys[0]}_late"],
            main_keys[1]: success_rate[f"{main_keys[1]}_late"],
        },
    }
    reaction_stats = {
        "Early": {
            main_keys[0]: reaction_time[f"{main_keys[0]}_early"],
            main_keys[1]: reaction_time[f"{main_keys[1]}_early"],
        },
        "Late": {
            main_keys[0]: reaction_time[f"{main_keys[0]}_late"],
            main_keys[1]: reaction_time[f"{main_keys[1]}_late"],
        },
    }
    cue_stats = {
        "Early": {
            main_keys[0]: cue_to_reward[f"{main_keys[0]}_early"],
            main_keys[1]: cue_to_reward[f"{main_keys[1]}_early"],
        },
        "Late": {
            main_keys[0]: cue_to_reward[f"{main_keys[0]}_late"],
            main_keys[1]: cue_to_reward[f"{main_keys[1]}_late"],
        },
    }
    within_stats = {
        "Early": {
            main_keys[0]: within_corr[f"{main_keys[0]}_early"],
            main_keys[1]: within_corr[f"{main_keys[1]}_early"],
        },
        "Late": {
            main_keys[0]: within_corr[f"{main_keys[0]}_late"],
            main_keys[1]: within_corr[f"{main_keys[1]}_late"],
        },
    }
    across_stats = {
        "Early": {
            main_keys[0]: across_corr[f"{main_keys[0]}_early"],
            main_keys[1]: across_corr[f"{main_keys[1]}_early"],
        },
        "Late": {
            main_keys[0]: across_corr[f"{main_keys[0]}_late"],
            main_keys[1]: across_corr[f"{main_keys[1]}_late"],
        },
    }

    success_anova, success_posthoc = t_utils.ANOVA_2way_posthoc(
        success_stats,
        groups_list=["Opto", "Session"],
        variable="Success_Rate",
        method=test_method,
    )
    reaction_anova, reaction_posthoc = t_utils.ANOVA_2way_posthoc(
        reaction_stats,
        groups_list=["Opto", "Session"],
        variable="Reaction_Time",
        method=test_method,
    )
    cue_anova, cue_posthoc = t_utils.ANOVA_2way_posthoc(
        cue_stats,
        groups_list=["Opto", "Session"],
        variable="Cue_Reward",
        method=test_method,
    )
    within_anova, within_posthoc = t_utils.ANOVA_2way_posthoc(
        within_stats,
        groups_list=["Opto", "Session"],
        variable="Within_Corr",
        method=test_method,
    )
    across_anova, across_posthoc = t_utils.ANOVA_2way_posthoc(
        across_stats,
        groups_list=["Opto", "Session"],
        variable="Across_Corr",
        method=test_method,
    )

    # Format the table
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        IJ
        """,
        figsize=(11, 15),
    )

    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title("Success Rate 2W ANOVA")
    A_table = axes2["A"].table(
        cellText=success_anova.values,
        colLabels=success_anova.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)

    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title("Success Rate Posthoc")
    B_table = axes2["B"].table(
        cellText=success_posthoc.values,
        colLabels=success_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)

    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title("Reaction time 2W ANOVA")
    C_table = axes2["C"].table(
        cellText=reaction_anova.values,
        colLabels=reaction_anova.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)

    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title("Reaction time Posthoc")
    D_table = axes2["D"].table(
        cellText=reaction_posthoc.values,
        colLabels=reaction_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)

    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title("Cue to reward 2W ANOVA")
    E_table = axes2["E"].table(
        cellText=cue_anova.values,
        colLabels=cue_anova.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)

    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title("Cue to reward Posthoc")
    F_table = axes2["F"].table(
        cellText=cue_posthoc.values,
        colLabels=cue_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)

    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title("Within corr 2W ANOVA")
    G_table = axes2["G"].table(
        cellText=within_anova.values,
        colLabels=within_anova.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)

    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title("Within corr Posthoc")
    H_table = axes2["H"].table(
        cellText=within_posthoc.values,
        colLabels=within_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)

    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title("Across corr 2W ANOVA")
    I_table = axes2["I"].table(
        cellText=across_anova.values,
        colLabels=across_anova.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)

    axes2["J"].axis("off")
    axes2["J"].axis("tight")
    axes2["J"].set_title("Across corr Posthoc")
    J_table = axes2["J"].table(
        cellText=across_posthoc.values,
        colLabels=across_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    J_table.auto_set_font_size(False)
    J_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = os.path.jon(save_path, f"{main_title}_bar_plots_stats")
        fig2.savefig(fname + ".pdf")
