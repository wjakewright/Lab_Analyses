import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Plotting.plot_multi_line_plot import plot_multi_line_plot
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils

sns.set_style("ticks")


def plot_paAIP2_population_movement_encoding(
    EGFP_data_list,
    paAIP2_data_list,
    norm=False,
    rwd=False,
    figsize=(10, 10),
    display_stats=False,
    save=False,
    save_path=None,
):
    """
    Function to compare the movement encoding properties of EGFP and paAIP2 groups

        INPUT PARAMETERS
            EGFP_data_list - list of paAIP2_Population_Data objects for EGFP mice

            paAIP2_data_list - list of paAIP2_Population_Data objects for paAIP2 mice

            norm - boolean specifying whether to normalize data to early sessions

            rwd - boolean specifying whether to use rMRNs

            figsize - tuple specifying the size of the figure

            save - boolean specifying whether to save the figure or not

            save_path - str specifying where to save the figure

    """
    COLORS = ["black", "mediumslateblue"]

    # Grab specific parameters
    sessions = EGFP_data_list[0].sessions

    # Organize data for plotting
    (
        EGFP_frac_mvmt,
        EGFP_frac_silent,
        EGFP_frac_rwd,
        EGFP_LMP_corr,
        EGFP_mvmt_reliability,
        EGFP_mvmt_specificity,
    ) = organize_mvmt_encoding_vars(sessions, EGFP_data_list, rwd_mvmt=rwd, norm=norm)

    (
        paAIP_frac_mvmt,
        paAIP_frac_silent,
        paAIP_frac_rwd,
        paAIP_LMP_corr,
        paAIP_mvmt_reliability,
        paAIP_mvmt_specificity,
    ) = organize_mvmt_encoding_vars(sessions, paAIP2_data_list, rwd_mvmt=rwd, norm=norm)

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABC
        DEF
        """,
        figsize=figsize,
    )
    if rwd:
        fig.suptitle("paAIP2 Population Mvmt Encoding RWD")
    else:
        fig.suptitle("paAIP2 Population Mvmt Encoding")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    # Fraction MRNs
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_frac_mvmt,
            "paAIP2": paAIP_frac_mvmt,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Frac. MRNs",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
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
    # Fraction Silent
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_frac_silent,
            "paAIP2": paAIP_frac_silent,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Frac. Silent",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
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
    # Fraction rMRNs
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_frac_rwd,
            "paAIP2": paAIP_frac_rwd,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Frac. rMRNs",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
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
    # LMP Correlation
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_LMP_corr,
            "paAIP2": paAIP_LMP_corr,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="LMP corr. (r)",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
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
    # Movement Reliability
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_mvmt_reliability,
            "paAIP2": paAIP_mvmt_reliability,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Movement reliability",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
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
    # Movement Specificity
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_mvmt_specificity,
            "paAIP2": paAIP_mvmt_specificity,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Movement specificity",
        xtitle="Session",
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["F"],
        legend=True,
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, "paAIP2_Population_Mvmt_Encoding")
        fig.savefig(fname + ".pdf")

    # Statistics section
    if display_stats == False:
        return

    ## Coded only for parametric testing
    MRNs_table, _, MRNs_posthoc = t_utils.ANOVA_2way_mixed_posthoc(
        data_dict={
            "EGFP": EGFP_frac_mvmt,
            "paAIP": paAIP_frac_mvmt,
        },
        method="fdr_tsbh",
        rm_vals=sessions,
        compare_type="between",
    )
    silent_table, _, silent_posthoc = t_utils.ANOVA_2way_mixed_posthoc(
        data_dict={
            "EGFP": EGFP_frac_silent,
            "paAIP": paAIP_frac_silent,
        },
        method="fdr_tsbh",
        rm_vals=sessions,
        compare_type="between",
    )
    rMRNs_table, _, rMRNs_posthoc = t_utils.ANOVA_2way_mixed_posthoc(
        data_dict={
            "EGFP": EGFP_frac_rwd,
            "paAIP": paAIP_frac_rwd,
        },
        method="fdr_tsbh",
        rm_vals=sessions,
        compare_type="between",
    )
    LMP_table, _, LMP_posthoc = t_utils.ANOVA_2way_mixed_posthoc(
        data_dict={
            "EGFP": EGFP_LMP_corr,
            "paAIP": paAIP_LMP_corr,
        },
        method="fdr_tsbh",
        rm_vals=sessions,
        compare_type="between",
    )
    reli_table, _, reli_posthoc = t_utils.ANOVA_2way_mixed_posthoc(
        data_dict={
            "EGFP": EGFP_mvmt_reliability,
            "paAIP": paAIP_mvmt_reliability,
        },
        method="fdr_tsbh",
        rm_vals=sessions,
        compare_type="between",
    )
    speci_table, _, speci_posthoc = t_utils.ANOVA_2way_mixed_posthoc(
        data_dict={
            "EGFP": EGFP_mvmt_specificity,
            "paAIP": paAIP_mvmt_specificity,
        },
        method="fdr_tsbh",
        rm_vals=sessions,
        compare_type="between",
    )

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
        figsize=(11, 17),
    )
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title("Frac MRNs 2W ANOVA")
    A_table = axes2["A"].table(
        cellText=MRNs_table.values,
        colLabels=MRNs_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)

    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title("Frac MRNs Posthoc")
    B_table = axes2["B"].table(
        cellText=MRNs_posthoc.values,
        colLabels=MRNs_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)

    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title("Frac Silent 2W ANOVA")
    C_table = axes2["C"].table(
        cellText=silent_table.values,
        colLabels=silent_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)

    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title("Frac Silent Posthoc")
    D_table = axes2["D"].table(
        cellText=silent_posthoc.values,
        colLabels=silent_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)

    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title("Frac rMRNs 2W ANOVA")
    E_table = axes2["E"].table(
        cellText=MRNs_table.values,
        colLabels=MRNs_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)

    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title("Frac rMRNs Posthoc")
    F_table = axes2["F"].table(
        cellText=rMRNs_posthoc.values,
        colLabels=rMRNs_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)

    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title("LMP Corr 2W ANOVA")
    G_table = axes2["G"].table(
        cellText=LMP_table.values,
        colLabels=LMP_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)

    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title("LMP Corr Posthoc")
    H_table = axes2["H"].table(
        cellText=LMP_posthoc.values,
        colLabels=LMP_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)

    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title("Reliability 2W ANOVA")
    I_table = axes2["I"].table(
        cellText=reli_table.values,
        colLabels=reli_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)

    axes2["J"].axis("off")
    axes2["J"].axis("tight")
    axes2["J"].set_title("Reliability Posthoc")
    J_table = axes2["J"].table(
        cellText=reli_posthoc.values,
        colLabels=reli_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    J_table.auto_set_font_size(False)
    J_table.set_fontsize(8)

    axes2["K"].axis("off")
    axes2["K"].axis("tight")
    axes2["K"].set_title("Specificity 2W ANOVA")
    K_table = axes2["K"].table(
        cellText=speci_table.values,
        colLabels=speci_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    K_table.auto_set_font_size(False)
    K_table.set_fontsize(8)

    axes2["L"].axis("off")
    axes2["L"].axis("tight")
    axes2["L"].set_title("Specificity Posthoc")
    L_table = axes2["L"].table(
        cellText=speci_posthoc.values,
        colLabels=speci_posthoc.columns,
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
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = os.path.jon(save_path, f"paAIP2_Population_Mvmt_Encoding_Stats")
        fig2.savefig(fname + ".pdf")


def plot_paAIP2_population_movement_dynamics(
    EGFP_data_list,
    paAIP2_data_list,
    norm=False,
    mvmt_only=False,
    rwd=False,
    figsize=(10, 10),
    display_stats=False,
    save=False,
    save_path=None,
):
    """
    Function to compare the population dynamics during movements between
    EGFP and paAIP2 groups

    INPUT PARAMETERS
        EGFP_data_list - list of paAIP2_Population_Data objects for EGFP mice

        paAIP2_data_list - list of paAIP2_Population_Data objects for paAIP2 mice

        norm - boolean specifying whether to normalize to early sessions

        rwd - boolean specifying whether to use only rewarded movements

        figsize - tuple specifying the size of the figure

        display_stats - boolean specifying whether to perform statistics

        save - boolean specifying whether to save the figure or not

        save_path - str specifying where to save the figure

    """
    COLORS = ["black", "mediumslateblue"]
    # Grab parameters
    sessions = EGFP_data_list[0].sessions
    sampling_rate = EGFP_data_list[0].parameters["Sampling Rate"]
    activity_window = EGFP_data_list[0].parameters["Activity Window"]

    # Organize data for plotting
    (
        EGFP_mvmt_traces,
        EGFP_mvmt_amplitude,
        EGFP_mvmt_avg_onset,
        EGFP_mvmt_onset_jitter,
        EGFP_activity_correlation,
    ) = organize_mvmt_activity_vars(
        sessions, EGFP_data_list, mvmt_only=mvmt_only, rwd_mvmt=rwd, norm=norm
    )
    (
        paAIP_mvmt_traces,
        paAIP_mvmt_amplitude,
        paAIP_mvmt_avg_onset,
        paAIP_mvmt_onset_jitter,
        paAIP_activity_correlation,
    ) = organize_mvmt_activity_vars(
        sessions, paAIP2_data_list, mvmt_only=mvmt_only, rwd_mvmt=rwd, norm=norm
    )

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFGH
        IJKL
        """,
        figsize=figsize,
    )
    fig.suptitle("paAIP2 Population Activity Dynamics")

    # Plot data onto axes
    # Heatmaps
    ## EGFP early
    plot_activity_heatmap(
        EGFP_mvmt_traces[0],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="EGFP Early",
        cbar_label="Estimated spikes",
        hmap_rage=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["A"],
        save=False,
        save_path=None,
    )

    ## EGFP middle
    plot_activity_heatmap(
        EGFP_mvmt_traces[1],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="EGFP Middle",
        cbar_label="Estimated spikes",
        hmap_rage=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["B"],
        save=False,
        save_path=None,
    )

    ## EGFP late
    plot_activity_heatmap(
        EGFP_mvmt_traces[2],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="EGFP Late",
        cbar_label="Estimated spikes",
        hmap_rage=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["C"],
        save=False,
        save_path=None,
    )

    ## paAIP2 early
    plot_activity_heatmap(
        paAIP_mvmt_traces[0],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="paAIP2 Early",
        cbar_label="Estimated spikes",
        hmap_rage=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["E"],
        save=False,
        save_path=None,
    )

    ## paAIP2 middle
    plot_activity_heatmap(
        paAIP_mvmt_traces[1],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="paAIP2 Middle",
        cbar_label="Estimated spikes",
        hmap_rage=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["F"],
        save=False,
        save_path=None,
    )

    ## paAIP2 late
    plot_activity_heatmap(
        paAIP_mvmt_traces[2],
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="paAIP2 Late",
        cbar_label="Estimated spikes",
        hmap_rage=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="hot",
        axis_width=2,
        minor_ticks="x",
        ax=axes["G"],
        save=False,
        save_path=None,
    )

    # Activity traces
    ## EGFP
    ### Organize means and sems
    egfp_e_mean = np.nanmean(EGFP_mvmt_traces[0], axis=1)
    egfp_e_sem = stats.sem(EGFP_mvmt_traces[0], axis=1, nan_policy="omit")
    egfp_m_mean = np.nanmean(EGFP_mvmt_traces[1], axis=1)
    egfp_m_sem = stats.sem(EGFP_mvmt_traces[1], axis=1, nan_policy="omit")
    egfp_l_mean = np.nanmean(EGFP_mvmt_traces[2], axis=1)
    egfp_l_sem = stats.sem(EGFP_mvmt_traces[2], axis=1, nan_policy="omit")

    plot_mean_activity_traces(
        means=[egfp_e_mean, egfp_m_mean, egfp_l_mean],
        sems=[egfp_e_sem, egfp_m_sem, egfp_l_sem],
        group_names=["Early", "Middle", "Late"],
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=[0],
        figsize=(5, 5),
        colors=["lightgrey", "grey", "black"],
        title="EGFP",
        ytitle="Estimated spikes",
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )

    ## paAIP2
    ### Organize means and sems
    pa_e_mean = np.nanmean(paAIP_mvmt_traces[0], axis=1)
    pa_e_sem = stats.sem(paAIP_mvmt_traces[0], axis=1, nan_policy="omit")
    pa_m_mean = np.nanmean(paAIP_mvmt_traces[1], axis=1)
    pa_m_sem = stats.sem(paAIP_mvmt_traces[1], axis=1, nan_policy="omit")
    pa_l_mean = np.nanmean(paAIP_mvmt_traces[2], axis=1)
    pa_l_sem = stats.sem(paAIP_mvmt_traces[2], axis=1, nan_policy="omit")

    plot_mean_activity_traces(
        means=[pa_e_mean, pa_m_mean, pa_l_mean],
        sems=[pa_e_sem, pa_m_sem, pa_l_sem],
        group_names=["Early", "Middle", "Late"],
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=[0],
        figsize=(5, 5),
        colors=["lightskyblue", "mediumslateblue", "mediumblue"],
        title="paAIP2",
        ytitle="Estimated spikes",
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )

    # Longitudinal quantification
    # Movement event amplitude
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_mvmt_amplitude,
            "paAIP2": paAIP_mvmt_amplitude,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Estimated spikes",
        xtitle="Session",
        line_colors=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["I"],
        legend=True,
        save=False,
        save_path=None,
    )

    # Movement activity onset
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_mvmt_avg_onset,
            "paAIP2": paAIP_mvmt_avg_onset,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Avg. onset (s)",
        xtitle="Session",
        line_colors=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["J"],
        legend=True,
        save=False,
        save_path=None,
    )

    # Movement onset jitter
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_mvmt_onset_jitter,
            "paAIP2": paAIP_mvmt_onset_jitter,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Onset jitter (s)",
        xtitle="Session",
        line_colors=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["K"],
        legend=True,
        save=False,
        save_path=None,
    )

    # Movement activity correlation
    plot_multi_line_plot(
        data_dict={
            "EGFP": EGFP_activity_correlation,
            "paAIP2": paAIP_activity_correlation,
        },
        x_vals=sessions,
        plot_ind=False,
        figsize=(5, 5),
        title=None,
        ytitle="Pouplation corr. (r)",
        xtitle="Session",
        line_colors=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["K"],
        legend=True,
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
        fname = os.path.join(save_path, "paAIP2_Population_Dynamics")
        fig.savefig(fname + ".pdf")

    # Statistics session
    if not display_stats:
        return

    # Perform anovas
    amp_table, _, amp_posthoc = t_utils.ANOVA_2way_mixed_posthoc(
        data_dict={
            "EGFP": EGFP_mvmt_amplitude,
            "paAIP2": paAIP_mvmt_amplitude,
        },
        method="fdr_tsbh",
        rm_vals=sessions,
        compare_type="between",
        test_type="parametric",
    )
    onset_table, _, onset_posthoc = t_utils.ANOVA_2way_mixed_posthoc(
        data_dict={
            "EGFP": EGFP_mvmt_avg_onset,
            "paAIP2": paAIP_mvmt_avg_onset,
        },
        method="fdr_tsbh",
        rm_vals=sessions,
        compare_type="between",
        test_type="parametric",
    )
    jitter_table, _, jitter_posthoc = t_utils.ANOVA_2way_mixed_posthoc(
        data_dict={
            "EGFP": EGFP_mvmt_onset_jitter,
            "paAIP2": paAIP_mvmt_onset_jitter,
        },
        method="fdr_tsbh",
        rm_vals=sessions,
        compare_type="between",
        test_type="parametric",
    )
    corr_table, _, corr_posthoc = t_utils.ANOVA_2way_mixed_posthoc(
        data_dict={
            "EGFP": EGFP_activity_correlation,
            "paAIP2": paAIP_activity_correlation,
        },
        method="fdr_tsbh",
        rm_vals=sessions,
        compare_type="between",
        test_type="parametric",
    )

    # Display the stats
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        EF
        HI
        """,
        figsize=figsize,
    )

    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title("Amplitude 2W ANOVA")
    A_table = axes2["A"].table(
        cellText=amp_table.values,
        colLabels=amp_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)

    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title("Amplitude Posthoc")
    B_table = axes2["B"].table(
        cellText=amp_posthoc.values,
        colLabels=amp_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)

    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title("Avg. Onset 2W ANOVA")
    C_table = axes2["C"].table(
        cellText=onset_table.values,
        colLabels=onset_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)

    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title("Avg Onset Posthoc")
    D_table = axes2["D"].table(
        cellText=onset_posthoc.values,
        colLabels=onset_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)

    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title("Onset Jitter 2W ANOVA")
    E_table = axes2["E"].table(
        cellText=jitter_table.values,
        colLabels=jitter_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)

    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title("Onset Jitter Posthoc")
    F_table = axes2["F"].table(
        cellText=jitter_posthoc.values,
        colLabels=jitter_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)

    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title("Activity Corr 2W ANOVA")
    G_table = axes2["G"].table(
        cellText=corr_table.values,
        colLabels=corr_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)

    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title("Activity Corr Posthoc")
    G_table = axes2["G"].table(
        cellText=corr_posthoc.values,
        colLabels=corr_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = os.path.jon(save_path, f"paAIP2_Population_Dynamics_Stats")
        fig2.savefig(fname + ".pdf")


def organize_mvmt_activity_vars(
    sessions, data_list, mvmt_only=False, rwd_mvmt=False, norm=False
):
    """Helper function to organize movement activity variables for plotting"""
    # Initialize outputs
    mvmt_traces = []
    mvmt_amplitude = []
    mvmt_avg_onset = []
    mvmt_onset_jitter = []
    activity_correlation = []

    for i, session in enumerate(sessions):
        # Temp variables
        m_traces = []
        m_amplitude = []
        m_avg_onset = []
        m_onset_jitter = []
        a_correlation = []

        for j, data in enumerate(data_list):
            mvmt_cells = data.mvmt_cells_spikes[i]
            ## Get relevant data
            if not rwd_mvmt:
                traces = data.rwd_movement_traces_spikes[session]
                amplitude = data.rwd_movement_amplitude_spikes[session]
                avg_onset = data.rwd_mean_onsets_spikes[session]
                onset_jitter = data.rwd_mvmt_onset_jitter[session]
                correlation = data.med_vector_correlation[session]
            else:
                traces = data.movement_traces_spikes[session]
                amplitude = data.movement_amplitude_spikes[session]
                avg_onset = data.mean_onsets_spikes[session]
                onset_jitter = data.mvmt_onset_jitter[session]
                correlation = data.med_vector_correlation[session]

            # Subselect if only mvmt cells
            if mvmt_only:
                traces = d_utils.subselect_data_by_idxs(traces, mvmt_cells)
                amplitude = d_utils.subselect_data_by_idxs(amplitude, mvmt_cells)
                avg_onset = d_utils.subselect_data_by_idxs(avg_onset, mvmt_cells)
                onset_jitter = d_utils.subselect_data_by_idxs(onset_jitter, mvmt_cells)
                correlation = d_utils.subselect_data_by_idxs(correlation, mvmt_cells)

            # Average traces for individual cells
            means = [np.nanmean(x, axis=1) for x in traces if type(x) == np.ndarray]
            means = np.vstack(means).T
            ## Store temporary variables
            m_traces.append(means)
            m_amplitude.append(amplitude)
            m_avg_onset.append(avg_onset)
            m_onset_jitter.append(onset_jitter)
            a_correlation.append(correlation)

        # Concatenate across datasets
        m_traces = np.hstack(m_traces)
        m_amplitude = np.concatenate(m_amplitude)
        m_avg_onset = np.concatenate(m_avg_onset)
        m_onset_jitter = np.concatenate(m_onset_jitter)
        a_correlation = np.concatenate(a_correlation)

        mvmt_traces.append(m_traces)
        mvmt_amplitude.append(m_amplitude)
        mvmt_avg_onset.append(m_avg_onset)
        mvmt_onset_jitter.append(m_onset_jitter)
        activity_correlation.append(a_correlation)

    # Convert to 2d arrays
    mvmt_amplitude = convert_to_2D(mvmt_amplitude)
    mvmt_avg_onset = convert_to_2D(mvmt_avg_onset)
    mvmt_onset_jitter = convert_to_2D(mvmt_onset_jitter)
    activity_correlation = convert_to_2D(activity_correlation)

    # Normalize if specified
    if norm:
        mvmt_amplitude = np.array(
            [
                d_utils.normalized_relative_difference(
                    mvmt_amplitude[0, :], mvmt_amplitude[i, :]
                )
                for i in range(mvmt_amplitude.shape[0])
            ]
        )
        mvmt_avg_onset = np.array(
            [
                d_utils.normalized_relative_difference(
                    mvmt_avg_onset[0, :], mvmt_avg_onset[i, :]
                )
                for i in range(mvmt_avg_onset.shape[0])
            ]
        )
        mvmt_onset_jitter = np.array(
            [
                d_utils.normalized_relative_difference(
                    mvmt_onset_jitter[0, :], mvmt_onset_jitter[i, :]
                )
                for i in range(mvmt_onset_jitter.shape[0])
            ]
        )
        activity_correlation = np.array(
            [
                d_utils.normalized_relative_difference(
                    activity_correlation[0, :], activity_correlation[i, :]
                )
                for i in range(activity_correlation.shape[0])
            ]
        )

    return (
        mvmt_traces,
        mvmt_amplitude,
        mvmt_avg_onset,
        mvmt_onset_jitter,
        activity_correlation,
    )


def organize_mvmt_encoding_vars(sessions, data_list, rwd_mvmt=False, norm=False):
    """Helper function to organize movement encoding varibles for plotting"""
    frac_mvmt = []
    frac_silent = []
    frac_rwd = []
    LMP_corr = []
    mvmt_reliability = []
    mvmt_specificity = []

    for i, session in enumerate(sessions):
        # Temp variables
        f_mvmt = []
        f_silent = []
        f_rwd = []
        LMP = []
        reliability = []
        specificity = []

        for j, data in enumerate(data_list):
            MRNs = data.fraction_MRNs_spikes[session]
            silent = data.fraction_silent_spikes[session]
            rMRNs = data.fraction_rMRNs_spikes[session]
            corr = data.movement_correlation[session]
            reli = data.movement_reliability[session]
            speci = data.movement_specificity[session]
            ## Subselect for MRNs or nMRNs
            if rwd_mvmt:
                rwd_mvmt_cells = data.rwd_mvmt_cells_spikes[i]
                corr = d_utils.subselect_data_by_idxs(corr, rwd_mvmt_cells)
                reli = d_utils.subselect_data_by_idxs(reli, rwd_mvmt_cells)
                speci = d_utils.subselect_data_by_idxs(speci, rwd_mvmt_cells)
            else:
                mvmt_cells = data.mvmt_cells_spikes[i]
                corr = d_utils.subselect_data_by_idxs(corr, mvmt_cells)
                reli = d_utils.subselect_data_by_idxs(reli, mvmt_cells)
                speci = d_utils.subselect_data_by_idxs(speci, mvmt_cells)

            # Store temporary variables
            f_mvmt.append(MRNs)
            f_silent.append(silent)
            f_rwd.append(rMRNs)
            LMP.append(corr)
            reliability.append(reli)
            specificity.append(speci)

        # Concatente across datasets
        f_mvmt = np.concatenate(f_mvmt)
        f_silent = np.concatenate(f_silent)
        f_rwd = np.concatenate(f_rwd)
        LMP = np.concatenate(LMP)
        reliability = np.concatenate(reliability)
        specificity = np.concatenate(specificity)

        frac_mvmt.append(f_mvmt)
        frac_silent.append(f_silent)
        frac_rwd.append(f_rwd)
        LMP_corr.append(LMP)
        mvmt_reliability.append(reliability)
        mvmt_specificity.append(specificity)

    # Convert data into 2d arrays
    frac_mvmt = convert_to_2D(frac_mvmt)
    frac_silent = convert_to_2D(frac_silent)
    frac_rwd = convert_to_2D(frac_rwd)
    LMP_corr = convert_to_2D(LMP_corr)
    mvmt_reliability = convert_to_2D(mvmt_reliability)
    mvmt_specificity = convert_to_2D(mvmt_specificity)

    if norm:
        frac_mvmt = np.array(
            [
                d_utils.normalized_relative_difference(frac_mvmt[0, :], frac_mvmt[i, :])
                for i in range(frac_mvmt.shape[0])
            ]
        )
        frac_silent = np.array(
            [
                d_utils.normalized_relative_difference(
                    frac_silent[0, :], frac_silent[i, :]
                )
                for i in range(frac_silent.shape[0])
            ]
        )
        frac_rwd = np.array(
            [
                d_utils.normalized_relative_difference(frac_rwd[0, :], frac_rwd[i, :])
                for i in range(frac_rwd.shape[0])
            ]
        )
        LMP_corr = np.array(
            [
                d_utils.normalized_relative_difference(LMP_corr[0, :], LMP_corr[i, :])
                for i in range(LMP_corr.shape[0])
            ]
        )
        mvmt_reliability = np.array(
            [
                d_utils.normalized_relative_difference(
                    mvmt_reliability[0, :], mvmt_reliability[i, :]
                )
                for i in range(mvmt_reliability.shape[0])
            ]
        )
        mvmt_specificity = np.array(
            [
                d_utils.normalized_relative_difference(
                    mvmt_specificity[0, :], mvmt_specificity[i, :]
                )
                for i in range(mvmt_specificity.shape[0])
            ]
        )

    return (
        frac_mvmt,
        frac_silent,
        frac_rwd,
        LMP_corr,
        mvmt_reliability,
        mvmt_specificity,
    )


def convert_to_2D(data_list):
    """Helper function to convert list of 1D arrays of different lengths into
    a 2D array"""
    # Get max value
    max_len = np.max([len(x) for x in data_list])
    padded_arrays = []
    for data in data_list:
        padded_data = d_utils.pad_array_to_length(data, length=max_len, value=np.nan)
        padded_arrays.append(padded_data)

    # Stack arrays
    output_array = np.vstack(padded_arrays)

    return output_array
