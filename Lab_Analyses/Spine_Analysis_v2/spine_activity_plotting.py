import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_box_plot import plot_box_plot
from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Plotting.plot_pie_chart import plot_pie_chart
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


def plot_basic_features(
    dataset,
    followup_dataset=None,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 4),
    hist_bins=25,
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """Function to plot the distribution of relative volumes and activity rates
    
        INPUT PARAMETERS
            dataset - Spine_Activity_Data object
            
            followup_dataset - optional Spine_Activity_Data object of the subsequent
                                session to use for volume comparison. Default is None
                                to use the followup volumes in the dataset

            exclude - str specifying a type of spine to exclude from analysis

            threshold - float or tuple of floats specifying the threshold cuttoffs 
                        for classifying plasticity

            figsize - tuple specifying the figure size

            hist_bins - int specifying how many bins to plot for the histogram

            showmeans - boolean specifying whether to show the mean on box plots

            test_type - str specifying whetehr to perform parametric or nonparameteric stats

            test_methods - str specifying the type of posthoc test to perform

            display_stats - boolean specifying whether to display stat results
            
            save - boolean specifying whether to save the figure or not
            
            save_path - str specifying where to save the path
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    spine_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }
    # Pull the relevant data
    initial_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    spine_activity_rate = dataset.spine_activity_rate

    # Calculate spine volumes
    ## Get followup volumes
    if followup_dataset == None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags
    ## Setup input lists
    volumes = [initial_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    ## Calculate
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=vol_norm, exclude=exclude
    )
    delta_volume = delta_volume[-1]

    # Classify plasticity
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=vol_norm
    )

    # Subselect data
    initial_volumes = d_utils.subselect_data_by_idxs(initial_volumes, spine_idxs)
    spine_activity_rate = d_utils.subselect_data_by_idxs(
        spine_activity_rate, spine_idxs
    )

    # Organize datad dictionaries
    initial_vol_dict = {}
    activity_dict = {}
    count_dict = {}
    for key, value in spine_groups.items():
        spines = eval(value)
        vol = initial_volumes[spines]
        activity = spine_activity_rate[spines]
        initial_vol_dict[key] = vol[~np.isnan(vol)]
        activity_dict[key] = activity[~np.isnan(activity)]
        count_dict[key] = np.sum(spines)

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        [["tl", "tm", "tr"], ["bl", "bm", "br"]], figsize=figsize,
    )
    fig.suptitle("Basic Spine Features")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ######################### Plat data onto the axes #############################
    # Relative volume distributions
    plot_histogram(
        data=delta_volume,
        bins=hist_bins,
        stat="probability",
        avlines=[1],
        title="\u0394 Volume",
        xtitle="\u0394 Volume",
        xlim=(0, 6),
        figsize=(5, 5),
        color="mediumblue",
        alpha=0.6,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["tl"],
        save=False,
        save_path=None,
    )

    # Initial volume vs Relative volume correlation
    plot_scatter_correlation(
        x_var=initial_volumes,
        y_var=delta_volume,
        CI=95,
        title="Initial vs \u0394 Volume",
        xtitle="Initial volume \u03BCm",
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
        ax=axes["tm"],
        save=False,
        save_path=None,
    )

    # Initial volume for spine types
    plot_box_plot(
        data_dict=initial_vol_dict,
        figsize=(5, 5),
        title="Initial Volumes",
        xtitle=None,
        ytitle="Initial volume \u03BCm",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["tr"],
        save=False,
        save_path=None,
    )

    # Percentage of different spine types
    plot_pie_chart(
        data_dict=count_dict,
        title="Fraction of plastic spines",
        figsize=(5, 5),
        colors=COLORS,
        alpha=1,
        edgecolor="white",
        txt_color="white",
        txt_size=10,
        legend="top",
        donut=0.6,
        linewidth=1.5,
        ax=axes["bl"],
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
        ax=axes["bm"],
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
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["br"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Spine_Activity_Figure_1")
        fig.savefig(fname + ".pdf")

    ########################### Statistics Section ###########################
    if display_stats == False:
        return

    ## Perform the statistics
    if test_type == "parametric":
        vol_f, vol_p, _, vol_test_df = t_utils.ANOVA_1way_posthoc(
            initial_vol_dict, test_method
        )
        activity_f, activity_p, _, activity_test_df = t_utils.ANOVA_1way_posthoc(
            activity_dict, test_method
        )
        test_title = f"One-Way ANOVA {test_method}"
    elif test_type == "nonparametric":
        vol_f, vol_p, vol_test_df = t_utils.kruskal_wallis_test(
            initial_vol_dict, "Conover", test_method,
        )
        activity_f, activity_p, activity_test_df = t_utils.kruskal_wallis_test(
            activity_dict, "Conover", test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"
    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic([["left", "right"]], figsize=(8, 4))
    ## Format the first table
    axes2["left"].axis("off")
    axes2["left"].axis("tight")
    axes2["left"].set_title(
        f"Initial Volume {test_title}\nF = {vol_f:.4}   p = {vol_p:.3E}"
    )
    left_table = axes2["left"].table(
        cellText=vol_test_df.values,
        colLabels=vol_test_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    left_table.auto_set_font_size(False)
    left_table.set_fontsize(8)
    ## Format the second table
    axes2["right"].axis("off")
    axes2["right"].axis("tight")
    axes2["right"].set_title(
        f"Event Rate {test_title}\nF = {activity_f:.4}    p = {activity_p:.3E}"
    )
    right_table = axes2["right"].table(
        cellText=activity_test_df.values,
        colLabels=vol_test_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    right_table.auto_set_font_size(False)
    right_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Spine_Activity_Figure_1_Stats")
        fig2.savefig(fname + ".pdf")


def plot_movement_related_activity(
    dataset,
    followup_dataset=None,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 6),
    hist_bins=25,
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """Function to plot movement-related activity of different spine classes
    
        INPUT PARAMETERS
            dataset - Spine_Activity_Data object
            
            followup_dataset - optional Spine_Activity_Data object of the subsequent 
                                session to use for volume comparision. Default is None,
                                to sue the followup volumes in the dataset
            
            exclude - str specifying type of spine to exclude from analysis
            
            threshold - float or tuple of floats specifying the threshold cutoffs for
                        classifying plasticity
                        
            figsize - tuple specifying the figure size
            
            hist_bins - int specifying how many  bins to plot for the histograms
            
            showmeans - boolean specifying whether to show mean on box plots
            
            test_type - str specifying whether to perform parametric or nonparametric stats
            
            test_method - str specifying the type of posthoc test to perform
            
            display_stats - boolean specifying whether to display stat results
            
            save - boolean specifying whether to save the figure or not
            
            save_path - str specifying where to save the figures
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    spine_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }

    # Pull relevant data
    sampling_rate = dataset.parameters["Sampling Rate"]
    activity_window = dataset.parameters["Activity Window"]
    if dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"
    ## Volume related information
    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset == None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags

    ## Movement identifiers
    movement_spines = dataset.movement_spines
    nonmovement_spines = dataset.nonmovement_spines
    rwd_movement_spines = dataset.rwd_movement_spines
    nonrwd_movement_spines = dataset.nonrwd_movement_spines

    ## Movement-related activity
    spine_movement_traces = dataset.spine_movement_traces
    spine_movement_calcium_traces = dataset.spine_movement_calcium_traces
    spine_movement_amplitude = dataset.spine_movement_amplitude
    spine_movement_calcium_amplitude = dataset.spine_movement_calcium_amplitude
    spine_movement_onset = dataset.spine_movement_onset
    spine_movement_calcium_onset = dataset.spine_movement_calcium_onset

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

    # Organize data
    ## Subselect present spines
    movement_spines = d_utils.subselect_data_by_idxs(movement_spines, spine_idxs)
    nonmovement_spines = d_utils.subselect_data_by_idxs(nonmovement_spines, spine_idxs)
    rwd_movement_spines = d_utils.subselect_data_by_idxs(
        rwd_movement_spines, spine_idxs
    )
    nonrwd_movement_spines = d_utils.subselect_data_by_idxs(
        nonrwd_movement_spines, spine_idxs
    )
    spine_movement_traces = d_utils.subselect_data_by_idxs(
        spine_movement_traces, spine_idxs
    )
    spine_movement_calcium_traces = d_utils.subselect_data_by_idxs(
        spine_movement_calcium_traces, spine_idxs
    )
    spine_movement_amplitude = d_utils.subselect_data_by_idxs(
        spine_movement_amplitude, spine_idxs
    )
    spine_movement_calcium_amplitude = d_utils.subselect_data_by_idxs(
        spine_movement_calcium_amplitude, spine_idxs
    )
    spine_movement_onset = d_utils.subselect_data_by_idxs(
        spine_movement_onset, spine_idxs
    )
    spine_movement_calcium_onset = d_utils.subselect_data_by_idxs(
        spine_movement_calcium_onset, spine_idxs
    )

    # Get the data for the correlation plot
    mrs_mvmt_amps = spine_movement_amplitude[movement_spines]
    mrs_mvmt_calcium_amps = spine_movement_calcium_amplitude[movement_spines]

    ## Seperate groups into dictionaries for grouped plots
    mvmt_fractions = {}
    nonmvmt_fractions = {}
    rwd_mvmt_fractions = {}
    nonrwd_mvmt_fractions = {}
    mrs_ind_mvmt_traces = {}
    mrs_ind_mvmt_calcium_traces = {}
    mrs_avg_mvmt_traces = {}
    mrs_avg_mvmt_calcium_traces = {}
    mrs_sem_mvmt_traces = {}
    mrs_sem_mvmt_calcium_traces = {}
    mrs_grouped_mvmt_amps = {}
    mrs_grouped_mvmt_calcium_amps = {}
    mrs_grouped_mvmt_onsets = {}
    mrs_grouped_mvmt_calcium_onsets = {}

    for key, value in spine_groups.items():
        ## Get spine types
        spines = eval(value)
        mvmt_spines = spines * movement_spines
        nmvmt_spines = spines * nonmovement_spines
        rwd_mvmt_spines = spines * rwd_movement_spines
        nrwd_mvmt_spines = spines * nonrwd_movement_spines
        ## Get fractions of the different types
        mvmt_fractions[key] = np.nansum(mvmt_spines)
        nonmvmt_fractions[key] = np.nansum(nmvmt_spines)
        rwd_mvmt_fractions[key] = np.nansum(rwd_mvmt_spines)
        nonrwd_mvmt_fractions[key] = np.nansum(nrwd_mvmt_spines)
        ## Grab grouped traces, amps, onsets
        mrs_traces = list(compress(spine_movement_traces, mvmt_spines))
        mrs_calcium_traces = list(compress(spine_movement_calcium_traces, mvmt_spines))
        ### Avg individual events
        trace_means = [
            np.nanmean(x, axis=1) for x in mrs_traces if type(x) == np.ndarray
        ]
        ca_trace_means = [
            np.nanmean(x, axis=1) for x in mrs_calcium_traces if type(x) == np.ndarray
        ]
        trace_means = np.vstack(trace_means)
        ca_trace_means = np.vstack(ca_trace_means)
        ### Add individual traces for heatmap plotting
        mrs_ind_mvmt_traces[key] = trace_means.T
        mrs_ind_mvmt_calcium_traces[key] = ca_trace_means.T
        ### Get avg traces
        group_trace_means = np.nanmean(trace_means, axis=0)
        group_ca_trace_means = np.nanmean(ca_trace_means, axis=0)
        group_trace_sem = stats.sem(trace_means, axis=0, nan_policy="omit")
        group_ca_trace_sem = stats.sem(ca_trace_means, axis=0, nan_policy="omit")
        mrs_avg_mvmt_traces[key] = group_trace_means
        mrs_avg_mvmt_calcium_traces[key] = group_ca_trace_means
        mrs_sem_mvmt_traces[key] = group_trace_sem
        mrs_sem_mvmt_calcium_traces[key] = group_ca_trace_sem
        ### Amps and Onsets
        mvmt_amps = spine_movement_amplitude[mvmt_spines]
        mvmt_onsets = spine_movement_onset[mvmt_spines]
        mvmt_ca_amps = spine_movement_calcium_amplitude[mvmt_spines]
        mvmt_ca_onsets = spine_movement_calcium_onset[mvmt_spines]
        mrs_grouped_mvmt_amps[key] = mvmt_amps[~np.isnan(mvmt_amps)]
        mrs_grouped_mvmt_calcium_amps[key] = mvmt_ca_amps[~np.isnan(mvmt_ca_amps)]
        mrs_grouped_mvmt_onsets[key] = mvmt_onsets[~np.isnan(mvmt_onsets)]
        mrs_grouped_mvmt_calcium_onsets[key] = mvmt_ca_onsets[~np.isnan(mvmt_ca_onsets)]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABGH.
        CDEF.
        IJKLM
        NOPQR
        """,
        figsize=figsize,
    )
    # fig, axes = plt.subplot_mosaic(
    #    """
    #    ABCDEF
    #    GHCDEF
    #    IJKLM.
    #    NOPQR.
    #    """,
    #   figsize=figsize,
    # )
    fig.suptitle("Movement-Related Spine Activity")

    ########################## Plot data onto the axes #############################
    ## Fractions of MRSs
    plot_pie_chart(
        mvmt_fractions,
        title="MRSs",
        figsize=(5, 5),
        colors=COLORS,
        alpha=1,
        edgecolor="white",
        txt_color="white",
        txt_size=9,
        legend=None,
        donut=0.6,
        linewidth=1.5,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    ## Fraction non MRSs
    plot_pie_chart(
        nonmvmt_fractions,
        title="nMRSs",
        figsize=(5, 5),
        colors=COLORS,
        alpha=1,
        edgecolor="white",
        txt_color="white",
        txt_size=9,
        legend=None,
        donut=0.6,
        linewidth=1.5,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    ## Fraction of reward MRSs
    plot_pie_chart(
        rwd_mvmt_fractions,
        figsize=(5, 5),
        title="rMRSs",
        colors=COLORS,
        alpha=1,
        edgecolor="white",
        txt_color="white",
        txt_size=9,
        legend=None,
        donut=0.6,
        linewidth=1.5,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    ## Fraction of non reward MRSs
    plot_pie_chart(
        nonrwd_mvmt_fractions,
        title="nrMRSs",
        colors=COLORS,
        alpha=1,
        edgecolor="white",
        txt_color="white",
        txt_size=9,
        legend=None,
        donut=0.6,
        linewidth=1.5,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    ## All MRS heatmap
    all_traces = np.hstack(list(mrs_ind_mvmt_traces.values()))
    plot_activity_heatmap(
        all_traces,
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="All MRSs",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="plasma",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    ## Enlarged MRS heatmap
    plot_activity_heatmap(
        mrs_ind_mvmt_traces["Enlarged"],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Enlarged MRSs",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="plasma",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    ## Shrunken MRS heatmap
    plot_activity_heatmap(
        mrs_ind_mvmt_traces["Shrunken"],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Shrunken MRSs",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="plasma",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Stable MRS heatmap
    plot_activity_heatmap(
        mrs_ind_mvmt_traces["Stable"],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Stable MRSs",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="plasma",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ## Movement-related GluSnFr traces
    plot_mean_activity_traces(
        means=list(mrs_avg_mvmt_traces.values()),
        sems=list(mrs_sem_mvmt_traces.values()),
        group_names=list(mrs_avg_mvmt_traces.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="iGluSnFr Traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    ## Movement-related calcium traces
    plot_mean_activity_traces(
        means=list(mrs_avg_mvmt_calcium_traces.values()),
        sems=list(mrs_sem_mvmt_calcium_traces.values()),
        group_names=list(mrs_avg_mvmt_calcium_traces.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Calcium Traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["N"],
        save=False,
        save_path=None,
    )
    ## GluSnFr amp correlation
    plot_scatter_correlation(
        x_var=mrs_mvmt_amps,
        y_var=delta_volume,
        CI=95,
        title="GluSnFr",
        xtitle=f"Event amplitude ({activity_type})",
        ytitle="\u0394 Volume",
        figsize=(5, 5),
        xlim=(0, None),
        ylim=(0, None),
        marker_size=25,
        face_color="cmap",
        edge_color="white",
        edge_width=0.3,
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
    ## Calcium amp correlation
    plot_scatter_correlation(
        x_var=mrs_mvmt_calcium_amps,
        y_var=delta_volume,
        CI=95,
        title="Calcium",
        xtitle=f"Event amplitude ({activity_type})",
        ytitle="\u0394 Volume",
        figsize=(5, 5),
        xlim=(0, None),
        ylim=(0, None),
        marker_size=25,
        face_color="cmap",
        edge_color="white",
        edge_width=0.3,
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
    ## Grouped GluSnFr amplitude
    plot_box_plot(
        mrs_grouped_mvmt_amps,
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
        b_width=0.6,
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    ## Grouped Calcium amplitude
    plot_box_plot(
        mrs_grouped_mvmt_calcium_amps,
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
        b_width=0.6,
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
        ax=axes["P"],
        save=False,
        save_path=None,
    )
    ## GluSnFr onset histogram
    plot_histogram(
        data=list(mrs_grouped_mvmt_onsets.values()),
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
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    ## GluSnFr onset histogram
    plot_histogram(
        data=list(mrs_grouped_mvmt_calcium_onsets.values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="Calcium",
        xtitle="Relative onset (s)",
        xlim=activity_window,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["Q"],
        save=False,
        save_path=None,
    )
    ## Grouped GluSnFr onsets
    plot_box_plot(
        mrs_grouped_mvmt_onsets,
        figsize=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Relative onset (s)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.6,
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
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    ## Grouped GluSnFr onsets
    plot_box_plot(
        mrs_grouped_mvmt_calcium_onsets,
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Relative onset (s)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.6,
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
        ax=axes["R"],
        save=False,
        save_path=None,
    )

    # fig.tight_layout()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Spine_Activity_Figure_2")
        fig.savefig(fname + ".pdf")

    ######################### Statistics Section ############################
    if display_stats == False:
        return

    ## Perform the statistics
    if test_type == "parametric":
        g_amp_f, g_amp_p, _, g_amp_df = t_utils.ANOVA_1way_posthoc(
            mrs_grouped_mvmt_amps, test_method,
        )
        c_amp_f, c_amp_p, _, c_amp_df = t_utils.ANOVA_1way_posthoc(
            mrs_grouped_mvmt_calcium_amps, test_method,
        )
        g_onset_f, g_onset_p, _, g_onset_df = t_utils.ANOVA_1way_posthoc(
            mrs_grouped_mvmt_onsets, test_method,
        )
        c_onset_f, c_onset_p, _, c_onset_df = t_utils.ANOVA_1way_posthoc(
            mrs_grouped_mvmt_calcium_onsets, test_method,
        )
        test_title = f"One-Way ANOVA {test_method}"
    elif test_type == "nonparametric":
        g_amp_f, g_amp_p, g_amp_df = t_utils.kruskal_wallis_test(
            mrs_grouped_mvmt_amps, "Conover", test_method,
        )
        c_amp_f, c_amp_p, c_amp_df = t_utils.kruskal_wallis_test(
            mrs_grouped_mvmt_calcium_amps, "Conover", test_method,
        )
        g_onset_f, g_onset_p, g_onset_df = t_utils.kruskal_wallis_test(
            mrs_grouped_mvmt_onsets, "Conover", test_method,
        )
        c_onset_f, c_onset_p, c_onset_df = t_utils.kruskal_wallis_test(
            mrs_grouped_mvmt_calcium_onsets, "Conover", test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"
    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        """,
        figsize=(8, 6),
    )
    ## Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(
        f"GluSnFr Amp {test_title}\nF = {g_amp_f:.4}  p = {g_amp_p:.3E}"
    )
    A_table = axes2["A"].table(
        cellText=g_amp_df.values,
        colLabels=g_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(
        f"Calcium Amp {test_title}\nF = {c_amp_f:.4}  p = {c_amp_p:.3E}"
    )
    B_table = axes2["B"].table(
        cellText=c_amp_df.values,
        colLabels=c_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(
        f"GluSnFr Onsets {test_title}\nF = {g_onset_f:.4}  p = {g_onset_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=g_onset_df.values,
        colLabels=g_onset_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(
        f"Calcium Onset {test_title}\nF = {c_onset_f:.4}  p = {c_onset_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=c_onset_df.values,
        colLabels=c_onset_df.columns,
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
        fname = os.path.join(save_path, "Spine_Activity_Figure_2_Stats")
        fig2.savefig(fname + ".pdf")


def plot_rewarded_movement_related_activity(
    dataset,
    followup_dataset=None,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 6),
    hist_bins=25,
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """
    Function to plot rewarded and non-rewarded movement related activity for different
    spine classes

    INPUT PARAMETERS
        dataset - Spine_Activity_Data object
            
        followup_dataset - optional Spine_Activity_Data object of the subsequent 
                            session to use for volume comparision. Default is None,
                            to sue the followup volumes in the dataset
        
        exclude - str specifying type of spine to exclude from analysis
        
        threshold - float or tuple of floats specifying the threshold cutoffs for
                    classifying plasticity
                    
        figsize - tuple specifying the figure size
        
        hist_bins - int specifying how many  bins to plot for the histograms
        
        showmeans - boolean specifying whether to show means on box plots
        
        test_type - str specifying whether to perform parametric or nonparametric stats
        
        test_method - str specifying the type of posthoc test to perform
        
        display_stats - boolean specifying whether to display stat results
        
        save - boolean specifying whether to save the figure or not
        
        save_path - str specifying where to save the figures
    
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    plastic_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }

    # Pull relevant data
    sampling_rate = dataset.parameters["Sampling Rate"]
    activity_window = dataset.parameters["Activity Window"]
    if dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"
    ## Volume related information
    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset == None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags
    ## Movement identifier
    movement_spines = dataset.movement_spines

    ## Movement-related activity
    spine_rwd_movement_traces = dataset.spine_rwd_movement_traces
    spine_rwd_movement_calcium_traces = dataset.spine_rwd_movement_calcium_traces
    spine_rwd_movement_amplitude = dataset.spine_rwd_movement_amplitude
    spine_rwd_movement_calcium_amplitude = dataset.spine_rwd_movement_calcium_amplitude
    spine_rwd_movement_onset = dataset.spine_rwd_movement_onset
    spine_rwd_movement_calcium_onset = dataset.spine_rwd_movement_calcium_onset
    spine_nonrwd_movement_traces = dataset.spine_nonrwd_movement_traces
    spine_nonrwd_movement_calcium_traces = dataset.spine_nonrwd_movement_calcium_traces
    spine_nonrwd_movement_amplitude = dataset.spine_nonrwd_movement_amplitude
    spine_nonrwd_movement_calcium_amplitude = (
        dataset.spine_nonrwd_movement_calcium_amplitude
    )
    spine_nonrwd_movement_onset = dataset.spine_nonrwd_movement_onset
    spine_nonrwd_movement_calcium_onset = dataset.spine_nonrwd_movement_calcium_onset

    # Calculate relative voluems
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
    movement_spines = d_utils.subselect_data_by_idxs(movement_spines, spine_idxs)
    spine_rwd_movement_traces = d_utils.subselect_data_by_idxs(
        spine_rwd_movement_traces, spine_idxs
    )
    spine_rwd_movement_calcium_traces = d_utils.subselect_data_by_idxs(
        spine_rwd_movement_calcium_traces, spine_idxs,
    )
    spine_rwd_movement_amplitude = d_utils.subselect_data_by_idxs(
        spine_rwd_movement_amplitude, spine_idxs,
    )
    spine_rwd_movement_calcium_amplitude = d_utils.subselect_data_by_idxs(
        spine_rwd_movement_calcium_amplitude, spine_idxs,
    )
    spine_rwd_movement_onset = d_utils.subselect_data_by_idxs(
        spine_rwd_movement_onset, spine_idxs,
    )
    spine_rwd_movement_calcium_onset = d_utils.subselect_data_by_idxs(
        spine_rwd_movement_calcium_onset, spine_idxs,
    )
    spine_nonrwd_movement_traces = d_utils.subselect_data_by_idxs(
        spine_nonrwd_movement_traces, spine_idxs,
    )
    spine_nonrwd_movement_calcium_traces = d_utils.subselect_data_by_idxs(
        spine_nonrwd_movement_calcium_traces, spine_idxs
    )
    spine_nonrwd_movement_amplitude = d_utils.subselect_data_by_idxs(
        spine_nonrwd_movement_amplitude, spine_idxs,
    )
    spine_nonrwd_movement_calcium_amplitude = d_utils.subselect_data_by_idxs(
        spine_nonrwd_movement_calcium_amplitude, spine_idxs,
    )
    spine_nonrwd_movement_onset = d_utils.subselect_data_by_idxs(
        spine_nonrwd_movement_onset, spine_idxs,
    )
    spine_nonrwd_movement_calcium_onset = d_utils.subselect_data_by_idxs(
        spine_nonrwd_movement_calcium_onset, spine_idxs,
    )

    ## Get data for direct rewarded vs nonrewarded comparisons
    ### Rewarded traces
    rewarded_traces = compress(spine_rwd_movement_traces, movement_spines)
    rewarded_trace_means = [
        np.nanmean(x, axis=1) for x in rewarded_traces if type(x) == np.ndarray
    ]
    rewarded_trace_means = np.vstack(rewarded_trace_means)
    rewarded_heatmap_traces = rewarded_trace_means.T
    rewarded_trace_group_mean = np.nanmean(rewarded_trace_means, axis=0)
    rewarded_trace_group_sem = stats.sem(
        rewarded_trace_means, axis=0, nan_policy="omit"
    )
    rewarded_ca_traces = compress(spine_rwd_movement_calcium_traces, movement_spines)
    rewarded_ca_trace_means = [
        np.nanmean(x, axis=1) for x in rewarded_ca_traces if type(x) == np.ndarray
    ]
    rewarded_ca_trace_means = np.vstack(rewarded_ca_trace_means)
    rewarded_ca_heatmap_traces = rewarded_ca_trace_means.T
    rewarded_ca_trace_group_mean = np.nanmean(rewarded_ca_trace_means, axis=0)
    rewarded_ca_trace_group_sem = stats.sem(
        rewarded_ca_trace_means, axis=0, nan_policy="omit"
    )
    ### Nonrewarded traces
    nonrewarded_traces = compress(spine_nonrwd_movement_traces, movement_spines)
    nonrewarded_trace_means = [
        np.nanmean(x, axis=1) for x in nonrewarded_traces if type(x) == np.ndarray
    ]
    nonrewarded_trace_means = np.vstack(nonrewarded_trace_means)
    nonrewarded_heatmap_traces = nonrewarded_trace_means.T
    nonrewarded_trace_group_mean = np.nanmean(nonrewarded_trace_means, axis=0)
    nonrewarded_trace_group_sem = stats.sem(
        nonrewarded_trace_means, axis=0, nan_policy="omit"
    )
    nonrewarded_ca_traces = compress(
        spine_nonrwd_movement_calcium_traces, movement_spines
    )
    nonrewarded_ca_trace_means = [
        np.nanmean(x, axis=1) for x in nonrewarded_ca_traces if type(x) == np.ndarray
    ]
    nonrewarded_ca_trace_means = np.vstack(nonrewarded_ca_trace_means)
    nonrewarded_ca_heatmap_traces = nonrewarded_ca_trace_means.T
    nonrewarded_ca_trace_group_mean = np.nanmean(nonrewarded_ca_trace_means, axis=0)
    nonrewarded_ca_trace_group_sem = stats.sem(
        nonrewarded_ca_trace_means, axis=0, nan_policy="omit"
    )
    rewarded_amps = spine_rwd_movement_amplitude[
        ~np.isnan(spine_rwd_movement_amplitude)
    ]
    nonrewarded_amps = spine_nonrwd_movement_amplitude[
        ~np.isnan(spine_nonrwd_movement_amplitude)
    ]
    rewarded_ca_amps = spine_rwd_movement_calcium_amplitude[
        ~np.isnan(spine_rwd_movement_calcium_amplitude)
    ]
    nonrewarded_ca_amps = spine_nonrwd_movement_calcium_amplitude[
        ~np.isnan(spine_nonrwd_movement_calcium_amplitude)
    ]

    ## Plastic spine groups
    rwd_traces = {}
    rwd_sems = {}
    rwd_amps = {}
    rwd_onsets = {}
    rwd_ca_traces = {}
    rwd_ca_sems = {}
    rwd_ca_amps = {}
    rwd_ca_onsets = {}
    nonrwd_traces = {}
    nonrwd_sems = {}
    nonrwd_amps = {}
    nonrwd_onsets = {}
    nonrwd_ca_traces = {}
    nonrwd_ca_sems = {}
    nonrwd_ca_amps = {}
    nonrwd_ca_onsets = {}

    for key, value in plastic_groups.items():
        # Get spine types
        spines = eval(value)
        spines = spines * movement_spines
        # Process traces
        r_traces = list(compress(spine_rwd_movement_traces, spines))
        r_ca_traces = list(compress(spine_rwd_movement_calcium_traces, spines))
        n_traces = list(compress(spine_nonrwd_movement_traces, spines))
        n_ca_traces = list(compress(spine_nonrwd_movement_calcium_traces, spines))
        r_trace_means = [
            np.nanmean(x, axis=1) for x in r_traces if type(x) == np.ndarray
        ]
        r_ca_trace_means = [
            np.nanmean(x, axis=1) for x in r_ca_traces if type(x) == np.ndarray
        ]
        n_trace_means = [
            np.nanmean(x, axis=1) for x in n_traces if type(x) == np.ndarray
        ]
        n_ca_trace_means = [
            np.nanmean(x, axis=1) for x in n_ca_traces if type(x) == np.ndarray
        ]
        r_means = np.vstack(r_trace_means)
        r_ca_means = np.vstack(r_ca_trace_means)
        n_means = np.vstack(n_trace_means)
        n_ca_means = np.vstack(n_ca_trace_means)
        rwd_traces[key] = np.nanmean(r_means, axis=0)
        rwd_sems[key] = stats.sem(r_means, axis=0, nan_policy="omit")
        rwd_ca_traces[key] = np.nanmean(r_ca_means, axis=0)
        rwd_ca_sems[key] = stats.sem(r_ca_means, axis=0, nan_policy="omit")
        nonrwd_traces[key] = np.nanmean(n_means, axis=0)
        nonrwd_sems[key] = stats.sem(n_means, axis=0)
        nonrwd_ca_traces[key] = np.nanmean(n_ca_means, axis=0)
        nonrwd_ca_sems[key] = stats.sem(n_ca_means, axis=0, nan_policy="omit")
        ## Amplitudes and onsets
        r_amps = spine_rwd_movement_amplitude[spines]
        rwd_amps[key] = r_amps[~np.isnan(r_amps)]
        r_ca_amps = spine_rwd_movement_calcium_amplitude[spines]
        rwd_ca_amps[key] = r_ca_amps[~np.isnan(r_ca_amps)]
        r_onsets = spine_rwd_movement_onset[spines]
        rwd_onsets[key] = r_onsets[~np.isnan(r_onsets)]
        r_ca_onsets = spine_rwd_movement_calcium_onset[spines]
        rwd_ca_onsets[key] = r_ca_onsets[~np.isnan(r_ca_onsets)]
        n_amps = spine_nonrwd_movement_amplitude[spines]
        nonrwd_amps[key] = n_amps[~np.isnan(n_amps)]
        n_ca_amps = spine_nonrwd_movement_calcium_amplitude[spines]
        nonrwd_ca_amps[key] = n_ca_amps[~np.isnan(n_ca_amps)]
        n_onsets = spine_nonrwd_movement_onset[spines]
        nonrwd_onsets[key] = n_onsets[~np.isnan(n_onsets)]
        n_ca_onsets = spine_nonrwd_movement_calcium_onset[spines]
        nonrwd_ca_onsets[key] = n_ca_onsets[~np.isnan(n_ca_onsets)]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDEF
        GHIJKL
        MNOPQR
        """,
        figsize=figsize,
    )
    fig.suptitle("Rewarded Movement-Related Spine Activity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ########################## Plot data onto the axes ##############################
    # Rewarded heatmap
    plot_activity_heatmap(
        rewarded_heatmap_traces,
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Rewarded Mvmts",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="plasma",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # Rewarded heatmap
    plot_activity_heatmap(
        nonrewarded_heatmap_traces,
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Non-rewarded Mvmts",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="plasma",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Rewarded vs non reward GluSnFr traces
    plot_mean_activity_traces(
        means=[rewarded_trace_group_mean, nonrewarded_trace_group_mean],
        sems=[rewarded_trace_group_sem, nonrewarded_trace_group_sem],
        group_names=["Rewarded", "Non-rewarded"],
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=["mediumblue", "firebrick"],
        title="GluSnFr Traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # Rewarded vs non reward Calcium traces
    plot_mean_activity_traces(
        means=[rewarded_ca_trace_group_mean, nonrewarded_ca_trace_group_mean],
        sems=[rewarded_ca_trace_group_sem, nonrewarded_ca_trace_group_sem],
        group_names=["Rewarded", "Non-rewarded"],
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=["mediumblue", "firebrick"],
        title="Calcium Traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Rewarded vs non reward amplitudes
    plot_box_plot(
        {"Rewarded": rewarded_amps, "Non-rewarded": nonrewarded_amps},
        figsize=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=(0, None),
        b_colors=["mediumblue", "firebrick"],
        b_edgecolors="white",
        b_err_colors=["mediumblue", "firebrick"],
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
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # Rewarded vs non reward calcium amplitudes
    plot_box_plot(
        {"Rewarded": rewarded_ca_amps, "Non-rewarded": nonrewarded_ca_amps},
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=(0, None),
        b_colors=["mediumblue", "firebrick"],
        b_edgecolors="white",
        b_err_colors=["mediumblue", "firebrick"],
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
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    # Plastic rewarded GluSnFr traces
    plot_mean_activity_traces(
        means=list(rwd_traces.values()),
        sems=list(rwd_sems.values()),
        group_names=list(rwd_traces.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Rewarded GluSnFr",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    # Plastic nonrewarded GluSnFr traces
    plot_mean_activity_traces(
        means=list(nonrwd_traces.values()),
        sems=list(nonrwd_sems.values()),
        group_names=list(nonrwd_traces.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Non-rewarded GluSnFr",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    # Plastic rewarded Calcium traces
    plot_mean_activity_traces(
        means=list(rwd_ca_traces.values()),
        sems=list(rwd_ca_sems.values()),
        group_names=list(rwd_ca_traces.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Rewarded Calcium",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    # Plastic nonrewarded Calcium traces
    plot_mean_activity_traces(
        means=list(nonrwd_ca_traces.values()),
        sems=list(nonrwd_ca_sems.values()),
        group_names=list(nonrwd_ca_traces.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=[0],
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Non-rewarded Calcium",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["P"],
        save=False,
        save_path=None,
    )
    # Plastic rewarded GluSnFr amps
    plot_box_plot(
        rwd_amps,
        figsize=(5, 5),
        title="Rewarded GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    # Plastic non-rewarded GluSnFr amps
    plot_box_plot(
        nonrwd_amps,
        figsize=(5, 5),
        title="Non-rewarded GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["N"],
        save=False,
        save_path=None,
    )
    # Plastic rewarded Calcium amps
    plot_box_plot(
        rwd_ca_amps,
        figsize=(5, 5),
        title="Rewarded Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Plastic rewarded Calcium amps
    plot_box_plot(
        nonrwd_ca_amps,
        figsize=(5, 5),
        title="Non-rewarded Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["Q"],
        save=False,
        save_path=None,
    )
    # Reward GluSnFr Onsets
    plot_histogram(
        data=list(rwd_onsets.values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="Rewarded GluSnFr",
        xtitle="Relative onset (s)",
        xlim=activity_window,
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
    # Non-eward GluSnFr Onsets
    plot_histogram(
        data=list(nonrwd_onsets.values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="Non-rewarded GluSnFr",
        xtitle="Relative onset (s)",
        xlim=activity_window,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    # Reward Calcium Onsets
    plot_histogram(
        data=list(rwd_ca_onsets.values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="Rewarded Calcium",
        xtitle="Relative onset (s)",
        xlim=activity_window,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    # non-eward Calcium Onsets
    plot_histogram(
        data=list(nonrwd_ca_onsets.values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="Non-rewarded Calcium",
        xtitle="Relative onset (s)",
        xlim=activity_window,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["R"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Spine_Activity_Figure_3")
        fig.savefig(fname + ".pdf")

    ################################# Statistics Section ##################################
    if display_stats == False:
        return

    # Perform the statistics
    if test_type == "parametric":
        r_n_amp_t, r_n_amp_p = stats.ttest_ind(
            rewarded_amps, nonrewarded_amps, nan_policy="omit"
        )
        r_n_ca_amp_t, r_n_ca_amp_p = stats.ttest_ind(
            rewarded_ca_amps, nonrewarded_ca_amps, nan_policy="omit"
        )
        t_title = "T-Test"
        rwd_amp_f, rwd_amp_p, _, rwd_amp_df = t_utils.ANOVA_1way_posthoc(
            rwd_amps, test_method,
        )
        rwd_ca_amp_f, rwd_ca_amp_p, _, rwd_ca_amp_df = t_utils.ANOVA_1way_posthoc(
            rwd_ca_amps, test_method
        )
        nonrwd_amp_f, nonrwd_amp_p, _, nonrwd_amp_df = t_utils.ANOVA_1way_posthoc(
            nonrwd_amps, test_method,
        )
        (
            nonrwd_ca_amp_f,
            nonrwd_ca_amp_p,
            _,
            nonrwd_ca_amp_df,
        ) = t_utils.ANOVA_1way_posthoc(nonrwd_ca_amps, test_method)
        rwd_onset_f, rwd_onset_p, _, rwd_onset_df = t_utils.ANOVA_1way_posthoc(
            rwd_onsets, test_method,
        )
        rwd_ca_onset_f, rwd_ca_onset_p, _, rwd_ca_onset_df = t_utils.ANOVA_1way_posthoc(
            rwd_ca_onsets, test_method
        )
        nonrwd_onset_f, nonrwd_onset_p, _, nonrwd_onset_df = t_utils.ANOVA_1way_posthoc(
            nonrwd_onsets, test_method,
        )
        (
            nonrwd_ca_onset_f,
            nonrwd_ca_onset_p,
            _,
            nonrwd_ca_onset_df,
        ) = t_utils.ANOVA_1way_posthoc(nonrwd_ca_onsets, test_method)
        test_title = f"One-Way ANOVA {test_method}"
    if test_type == "nonparametric":
        r_n_amp_t, r_n_amp_p = stats.mannwhitneyu(
            rewarded_amps[~np.isnan(rewarded_amps)],
            nonrewarded_amps[~np.isnan(nonrewarded_amps)],
        )
        r_n_ca_amp_t, r_n_ca_amp_p = stats.mannwhitneyu(
            rewarded_ca_amps[~np.isnan(rewarded_ca_amps)],
            nonrewarded_ca_amps[~np.isnan(nonrewarded_ca_amps)],
        )
        t_title = "Mann-Whitney U"
        rwd_amp_f, rwd_amp_p, rwd_amp_df = t_utils.kruskal_wallis_test(
            rwd_amps, "Conover", test_method,
        )
        rwd_ca_amp_f, rwd_ca_amp_p, rwd_ca_amp_df = t_utils.kruskal_wallis_test(
            rwd_ca_amps, "Conover", test_method
        )
        nonrwd_amp_f, nonrwd_amp_p, nonrwd_amp_df = t_utils.kruskal_wallis_test(
            nonrwd_amps, "Conover", test_method,
        )
        (
            nonrwd_ca_amp_f,
            nonrwd_ca_amp_p,
            nonrwd_ca_amp_df,
        ) = t_utils.kruskal_wallis_test(nonrwd_ca_amps, "Conover", test_method)
        rwd_onset_f, rwd_onset_p, rwd_onset_df = t_utils.kruskal_wallis_test(
            rwd_onsets, "Conover", test_method,
        )
        rwd_ca_onset_f, rwd_ca_onset_p, rwd_ca_onset_df = t_utils.kruskal_wallis_test(
            rwd_ca_onsets, "Conover", test_method
        )
        nonrwd_onset_f, nonrwd_onset_p, nonrwd_onset_df = t_utils.kruskal_wallis_test(
            nonrwd_onsets, "Conover", test_method,
        )
        (
            nonrwd_ca_onset_f,
            nonrwd_ca_onset_p,
            nonrwd_ca_onset_df,
        ) = t_utils.kruskal_wallis_test(nonrwd_ca_onsets, "Conover", test_method)
        test_title = f"Kruskal-Wallis {test_method}"

    # Display the statistics
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        IJ
        """,
        figsize=(8, 12),
    )
    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(
        f"Rwd vs Non-rwd GluSnFr Amplitude\n{t_title}\nt = {r_n_amp_t:.4}  p = {r_n_amp_p:.3E}"
    )
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(
        f"Rwd vs Non-rwd Calcium Amplitude\n{t_title}\nt = {r_n_ca_amp_t:.4}  p = {r_n_ca_amp_p:.3E}"
    )
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(
        f"Rewarded GluSnFr Amp\n {test_title}\nF = {rwd_amp_f:.4}  p = {rwd_amp_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=rwd_amp_df.values,
        colLabels=rwd_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(
        f"Rewarded Calcium Amp\n {test_title}\nF = {rwd_ca_amp_f:.4}  p = {rwd_ca_amp_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=rwd_ca_amp_df.values,
        colLabels=rwd_ca_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(
        f"Non-rewarded GluSnFr Amp\n {test_title}\nF = {nonrwd_amp_f:.4}  p = {nonrwd_amp_p:.3E}"
    )
    E_table = axes2["E"].table(
        cellText=nonrwd_amp_df.values,
        colLabels=nonrwd_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title(
        f"Non-rewarded Calcium Amp\n {test_title}\nF = {nonrwd_ca_amp_f:.4}  p = {nonrwd_ca_amp_p:.3E}"
    )
    F_table = axes2["F"].table(
        cellText=nonrwd_ca_amp_df.values,
        colLabels=nonrwd_ca_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)
    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title(
        f"Rewarded GluSnFr Onset\n {test_title}\nF = {rwd_onset_f:.4}  p = {rwd_onset_p:.3E}"
    )
    G_table = axes2["G"].table(
        cellText=rwd_onset_df.values,
        colLabels=rwd_onset_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title(
        f"Rewarded Calcium Onset\n {test_title}\nF = {rwd_ca_onset_f:.4}  p = {rwd_ca_onset_p:.3E}"
    )
    H_table = axes2["H"].table(
        cellText=rwd_ca_onset_df.values,
        colLabels=rwd_ca_onset_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)
    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title(
        f"Non-rewarded GluSnFr Onset\n {test_title}\nF = {nonrwd_onset_f:.4}  p = {nonrwd_onset_p:.3E}"
    )
    I_table = axes2["I"].table(
        cellText=nonrwd_onset_df.values,
        colLabels=nonrwd_onset_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)
    axes2["J"].axis("off")
    axes2["J"].axis("tight")
    axes2["J"].set_title(
        f"Non-rewarded Calcium Onset\n {test_title}\nF = {nonrwd_ca_onset_f:.4}  p = {nonrwd_ca_onset_p:.3E}"
    )
    J_table = axes2["J"].table(
        cellText=nonrwd_ca_onset_df.values,
        colLabels=nonrwd_ca_onset_df.columns,
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
        fname = os.path.join(save_path, "Spine_Activity_Figure_3_Stats")
        fig2.savefig(fname + ".pdf")


def plot_spine_movement_encoding(
    dataset,
    followup_dataset=None,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 4),
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    vol_norm=False,
    save=False,
    save_path=None,
):
    """Function to plot the movement encoding related variables for the different 
        groups

        INPUT PARAMETERS
            dataset - Spine_Activity_Data object

            followup_dataset - optional Spine_Activity_Data object of the subsequent
                                session to use for volume comparison. Default is None,
                                to use the followup volumes in the dataset
            
            exclude - str specifying type of spine to exclude from analysis

            threshold - float or tuple of floats specifying the threshold cutoffs for 
                        classifying plasticity
            
            figsize - tuple specifying the figure size

            showmeans - boolean specifying whether to show the means on box plots

            test_type - str specifying whether to perform parametric or nonparametric stats

            display_stats - boolean specifying whether to display stat results

            save - boolean specifying whether to save the figure or not

            save_path - str specifying where to save the figures
        
    """
    COLORS = ["darkorange", "darkviolet", "silver"]
    plastic_groups = {
        "Enlarged": "enlarged_spines",
        "Shrunken": "shrunken_spines",
        "Stable": "stable_spines",
    }

    # Pull relevant data
    sampling_rate = dataset.parameters["Sampling Rate"]
    activity_window = dataset.parameters["Activity Window"]
    if dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"
    ## Volume related information
    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset == None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags
    ## Movement-encoding variables
    spine_movement_correlation = dataset.spine_movement_correlation
    spine_movement_stereotypy = dataset.spine_movement_stereotypy
    spine_movement_reliability = dataset.spine_movement_reliability
    spine_movement_specificity = dataset.spine_movement_specificity
    spine_LMP_reliability = dataset.spine_LMP_reliability
    spine_LMP_specificity = dataset.spine_LMP_specificity
    spine_rwd_movement_correlation = dataset.spine_rwd_movement_correlation
    spine_rwd_movement_stereotypy = dataset.spine_rwd_movement_stereotypy
    spine_rwd_movement_reliability = dataset.spine_rwd_movement_reliability
    spine_rwd_movement_specificity = dataset.spine_rwd_movement_specificity
    spine_fraction_rwd_mvmts = dataset.spine_fraction_rwd_mvmts

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

    # Organize the data
    ## Subselect present spines
    spine_movement_correlation = d_utils.subselect_data_by_idxs(
        spine_movement_correlation, spine_idxs,
    )
    spine_movement_stereotypy = d_utils.subselect_data_by_idxs(
        spine_movement_stereotypy, spine_idxs,
    )
    spine_movement_reliability = d_utils.subselect_data_by_idxs(
        spine_movement_reliability, spine_idxs,
    )
    spine_movement_specificity = d_utils.subselect_data_by_idxs(
        spine_movement_specificity, spine_idxs,
    )
    spine_LMP_reliability = d_utils.subselect_data_by_idxs(
        spine_LMP_reliability, spine_idxs,
    )
    spine_LMP_specificity = d_utils.subselect_data_by_idxs(
        spine_LMP_specificity, spine_idxs,
    )
    spine_rwd_movement_correlation = d_utils.subselect_data_by_idxs(
        spine_rwd_movement_correlation, spine_idxs,
    )
    spine_rwd_movement_stereotypy = d_utils.subselect_data_by_idxs(
        spine_rwd_movement_stereotypy, spine_idxs
    )
    spine_rwd_movement_reliability = d_utils.subselect_data_by_idxs(
        spine_rwd_movement_reliability, spine_idxs,
    )
    spine_rwd_movement_specificity = d_utils.subselect_data_by_idxs(
        spine_rwd_movement_specificity, spine_idxs,
    )
    spine_fraction_rwd_mvmts = d_utils.subselect_data_by_idxs(
        spine_fraction_rwd_mvmts, spine_idxs,
    )

    ## Seperate groups
    group_mvmt_corr = {}
    group_mvmt_stero = {}
    group_mvmt_reli = {}
    group_mvmt_spec = {}
    group_LMP_reli = {}
    group_LMP_spec = {}
    group_rwd_mvmt_corr = {}
    group_rwd_mvmt_stero = {}
    group_rwd_mvmt_reli = {}
    group_rwd_mvmt_spec = {}
    group_frac_rwd = {}
    for key, value in plastic_groups.items():
        spines = eval(value)
        group_mvmt_corr[key] = spine_movement_correlation[spines]
        group_mvmt_stero[key] = spine_movement_stereotypy[spines]
        group_mvmt_reli[key] = spine_movement_reliability[spines]
        group_mvmt_spec[key] = spine_movement_specificity[spines]
        group_LMP_reli[key] = spine_LMP_reliability[spines]
        group_LMP_spec[key] = spine_LMP_specificity[spines]
        group_rwd_mvmt_corr[key] = spine_rwd_movement_correlation[spines]
        group_rwd_mvmt_stero[key] = spine_rwd_movement_stereotypy[spines]
        group_rwd_mvmt_reli[key] = spine_rwd_movement_reliability[spines]
        group_rwd_mvmt_spec[key] = spine_rwd_movement_specificity[spines]
        group_frac_rwd[key] = spine_fraction_rwd_mvmts[spines]

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE
        GHIJF
        K....
        """,
        figsize=figsize,
    )
    fig.suptitle("Spine Movement Encoding")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ############################## Plot data onto the axes ###########################
    ## Spine movement correlation
    plot_box_plot(
        group_mvmt_corr,
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
        b_width=0.55,
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
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    ## Spine movement stereotypy
    plot_box_plot(
        group_mvmt_stero,
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
        b_width=0.55,
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
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    ## Spine movement reliability
    plot_box_plot(
        group_mvmt_reli,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="Movement reliability",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    ## Spine movement reliability
    plot_box_plot(
        group_mvmt_spec,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="Movement specificity",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    ## Spine LMP reliability
    plot_box_plot(
        group_LMP_reli,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="Learned movement reliability",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
    ## Spine LMP specificity
    plot_box_plot(
        group_LMP_spec,
        figsize=(5, 5),
        title="All Mvmts",
        xtitle=None,
        ytitle="Learned movement specificity",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ## Spine reward movement correlation
    plot_box_plot(
        group_rwd_mvmt_corr,
        figsize=(5, 5),
        title="Rewarded Mvmts",
        xtitle=None,
        ytitle="LMP correlation (r)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
    ## Spine reward movement stereotypy
    plot_box_plot(
        group_rwd_mvmt_stero,
        figsize=(5, 5),
        title="Rewarded Mvmts",
        xtitle=None,
        ytitle="Movement stereotypy",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    ## Spine reward movement reliability
    plot_box_plot(
        group_rwd_mvmt_reli,
        figsize=(5, 5),
        title="Rewarded Mvmts",
        xtitle=None,
        ytitle="Movement reliability",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
    ## Spine reward movement specificity
    plot_box_plot(
        group_rwd_mvmt_spec,
        figsize=(5, 5),
        title="Rewarded Mvmts",
        xtitle=None,
        ytitle="Movement specificity",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
    ## Spine fraction rwd movements
    plot_box_plot(
        group_frac_rwd,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction rwd movements",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="white",
        b_err_colors=COLORS,
        m_color="white",
        m_width=1.5,
        b_width=0.55,
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Spine_Activity_Figure_4")
        fig.savefig(fname + ".pdf")

    ######################### Statistics Section ################################
    if display_stats == False:
        return

    # Perform the statistics
    if test_type == "parametric":
        mvmt_corr_f, mvmt_corr_p, _, mvmt_corr_df = t_utils.ANOVA_1way_posthoc(
            group_mvmt_corr, test_method,
        )
        mvmt_stereo_f, mvmt_stereo_p, _, mvmt_stereo_df = t_utils.ANOVA_1way_posthoc(
            group_mvmt_stero, test_method,
        )
        mvmt_rel_f, mvmt_rel_p, _, mvmt_rel_df = t_utils.ANOVA_1way_posthoc(
            group_mvmt_reli, test_method,
        )
        mvmt_spec_f, mvmt_spec_p, _, mvmt_spec_df = t_utils.ANOVA_1way_posthoc(
            group_mvmt_spec, test_method,
        )
        LMP_rel_f, LMP_rel_p, _, LMP_rel_df = t_utils.ANOVA_1way_posthoc(
            group_LMP_reli, test_method,
        )
        LMP_spec_f, LMP_spec_p, _, LMP_spec_df = t_utils.ANOVA_1way_posthoc(
            group_LMP_spec, test_method,
        )
        rwd_corr_f, rwd_corr_p, _, rwd_corr_df = t_utils.ANOVA_1way_posthoc(
            group_rwd_mvmt_corr, test_method,
        )
        rwd_stereo_f, rwd_stereo_p, _, rwd_stereo_df = t_utils.ANOVA_1way_posthoc(
            group_rwd_mvmt_stero, test_method,
        )
        rwd_rel_f, rwd_rel_p, _, rwd_rel_df = t_utils.ANOVA_1way_posthoc(
            group_rwd_mvmt_reli, test_method,
        )
        rwd_spec_f, rwd_spec_p, _, rwd_spec_df = t_utils.ANOVA_1way_posthoc(
            group_rwd_mvmt_spec, test_method,
        )
        frac_f, frac_p, _, frac_df = t_utils.ANOVA_1way_posthoc(
            group_frac_rwd, test_method,
        )
        test_title = f"One-Way ANOVA {test_method}"
    elif test_type == "nonparametric":
        mvmt_corr_f, mvmt_corr_p, mvmt_corr_df = t_utils.kruskal_wallis_test(
            group_mvmt_corr, "Conover", test_method,
        )
        mvmt_stereo_f, mvmt_stereo_p, mvmt_stereo_df = t_utils.kruskal_wallis_test(
            group_mvmt_stero, "Conover", test_method,
        )
        mvmt_rel_f, mvmt_rel_p, mvmt_rel_df = t_utils.kruskal_wallis_test(
            group_mvmt_reli, "Conover", test_method,
        )
        mvmt_spec_f, mvmt_spec_p, mvmt_spec_df = t_utils.kruskal_wallis_test(
            group_mvmt_spec, "Conover", test_method,
        )
        LMP_rel_f, LMP_rel_p, LMP_rel_df = t_utils.kruskal_wallis_test(
            group_LMP_reli, "Conover", test_method,
        )
        LMP_spec_f, LMP_spec_p, LMP_spec_df = t_utils.kruskal_wallis_test(
            group_LMP_spec, "Conover", test_method,
        )
        rwd_corr_f, rwd_corr_p, rwd_corr_df = t_utils.kruskal_wallis_test(
            group_rwd_mvmt_corr, "Conover", test_method,
        )
        rwd_stereo_f, rwd_stereo_p, rwd_stereo_df = t_utils.kruskal_wallis_test(
            group_rwd_mvmt_stero, "Conover", test_method,
        )
        rwd_rel_f, rwd_rel_p, rwd_rel_df = t_utils.kruskal_wallis_test(
            group_rwd_mvmt_reli, "Conover", test_method,
        )
        rwd_spec_f, rwd_spec_p, rwd_spec_df = t_utils.kruskal_wallis_test(
            group_rwd_mvmt_spec, "Conover", test_method,
        )
        frac_f, frac_p, frac_df = t_utils.kruskal_wallis_test(
            group_frac_rwd, "Conover", test_method,
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
        K.
        """,
        figsize=(8, 12),
    )
    ## Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title(
        f"Movement Correlation\n {test_title}\nF = {mvmt_corr_f:.4}  p = {mvmt_corr_p:.3E}"
    )
    A_table = axes2["A"].table(
        cellText=mvmt_corr_df.values,
        colLabels=mvmt_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(
        f"Movement Stereotypy\n {test_title}\nF = {mvmt_stereo_f:.4}  p = {mvmt_stereo_p:.3E}"
    )
    B_table = axes2["B"].table(
        cellText=mvmt_stereo_df.values,
        colLabels=mvmt_stereo_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title(
        f"Movement Reliability\n {test_title}\nF = {mvmt_rel_f:.4}  p = {mvmt_rel_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=mvmt_rel_df.values,
        colLabels=mvmt_rel_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title(
        f"Movement Specificity\n {test_title}\nF = {mvmt_spec_f:.4}  p = {mvmt_spec_p:.3E}"
    )
    D_table = axes2["D"].table(
        cellText=mvmt_spec_df.values,
        colLabels=mvmt_spec_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title(
        f"Learned Movement Reliability\n {test_title}\nF = {LMP_rel_f:.4}  p = {LMP_rel_p:.3E}"
    )
    E_table = axes2["E"].table(
        cellText=LMP_rel_df.values,
        colLabels=LMP_rel_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title(
        f"Learned Movement Specificity\n {test_title}\nF = {LMP_spec_f:.4}  p = {LMP_spec_p:.3E}"
    )
    F_table = axes2["F"].table(
        cellText=LMP_spec_df.values,
        colLabels=LMP_spec_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)
    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title(
        f"Rewarded Movement Correlation\n {test_title}\nF = {rwd_corr_f:.4}  p = {rwd_corr_p:.3E}"
    )
    G_table = axes2["G"].table(
        cellText=rwd_corr_df.values,
        colLabels=rwd_corr_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title(
        f"Rewarded Movement Stereotypy\n {test_title}\nF = {rwd_stereo_f:.4}  p = {rwd_stereo_p:.3E}"
    )
    H_table = axes2["H"].table(
        cellText=rwd_stereo_df.values,
        colLabels=rwd_stereo_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)
    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title(
        f"Rewarded Movement Reliability\n {test_title}\nF = {rwd_rel_f:.4}  p = {rwd_rel_p:.3E}"
    )
    I_table = axes2["I"].table(
        cellText=rwd_rel_df.values,
        colLabels=rwd_rel_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)
    axes2["J"].axis("off")
    axes2["J"].axis("tight")
    axes2["J"].set_title(
        f"Rewarded Movement Specificity\n {test_title}\nF = {rwd_spec_f:.4}  p = {rwd_spec_p:.3E}"
    )
    J_table = axes2["J"].table(
        cellText=rwd_spec_df.values,
        colLabels=rwd_spec_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    J_table.auto_set_font_size(False)
    J_table.set_fontsize(8)
    axes2["K"].axis("off")
    axes2["K"].axis("tight")
    axes2["K"].set_title(
        f"Fraction Rewarded Movements\n {test_title}\nF = {frac_f:.4}  p = {frac_p:.3E}"
    )
    K_table = axes2["K"].table(
        cellText=frac_df.values,
        colLabels=frac_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    K_table.auto_set_font_size(False)
    K_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Spine_Activity_Figure_4_Stats")
        fig2.savefig(fname + ".pdf")

