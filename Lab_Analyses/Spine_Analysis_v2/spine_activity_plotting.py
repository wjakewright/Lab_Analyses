import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
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
    mean_type="median",
    err_type="CI",
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
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

            mean_type - str specifying the mean type for bar plots

            err_type - str specifying the error type for bar plots

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
        followup_flags = followup_dataset.followup_flags
    ## Setup input lists
    volumes = [initial_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    ## Calculate
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=False, exclude=exclude
    )
    delta_volume = delta_volume[-1]

    # Classify plasticity
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=False
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
        xlim=None,
        figsize=(5, 5),
        color="mediumblue",
        alpha=0.7,
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
        ax=axes["tm"],
        save=False,
        save_path=None,
    )

    # Initial volume for spine types
    plot_swarm_bar_plot(
        data_dict=initial_vol_dict,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="Initial Volumes",
        xtitle=None,
        ytitle="Initial volume \u03BCm",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
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
        alpha=0.7,
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
        ax=axes["bm"],
        save=False,
        save_path=None,
    )

    # Activity rates for spine types
    plot_swarm_bar_plot(
        data_dict=activity_dict,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="Event rates",
        xtitle=None,
        ytitle="Event rate (events/min)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=COLORS,
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
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
    axes2["left"].axes("tight")
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
        cellLabels=vol_test_df.columns,
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
    mean_type="median",
    err_type="CI",
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
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
            
            mean_type - str specifying the mean type for bar plots
            
            err_type - str specifying the error type for bar plots
            
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
        followup_volumes = followup_dataset.followup_volumes
        followup_flags = followup_dataset.followup_flags

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
        volumes, flags, norm=False, exclude=exclude,
    )
    enlarged_spines, shrunken_spines, stable_spined = classify_plasticity(
        delta_volume, threshold=threshold, norm=False,
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
        mvmt_fractions[key] = np.sum(mvmt_spines)
        nonmvmt_fractions[key] = np.sum(nmvmt_spines)
        rwd_mvmt_fractions[key] = np.sum(rwd_mvmt_spines)
        nonrwd_mvmt_fractions[key] = np.sum(nrwd_mvmt_spines)
        ## Grab grouped traces, amps, onsets
        mrs_traces = compress(spine_movement_traces, mvmt_spines)
        mrs_calcium_traces = compress(spine_movement_calcium_traces, mvmt_spines)
        ### Avg individual events
        trace_means = [
            np.mnanmean(x, axis=1) for x in mrs_traces if type(x) == np.ndarray
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
        ABCDEF
        GHCDEF
        IJKLM.
        NOPQR.
        """,
        figsize=figsize,
    )
    fig.suptitle("Movement-Related Spine Activity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ########################## Plot data onto the axes #############################
    ## Fractions of MRSs
    plot_pie_chart(
        mvmt_fractions,
        title="MRSs",
        figsize=(5, 5),
        colors=COLORS,
        alpha=0.7,
        edgecolor="white",
        txt_color="white",
        txt_size=9,
        legend="top",
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
        alpha=0.7,
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
        alpha=0.7,
        edgecolor="white",
        txt_color="white",
        txt_size=9,
        legende=None,
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
        alpha=0.7,
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
    all_traces = np.hstack(mrs_ind_mvmt_traces.values())
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
        cbar_labl=activity_type,
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
        xlim=None,
        ylim=None,
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
        xlim=None,
        ylim=None,
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
    plot_swarm_bar_plot(
        mrs_grouped_mvmt_amps,
        mean_type=mean_type,
        err_type=err_type,
        figsze=(5, 5),
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    ## Grouped Calcium amplitude
    plot_swarm_bar_plot(
        mrs_grouped_mvmt_calcium_amps,
        mean_type=mean_type,
        err_type=err_type,
        figsze=(5, 5),
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
        xlim=None,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.3,
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
        xlim=None,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["Q"],
        save=False,
        save_path=None,
    )
    ## Grouped GluSnFr onsets
    plot_swarm_bar_plot(
        mrs_grouped_mvmt_onsets,
        mean_type=mean_type,
        err_type=err_type,
        figsze=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Relative onset (s)",
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
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    ## Grouped GluSnFr onsets
    plot_swarm_bar_plot(
        mrs_grouped_mvmt_calcium_onsets,
        mean_type=mean_type,
        err_type=err_type,
        figsze=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Relative onset (s)",
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
        ax=axes["R"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Spine_Activity_Figure_2")
        fig.save_fig(fname + ".pdf")

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
    axes2["A"].axes("tight")
    axes2["A"].set_title(
        f"GluSnFr Amp {test_title}\nF = {g_amp_f:.4}  p = {g_amp_p:.3E}"
    )
    A_table = axes2["A"].table(
        cellText=g_amp_df.values,
        colLabels=g_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.autoset_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axes("tight")
    axes2["B"].set_title(
        f"Calcium Amp {test_title}\nF = {c_amp_f:.4}  p = {c_amp_p:.3E}"
    )
    B_table = axes2["B"].table(
        cellText=c_amp_df.values,
        colLabels=c_amp_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.autoset_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axes("tight")
    axes2["C"].set_title(
        f"GluSnFr Onsets {test_title}\nF = {g_onset_f:.4}  p = {g_onset_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=g_onset_df.values,
        colLabels=g_onset_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.autoset_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axes("tight")
    axes2["D"].set_title(
        f"Calcium Onset {test_title}\nF = {c_onset_f:.4}  p = {c_onset_p:.3E}"
    )
    C_table = axes2["C"].table(
        cellText=c_onset_df.values,
        colLabels=c_onset_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.autoset_font_size(False)
    C_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Spine_Activity_Figure_2_Stats")
        fig2.savefig(fname + ".pdf")


def plot_reward_movement_related_activity(
    dataset,
    followup_dataset=None,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 6),
    hist_bins=25,
    mean_type="median",
    err_type="CI",
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
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
        
        mean_type - str specifying the mean type for bar plots
        
        err_type - str specifying the error type for bar plots
        
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
        "Stable": "stable_spines,",
    }
    mvmt_groups = {
        "Rewarded": "rwd_movement_spines",
        "Non-rewarded": "nonrwd_movement_spines",
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
        followup_volumes = followup_dataset.followup_volumes
        followup_flags = followup_dataset.followup_flags
    ## Movement identifiers
    rwd_movement_spines = dataset.rwd_movement_spines
    nonrwd_movement_spines = dataset.nonrwd_movement_spines
    ## Movement-related activity
    spine_movement_traces = dataset.spine_movement_traces
    spine_movement_calcium_traces = dataset.spine_movement_calcium_traces
    spine_movement_amplitude = dataset.spine_movement_amplitude
    spine_movement_calcium_amplitude = dataset.spine_movement_calcium_amplitude
    spine_movement_onset = dataset.spine_movement_onset
    spine_movement_calcium_onset = dataset.spine_movement_calcium_onset

    # Calculate the relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=False, exclude=exclude,
    )
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=False,
    )

    # Organize data
    ## Subselct present spines
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

    ## Seperate groups
    ### Rewarded vs nonrewarded groups
    rwd_nonrwd_ind_traces = {}
    rwd_nonrwd_avg_traces = {}
    rwd_nonrwd_sem_traces = {}
    rwd_nonrwd_avg_calcium_traces = {}
    rwd_nonrwd_sem_calcium_traces = {}
    rwd_nonrwd_amps = {}
    rwd_nonrwd_calcium_amps = {}
    ### Rewarded plastic spines
    group_avg_traces = {}
    group_sem_traces = {}
    group_amps = {}
    group_onsets = {}
    group_avg_calcium_traces = {}
    group_sem_calcium_traces = {}
    group_calcium_amps = {}
    group_calcium_onsets = {}

    for mvmt_key, mvmt_value in mvmt_groups.items():
        mvmt_spines = eval(mvmt_value)
        ## Grab grouped mvmt data
        mvmt_traces = compress(spine_movement_traces, mvmt_spines)
        mvmt_calcium_traces = compress(spine_movement_calcium_traces, mvmt_spines)
        ## Avg individual events
        mvmt_trace_means = [
            np.nanmean(x, axis=1) for x in mvmt_traces if type(x) == np.ndarray
        ]
        mvmt_ca_trace_means = [
            np.nanmean(x, axis=1) for x in mvmt_calcium_traces if type(x) == np.ndarray
        ]
        mvmt_trace_means = np.vstack(mvmt_trace_means)
        mvmt_ca_trace_means = np.vstack(mvmt_ca_trace_means)
        ## Add individual traces
        rwd_nonrwd_ind_traces[mvmt_key] = mvmt_trace_means.T
        ## Get avg traces
        group_trace_means = np.nanmean(mvmt_trace_means, axis=0)
        group_ca_trace_means = np.nanmean(mvmt_ca_trace_means, axis=0)
        group_trace_sem = stats.sem(mvmt_trace_means, axis=0, nan_policy="omit")
        group_ca_trace_sem = stats.sem(mvmt_ca_trace_means, axis=0, nan_policy="omit")
        rwd_nonrwd_avg_traces[mvmt_key] = group_trace_means
        rwd_nonrwd_avg_calcium_traces[mvmt_key] = group_ca_trace_means
        rwd_nonrwd_sem_traces[mvmt_key] = group_trace_sem
        rwd_nonrwd_sem_calcium_traces[mvmt_key] = group_ca_trace_sem
        ## Amplitudes
        amps = spine_movement_amplitude[mvmt_spines]
        ca_amps = spine_movement_calcium_amplitude[mvmt_spines]
        rwd_nonrwd_amps[mvmt_key] = amps[~np.isnan(amps)]
        rwd_nonrwd_calcium_amps[mvmt_key] = ca_amps[~np.isnan(ca_amps)]
        onsets = spine_movement_onset[mvmt_spines]
        ca_onsets = spine_movement_calcium_onset[mvmt_spines]

        # Get plastic spine data
        avg_traces = {}
        avg_ca_traces = {}
        sem_traces = {}
        sem_ca_traces = {}
        plastic_group_amps = {}
        plastic_group_ca_amps = {}
        plastic_group_onsets = {}
        plastic_group_ca_onsets = {}

        mvmt_spine_idxs = np.nonzero(mvmt_spines)[0]
        for plastic_key, plastic_value in plastic_groups.items():
            plastic_spines = eval(plastic_value)[mvmt_spine_idxs]
            plastic_traces = mvmt_trace_means[plastic_spines, :]
            plastic_ca_traces = mvmt_ca_trace_means[plastic_spines, :]
            avg_plastic_trace = np.nanmean(plastic_traces, axis=0)
            avg_plastic_ca_trace = np.nanmean(plastic_ca_traces, axis=0)
            sem_plastic_traces = stats.sem(plastic_traces, axis=0, nan_policy="omit")
            sem_plastic_ca_traces = stats.sem(
                plastic_ca_traces, axis=0, nan_policy="omit"
            )
            plastic_amps = amps[plastic_spines]
            plastic_ca_amps = ca_amps[plastic_spines]
            plastic_onsets = onsets[plastic_spines]
            plastic_ca_onsets = ca_onsets[plastic_spines]
            avg_traces[plastic_key] = avg_plastic_trace
            avg_ca_traces[plastic_key] = avg_plastic_ca_trace
            sem_traces[plastic_key] = sem_plastic_traces
            sem_ca_traces[plastic_key] = sem_plastic_ca_traces
            plastic_group_amps[plastic_key] = plastic_amps[~np.isnan(plastic_amps)]
            plastic_group_ca_amps[plastic_key] = plastic_ca_amps[
                ~np.isnan(plastic_ca_amps)
            ]
            plastic_group_onsets[plastic_key] = plastic_onsets[
                ~np.isnan(plastic_onsets)
            ]
            plastic_group_ca_onsets[plastic_key] = plastic_ca_onsets[
                ~np.isnan(plastic_ca_onsets)
            ]
        group_avg_traces[mvmt_key] = avg_traces
        group_sem_traces[mvmt_key] = sem_traces
        group_amps[mvmt_key] = plastic_group_amps
        group_onsets[mvmt_key] = plastic_group_onsets
        group_avg_calcium_traces[mvmt_key] = avg_ca_traces
        group_sem_calcium_traces[mvmt_key] = sem_ca_traces
        group_calcium_amps[mvmt_key] = plastic_group_ca_amps
        group_calcium_onsets[mvmt_key] = plastic_group_ca_onsets

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

    ######################## Plot data onto the axes #########################
    ## Rewarded movement heatmap
    plot_activity_heatmap(
        rwd_nonrwd_ind_traces["Rewarded"],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Rewarded",
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
    ## Non-rewarded heatmap
    plot_activity_heatmap(
        rwd_nonrwd_ind_traces["Non-rewarded"],
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Rewarded",
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
    ## Reward and non-reward GluSnFr Traces
    plot_mean_activity_traces(
        means=list(rwd_nonrwd_avg_traces.values()),
        sems=list(rwd_nonrwd_sem_traces.values()),
        group_names=list(rwd_nonrwd_avg_traces.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
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
    ## Reward and non-reward Calcium Traces
    plot_mean_activity_traces(
        means=list(rwd_nonrwd_avg_calcium_traces.values()),
        sems=list(rwd_nonrwd_sem_calcium_traces.values()),
        group_names=list(rwd_nonrwd_avg_calcium_traces.keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
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
    ## Reward and non-reward GluSnFr Amplitudes
    plot_swarm_bar_plot(
        rwd_nonrwd_amps,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="GluSnFr",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=["mediumblue", "firebrick"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=["mediumblue", "firebrick"],
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    ## Reward and non-reward Calcium Amplitudes
    plot_swarm_bar_plot(
        rwd_nonrwd_calcium_amps,
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"Event amplitude ({activity_type})",
        ylim=None,
        b_colors=["mediumblue", "firebrick"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.3,
        s_colors=["mediumblue", "firebrick"],
        s_size=5,
        s_alpha=0.7,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ## Grouped rewarded movement GluSnFr traces
    plot_mean_activity_traces(
        means=list(group_avg_traces["Rewarded"].values()),
        sems=list(group_sem_traces["Rewarded"].values()),
        group_names=list(group_avg_traces["Rewarded"].keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Rewarded GluSnFr Traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["G"],
        save=False,
    )
    ## Grouped non-rewarded movement GluSnFr traces
    plot_mean_activity_traces(
        means=list(group_avg_traces["Non-rewarded"].values()),
        sems=list(group_sem_traces["Non-rewarded"].values()),
        group_names=list(group_avg_traces["Non-rewarded"].keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Non-rewarded GluSnFr Traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["M"],
        save=False,
    )
    ## Grouped rewarded movement Calcium traces
    plot_mean_activity_traces(
        means=list(group_avg_calcium_traces["Rewarded"].values()),
        sems=list(group_sem_calcium_traces["Rewarded"].values()),
        group_names=list(group_avg_calcium_traces["Rewarded"].keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Rewarded Calcium Traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["J"],
        save=False,
    )
    ## Grouped non-rewarded movement GluSnFr traces
    plot_mean_activity_traces(
        means=list(group_avg_calcium_traces["Non-rewarded"].values()),
        sems=list(group_sem_calcium_traces["Non-rewarded"].values()),
        group_names=list(group_avg_calcium_traces["Non-rewarded"].keys()),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="Non-rewarded Calcium Traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["P"],
        save=False,
    )
    ## Grouped rewarded GluSnFr amplitudes
    plot_swarm_bar_plot(
        group_amps["Rewarded"],
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
        ax=axes["H"],
        save=False,
        save_path=True,
    )
    ## Grouped non-rewarded GluSnFr amplitudes
    plot_swarm_bar_plot(
        group_amps["Non-rewarded"],
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
        ax=axes["N"],
        save=False,
        save_path=True,
    )
    ## Grouped rewarded Calcium amplitudes
    plot_swarm_bar_plot(
        group_calcium_amps["Rewarded"],
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
        ax=axes["K"],
        save=False,
        save_path=True,
    )
    ## Grouped non-rewarded Calcium amplitudes
    plot_swarm_bar_plot(
        group_calcium_amps["Non-rewarded"],
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
        ax=axes["Q"],
        save=False,
        save_path=True,
    )
    ## Grouped rewarded GluSnFr onsets
    plot_histogram(
        data=list(group_onsets["Rewarded"].values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="GluSnFr",
        xtitle="Relative onset (s)",
        xlime=None,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    ## Grouped non-rewarded GluSnFr onsets
    plot_histogram(
        data=list(group_onsets["Non-rewarded"].values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="GluSnFr",
        xtitle="Relative onset (s)",
        xlime=None,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    ## Grouped rewarded Calcium onsets
    plot_histogram(
        data=list(group_calcium_onsets["Rewarded"].values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="Calcium",
        xtitle="Relative onset (s)",
        xlime=None,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    ## Grouped rewarded GluSnFr onsets
    plot_histogram(
        data=list(group_calcium_onsets["Non-rewarded"].values()),
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="Calcium",
        xtitle="Relative onset (s)",
        xlime=None,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.3,
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
        fig.save_fig(fname + ".pdf")

    ########################### Statistics Section #################################
    if display_stats == False:
        return

    ## Perform the statistics
    ## T-Tests
    rwd_nonrwd_amp_t, rwd_nonrwd_amp_p = stats.ttest_ind(
        rwd_nonrwd_amps["Rewarded"], rwd_nonrwd_amps["Non-rewarded"]
    )
    rwd_nonrwd_ca_amp_t, rwd_nonrwd_ca_amp_p = stats.ttest_ind(
        rwd_nonrwd_calcium_amps["Rewarded"], rwd_nonrwd_calcium_amps["Non-rewarded"],
    )
    ## F-tests
    if test_type == "parametric":
        rwd_amp_f, rwd_amp_p, _, rwd_amp_df = t_utils.ANOVA_1way_posthoc(
            group_amps["Rewarded"], test_method,
        )
        rwd_ca_amp_f, rwd_ca_amp_p, _, rwd_ca_amp_df = t_utils.ANOVA_1way_posthoc(
            group_calcium_amps["Rewarded"], test_method,
        )
        rwd_onset_f, rwd_onset_p, _, rwd_onset_df = t_utils.ANOVA_1way_posthoc(
            group_onsets["Rewarded"], test_method,
        )
        rwd_ca_onset_f, rwd_ca_onset_p, _, rwd_ca_onset_df = t_utils.ANOVA_1way_posthoc(
            group_calcium_onsets["Rewarded"], test_method,
        )
        nonrwd_amp_f, nonrwd_amp_p, _, nonrwd_amp_df = t_utils.ANOVA_1way_posthoc(
            group_amps["Non-rewarded"], test_method,
        )
        (
            nonrwd_ca_amp_f,
            nonrwd_ca_amp_p,
            _,
            nonrwd_ca_amp_df,
        ) = t_utils.ANOVA_1way_posthoc(group_calcium_amps["Non-rewarded"], test_method,)
        nonrwd_onset_f, nonrwd_onset_p, _, nonrwd_onset_df = t_utils.ANOVA_1way_posthoc(
            group_onsets["Non-rewarded"], test_method,
        )
        (
            nonrwd_ca_onset_f,
            nonrwd_ca_onset_p,
            _,
            nonrwd_ca_onset_df,
        ) = t_utils.ANOVA_1way_posthoc(
            group_calcium_onsets["Non-rewarded"], test_method,
        )
        test_title = f"One-Way ANOVA {test_method}"
    elif test_type == "nonparametric":
        rwd_amp_f, rwd_amp_p, rwd_amp_df = t_utils.kruskal_wallis_test(
            group_amps["Rewarded"], "Conover", test_method,
        )
        rwd_ca_amp_f, rwd_ca_amp_p, rwd_ca_amp_df = t_utils.kruskal_wallis_test(
            group_calcium_amps["Rewarded"], "Conover", test_method,
        )
        rwd_onset_f, rwd_onset_p, rwd_onset_df = t_utils.kruskal_wallis_test(
            group_onsets["Rewarded"], "Conover", test_method,
        )
        rwd_ca_onset_f, rwd_ca_onset_p, rwd_ca_onset_df = t_utils.kruskal_wallis_test(
            group_calcium_onsets["Rewarded"], "Conover", test_method,
        )
        nonrwd_amp_f, nonrwd_amp_p, nonrwd_amp_df = t_utils.kruskal_wallis_test(
            group_amps["Non-rewarded"], "Conover", test_method,
        )
        (
            nonrwd_ca_amp_f,
            nonrwd_ca_amp_p,
            nonrwd_ca_amp_df,
        ) = t_utils.kruskal_wallis_test(
            group_calcium_amps["Non-rewarded"], "Conover", test_method,
        )
        nonrwd_onset_f, nonrwd_onset_p, nonrwd_onset_df = t_utils.kruskal_wallis_test(
            group_onsets["Non-rewarded"], "Conover", test_method,
        )
        (
            nonrwd_ca_onset_f,
            nonrwd_ca_onset_p,
            nonrwd_ca_onset_df,
        ) = t_utils.kruskal_wallis_test(
            group_calcium_onsets["Non-rewarded"], "Conover", test_method,
        )
        test_title = f"Kruskal-Wallis {test_method}"
    # Display the statistics
