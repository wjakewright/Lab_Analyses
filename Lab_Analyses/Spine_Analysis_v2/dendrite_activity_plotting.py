import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils

sns.set()
sns.set_style("ticks")


def plot_activity_features(
    dataset,
    followup_dataset=None,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 8),
    hist_bins=10,
    save=False,
    save_path=None,
):
    """Function to plot the activity-related variables of parent dendrites
    
        INPUT PARAMETERS
            dataset - Spine_Activity_Data object

            followup_dataset - optional Spine_Activity_Data object of the subsequent
                                session to use for volume comparision. Default is None
                                to use the followup volumes in the dataset
            
            exclude - str specifying the type of spine to exclude from analysis

            threshold - float or tuple of floats specifying the threshold cuttoff
                        for classifying plasticity
                    
            figsize - tuple specifying the figure size

            hist_bins - int specifying how many bins to plot for the histogram

            save - boolean specifying whether to save the figure or not

            save_path - str specifying where to save the data
    """
    COLORS = ["darkorange", "darkviolet", "silver"]

    # Pull relevant data
    sampling_rate = dataset.parameters["Sampling Rate"]
    activity_window = dataset.parameters["Activity Window"]
    if dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"
    dendrite_number = dataset.dendrite_number
    ## Volume related information
    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset == None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = followup_dataset.spine_volumes
        followup_flags = followup_dataset.spine_flags

    ## Activity related variables
    dendrite_activity_rate = dataset.dendrite_activity_rate
    dendrite_movement_traces = dataset.dendrite_movement_traces
    dendrite_rwd_movement_traces = dataset.dendrite_rwd_movement_traces
    dendrite_nonrwd_movement_traces = dataset.dendrite_nonrwd_movement_traces
    dendrite_movement_amplitude = dataset.dendrite_movement_amplitude
    dendrite_rwd_movement_amplitude = dataset.dendrite_rwd_movement_amplitude
    dendrite_nonrwd_movement_amplitude = dataset.dendrite_nonrwd_movement_amplitude
    dendrite_movement_onset = dataset.dendrite_movement_onset
    dendrite_rwd_movement_onset = dataset.dendrite_rwd_movement_onset
    dendrite_nonrwd_movement_onset = dataset.dendrite_nonrwd_movement_onset

    # Calculate relative volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=False, exclude=exclude
    )
    delta_volume = delta_volume[-1]
    ## Classify
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=False
    )

    # Subselect present spines
    dendrite_number = d_utils.subselect_data_by_idxs(dendrite_number, spine_idxs)
    dendrite_activity_rate = d_utils.subselect_data_by_idxs(
        dendrite_activity_rate, spine_idxs
    )
    dendrite_movement_traces = d_utils.subselect_data_by_idxs(
        dendrite_movement_traces, spine_idxs
    )
    dendrite_rwd_movement_traces = d_utils.subselect_data_by_idxs(
        dendrite_rwd_movement_traces, spine_idxs
    )
    dendrite_nonrwd_movement_traces = d_utils.subselect_data_by_idxs(
        dendrite_nonrwd_movement_traces, spine_idxs
    )
    dendrite_movement_amplitude = d_utils.subselect_data_by_idxs(
        dendrite_movement_amplitude, spine_idxs
    )
    dendrite_rwd_movement_amplitude = d_utils.subselect_data_by_idxs(
        dendrite_rwd_movement_amplitude, spine_idxs
    )
    dendrite_nonrwd_movement_amplitude = d_utils.subselect_data_by_idxs(
        dendrite_nonrwd_movement_amplitude, spine_idxs
    )
    dendrite_movement_onset = d_utils.subselect_data_by_idxs(
        dendrite_movement_onset, spine_idxs
    )
    dendrite_rwd_movement_onset = d_utils.subselect_data_by_idxs(
        dendrite_rwd_movement_onset, spine_idxs
    )
    dendrite_nonrwd_movement_onset = d_utils.subselect_data_by_idxs(
        dendrite_nonrwd_movement_onset, spine_idxs
    )

    # Organize data based on fraction of plastic spine groups
    unique_dend = np.unique(dendrite_number)
    fraction_enlarged = np.zeros(len(unique_dend))
    fraction_shrunken = np.zeros(len(unique_dend))
    fraction_stable = np.zeros(len(unique_dend))
    activity_rate = np.zeros(len(unique_dend))
    mvmt_amplitude = np.zeros(len(unique_dend))
    mvmt_onset = np.zeros(len(unique_dend))
    rwd_mvmt_amplitude = np.zeros(len(unique_dend))
    rwd_mvmt_onset = np.zeros(len(unique_dend))
    nonrwd_mvmt_amplitude = np.zeros(len(unique_dend))
    nonrwd_mvmt_onset = np.zeros(len(unique_dend))

    for i, dend in enumerate(unique_dend):
        spines = np.nonzero(dendrite_number == dend)[0]
        fraction_enlarged[i] = np.nansum(enlarged_spines[spines]) / len(spines)
        fraction_shrunken[i] = np.nansum(shrunken_spines[spines]) / len(spines)
        fraction_stable[i] = np.nansum(stable_spines[spines]) / len(spines)
        activity_rate[i] = np.nanmean(dendrite_activity_rate[spines])
        mvmt_amplitude[i] = np.nanmean(dendrite_movement_amplitude[spines])
        mvmt_onset[i] = np.nanmean(dendrite_movement_onset[spines])
        rwd_mvmt_amplitude[i] = np.nanmean(dendrite_rwd_movement_amplitude[spines])
        rwd_mvmt_onset[i] = np.nanmean(dendrite_rwd_movement_onset[spines])
        nonrwd_mvmt_amplitude[i] = np.nanmean(
            dendrite_nonrwd_movement_amplitude[spines]
        )
        nonrwd_mvmt_onset[i] = np.nanmean(dendrite_nonrwd_movement_onset[spines])

    # Organize data for heatmaps
    trace_means = [
        np.nanmean(x, axis=1) for x in dendrite_movement_traces if type(x) == np.ndarray
    ]
    trace_means = np.vstack(trace_means)
    heatmap_traces = trace_means.T
    rwd_means = [
        np.nanmean(x, axis=1)
        for x in dendrite_rwd_movement_traces
        if type(x) == np.ndarray
    ]
    rwd_means = np.vstack(rwd_means)
    heatmap_rwd_traces = rwd_means.T
    nonrwd_means = [
        np.nanmean(x, axis=1)
        for x in dendrite_nonrwd_movement_traces
        if type(x) == np.ndarray
    ]
    nonrwd_means = np.vstack(nonrwd_means)
    heatmap_nonrwd_traces = nonrwd_means.T

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE
        FGHIJ
        KLMNO
        PQRST
        """,
        figsize=figsize,
    )
    fig.suptitle("Dendrite Activity")
    fig.subplot_adjust(hspace=1, wspace=0.5)

    #################### Plot data onto the axes #######################
    # Fraction enlarged histogram
    plot_histogram(
        data=fraction_enlarged,
        bins=hist_bins,
        avlines=None,
        title="Enlarged Spines",
        xtitle="Fraction of Spines",
        xlim=None,
        figsize=(5, 5),
        color=COLORS[0],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # Fraction shrunken histogram
    plot_histogram(
        data=fraction_shrunken,
        bins=hist_bins,
        avlines=None,
        title="Shrunken Spines",
        xtitle="Fraction of Spines",
        xlim=None,
        figsize=(5, 5),
        color=COLORS[1],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Fraction stable histogram
    plot_histogram(
        data=fraction_stable,
        bins=hist_bins,
        avlines=None,
        title="Stable Spines",
        xtitle="Fraction of Spines",
        xlim=None,
        figsize=(5, 5),
        color=COLORS[2],
        alpha=0.3,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # Activity rate vs fraction enlarged
    plot_scatter_correlation(
        x_var=activity_rate,
        y_var=fraction_enlarged,
        CI=95,
        title="Enlarged",
        xtitle="Event rate (events/min)",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # Activity rate vs fraction shrunken
    plot_scatter_correlation(
        x_var=activity_rate,
        y_var=fraction_shrunken,
        CI=95,
        title="Shrunken",
        xtitle="Event rate (events/min)",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Movement heatmap
    plot_activity_heatmap(
        heatmap_traces,
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="All Movements",
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
    # Rewarded Movement heatmap
    plot_activity_heatmap(
        heatmap_rwd_traces,
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Rewarded Movements",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="plasma",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Non rewarded Movement heatmap
    plot_activity_heatmap(
        heatmap_nonrwd_traces,
        figsize=(4, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Non-rewarded Movements",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="plasma",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["P"],
        save=False,
        save_path=None,
    )
    # Mvmt amplitude vs fraction enlarged
    plot_scatter_correlation(
        x_var=mvmt_amplitude,
        y_var=fraction_enlarged,
        CI=95,
        title="All Mvmts",
        xtitle=f"Event amplitude ({activity_type})",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    # Mvmt amplitude vs fraction shrunken
    plot_scatter_correlation(
        x_var=mvmt_amplitude,
        y_var=fraction_shrunken,
        CI=95,
        title="All Mvmts",
        xtitle=f"Event amplitude ({activity_type})",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    # Mvmt onset vs fraction enlarged
    plot_scatter_correlation(
        x_var=mvmt_onset,
        y_var=fraction_enlarged,
        CI=95,
        title="All Mvmts",
        xtitle=f"Relative onset (s)",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    # Mvmt onset vs fraction shrunken
    plot_scatter_correlation(
        x_var=mvmt_onset,
        y_var=fraction_shrunken,
        CI=95,
        title="All Mvmts",
        xtitle=f"Relative onset (s)",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt amplitude vs fraction enlarged
    plot_scatter_correlation(
        x_var=rwd_mvmt_amplitude,
        y_var=fraction_enlarged,
        CI=95,
        title="Rewarded Mvmts",
        xtitle=f"Event amplitude ({activity_type})",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt amplitude vs fraction shrunken
    plot_scatter_correlation(
        x_var=rwd_mvmt_amplitude,
        y_var=fraction_shrunken,
        CI=95,
        title="Rewarded Mvmts",
        xtitle=f"Event amplitude ({activity_type})",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    # Nonrwd Mvmt amplitude vs fraction enlarged
    plot_scatter_correlation(
        x_var=nonrwd_mvmt_amplitude,
        y_var=fraction_enlarged,
        CI=95,
        title="Non-rewarded Mvmts",
        xtitle=f"Event amplitude ({activity_type})",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["Q"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt amplitude vs fraction shrunken
    plot_scatter_correlation(
        x_var=nonrwd_mvmt_amplitude,
        y_var=fraction_shrunken,
        CI=95,
        title="Non-rewarded Mvmts",
        xtitle=f"Event amplitude ({activity_type})",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["R"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt onset vs fraction enlarged
    plot_scatter_correlation(
        x_var=rwd_mvmt_onset,
        y_var=fraction_enlarged,
        CI=95,
        title="Rewarded Mvmts",
        xtitle=f"Relative onset (s)",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["N"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt onset vs fraction shrunken
    plot_scatter_correlation(
        x_var=rwd_mvmt_onset,
        y_var=fraction_shrunken,
        CI=95,
        title="Rewarded Mvmts",
        xtitle=f"Relative onset (s)",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    # NonRwd Mvmt onset vs fraction enlarged
    plot_scatter_correlation(
        x_var=nonrwd_mvmt_onset,
        y_var=fraction_enlarged,
        CI=95,
        title="Non-rewarded Mvmts",
        xtitle=f"Relative onset (s)",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["S"],
        save=False,
        save_path=None,
    )
    # NonRwd Mvmt onset vs fraction shrunken
    plot_scatter_correlation(
        x_var=nonrwd_mvmt_onset,
        y_var=fraction_shrunken,
        CI=95,
        title="Non-rewarded Mvmts",
        xtitle=f"Relative onset (s)",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["T"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Dendrite_Activity_Figure_1")
        fig.savefig(fname + ".pdf")


def plot_dendrite_movement_encoding(
    dataset,
    followup_dataset=None,
    exclude="Shaft Spine",
    threshold=0.3,
    figsize=(10, 10),
    save=False,
    save_path=None,
):
    """Function to plot the movement encoding related variables for the different
        dendrites relative to fraction of plastic spines

        INPUT PARAMETERS
            dataset - Spine_Activity_Data object

            followup_dataset - optional Spine_Activity_Data object of the subsequent
                                session to use for volume comparision. Default is None
                                to use the followup volumes in the dataset
            
            exclude - str specifying the type of spine to exclude from analysis

            threshold - float or tuple of floats specifyuing the threshold cutoff for
                        classifying plasticity
            
            figsize - tuple specifying the figure size

            save - boolean specifying whether to save the data or not

            save_path - str specifyng where to save the data
        
    """
    COLORS = ["darkorange", "darkviolet"]

    # Pull relevant data
    dendrite_number = dataset.dendrite_number
    spine_volumes = dataset.spine_volumes
    spine_flags = dataset.spine_flags
    if followup_dataset == None:
        followup_volumes = dataset.followup_volumes
        followup_flags = dataset.followup_flags
    else:
        followup_volumes = dataset.spine_volumes
        followup_flags = dataset.spine_flags
    ## Movement encoding variables
    dendrite_movement_correlation = dataset.dendrite_movement_correlation
    dendrite_movement_stereotypy = dataset.dendrite_movement_stereotypy
    dendrite_movement_reliability = dataset.dendrite_movement_reliability
    dendrite_movement_specificity = dataset.dendrite_movement_specificity
    dendrite_LMP_reliability = dataset.dendrite_LMP_reliability
    dendrite_LMP_specificity = dataset.dendrite_LMP_specificity
    dendrite_rwd_movement_correlation = dataset.dendrite_rwd_movement_correlation
    dendrite_rwd_movement_stereotypy = dataset.dendrite_rwd_movement_stereotypy
    dendrite_rwd_movement_reliability = dataset.dendrite_rwd_movement_reliablity
    dendrite_rwd_movement_specificity = dataset.dendrite_rwd_movement_specificity

    # Calculate volumes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]
    delta_volume, spine_idxs = calculate_volume_change(
        volumes, flags, norm=False, exclude=exclude,
    )
    delta_volume = delta_volume[-1]
    enlarged_spines, shrunken_spines, stable_spines = classify_plasticity(
        delta_volume, threshold=threshold, norm=False
    )

    # Subselect present spines
    dendrite_movement_correlation = d_utils.subselect_data_by_idxs(
        dendrite_movement_correlation, spine_idxs
    )
    dendrite_movement_stereotypy = d_utils.subselect_data_by_idxs(
        dendrite_movement_stereotypy, spine_idxs
    )
    dendrite_movement_reliability = d_utils.subselect_data_by_idxs(
        dendrite_movement_reliability, spine_idxs
    )
    dendrite_movement_specificity = d_utils.subselect_data_by_idxs(
        dendrite_movement_specificity, spine_idxs
    )
    dendrite_LMP_reliability = d_utils.subselect_data_by_idxs(
        dendrite_LMP_reliability, spine_idxs
    )
    dendrite_LMP_specificity = d_utils.subselect_data_by_idxs(
        dendrite_LMP_specificity, spine_idxs
    )
    dendrite_rwd_movement_correlation = d_utils.subselect_data_by_idxs(
        dendrite_rwd_movement_correlation, spine_idxs
    )
    dendrite_rwd_movement_stereotypy = d_utils.subselect_data_by_idxs(
        dendrite_rwd_movement_stereotypy, spine_idxs
    )
    dendrite_rwd_movement_reliability = d_utils.subselect_data_by_idxs(
        dendrite_rwd_movement_reliability, spine_idxs
    )
    dendrite_rwd_movement_specificity = d_utils.subselect_data_by_idxs(
        dendrite_rwd_movement_specificity, spine_idxs
    )

    # Organize based on fraction of plastic spines
    unique_dend = np.unique(dendrite_number)
    fraction_enlarged = np.zeros(len(unique_dend))
    fraction_shrunken = np.zeros(len(unique_dend))
    mvmt_corr = np.zeros(len(unique_dend))
    mvmt_stereo = np.zeros(len(unique_dend))
    mvmt_reli = np.zeros(len(unique_dend))
    mvmt_spec = np.zeros(len(unique_dend))
    LMP_reli = np.zeros(len(unique_dend))
    LMP_spec = np.zeros(len(unique_dend))
    rwd_mvmt_corr = np.zeros(len(unique_dend))
    rwd_mvmt_stereo = np.zeros(len(unique_dend))
    rwd_mvmt_reli = np.zeros(len(unique_dend))
    rwd_mvmt_spec = np.zeros(len(unique_dend))

    for i, dend in enumerate(unique_dend):
        spines = np.nonzero(dendrite_number == dend)[0]
        fraction_enlarged[i] = np.nansum(enlarged_spines[spines]) / len(spines)
        fraction_shrunken[i] = np.nansum(shrunken_spines[spines]) / len(spines)
        mvmt_corr[i] = np.nanmean(dendrite_movement_correlation[spines])
        mvmt_stereo[i] = np.nanmean(dendrite_movement_stereotypy[spines])
        mvmt_reli[i] = np.nanmean(dendrite_movement_reliability[spines])
        mvmt_spec[i] = np.nanmean(dendrite_movement_specificity[spines])
        LMP_reli[i] = np.nanmean(dendrite_LMP_reliability[spines])
        LMP_spec[i] = np.nanmean(dendrite_LMP_specificity[spines])
        rwd_mvmt_corr[i] = np.nanmean(dendrite_rwd_movement_correlation[spines])
        rwd_mvmt_stereo[i] = np.nanmean(dendrite_rwd_movement_stereotypy[spines])
        rwd_mvmt_reli[i] = np.nanmean(dendrite_rwd_movement_reliability[spines])
        rwd_mvmt_spec[i] = np.nanmean(dendrite_rwd_movement_specificity[spines])

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFGH
        IJKL
        MNOP
        QRST
        """,
        figsize=figsize,
    )
    fig.suptitle("Dendrite Movement Encoding")
    fig.subplot_adjust(hspace=1, wspace=0.5)

    #################### Plot data onto the axes ########################
    # Mvmt corr vs fraction enlarged
    plot_scatter_correlation(
        x_var=mvmt_corr,
        y_var=fraction_enlarged,
        CI=95,
        title="All Mvmts",
        xtitle="LMP correlation (r)",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # Mvmt corr vs fraction shrunken
    plot_scatter_correlation(
        x_var=mvmt_corr,
        y_var=fraction_shrunken,
        CI=95,
        title="All Mvmts",
        xtitle="LMP correlation (r)",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Mvmt stereotypy vs fraction enlarged
    plot_scatter_correlation(
        x_var=mvmt_stereo,
        y_var=fraction_enlarged,
        CI=95,
        title="All Mvmts",
        xtitle="Movement stereotypy",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # Mvmt stereotypy vs fraction shrunken
    plot_scatter_correlation(
        x_var=mvmt_stereo,
        y_var=fraction_shrunken,
        CI=95,
        title="All Mvmts",
        xtitle="Movement stereotypy",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # Mvmt reliability vs fraction enlarged
    plot_scatter_correlation(
        x_var=mvmt_reli,
        y_var=fraction_enlarged,
        CI=95,
        title="All Mvmts",
        xtitle="Movement reliability",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Mvmt reliability vs fraction shrunken
    plot_scatter_correlation(
        x_var=mvmt_reli,
        y_var=fraction_shrunken,
        CI=95,
        title="All Mvmts",
        xtitle="Movement reliability",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    # Mvmt specificity vs fraction enlarged
    plot_scatter_correlation(
        x_var=mvmt_spec,
        y_var=fraction_enlarged,
        CI=95,
        title="All Mvmts",
        xtitle="Movement specificity",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    # Mvmt specificity vs fraction shrunken
    plot_scatter_correlation(
        x_var=mvmt_spec,
        y_var=fraction_shrunken,
        CI=95,
        title="All Mvmts",
        xtitle="Movement specificity",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt corr vs fraction enlarged
    plot_scatter_correlation(
        x_var=rwd_mvmt_corr,
        y_var=fraction_enlarged,
        CI=95,
        title="Rewarded Mvmts",
        xtitle="LMP correlation (r)",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt corr vs fraction shrunken
    plot_scatter_correlation(
        x_var=rwd_mvmt_corr,
        y_var=fraction_shrunken,
        CI=95,
        title="Rewarded Mvmts",
        xtitle="LMP correlation (r)",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt stereotypy vs fraction enlarged
    plot_scatter_correlation(
        x_var=rwd_mvmt_stereo,
        y_var=fraction_enlarged,
        CI=95,
        title="Rewarded Mvmts",
        xtitle="Movement stereotypy",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt stereotypy vs fraction shrunken
    plot_scatter_correlation(
        x_var=rwd_mvmt_stereo,
        y_var=fraction_shrunken,
        CI=95,
        title="Rewarded Mvmts",
        xtitle="Movement stereotypy",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt reliability vs fraction enlarged
    plot_scatter_correlation(
        x_var=rwd_mvmt_reli,
        y_var=fraction_enlarged,
        CI=95,
        title="Rewarded Mvmts",
        xtitle="Movement reliability",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt reliability vs fraction shrunken
    plot_scatter_correlation(
        x_var=rwd_mvmt_reli,
        y_var=fraction_shrunken,
        CI=95,
        title="Rewarded Mvmts",
        xtitle="Movement reliability",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["N"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt specificity vs fraction enlarged
    plot_scatter_correlation(
        x_var=rwd_mvmt_spec,
        y_var=fraction_enlarged,
        CI=95,
        title="Rewarded Mvmts",
        xtitle="Movement specificity",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    # Rwd Mvmt specificity vs fraction shrunken
    plot_scatter_correlation(
        x_var=rwd_mvmt_spec,
        y_var=fraction_shrunken,
        CI=95,
        title="Rewarded Mvmts",
        xtitle="Movement specificity",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["P"],
        save=False,
        save_path=None,
    )
    # LMP reliability vs fraction enlarged
    plot_scatter_correlation(
        x_var=LMP_reli,
        y_var=fraction_enlarged,
        CI=95,
        title="All Mvmts",
        xtitle="Learned Movement reliability",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["Q"],
        save=False,
        save_path=None,
    )
    # LMP reliability vs fraction shrunken
    plot_scatter_correlation(
        x_var=LMP_reli,
        y_var=fraction_shrunken,
        CI=95,
        title="All Mvmts",
        xtitle="Learned Movement reliability",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["R"],
        save=False,
        save_path=None,
    )
    # LMP specificity vs fraction enlarged
    plot_scatter_correlation(
        x_var=LMP_spec,
        y_var=fraction_enlarged,
        CI=95,
        title="All Mvmts",
        xtitle="Learned Movement specificity",
        ytitle="Enlarged spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[0],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[0],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["S"],
        save=False,
        save_path=None,
    )
    # LMP specificity vs fraction shrunken
    plot_scatter_correlation(
        x_var=LMP_spec,
        y_var=fraction_shrunken,
        CI=95,
        title="All Mvmts",
        xtitle="Learned Movement specificity",
        ytitle="Shrunken spine fraction",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=25,
        face_color=COLORS[1],
        edge_color="white",
        edge_width=0.3,
        line_color=COLORS[1],
        s_alpha=0.5,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["T"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Dendrite_Activity_Figure_2")
        fig.savefig(fname + ".pdf")

