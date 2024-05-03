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
from Lab_Analyses.Plotting.plot_pie_chart import plot_pie_chart
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    find_present_spines,
    load_spine_datasets,
)
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import (
    calculate_spine_dynamics,
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils

sns.set()
sns.set_style("ticks")


def plot_structural_plasticity(
    apical_datasets,
    basal_datasets,
    figsize=(7, 7),
    threshold=0.3,
    showmeans=False,
    hist_bins=30,
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to plot the spine plasticity of apical and basal dendrites

    INPUT PARAMETERS
        apical_datasets - list of Spine_Acativity_Data objects for each session
                          for apical datasets

        basal_datasets - list of Spine_Activity_Data objects for each session
                          for basal datasets

        figsize - tuple specifying the figure size

        showmeans - boolean specifying whether to show means on box plots

        mean_type - str specifying the central tendency for the data to use

        err_type - str specifying what to use to generate the error bars

        hist_bins - int specifying the number of bins for histograms

        test_type - str specifying whether to perform parametric or nonparametric stats

        display_stats - boolean specifying whether to perform stats or not

        save - boolean specifying to save the figure or not

        save_path - str specifying the path where to save the figure


    """
    COLORS = ["goldenrod", "mediumseagreen"]

    # Pull the relevant data
    apical_volumes = [x.spine_volumes for x in apical_datasets]

    apical_mice = apical_datasets[0].mouse_id

    basal_volumes = [x.spine_volumes for x in basal_datasets]

    basal_mice = basal_datasets[0].mouse_id

    # Calulate the spine dynamics
    ## Have to do it FOV by FOV
    apical_density = {"Early": [], "Middle": [], "Late": []}
    apical_new_spines = {"Early": [], "Middle": [], "Late": []}
    apical_elim_spines = {"Early": [], "Middle": [], "Late": []}
    apical_relative_volumes = {"Early": [], "Middle": [], "Late": []}
    apical_enlarged_spines = {"Early": [], "Middle": [], "Late": []}
    apical_shrunken_spines = {"Early": [], "Middle": [], "Late": []}
    apical_enlarged_volumes = {"Early": [], "Middle": [], "Late": []}
    apical_shrunken_volumes = {"Early": [], "Middle": [], "Late": []}
    basal_density = {"Early": [], "Middle": [], "Late": []}
    basal_new_spines = {"Early": [], "Middle": [], "Late": []}
    basal_elim_spines = {"Early": [], "Middle": [], "Late": []}
    basal_relative_volumes = {"Early": [], "Middle": [], "Late": []}
    basal_enlarged_spines = {"Early": [], "Middle": [], "Late": []}
    basal_shrunken_spines = {"Early": [], "Middle": [], "Late": []}
    basal_enlarged_volumes = {"Early": [], "Middle": [], "Late": []}
    basal_shrunken_volumes = {"Early": [], "Middle": [], "Late": []}
    apical_mice = list(set(apical_mice))
    basal_mice = list(set(basal_mice))

    # Go through each apical datasets
    for mouse in apical_mice:
        temp_datasets = load_spine_datasets(
            mouse, ["Early", "Middle", "Late"], fov_type="apical"
        )
        for FOV, dataset in temp_datasets.items():
            temp_groupings = []
            temp_flags = []
            temp_positions = []

            for session in ["Early", "Middle", "Late"]:
                temp_data = dataset[session]
                temp_groupings.append(temp_data.spine_groupings)
                temp_flags.append(temp_data.spine_flags)
                temp_positions.append(temp_data.spine_positions)
                delta, _ = calculate_volume_change(
                    [
                        temp_data.corrected_spine_volume,
                        temp_data.corrected_followup_volume,
                    ],
                    [
                        temp_data.spine_flags,
                        temp_data.followup_flags,
                    ],
                    exclude="Shaft Spine",
                    norm=False,
                )
                e, s, _ = classify_plasticity(
                    delta[-1],
                    threshold=threshold,
                    norm=False,
                )
                apical_relative_volumes[session].append(np.nanmean(delta[-1]))
                apical_enlarged_spines[session].append(np.nansum(e) / len(e))
                apical_shrunken_spines[session].append(np.nansum(s) / len(s))
                apical_enlarged_volumes[session].append(
                    np.nanmean(np.array(delta[-1])[e])
                )
                apical_shrunken_volumes[session].append(
                    np.nanmean(np.array(delta[-1])[s])
                )
            temp_density, temp_new, temp_elim = calculate_spine_dynamics(
                temp_flags,
                temp_positions,
                temp_groupings,
            )
            for j, session in enumerate(["Early", "Middle", "Late"]):
                apical_density[session].append(temp_density[j])
                apical_new_spines[session].append(temp_new[j])
                apical_elim_spines[session].append(temp_elim[j])
    for mouse in basal_mice:
        temp_datasets = load_spine_datasets(
            mouse, ["Early", "Middle", "Late"], fov_type="basal"
        )
        for FOV, dataset in temp_datasets.items():
            temp_groupings = []
            temp_flags = []
            temp_positions = []
            for session in ["Early", "Middle", "Late"]:
                temp_data = dataset[session]
                temp_groupings.append(temp_data.spine_groupings)
                temp_flags.append(temp_data.spine_flags)
                temp_positions.append(temp_data.spine_positions)
                delta, _ = calculate_volume_change(
                    [
                        temp_data.corrected_spine_volume,
                        temp_data.corrected_followup_volume,
                    ],
                    [
                        temp_data.spine_flags,
                        temp_data.followup_flags,
                    ],
                    exclude="Shaft Spine",
                    norm=False,
                )
                e, s, _ = classify_plasticity(
                    delta[-1],
                    threshold=threshold,
                    norm=False,
                )
                basal_relative_volumes[session].append(np.nanmean(delta[-1]))
                basal_enlarged_spines[session].append(np.nansum(e) / len(e))
                basal_shrunken_spines[session].append(np.nansum(s) / len(s))
                basal_enlarged_volumes[session].append(
                    np.nanmean(np.array(delta[-1])[e])
                )
                basal_shrunken_volumes[session].append(
                    np.nanmean(np.array(delta[-1])[s])
                )
            temp_density, temp_new, temp_elim = calculate_spine_dynamics(
                temp_flags,
                temp_positions,
                temp_groupings,
            )
            for j, session in enumerate(["Early", "Middle", "Late"]):
                basal_density[session].append(temp_density[j])
                basal_new_spines[session].append(temp_new[j])
                basal_elim_spines[session].append(temp_elim[j])

    # concatenate all values in the dictionaries
    for session in ["Early", "Middle", "Late"]:
        apical_density[session] = np.concatenate(apical_density[session])
        apical_new_spines[session] = np.concatenate(apical_new_spines[session])
        apical_elim_spines[session] = np.concatenate(apical_elim_spines[session])
        basal_density[session] = np.concatenate(basal_density[session])
        basal_new_spines[session] = np.concatenate(basal_new_spines[session])
        basal_elim_spines[session] = np.concatenate(basal_elim_spines[session])

    # Make the plot
    fig, axes = plt.subplot_mosaic(
        """
        ABC
        DEF
        GHI
        """,
        figsize=figsize,
    )
    fig.suptitle("Apical vs Basal Structural Plasticity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ######################### Plot data onto axes ###############################
    # Spine density
    plot_multi_line_plot(
        data_dict={
            "Apical": np.vstack(list(apical_density.values())),
            "Basal": np.vstack(list(basal_density.values())),
        },
        x_vals=range(3),
        plot_ind=False,
        figsize=(5, 5),
        title="Spine Density",
        ytitle="Spine density (\u03BCm)",
        xtitle="Session",
        ylim=(0.2, 0.9),
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
    axes["A"].set_xticklabels(["Early", "Middle", "Late"])
    # Fraction new spines
    plot_multi_line_plot(
        data_dict={
            "Apical": np.vstack(list(apical_new_spines.values())),
            "Basal": np.vstack(list(basal_new_spines.values())),
        },
        x_vals=range(3),
        plot_ind=False,
        figsize=(5, 5),
        title="Fraction new spines",
        ytitle="Fraction of new spines",
        xtitle="Session",
        ylim=None,
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
    axes["B"].set_xticklabels(["Early", "Middle", "Late"])
    # Fraction eliminated spines
    plot_multi_line_plot(
        data_dict={
            "Apical": np.vstack(list(apical_elim_spines.values())),
            "Basal": np.vstack(list(basal_elim_spines.values())),
        },
        x_vals=range(3),
        plot_ind=False,
        figsize=(5, 5),
        title="Fraction eliminated",
        ytitle="Fraction of elim. spines",
        xtitle="Session",
        ylim=None,
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
    axes["C"].set_xticklabels(["Early", "Middle", "Late"])
    # Initial spine volume
    plot_histogram(
        data=list((apical_volumes[0], basal_volumes[0])),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Initial Spine Volume",
        xtitle="Spine Volume",
        xlim=(0, None),
        figsize=(5, 5),
        color=COLORS,
        alpha=0.4,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    ## Inset box plot
    ax_D_inset = axes["D"].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_D_inset)
    plot_box_plot(
        data_dict={"Apical": apical_volumes[0], "Basal": basal_volumes[0]},
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Volume",
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
        ax=ax_D_inset,
        save=False,
        save_path=None,
    )
    # Fraction LTP
    plot_multi_line_plot(
        data_dict={
            "Apical": np.vstack(list(apical_enlarged_spines.values())),
            "Basal": np.vstack(list(basal_enlarged_spines.values())),
        },
        x_vals=range(3),
        plot_ind=False,
        figsize=(5, 5),
        title="Fraction sLTP",
        ytitle="Fraction of sLTP",
        xtitle="Session",
        ylim=None,
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
    axes["E"].set_xticklabels(["Early", "Middle", "Late"])
    # Fraction LTD
    plot_multi_line_plot(
        data_dict={
            "Apical": np.vstack(list(apical_shrunken_spines.values())),
            "Basal": np.vstack(list(basal_shrunken_spines.values())),
        },
        x_vals=range(3),
        plot_ind=False,
        figsize=(5, 5),
        title="Fraction sLTD",
        ytitle="Fraction of sLTD",
        xtitle="Session",
        ylim=None,
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
    axes["F"].set_xticklabels(["Early", "Middle", "Late"])
    # Relative volume
    plot_multi_line_plot(
        data_dict={
            "Apical": np.vstack(list(apical_relative_volumes.values())),
            "Basal": np.vstack(list(basal_relative_volumes.values())),
        },
        x_vals=range(3),
        plot_ind=False,
        figsize=(5, 5),
        title="Avg. Relative Volumes",
        ytitle="Average relative volume",
        xtitle="Session",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["G"],
        legend=True,
        save=False,
        save_path=None,
    )
    axes["G"].set_xticklabels(["Early", "Middle", "Late"])
    # LTP spine volumes
    plot_multi_line_plot(
        data_dict={
            "Apical": np.vstack(list(apical_enlarged_volumes.values())),
            "Basal": np.vstack(list(basal_enlarged_volumes.values())),
        },
        x_vals=range(3),
        plot_ind=False,
        figsize=(5, 5),
        title="sLTP Relative Volume",
        ytitle="sLTP relative volume",
        xtitle="Session",
        ylim=None,
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["H"],
        legend=True,
        save=False,
        save_path=None,
    )
    axes["H"].set_xticklabels(["Early", "Middle", "Late"])
    # LTD spine volumes
    plot_multi_line_plot(
        data_dict={
            "Apical": np.vstack(list(apical_shrunken_volumes.values())),
            "Basal": np.vstack(list(basal_shrunken_volumes.values())),
        },
        x_vals=range(3),
        plot_ind=False,
        figsize=(5, 5),
        title="sLTD Relative Volume",
        ytitle="sLTD relative volume",
        xtitle="Session",
        ylim=None,
        line_color=COLORS,
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
    axes["I"].set_xticklabels(["Early", "Middle", "Late"])

    fig.tight_layout()
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Apical_vs_Basal_Figure_1")
        fig.savefig(fname + ".pdf")

    ######################## Statistics Section #############################
    if display_stats == False:
        return

    print("Need to code in the statistics")


def plot_movement_related_activity(
    apical_dataset,
    basal_dataset,
    figsize=(10, 6),
    hist_bins=30,
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to plot movement-related activity of apical and basal spines

    INPUT PARAMETERS
        apical_dataset - Spine_Activity_Data object for apical data

        basal_dataset - Spine_Activity_Data object for basal data

        figsize - tuple specifying the size of the figure

        hist_bins - int specifying the number of bins for histogram

        showmeans - boolean specifying whether to show means on box plots

        test_type - str specifying whether to perform parametric or nonparametric stats

        test_method - str specifying the type of posthoc test to perform

        display_stats - boolean specifying whether to display stat results

        save - boolean specifying whether to save the figure or not

        save_path - str specifying where to save the figure

    """
    COLORS = ["goldenrod", "mediumseagreen"]

    # Pull relevant data
    ## Parameters
    sampling_rate = apical_dataset.parameters["Sampling Rate"]
    activity_window = apical_dataset.parameters["Activity Window"]
    if apical_dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"
    ## Present spine idxs
    apical_present = find_present_spines(apical_dataset.spine_flags)
    basal_present = find_present_spines(basal_dataset.spine_flags)
    ## Event rates
    apical_event_rate = d_utils.subselect_data_by_idxs(
        apical_dataset.spine_activity_rate, apical_present
    )
    basal_event_rate = d_utils.subselect_data_by_idxs(
        basal_dataset.spine_activity_rate, basal_present
    )
    ## Movement identifiers
    apical_MRSs = d_utils.subselect_data_by_idxs(
        apical_dataset.movement_spines, apical_present
    )
    apical_nMRSs = d_utils.subselect_data_by_idxs(
        apical_dataset.nonmovement_spines, apical_present
    )
    basal_MRSs = d_utils.subselect_data_by_idxs(
        basal_dataset.movement_spines, basal_present
    )
    basal_nMRSs = d_utils.subselect_data_by_idxs(
        basal_dataset.nonmovement_spines, basal_present
    )
    ## Movement related activity
    apical_glu_traces = d_utils.subselect_data_by_idxs(
        apical_dataset.spine_movement_traces, apical_present
    )
    basal_glu_traces = d_utils.subselect_data_by_idxs(
        basal_dataset.spine_movement_traces, basal_present
    )
    apical_ca_traces = d_utils.subselect_data_by_idxs(
        apical_dataset.spine_movement_calcium_traces, apical_present
    )
    basal_ca_traces = d_utils.subselect_data_by_idxs(
        basal_dataset.spine_movement_calcium_traces, basal_present
    )
    apical_glu_amp = d_utils.subselect_data_by_idxs(
        apical_dataset.spine_movement_amplitude, apical_present
    )
    basal_glu_amp = d_utils.subselect_data_by_idxs(
        basal_dataset.spine_movement_amplitude, basal_present
    )
    apical_ca_amp = d_utils.subselect_data_by_idxs(
        apical_dataset.spine_movement_calcium_amplitude, apical_present
    )
    basal_ca_amp = d_utils.subselect_data_by_idxs(
        basal_dataset.spine_movement_calcium_amplitude, basal_present
    )
    apical_onset = d_utils.subselect_data_by_idxs(
        apical_dataset.spine_movement_onset, apical_present
    )
    basal_onset = d_utils.subselect_data_by_idxs(
        basal_dataset.spine_movement_onset, basal_present
    )

    # Get mean and sem traces for plotting
    ## Apical GluSnFR
    a_glu_traces = list(compress(apical_glu_traces, apical_MRSs))
    a_glu_means = [np.nanmean(x, axis=1) for x in a_glu_traces if type(x) == np.ndarray]
    a_glu_means = np.vstack(a_glu_means)
    apical_glu_hmap = a_glu_means.T
    apical_grouped_glu_traces = np.nanmean(a_glu_means, axis=0)
    apical_grouped_glu_sems = stats.sem(a_glu_means, axis=0, nan_policy="omit")
    ## Basal GluSnFR
    b_glu_traces = list(compress(basal_glu_traces, basal_MRSs))
    b_glu_means = [np.nanmean(x, axis=1) for x in b_glu_traces if type(x) == np.ndarray]
    b_glu_means = np.vstack(b_glu_means)
    basal_glu_hmap = b_glu_means.T
    basal_grouped_glu_traces = np.nanmean(b_glu_means, axis=0)
    basal_grouped_glu_sems = stats.sem(b_glu_means, axis=0, nan_policy="omit")
    ## Apical Calcium
    a_ca_traces = list(compress(apical_ca_traces, basal_MRSs))
    a_ca_means = [np.nanmean(x, axis=1) for x in a_ca_traces if type(x) == np.ndarray]
    a_ca_means = np.vstack(a_ca_means)
    apical_ca_hmap = a_ca_means.T
    apical_grouped_ca_traces = np.nanmean(a_ca_means, axis=0)
    apical_grouped_ca_sems = stats.sem(a_ca_means, axis=0, nan_policy="omit")
    ## Basal GluSnFR
    b_ca_traces = list(compress(basal_ca_traces, basal_MRSs))
    b_ca_means = [np.nanmean(x, axis=1) for x in b_ca_traces if type(x) == np.ndarray]
    b_ca_means = np.vstack(b_ca_means)
    basal_ca_hmap = b_ca_means.T
    basal_grouped_ca_traces = np.nanmean(b_ca_means, axis=0)
    basal_grouped_ca_sems = stats.sem(b_ca_means, axis=0, nan_policy="omit")

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE
        FGHI.
        JK...
        """,
        figsize=figsize,
    )
    fig.suptitle("Apical vs Basal Movement-related activity")

    ###################### Plot data onto the axes ###########################
    ## Event rates
    plot_box_plot(
        data_dict={
            "Apical": apical_event_rate,
            "Basal": basal_event_rate,
        },
        figsize=(5, 5),
        title="Event rates",
        xtitle=None,
        ytitle=f"Event rate (events/min)",
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
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    ## MRS event event amplitude
    plot_box_plot(
        data_dict={
            "Apical": apical_glu_amp[apical_MRSs],
            "Basal": basal_glu_amp[basal_MRSs],
        },
        figsize=(5, 5),
        title="GluSnFR Amplitude",
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
    ## MRS calcium event event amplitude
    plot_box_plot(
        data_dict={
            "Apical": apical_ca_amp[apical_MRSs],
            "Basal": basal_ca_amp[basal_MRSs],
        },
        figsize=(5, 5),
        title="Calcium Amplitude",
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    ## MRS event onset amplitude
    plot_box_plot(
        data_dict={
            "Apical": apical_onset[apical_MRSs],
            "Basal": basal_onset[basal_MRSs],
        },
        figsize=(5, 5),
        title="GluSnFR Onset",
        xtitle=None,
        ytitle=f"Event onset (s)",
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
    ## Fraction Apical MRSs
    plot_pie_chart(
        data_dict={"MRSs": np.nansum(apical_MRSs), "nMRSs": np.nansum(apical_nMRSs)},
        title="Apical",
        figsize=(5, 5),
        colors=[COLORS[0], "silver"],
        alpha=0.9,
        edgecolor="white",
        txt_color="white",
        txt_size=9,
        legend=True,
        donut=0.6,
        linewidth=1.5,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    ## Fraction Basal MRSs
    plot_pie_chart(
        data_dict={"MRSs": np.nansum(basal_MRSs), "nMRSs": np.nansum(basal_nMRSs)},
        title="Basal",
        figsize=(5, 5),
        colors=[COLORS[1], "silver"],
        alpha=0.9,
        edgecolor="white",
        txt_color="white",
        txt_size=9,
        legend=True,
        donut=0.6,
        linewidth=1.5,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    ## Apical heatmap
    plot_activity_heatmap(
        apical_glu_hmap,
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Apical",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="YlOrBr",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    ## Basal heatmap
    plot_activity_heatmap(
        basal_glu_hmap,
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Basal",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="Greens",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # GluSnFR traces
    plot_mean_activity_traces(
        means=[apical_grouped_glu_traces, basal_grouped_glu_traces],
        sems=[apical_grouped_glu_sems, basal_grouped_glu_sems],
        group_names=["Apical", "Basal"],
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=COLORS,
        title="GluSnFR Traces",
        ytitle=activity_type,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    # Calcium traces
    plot_mean_activity_traces(
        means=[apical_grouped_ca_traces, basal_grouped_ca_traces],
        sems=[apical_grouped_ca_sems, basal_grouped_ca_sems],
        group_names=["Apical", "Basal"],
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
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    # Onset histogram
    plot_histogram(
        data=[apical_onset, basal_onset],
        bins=hist_bins,
        stat="probability",
        avlines=[0],
        title="GluSnFR Onsets",
        xtitle="Event onset (s)",
        xlim=activity_window,
        color=COLORS,
        alpha=0.4,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Apical_vs_Basal_Figure_2")
        fig.savefig(fname + ".pdf")

    ####################### Statistics Section #######################
    if display_stats == False:
        return

    # Peform stats
    if test_type == "parametric":
        rate_t, rate_p = stats.ttest_ind(
            apical_event_rate,
            basal_ca_amp,
            nan_policy="omit",
        )
        amp_t, amp_p = stats.ttest_ind(apical_glu_amp, basal_glu_amp, nan_policy="omit")
        onset_t, onset_p = stats.ttest_ind(
            apical_onset,
            basal_onset,
            nan_policy="omit",
        )
        ca_amp_t, ca_amp_p = stats.ttest_ind(
            apical_ca_amp, basal_ca_amp, nan_policy="omit"
        )
        test_title = "T-Test"
    elif test_type == "nonparametric":
        rate_t, rate_p = stats.mannwhitneyu(
            apical_event_rate[~np.isnan(apical_event_rate)],
            basal_event_rate[~np.isnan(basal_event_rate)],
        )
        amp_t, amp_p = stats.mannwhitneyu(
            apical_glu_amp[~np.isnan(apical_glu_amp)],
            basal_glu_amp[~np.isnan(basal_glu_amp)],
        )
        onset_t, onset_p = stats.mannwhitneyu(
            apical_onset[~np.isnan(apical_onset)],
            basal_onset[~np.isnan(basal_onset)],
        )
        ca_amp_t, ca_amp_p = stats.mannwhitneyu(
            apical_ca_amp[~np.isnan(apical_ca_amp)],
            basal_ca_amp[~np.isnan(basal_ca_amp)],
        )
        test_title = "Mann-Whitney U"

    # Organize the results
    results_dict = {
        "test": ["Event Rate", "GluSnFR Amp", "Onset", "Calcium Amp"],
        "stat": [rate_t, amp_t, onset_t, ca_amp_t],
        "p-val": [rate_p, amp_p, onset_p, ca_amp_p],
    }
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.update(results_df[["p-val"]].applymap("{:.4E}".format))

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
        fname = os.path.join(save_path, "Apical_vs_Basal_Figure_2_stats")
        fig2.savefig(fname + ".pdf")


def plot_spine_movement_encoding(
    apical_dataset,
    basal_dataset,
    figsize=(10, 4),
    showmeans=False,
    test_type="nonparametric",
    test_method="holm-sidak",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Figure ploting spine movement encoding variables for apical and basal data

    INPUT PARAMETERS
        apical_dataset - Spine_Activity_Data object for apical data

        basal_dataset - Spine_Activity_Data object for basal data

        figsize - tuple specifying the size of the figure

        showmeans - boolean specifying whether to show means on box plots

        test_type - str specifying whetehr to perform parametric or nonparametric tests

        test_method - str specifying the type of posthoc test to do

        display_stats - boolean specifying whether to show stats

        save - boolean specifying whether to save the figure

        save_path - str spcecifying the path to save the figure

    """
    COLORS = ["goldenrod", "mediumseagreen"]

    # Pull relevant data
    apical_LMP = apical_dataset.spine_movement_correlation
    apical_sterotypy = apical_dataset.spine_movement_stereotypy
    apical_reliability = apical_dataset.spine_movement_reliability
    apical_specificity = apical_dataset.spine_movement_specificity
    apical_fraction_rwd_mvmt = apical_dataset.spine_fraction_rwd_mvmts

    basal_LMP = basal_dataset.spine_movement_correlation
    basal_sterotypy = basal_dataset.spine_movement_stereotypy
    basal_reliability = basal_dataset.spine_movement_reliability
    basal_specificity = basal_dataset.spine_movement_specificity
    basal_fraction_rwd_mvmt = basal_dataset.spine_fraction_rwd_mvmts

    # Present spines
    apical_present = find_present_spines(apical_dataset.spine_flags)
    basal_present = find_present_spines(basal_dataset.spine_flags)

    # Subselect only the present spines
    apical_LMP = d_utils.subselect_data_by_idxs(apical_LMP, apical_present)
    apical_sterotypy = d_utils.subselect_data_by_idxs(apical_sterotypy, apical_present)
    apical_reliability = d_utils.subselect_data_by_idxs(
        apical_reliability, apical_present
    )
    apical_specificity = d_utils.subselect_data_by_idxs(
        apical_specificity, apical_present
    )
    apical_fraction_rwd_mvmt = d_utils.subselect_data_by_idxs(
        apical_fraction_rwd_mvmt, apical_present
    )

    basal_LMP = d_utils.subselect_data_by_idxs(basal_LMP, basal_present)
    basal_sterotypy = d_utils.subselect_data_by_idxs(basal_sterotypy, basal_present)
    basal_reliability = d_utils.subselect_data_by_idxs(basal_reliability, basal_present)
    basal_specificity = d_utils.subselect_data_by_idxs(basal_specificity, basal_present)
    basal_fraction_rwd_mvmt = d_utils.subselect_data_by_idxs(
        basal_fraction_rwd_mvmt, basal_present
    )

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """ABCDE""",
        figsize=figsize,
    )
    fig.suptitle("Apical vs Basal Spine Movement Encoding")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ######################## Plot data onto the axes #################################
    # LMP correlation
    plot_box_plot(
        data_dict={"Apical": apical_LMP, "Basal": basal_LMP},
        figsize=(5, 5),
        title="LMP",
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
    # Stereotypy
    plot_box_plot(
        data_dict={
            "Apical": apical_sterotypy,
            "Basal": basal_sterotypy,
        },
        figsize=(5, 5),
        title="Stereotypy",
        xtitle=None,
        ytitle="Movement stereotypy (r)",
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
    # Reliability
    plot_box_plot(
        data_dict={"Apical": apical_reliability, "Basal": basal_reliability},
        figsize=(5, 5),
        title="Reliability",
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
    #  Specificity
    plot_box_plot(
        data_dict={"Apical": apical_specificity, "Basal": basal_specificity},
        figsize=(5, 5),
        title="Specificity",
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
    # Fraction Rewarded
    plot_box_plot(
        data_dict={
            "Apical": apical_fraction_rwd_mvmt,
            "Basal": basal_fraction_rwd_mvmt,
        },
        figsize=(5, 5),
        title="Fraction Rewarded",
        xtitle=None,
        ytitle="Fraction of rewarded movements",
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
        fname = os.path.join(save_path, "Apical_vs_Basal_Figure_3")
        fig.savefig(fname + ".pdf")

    ################################# Statistics Section ################################
    if display_stats == False:
        return

    # Perform the statistics
    if test_type == "parametric":
        LMP_t, LMP_p = stats.ttest_ind(apical_LMP, basal_LMP, nan_policy="omit")
        stereo_t, stereo_p = stats.ttest_ind(
            apical_sterotypy,
            basal_sterotypy,
            nan_policy="omit",
        )
        reli_t, reli_p = stats.ttest_ind(
            apical_reliability,
            basal_reliability,
            nan_policy="omit",
        )
        (
            speci_t,
            speci_p,
        ) = stats.ttest_ind(
            apical_specificity,
            basal_specificity,
            nan_policy="omit",
        )
        rwd_t, rwd_p = stats.ttest_ind(
            apical_fraction_rwd_mvmt, basal_fraction_rwd_mvmt, nan_policy="omit"
        )
        test_title = "T-Test"
    elif test_type == "nonparametric":
        LMP_t, LMP_p = stats.mannwhitneyu(
            apical_LMP[~np.isnan(apical_LMP)], basal_LMP[~np.isnan(basal_LMP)]
        )
        stereo_t, stereo_p = stats.mannwhitneyu(
            apical_sterotypy[~np.isnan(apical_sterotypy)],
            basal_sterotypy[~np.isnan(basal_sterotypy)],
        )
        reli_t, reli_p = stats.mannwhitneyu(
            apical_reliability[~np.isnan(apical_reliability)],
            basal_reliability[~np.isnan(basal_reliability)],
        )
        speci_t, speci_p = stats.mannwhitneyu(
            apical_specificity[~np.isnan(apical_specificity)],
            basal_specificity[~np.isnan(basal_specificity)],
        )
        rwd_t, rwd_p = stats.mannwhitneyu(
            apical_fraction_rwd_mvmt[~np.isnan(apical_fraction_rwd_mvmt)],
            basal_fraction_rwd_mvmt[~np.isnan(basal_fraction_rwd_mvmt)],
        )
        test_title = "Mann-Whitney U"

    # Organize the results
    result_dict = {
        "Comparison": [
            "LMP",
            "Stereotypy",
            "Reliability",
            "Specificity",
            "Frac. Rewarded",
        ],
        "stat": [
            LMP_t,
            stereo_t,
            reli_t,
            speci_t,
            rwd_t,
        ],
        "p-val": [
            LMP_p,
            stereo_p,
            reli_p,
            speci_p,
            rwd_p,
        ],
    }
    results_df = pd.DataFrame.from_dict(result_dict)
    results_df.update(results_df[["p-val"]].applymap("{:.4E}".format))

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
        fname = os.path.join(save_path, "Apical_vs_Basal_Figure_3_stats")
        fig2.savefig(fname + ".pdf")


def plot_local_coactivity(
    apical_dataset,
    basal_dataset,
    figsize=(10, 8),
    showmeans=False,
    mean_type="median",
    err_type="CI",
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to plot local coactivity properties of apical and basal spines

    INPUT PARAMETERS
        apical_dataset - Local_Coactivity_Data object for apical data

        basal_dataset - Local_Coactivity_Data object for basal data

        figsize - tuple specifying the size of the figure

        showmeans - boolean specifying whehter to show means on box plots

        mean_type - str specifying the central tendency for bar plots

        err_type - str specifying the error of bar plots

        test_type - str specifying whether to perform parametric or nonparametric tests

        display_stats - boolean specifying whether to display stat results

        save - boolean specifying whether to save the figure

        save_path - str specifying the path to save the figure to

    """
    COLORS = ["goldenrod", "mediumseagreen"]

    # Find the present spines
    apical_present = find_present_spines(apical_dataset.spine_flags)
    basal_present = find_present_spines(basal_dataset.spine_flags)
    distance_bins = apical_dataset.parameters["position bins"][1:]

    # Pull relevant data
    apical_distance_coactivity = apical_dataset.distance_coactivity_rate
    basal_distance_coactivity = basal_dataset.distance_coactivity_rate
    apical_distance_coactivity_norm = apical_dataset.distance_coactivity_rate_norm
    basal_distance_coactivity_norm = basal_dataset.distance_coactivity_rate_norm
    apical_local_coactivity = apical_dataset.avg_local_coactivity_rate
    basal_local_coactivity = basal_dataset.avg_local_coactivity_rate
    apical_shuff_coactivity = apical_dataset.shuff_local_coactivity_rate
    basal_shuff_coactivity = basal_dataset.shuff_local_coactivity_rate
    apical_local_coactivity_norm = apical_dataset.avg_local_coactivity_rate_norm
    basal_local_coactivity_norm = basal_dataset.avg_local_coactivity_rate_norm
    apical_shuff_coactivity_norm = apical_dataset.shuff_local_coactivity_rate_norm
    basal_shuff_coactivity_norm = basal_dataset.shuff_local_coactivity_rate_norm
    apical_near_vs_dist = apical_dataset.near_vs_dist_coactivity
    basal_near_vs_dist = basal_dataset.near_vs_dist_coactivity
    apical_near_vs_dist_norm = apical_dataset.near_vs_dist_coactivity_norm
    basal_near_vs_dist_norm = basal_dataset.near_vs_dist_coactivity_norm
    apical_fraction_coactive = apical_dataset.fraction_spine_coactive
    basal_fraction_coactive = basal_dataset.fraction_spine_coactive
    apical_fraction_participating = apical_dataset.fraction_coactivity_participation
    basal_fraction_participating = basal_dataset.fraction_coactivity_participation
    apical_coactive_num = apical_dataset.coactive_spine_num
    basal_coactive_num = basal_dataset.coactive_spine_num
    # apical_coactive_spines = apical_dataset.coactive_spines
    # basal_coactive_spines = basal_dataset.coactive_spines
    # apical_coactive_spines_norm = apical_dataset.coactive_spines_norm
    # basal_coactive_spines_norm = basal_dataset.coactive_spines_norm

    # Subselect for present spines
    apical_distance_coactivity = d_utils.subselect_data_by_idxs(
        apical_distance_coactivity, apical_present
    )
    basal_distance_coactivity = d_utils.subselect_data_by_idxs(
        basal_distance_coactivity, basal_present
    )
    apical_distance_coactivity_norm = d_utils.subselect_data_by_idxs(
        apical_distance_coactivity_norm, apical_present
    )
    basal_distance_coactivity_norm = d_utils.subselect_data_by_idxs(
        basal_distance_coactivity_norm, basal_present
    )
    apical_local_coactivity = d_utils.subselect_data_by_idxs(
        apical_local_coactivity, apical_present
    )
    basal_local_coactivity = d_utils.subselect_data_by_idxs(
        basal_local_coactivity, basal_present
    )
    apical_shuff_coactivity = d_utils.subselect_data_by_idxs(
        apical_shuff_coactivity, apical_present
    )
    basal_shuff_coactivity = d_utils.subselect_data_by_idxs(
        basal_shuff_coactivity, basal_present
    )
    apical_local_coactivity_norm = d_utils.subselect_data_by_idxs(
        apical_local_coactivity_norm, apical_present
    )
    basal_local_coactivity_norm = d_utils.subselect_data_by_idxs(
        basal_local_coactivity_norm, basal_present
    )
    apical_shuff_coactivity_norm = d_utils.subselect_data_by_idxs(
        apical_shuff_coactivity_norm, apical_present
    )
    basal_shuff_coactivity_norm = d_utils.subselect_data_by_idxs(
        basal_shuff_coactivity_norm, basal_present
    )
    apical_near_vs_dist = d_utils.subselect_data_by_idxs(
        apical_near_vs_dist, apical_present
    )
    basal_near_vs_dist = d_utils.subselect_data_by_idxs(
        basal_near_vs_dist, basal_present
    )
    apical_near_vs_dist_norm = d_utils.subselect_data_by_idxs(
        apical_near_vs_dist_norm, apical_present
    )
    basal_near_vs_dist_norm = d_utils.subselect_data_by_idxs(
        basal_near_vs_dist_norm, basal_present
    )
    apical_fraction_coactive = d_utils.subselect_data_by_idxs(
        apical_fraction_coactive, apical_present
    )
    basal_fraction_coactive = d_utils.subselect_data_by_idxs(
        basal_fraction_coactive, basal_present
    )
    apical_fraction_participating = d_utils.subselect_data_by_idxs(
        apical_fraction_participating, apical_present
    )
    basal_fraction_participating = d_utils.subselect_data_by_idxs(
        basal_fraction_participating, basal_present
    )
    apical_coactive_num = d_utils.subselect_data_by_idxs(
        apical_coactive_num, apical_present
    )
    basal_coactive_num = d_utils.subselect_data_by_idxs(
        basal_coactive_num, basal_present
    )
    # apical_coactive_spines = d_utils.subselect_data_by_idxs(
    #     apical_coactive_spines, apical_present
    # )
    # basal_coactive_spines = d_utils.subselect_data_by_idxs(
    #     basal_coactive_spines, basal_present
    # )
    # apical_coactive_spines_norm = d_utils.subselect_data_by_idxs(
    #     apical_coactive_spines_norm, apical_present
    # )
    # basal_coactive_spines_norm = d_utils.subselect_data_by_idxs(
    #     basal_coactive_spines_norm, basal_present
    # )

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE.
        FGHIJ.
        .K..LM
        """,
        figsize=figsize,
        width_ratios=[2, 1, 2, 2, 1, 1],
    )
    fig.suptitle("Apical vs Basal Local Coactivity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ########################## Plot data onto axes ############################
    # Distance coactivity rate
    plot_multi_line_plot(
        data_dict={
            "Apical": apical_distance_coactivity,
            "Basal": basal_distance_coactivity,
        },
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        mean_type="median",
        title="Raw Coactivity",
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
    # Distance coactivity rate norm
    plot_multi_line_plot(
        data_dict={
            "Apical": apical_distance_coactivity_norm,
            "Basal": basal_distance_coactivity_norm,
        },
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        mean_type="median",
        title="Norm. Coactivity",
        ytitle="Norm. coactivity rate",
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
    # Local coactivity rate bar plot
    plot_box_plot(
        data_dict={
            "Apical": apical_local_coactivity,
            "Basal": basal_local_coactivity,
        },
        figsize=(5, 5),
        title="Raw",
        xtitle=None,
        ytitle=f"Local coactivity rate (events/min)",
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
    # Local normalized coactivity rate bar plot
    plot_box_plot(
        data_dict={
            "Apical": apical_local_coactivity_norm,
            "Basal": basal_local_coactivity_norm,
        },
        figsize=(5, 5),
        title="Norm.",
        xtitle=None,
        ytitle=f"Local norm. coactivity rate",
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
    # Near vs dist coactivity
    plot_box_plot(
        data_dict={
            "Apical": apical_near_vs_dist,
            "Basal": basal_near_vs_dist,
        },
        figsize=(5, 5),
        title="Raw",
        xtitle=None,
        ytitle=f"Relative coactivity\n (near - dist.)",
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
    # Near vs dist coactivity norm
    plot_box_plot(
        data_dict={
            "Apical": apical_near_vs_dist_norm,
            "Basal": basal_near_vs_dist_norm,
        },
        figsize=(5, 5),
        title="Norm.",
        xtitle=None,
        ytitle=f"Relative norm. coactivity\n (near - dist.)",
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
    # Fraction coactive
    plot_box_plot(
        data_dict={
            "Apical": apical_fraction_coactive,
            "Basal": basal_fraction_coactive,
        },
        figsize=(5, 5),
        title="",
        xtitle=None,
        ytitle=f"Fraction of events coactive",
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Fraction of coactivity participating
    plot_box_plot(
        data_dict={
            "Apical": apical_fraction_participating,
            "Basal": basal_fraction_participating,
        },
        figsize=(5, 5),
        title="Raw",
        xtitle=None,
        ytitle=f"Fraction of coactivity\n particpating",
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
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    # Coactive spine number
    plot_box_plot(
        data_dict={
            "Apical": apical_coactive_num,
            "Basal": basal_coactive_num,
        },
        figsize=(5, 5),
        title="Raw",
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
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    # Apical local vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list(
            (
                apical_local_coactivity,
                apical_shuff_coactivity,
            )
        ),
        plot_ind=True,
        title="Apical",
        xtitle=f"Local coactivity (event/min)",
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
            "data": apical_local_coactivity,
            "shuff": apical_shuff_coactivity.flatten().astype(np.float32),
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
    # Basal local vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list(
            (
                basal_local_coactivity,
                basal_shuff_coactivity,
            )
        ),
        plot_ind=True,
        title="Basal",
        xtitle=f"Local coactivity (event/min)",
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
            "data": basal_local_coactivity,
            "shuff": basal_shuff_coactivity.flatten().astype(np.float32),
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
    # Apical Norm local vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list(
            (
                apical_local_coactivity_norm,
                apical_shuff_coactivity_norm,
            )
        ),
        plot_ind=True,
        title="Apical",
        xtitle=f"Local norm. coactivity",
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
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_h_inset = axes["H"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_h_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": apical_local_coactivity_norm,
            "shuff": apical_shuff_coactivity_norm.flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        ytitle="Norm. coactivity rate",
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
        ax=ax_h_inset,
        save=False,
        save_path=None,
    )
    # Basal Norm local vs chance
    ## histogram
    plot_cummulative_distribution(
        data=list(
            (
                basal_local_coactivity_norm,
                basal_shuff_coactivity_norm,
            )
        ),
        plot_ind=True,
        title="Basal",
        xtitle=f"Local norm. coactivity",
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
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_i_inset = axes["I"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_i_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": basal_local_coactivity_norm,
            "shuff": basal_shuff_coactivity_norm.flatten().astype(np.float32),
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        ytitle="Norm. coactivity rate",
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
        ax=ax_i_inset,
        save=False,
        save_path=None,
    )

    fig.tight_layout()
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Apical_vs_Basal_Figure_4")
        fig.savefig(fname + ".pdf")

    ##################### Statistics Section ########################
    if display_stats == False:
        return
    # Perform tests
    if test_type == "parametric":
        coactivity_t, coactivity_p = stats.ttest_ind(
            apical_local_coactivity,
            basal_local_coactivity,
            nan_policy="omit",
        )
        coactivity_norm_t, coactivity_norm_p = stats.ttest_ind(
            apical_local_coactivity_norm,
            basal_local_coactivity_norm,
            nan_policy="omit",
        )
        near_dist_t, near_dist_p = stats.ttest_ind(
            apical_near_vs_dist,
            basal_near_vs_dist,
            nan_policy="omit",
        )
        near_dist_norm_t, near_dist_norm_p = stats.ttest_ind(
            apical_near_vs_dist_norm, basal_near_vs_dist_norm, nan_policy="omit"
        )
        frac_coactive_t, frac_coactive_p = stats.ttest_ind(
            apical_fraction_coactive,
            basal_fraction_participating,
            nan_policy="omit",
        )
        frac_part_t, frac_part_p = stats.ttest_ind(
            apical_fraction_participating,
            basal_fraction_participating,
            nan_policy="omit",
        )
        num_t, num_p = stats.ttest_ind(
            apical_coactive_num,
            basal_coactive_num,
            nan_policy="omit",
        )
        test_title = "T-Test"
    elif test_type == "nonparametric":
        coactivity_t, coactivity_p = stats.mannwhitneyu(
            apical_local_coactivity[~np.isnan(apical_local_coactivity)],
            basal_local_coactivity[~np.isnan(basal_local_coactivity)],
        )
        coactivity_norm_t, coactivity_norm_p = stats.mannwhitneyu(
            apical_local_coactivity_norm[~np.isnan(apical_local_coactivity_norm)],
            basal_local_coactivity_norm[~np.isnan(basal_local_coactivity_norm)],
        )
        near_dist_t, near_dist_p = stats.mannwhitneyu(
            apical_near_vs_dist[~np.isnan(apical_near_vs_dist)],
            basal_near_vs_dist[~np.isnan(basal_near_vs_dist)],
        )
        near_dist_norm_t, near_dist_norm_p = stats.mannwhitneyu(
            apical_near_vs_dist_norm[~np.isnan(apical_near_vs_dist_norm)],
            basal_near_vs_dist_norm[~np.isnan(basal_near_vs_dist_norm)],
        )
        frac_coactive_t, frac_coactive_p = stats.mannwhitneyu(
            apical_fraction_coactive[~np.isnan(apical_fraction_coactive)],
            basal_fraction_coactive[~np.isnan(basal_fraction_coactive)],
        )
        frac_part_t, frac_part_p = stats.mannwhitneyu(
            apical_fraction_participating[~np.isnan(apical_fraction_participating)],
            basal_fraction_participating[~np.isnan(basal_fraction_participating)],
        )
        num_t, num_p = stats.mannwhitneyu(
            apical_coactive_num[~np.isnan(apical_coactive_num)],
            basal_coactive_num[~np.isnan(basal_coactive_num)],
        )
        test_title = "Mann-Whitney U"

    # Comparisons against chance
    apical_above, apical_below = t_utils.test_against_chance(
        apical_local_coactivity, apical_shuff_coactivity
    )
    apical_above_norm, apical_below_norm = t_utils.test_against_chance(
        apical_local_coactivity_norm,
        apical_shuff_coactivity_norm,
    )
    basal_above, basal_below = t_utils.test_against_chance(
        basal_local_coactivity,
        basal_shuff_coactivity,
    )
    basal_above_norm, basal_below_norm = t_utils.test_against_chance(
        basal_local_coactivity_norm,
        basal_shuff_coactivity_norm,
    )

    # Organize results
    results_dict = {
        "Comparison": [
            "Local CoA",
            "Local Norm. CoA",
            "Near vs Dist",
            "Near vs Dist Norm",
            "Frac. CoA",
            "Frac Part.",
            "Coactive Num.",
        ],
        "stat": [
            coactivity_t,
            coactivity_norm_t,
            near_dist_t,
            near_dist_norm_t,
            frac_coactive_t,
            frac_part_t,
            num_t,
        ],
        "p-val": [
            coactivity_p,
            coactivity_norm_p,
            near_dist_p,
            near_dist_norm_p,
            frac_coactive_p,
            frac_part_p,
            num_p,
        ],
    }
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.update(results_df[["p-val"]].applymap("{:.4E}".format))

    chance_dict = {
        "Comparision": ["Apical", "Basal", "Apical Norm.", "Basal Norm."],
        "p-val above": [apical_above, basal_above, apical_above_norm, basal_above_norm],
        "p-val below": [apical_below, basal_below, apical_below_norm, basal_below_norm],
    }
    chance_df = pd.DataFrame.from_dict(chance_dict)

    # Diplay stats
    fig2, axes2 = plt.subplot_mosaic(
        """AB""",
        figsize=(5, 4),
    )

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

    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title(f"Chance result")
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
        fname = os.path.join(save_path, "Apical_vs_Basal_Figure_4_stats")
        fig2.savefig(fname + ".pdf")


def plot_dendrite_coactivity(
    apical_dataset,
    basal_dataset,
    figsize=(10, 10),
    showmeans=False,
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function plotting dendritic coactivity variables for apical and basal spines

    INPUT PARAMETERS
        apical_dataset - Dendritic_Coactivity_Data object for apical data

        basal_dataset - Dendritic_Coactivity_Data object for basal data

        figsize - tuple specifying the figure size

        showmeans - boolean specifying whether to show means on the box plots

        test_type - str specifying whehter to perform parametric or nonparametric tests

        display_stats - boolean specifying whether to display the stat results

        save - boolean specifying whether to save the figure or not

        save_path - str specifying where to save the figure

    """
    COLORS = ["goldenrod", "mediumseagreen"]

    # Find the present spines
    apical_present = find_present_spines(apical_dataset.spine_flags)
    basal_present = find_present_spines(basal_dataset.spine_flags)

    # Pull relevant data
    ## Event rate related data
    apical_coactivity_rate = d_utils.subselect_data_by_idxs(
        apical_dataset.all_dendrite_coactivity_rate, apical_present
    )
    basal_coactivity_rate = d_utils.subselect_data_by_idxs(
        basal_dataset.all_dendrite_coactivity_rate, basal_present
    )
    apical_conj_coactivity_rate = d_utils.subselect_data_by_idxs(
        apical_dataset.conj_dendrite_coactivity_rate, apical_present
    )
    basal_conj_coactivity_rate = d_utils.subselect_data_by_idxs(
        basal_dataset.conj_dendrite_coactivity_rate, basal_present
    )
    apical_nonconj_coactivity_rate = d_utils.subselect_data_by_idxs(
        apical_dataset.nonconj_dendrite_coactivity_rate, apical_present
    )
    basal_nonconj_coactivity_rate = d_utils.subselect_data_by_idxs(
        basal_dataset.nonconj_dendrite_coactivity_rate, basal_present
    )
    apical_frac_conj = d_utils.subselect_data_by_idxs(
        apical_dataset.fraction_conj_events, apical_present
    )
    basal_frac_conj = d_utils.subselect_data_by_idxs(
        basal_dataset.fraction_conj_events, basal_present
    )
    ## Event properties
    sampling_rate = apical_dataset.parameters["Sampling Rate"]
    activity_window = apical_dataset.parameters["Activity Window"]
    if apical_dataset.parameters["zscore"]:
        activity_type = "zscore"
    else:
        activity_type = "\u0394F/F"

    apical_glu_traces = d_utils.subselect_data_by_idxs(
        apical_dataset.all_spine_coactive_traces, apical_present
    )
    basal_glu_traces = d_utils.subselect_data_by_idxs(
        basal_dataset.all_spine_coactive_traces, basal_present
    )
    apical_ca_traces = d_utils.subselect_data_by_idxs(
        apical_dataset.all_spine_coactive_calcium_traces, apical_present
    )
    basal_ca_traces = d_utils.subselect_data_by_idxs(
        basal_dataset.all_spine_coactive_calcium_traces, basal_present
    )
    apical_dend_traces = d_utils.subselect_data_by_idxs(
        apical_dataset.all_dendrite_coactive_traces, apical_present
    )
    basal_dend_traces = d_utils.subselect_data_by_idxs(
        basal_dataset.all_dendrite_coactive_traces, basal_present
    )
    apical_glu_amp = d_utils.subselect_data_by_idxs(
        apical_dataset.all_spine_coactive_amplitude, apical_present
    )
    basal_glu_amp = d_utils.subselect_data_by_idxs(
        basal_dataset.all_spine_coactive_amplitude, basal_present
    )
    apical_ca_amp = d_utils.subselect_data_by_idxs(
        apical_dataset.all_spine_coactive_calcium_amplitude, apical_present
    )
    basal_ca_amp = d_utils.subselect_data_by_idxs(
        basal_dataset.all_spine_coactive_calcium_amplitude, basal_present
    )
    apical_dend_amp = d_utils.subselect_data_by_idxs(
        apical_dataset.all_dendrite_coactive_amplitude, apical_present
    )
    basal_dend_amp = d_utils.subselect_data_by_idxs(
        basal_dataset.all_dendrite_coactive_amplitude, basal_present
    )
    apical_onset = d_utils.subselect_data_by_idxs(
        apical_dataset.all_relative_onsets, apical_present
    )
    basal_onset = d_utils.subselect_data_by_idxs(
        basal_dataset.all_relative_onsets, basal_present
    )

    # Process traces for plotting
    a_glu_means = [
        np.nanmean(x, axis=1) for x in apical_glu_traces if type(x) == np.ndarray
    ]
    a_glu_means = np.vstack(a_glu_means)
    apical_glu_means = np.nanmean(a_glu_means, axis=0)
    apical_glu_sems = stats.sem(a_glu_means, axis=0, nan_policy="omit")
    b_glu_means = [
        np.nanmean(x, axis=1) for x in basal_glu_traces if type(x) == np.ndarray
    ]
    b_glu_means = np.vstack(b_glu_means)
    basal_glu_means = np.nanmean(b_glu_means, axis=0)
    basal_glu_sems = stats.sem(b_glu_means, axis=0, nan_policy="omit")
    a_ca_means = [
        np.nanmean(x, axis=1) for x in apical_ca_traces if type(x) == np.ndarray
    ]
    a_ca_means = np.vstack(a_ca_means)
    apical_ca_means = np.nanmean(a_ca_means, axis=0)
    apical_ca_sems = stats.sem(a_ca_means, axis=0, nan_policy="omit")
    b_ca_means = [
        np.nanmean(x, axis=1) for x in basal_ca_traces if type(x) == np.ndarray
    ]
    b_ca_means = np.vstack(b_ca_means)
    basal_ca_means = np.nanmean(b_ca_means, axis=0)
    basal_ca_sems = stats.sem(b_ca_means, axis=0, nan_policy="omit")
    a_dend_means = [
        np.nanmean(x, axis=1) for x in apical_dend_traces if type(x) == np.ndarray
    ]
    a_dend_means = np.vstack(a_dend_means)
    apical_dend_means = np.nanmean(a_dend_means, axis=0)
    apical_dend_sems = stats.sem(a_dend_means, axis=0, nan_policy="omit")
    b_dend_means = [
        np.nanmean(x, axis=1) for x in basal_dend_traces if type(x) == np.ndarray
    ]
    b_dend_means = np.vstack(b_dend_means)
    basal_dend_means = np.nanmean(b_dend_means, axis=0)
    basal_dend_sems = stats.sem(b_dend_means, axis=0, nan_policy="omit")

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFGH
        JKLM
        """,
        figsize=figsize,
    )

    fig.suptitle("Apical vs Basal Dendrite Coactivity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ########################### Plot data onto axes ###########################
    # All coactivity box plot
    plot_box_plot(
        data_dict={"Apical": apical_coactivity_rate, "Basal": basal_coactivity_rate},
        figsize=(5, 5),
        title="All events",
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
    # All conj coactivity box plot
    plot_box_plot(
        data_dict={
            "Apical": apical_conj_coactivity_rate,
            "Basal": basal_conj_coactivity_rate,
        },
        figsize=(5, 5),
        title="With Local Coactivity",
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
    # All nonconj coactivity box plot
    plot_box_plot(
        data_dict={
            "Apical": apical_nonconj_coactivity_rate,
            "Basal": basal_nonconj_coactivity_rate,
        },
        figsize=(5, 5),
        title="Without Local Coactivity",
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
    # Fraction with conj coactivity
    plot_box_plot(
        data_dict={"Apical": apical_frac_conj, "Basal": basal_frac_conj},
        figsize=(5, 5),
        title="Fraction with local",
        xtitle=None,
        ytitle=f"Fraction with local coactivity",
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
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # GluSnFR amplitude
    plot_box_plot(
        data_dict={"Apical": apical_glu_amp, "Basal": basal_glu_amp},
        figsize=(5, 5),
        title="GluSnFR",
        xtitle=None,
        ytitle=f"{activity_type}",
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
    # Calcium amplitude
    plot_box_plot(
        data_dict={"Apical": apical_ca_amp, "Basal": basal_ca_amp},
        figsize=(5, 5),
        title="Calcium",
        xtitle=None,
        ytitle=f"{activity_type}",
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
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    # Dendrite amplitude
    plot_box_plot(
        data_dict={"Apical": apical_dend_amp, "Basal": basal_dend_amp},
        figsize=(5, 5),
        title="Dendrite",
        xtitle=None,
        ytitle=f"{activity_type}",
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
    # Onsets
    plot_box_plot(
        data_dict={"Apical": apical_onset, "Basal": basal_onset},
        figsize=(5, 5),
        title="Onsets",
        xtitle=None,
        ytitle="Realtive onset (s)",
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

    # GluSnFr traces
    plot_mean_activity_traces(
        means=list((apical_glu_means, basal_glu_means)),
        sems=list((apical_glu_sems, basal_glu_sems)),
        group_names=["Apical", "Basal"],
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
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Calcium traces
    plot_mean_activity_traces(
        means=list((apical_ca_means, basal_ca_means)),
        sems=list((apical_ca_sems, basal_ca_sems)),
        group_names=["Apical", "Basal"],
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
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    # Dendrite traces
    plot_mean_activity_traces(
        means=list((apical_dend_means, basal_dend_means)),
        sems=list((apical_dend_sems, basal_dend_sems)),
        group_names=["Apical", "Basal"],
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
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    plot_histogram(
        data=list((apical_onset, basal_onset)),
        bins=30,
        stat="probability",
        avlines=[0],
        title=None,
        xtitle="Relative onset (s)",
        xlim=None,
        figsize=(5, 5),
        color=COLORS,
        alpha=0.4,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Apical_vs_Basal_Figure_5")
        fig.savefig(fname + ".pdf")

    ######################## Statistics Section #############################
    if display_stats == False:
        return

    # Perform tests
    if test_type == "parametric":
        coactivity_t, coactivity_p = stats.ttest_ind(
            apical_coactivity_rate, basal_coactivity_rate, nan_policy="omit"
        )
        conj_t, conj_p = stats.ttest_ind(
            apical_conj_coactivity_rate,
            basal_conj_coactivity_rate,
            nan_policy="omit",
        )
        nonconj_t, nonconj_p = stats.ttest_ind(
            apical_nonconj_coactivity_rate,
            basal_nonconj_coactivity_rate,
            nan_policy="omit",
        )
        frac_t, frac_p = stats.ttest_ind(
            apical_frac_conj, basal_frac_conj, nan_policy="omit"
        )
        glu_t, glu_p = stats.ttest_ind(apical_glu_amp, basal_glu_amp, nan_policy="omit")
        ca_t, ca_p = stats.ttest_ind(apical_ca_amp, basal_ca_amp, nan_policy="omit")
        dend_t, dend_p = stats.ttest_ind(
            apical_dend_amp,
            basal_dend_amp,
            nan_policy="omit",
        )
        onset_t, onset_p = stats.ttest_ind(apical_onset, basal_onset, nan_policy="omit")
        test_title = "T-Test"
    elif test_type == "nonparametric":
        coactivity_t, coactivity_p = stats.mannwhitneyu(
            apical_coactivity_rate[~np.isnan(apical_coactivity_rate)],
            basal_coactivity_rate[~np.isnan(basal_coactivity_rate)],
        )
        conj_t, conj_p = stats.mannwhitneyu(
            apical_conj_coactivity_rate[~np.isnan(apical_conj_coactivity_rate)],
            basal_conj_coactivity_rate[~np.isnan(basal_conj_coactivity_rate)],
        )
        nonconj_t, nonconj_p = stats.mannwhitneyu(
            apical_nonconj_coactivity_rate[~np.isnan(apical_nonconj_coactivity_rate)],
            basal_nonconj_coactivity_rate[~np.isnan(basal_nonconj_coactivity_rate)],
        )
        frac_t, frac_p = stats.mannwhitneyu(
            apical_frac_conj[~np.isnan(apical_frac_conj)],
            basal_frac_conj[~np.isnan(basal_frac_conj)],
        )
        glu_t, glu_p = stats.mannwhitneyu(
            apical_glu_amp[~np.isnan(apical_glu_amp)],
            basal_glu_amp[~np.isnan(basal_glu_amp)],
        )
        ca_t, ca_p = stats.mannwhitneyu(
            apical_ca_amp[~np.isnan(apical_ca_amp)],
            basal_ca_amp[~np.isnan(basal_ca_amp)],
        )
        dend_t, dend_p = stats.mannwhitneyu(
            apical_dend_amp[~np.isnan(apical_dend_amp)],
            basal_dend_amp[~np.isnan(basal_dend_amp)],
        )
        onset_t, onset_p = stats.mannwhitneyu(
            apical_onset[~np.isnan(apical_onset)], basal_onset[~np.isnan(basal_onset)]
        )
        test_title = "Mann-Whitney U"

        # Organize the results
        results_dict = {
            "Comparison": [
                "Coactivity rate",
                "Conj rate",
                "Nonconj rate",
                "Frac Coactive",
                "Glu amp",
                "Ca amp",
                "Dend amp",
                "Onset",
            ],
            "Stat": [
                coactivity_t,
                conj_t,
                nonconj_t,
                frac_t,
                glu_t,
                ca_t,
                dend_t,
                onset_t,
            ],
            "P-Val": [
                coactivity_p,
                conj_p,
                nonconj_p,
                frac_p,
                glu_p,
                ca_p,
                dend_p,
                onset_p,
            ],
        }
        results_df = pd.DataFrame.from_dict(results_dict)

        fig2, axes2 = plt.subplot_mosaic("""A""", figsize=(5, 4))

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
            fname = os.path.join(save_path, "Apical_vs_Basal_Figure_5_Stats")
            fig2.savefig(fname + ".pdf")
