import os
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

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
    calculate_fraction_plastic,
    calculate_spine_dynamics,
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils

sns.set()
sns.set_style("ticks")


def plot_longitudinal_structural_plasticity(
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
                e, s, _ = calculate_fraction_plastic(
                    [
                        temp_data.corrected_spine_volume,
                        temp_data.corrected_followup_volume,
                    ],
                    [
                        temp_data.spine_flags,
                        temp_data.followup_flags,
                    ],
                    threshold=threshold,
                    exclude="Shaft Spine",
                )
                apical_relative_volumes[session].append(np.nanmean(delta[-1]))
                apical_enlarged_spines[session].append(e)
                apical_shrunken_spines[session].append(s)
                apical_enlarged_volumes[session].append(np.nanmean(delta[-1]))
                apical_shrunken_volumes[session].append(np.nanmean(delta[-1]))
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
                e, s, _ = calculate_fraction_plastic(
                    [
                        temp_data.corrected_spine_volume,
                        temp_data.corrected_followup_volume,
                    ],
                    [
                        temp_data.spine_flags,
                        temp_data.followup_flags,
                    ],
                    threshold=threshold,
                    exclude="Shaft Spine",
                )
                basal_relative_volumes[session].append(np.nanmean(delta[-1]))
                basal_enlarged_spines[session].append(e)
                basal_shrunken_spines[session].append(s)
                basal_enlarged_volumes[session].append(np.nanmean(delta[-1]))
                basal_shrunken_volumes[session].append(np.nanmean(delta[-1]))
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
    print("APICAL")
    print(apical_enlarged_spines)
    print("BASAL")
    print(basal_enlarged_spines)
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
        ylim=(0.05, 0.2),
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
        fname = os.path.join(save_path, "Apical_vs_Basal_Longitudinal_Plasticity")
        fig.savefig(fname + ".pdf")

    ######################## Statistics Section #############################
    if display_stats == False:
        return

    print("Need to code in the statistics")


def plot_structural_plasticity(
    apical_dataset,
    basal_dataset,
    figsize=(7, 7),
    threshold=0.3,
    showmeans=True,
    mean_type="mean",
    err_type="sem",
    hist_bins=30,
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to plot spine plasticity of apical and basal dendrites over a single
    session period

    INPUT PARAMETERS
        apical_dataset - Spine_Activity_Data object for apical data

        basal_dataset - Spine_Activity_Data object for basal data

        figsize - tuple specifying the size of the figure to plot

        threshold - float or tuple of floats specifying plasticity thresholds

        showmeans - boolean specifying whether to show means on box plots

        mean_type - str specifying the mean type to use for bar plots

        err_type - str specifying the err type to use for bar plots

        hist_bins - int specifying how many bins for histogram

        test_type - str specifying the type of statistics to perform

        display_stats - boolean specifying whether to perform stats

        save - boolean specifying whether to save the data or not

        save_path - str specifying where to save the figure

    """
    COLORS = ["goldenrod", "mediumseagreen"]
    apical_dendrites = apical_dataset.dendrite_number
    basal_dendrites = basal_dataset.dendrite_number

    # Get spine flags
    apical_flags = apical_dataset.spine_flags
    apical_followup_flags = apical_dataset.followup_flags
    basal_flags = basal_dataset.spine_flags
    basal_followup_flags = basal_dataset.followup_flags

    # Print base numbers
    apical_present = find_present_spines(apical_flags)
    basal_present = find_present_spines(basal_flags)

    print(f"Apical Spines: {np.sum(apical_present)}")
    print(f"Basal Spines: {np.sum(basal_present)}")

    # Get spine volumes
    apical_volumes = apical_dataset.spine_volumes
    apical_followup_volumes = apical_dataset.followup_volumes
    basal_volumes = basal_dataset.spine_volumes
    basal_followup_volumes = basal_dataset.followup_volumes

    # Calculate volume changes
    apical_vol_list = [apical_volumes, apical_followup_volumes]
    apical_flag_list = [apical_flags, apical_followup_flags]
    basal_vol_list = [basal_volumes, basal_followup_volumes]
    basal_flag_list = [basal_flags, basal_followup_flags]

    apical_delta_volume, apical_idx = calculate_volume_change(
        apical_vol_list,
        apical_flag_list,
        norm=False,
        exclude="Shaft Spine",
    )
    apical_delta_volume = apical_delta_volume[-1]

    basal_delta_volume, basal_idx = calculate_volume_change(
        basal_vol_list, basal_flag_list, norm=False, exclude="Shaft Spine"
    )
    basal_delta_volume = basal_delta_volume[-1]

    # Organize plasticity per dendrite
    ## Get uniuque and dendrites corresponding to plastic spines
    unique_apical_dendrites = np.unique(apical_dendrites)
    unique_basal_dendrites = np.unique(basal_dendrites)

    # Set up plasticity variables
    apical_frac_LTP = []
    apical_frac_LTD = []
    apical_frac_stable = []
    basal_frac_LTP = []
    basal_frac_LTD = []
    basal_frac_stable = []

    for i, a_dend in enumerate(unique_apical_dendrites):
        ## Deal with plasticity first
        if np.isnan(a_dend):
            continue
        dend_idxs = np.nonzero(apical_dendrites == a_dend)[0]
        temp_flag_list = [
            d_utils.subselect_data_by_idxs(apical_flags, dend_idxs),
            d_utils.subselect_data_by_idxs(apical_followup_flags, dend_idxs),
        ]
        temp_vol_list = [
            d_utils.subselect_data_by_idxs(apical_volumes, dend_idxs),
            d_utils.subselect_data_by_idxs(apical_followup_volumes, dend_idxs),
        ]
        a_ltp, a_ltd, a_stable = calculate_fraction_plastic(
            temp_vol_list, temp_flag_list, threshold=threshold, exclude="Shaft Spine"
        )

        apical_frac_LTP.append(a_ltp)
        apical_frac_LTD.append(a_ltd)
        apical_frac_stable.append(a_stable)

    for b_dend in unique_basal_dendrites:
        ## Deal with plasticity first
        if np.isnan(b_dend):
            continue
        dend_idxs = np.nonzero(basal_dendrites == b_dend)[0]
        temp_flag_list = [
            d_utils.subselect_data_by_idxs(basal_flags, dend_idxs),
            d_utils.subselect_data_by_idxs(basal_followup_flags, dend_idxs),
        ]
        temp_vol_list = [
            d_utils.subselect_data_by_idxs(basal_volumes, dend_idxs),
            d_utils.subselect_data_by_idxs(basal_followup_volumes, dend_idxs),
        ]
        b_ltp, b_ltd, b_stable = calculate_fraction_plastic(
            temp_vol_list, temp_flag_list, threshold=threshold, exclude="Shaft Spine"
        )

        basal_frac_LTP.append(b_ltp)
        basal_frac_LTD.append(b_ltd)
        basal_frac_stable.append(b_stable)

    apical_frac_LTP = np.array(apical_frac_LTP)
    apical_frac_LTD = np.array(apical_frac_LTD)
    basal_frac_LTP = np.array(basal_frac_LTP)
    basal_frac_LTD = np.array(basal_frac_LTD)

    apical_frac_stable = np.array(apical_frac_stable)
    basal_frac_stable = np.array(basal_frac_stable)

    print(f"Apical LTP: {np.nanmean(apical_frac_LTP)}")
    print(f"Apical LTD: {np.nanmean(apical_frac_LTD)}")
    print(f"Apical Stable: {np.nanmean(apical_frac_stable)}")
    print(f"Basal LTP: {np.nanmean(basal_frac_LTP)}")
    print(f"Basal LTD: {np.nanmean(basal_frac_LTD)}")
    print(f"Basal Stable: {np.nanmean(basal_frac_stable)}")

    # Calculate spine dynamics
    apical_mice = list(set(apical_dataset.mouse_id))
    basal_mice = list(set(basal_dataset.mouse_id))

    apical_density = []
    apical_new_spines = []
    apical_elim_spines = []
    basal_density = []
    basal_new_spines = []
    basal_elim_spines = []
    apical_max_positions = []
    basal_max_positions = []

    for mouse in apical_mice:
        temp_data = load_spine_datasets(mouse, ["Early"], fov_type="apical")
        for FOV, dataset in temp_data.items():
            t_data = dataset["Early"]
            temp_groupings = [t_data.spine_groupings, t_data.followup_groupings]
            temp_flags = [t_data.spine_flags, t_data.followup_flags]
            temp_positions = [t_data.spine_positions, t_data.followup_positions]
            apical_max_positions.append(np.nanmax(t_data.spine_positions))
            temp_density, temp_new, temp_elim = calculate_spine_dynamics(
                temp_flags,
                temp_positions,
                temp_groupings,
            )
            apical_density.append(temp_density[0])
            apical_new_spines.append(temp_new[-1])
            apical_elim_spines.append(temp_elim[-1])
    for mouse in basal_mice:
        temp_data = load_spine_datasets(mouse, ["Early"], fov_type="basal")
        for FOV, dataset in temp_data.items():
            t_data = dataset["Early"]
            temp_groupings = [t_data.spine_groupings, t_data.followup_groupings]
            temp_flags = [t_data.spine_flags, t_data.followup_flags]
            temp_positions = [t_data.spine_positions, t_data.followup_positions]
            basal_max_positions.append(np.nanmax(t_data.spine_positions))
            temp_density, temp_new, temp_elim = calculate_spine_dynamics(
                temp_flags,
                temp_positions,
                temp_groupings,
            )
            basal_density.append(temp_density[0])
            basal_new_spines.append(temp_new[-1])
            basal_elim_spines.append(temp_elim[-1])

    apical_density = np.concatenate(apical_density)
    apical_new_spines = np.concatenate(apical_new_spines)
    apical_elim_spines = np.concatenate(apical_elim_spines)
    basal_density = np.concatenate(basal_density)
    basal_new_spines = np.concatenate(basal_new_spines)
    basal_elim_spines = np.concatenate(basal_elim_spines)

    print(f"Apical dend. n: {len(apical_density)}")
    print(f"Basal dend n: {len(basal_density)}")

    apical_dict = {
        "Density": apical_density,
        "sLTP_fraction": apical_frac_LTP,
        "sLTD_fraction": apical_frac_LTD,
    }
    apical_df = pd.DataFrame.from_dict(apical_dict)
    basal_dict = {
        "Density": basal_density,
        "sLTP_fraction": basal_frac_LTP,
        "sLTD_fraction": basal_frac_LTD,
    }
    basal_df = pd.DataFrame.from_dict(basal_dict)

    apical_df.to_csv("apical_structural_properties.csv", index=False)
    basal_df.to_csv("basal_structural_properties.csv", index=False)

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABCDE
        FG...
        """,
        figsize=figsize,
        width_ratios=[1, 1, 1, 2, 2],
    )
    fig.suptitle(f"Apical vs Basal Structural Plasticity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    # dynamics_df = pd.DataFrame(
    #    {
    #        "Density": apical_density,
    #        "New spines": apical_new_spines,
    #        "Elim spines": apical_elim_spines,
    #    }
    # )
    # volume_df = pd.DataFrame(
    #    {
    #        "Volume": d_utils.subselect_data_by_idxs(apical_volumes, apical_idx),
    #        "Delta Volume": apical_delta_volume,
    #    }
    # )
    # dynamics_df.to_csv("dynamics.csv", index=False)
    # volume_df.to_csv("volume.csv", index=False)

    ########################## Plot data onto axes ############################
    # Spine density
    plot_swarm_bar_plot(
        data_dict={
            "Apical": apical_density,
            "Basal": basal_density,
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Spine density (spines/um)",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    # Fraction of new spines
    plot_swarm_bar_plot(
        data_dict={
            "Apical": apical_new_spines,
            "Basal": basal_new_spines,
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction new spines",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    # Fraction of elim spines
    plot_swarm_bar_plot(
        data_dict={
            "Apical": apical_elim_spines,
            "Basal": basal_elim_spines,
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction eliminated spines",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # Fraction of LTP spines
    plot_swarm_bar_plot(
        data_dict={
            "Apical": apical_frac_LTP,
            "Basal": basal_frac_LTP,
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction sLTP spines",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    # Fraction of LTD spines
    plot_swarm_bar_plot(
        data_dict={
            "Apical": apical_frac_LTD,
            "Basal": basal_frac_LTD,
        },
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction sLTD spines",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=COLORS,
        s_size=5,
        s_alpha=1,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )

    # Initial spine volume
    plot_histogram(
        data=list(
            (
                d_utils.subselect_data_by_idxs(apical_volumes, apical_present),
                d_utils.subselect_data_by_idxs(basal_volumes, basal_present),
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Initial Spine Volume",
        xtitle="Spine Volume (um)",
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
    ax_D_inset = axes["D"].inset_axes([0.65, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_D_inset)
    plot_box_plot(
        data_dict={
            "Apical": d_utils.subselect_data_by_idxs(apical_volumes, apical_present),
            "Basal": d_utils.subselect_data_by_idxs(basal_volumes, basal_present),
        },
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
    # delta spine volume
    plot_histogram(
        data=list(
            (
                apical_delta_volume,
                basal_delta_volume,
            )
        ),
        bins=hist_bins,
        stat="probability",
        avlines=None,
        title="Volume Change",
        xtitle="Volume Change",
        xlim=(0, None),
        figsize=(5, 5),
        color=COLORS,
        alpha=0.4,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Inset box plot
    ax_E_inset = axes["E"].inset_axes([0.65, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_E_inset)
    plot_box_plot(
        data_dict={
            "Apical": apical_delta_volume,
            "Basal": basal_delta_volume,
        },
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Volume Change",
        ylim=(0, 3),
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
        ax=ax_E_inset,
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Apical_vs_Basal_Structural_Plasticity")
        fig.savefig(fname + ".pdf")

    ############################# Statistics Section ###############################
    if display_stats == False:
        return
    p_apical_vols = d_utils.subselect_data_by_idxs(apical_volumes, apical_present)
    p_basal_vols = d_utils.subselect_data_by_idxs(basal_volumes, basal_present)
    # Peform the statistics
    if test_type == "parametric":
        density_t, density_p = stats.ttest_ind(
            apical_density, basal_density, nan_policy="omit"
        )
        new_t, new_p = stats.ttest_ind(
            apical_new_spines, basal_new_spines, nan_policy="omit"
        )
        elim_t, elim_p = stats.ttest_ind(
            apical_elim_spines, basal_elim_spines, nan_policy="omit"
        )
        ltp_t, ltp_p = stats.ttest_ind(
            apical_frac_LTP, basal_frac_LTP, nan_policy="omit"
        )
        ltd_t, ltd_p = stats.ttest_ind(
            apical_frac_LTD, basal_frac_LTD, nan_policy="omit"
        )
        vol_t, vol_p = stats.ttest_ind(
            p_apical_vols,
            p_basal_vols,
            nan_policy="omit",
        )
        rel_t, rel_p = stats.ttest_ind(
            apical_delta_volume, basal_delta_volume, nan_policy="omit"
        )
        test_title = "T-Test"
    elif test_type == "nonparametric":
        density_t, density_p = stats.mannwhitneyu(
            apical_density[~np.isnan(apical_density)],
            basal_density[~np.isnan(basal_density)],
        )
        new_t, new_p = stats.mannwhitneyu(
            apical_new_spines[~np.isnan(apical_new_spines)],
            basal_new_spines[~np.isnan(basal_new_spines)],
        )
        elim_t, elim_p = stats.mannwhitneyu(
            apical_elim_spines[~np.isnan(apical_elim_spines)],
            basal_elim_spines[~np.isnan(basal_elim_spines)],
        )
        ltp_t, ltp_p = stats.mannwhitneyu(
            apical_frac_LTP[~np.isnan(apical_frac_LTP)],
            basal_frac_LTP[~np.isnan(basal_frac_LTP)],
        )
        ltd_t, ltd_p = stats.mannwhitneyu(
            apical_frac_LTD[~np.isnan(apical_frac_LTD)],
            basal_frac_LTD[~np.isnan(basal_frac_LTD)],
        )
        vol_t, vol_p = stats.mannwhitneyu(
            p_apical_vols[~np.isnan(p_apical_vols)],
            p_basal_vols[~np.isnan(p_basal_vols)],
        )
        rel_t, rel_p = stats.mannwhitneyu(
            apical_delta_volume[~np.isnan(apical_delta_volume)],
            basal_delta_volume[~np.isnan(basal_delta_volume)],
        )
        test_title = "Mann-Whitney U"

    # Organize the results
    results_dict = {
        "Comparison": [
            "Density",
            "New Spines",
            "Elim. Spines",
            "Fraction LTP",
            "Fraction LTD",
            "Initial Vol.",
            "Relative Vol.",
        ],
        "stat": [
            density_t,
            new_t,
            elim_t,
            ltp_t,
            ltd_t,
            vol_t,
            rel_t,
        ],
        "p-val": [
            density_p,
            new_p,
            elim_p,
            ltp_p,
            ltd_p,
            vol_p,
            rel_p,
        ],
    }

    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.update(results_df[["stat"]].applymap("{:.4}".format))
    results_df.update(results_df[["p-val"]].applymap("{:.4E}".format))

    fig2, axes2 = plt.subplot_mosaic("""A""", figsize=(6, 6))

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
        fname = os.path.join(save_path, "Apical_vs_Basal_Structural_Plasticity_Stats")
        fig2.savefig(fname + ".pdf")


def plot_movement_related_activity(
    apical_dataset,
    basal_dataset,
    MRSs=None,
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
    # apical_ids = np.array(d_utils.code_str_to_int(apical_dataset.mouse_id))
    # basal_ids = np.array(d_utils.code_str_to_int(basal_dataset.mouse_id))
    all_ids = apical_dataset.mouse_id + basal_dataset.mouse_id
    all_ids_coded = np.array(d_utils.code_str_to_int(all_ids))
    apical_ids = all_ids_coded[: len(apical_dataset.mouse_id)]
    basal_ids = all_ids_coded[len(apical_dataset.mouse_id) :]
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

    if MRSs == "MRS":
        apical_present = apical_present * apical_dataset.movement_spines
        basal_present = basal_present * basal_dataset.movement_spines
    elif MRSs == "nMRS":
        apical_present = apical_present * apical_dataset.nonmovement_spines
        basal_present = basal_present * basal_dataset.nonmovement_spines

    ## IDs
    apical_ids = d_utils.subselect_data_by_idxs(apical_ids, apical_present)
    basal_ids = d_utils.subselect_data_by_idxs(basal_ids, basal_present)
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

    ## Output some data as CVS for collaboration
    event_rate_dict = {
        "Apical": apical_event_rate,
        "Basal": basal_event_rate,
    }
    event_rate_df = pd.DataFrame(
        dict([(key, pd.Series(value)) for key, value in event_rate_dict.items()])
    )

    event_rate_df.to_csv("spine_event_rate.csv", index=False)

    apical_dend_rate = d_utils.subselect_data_by_idxs(
        apical_dataset.dendrite_activity_rate,
        apical_present,
    )
    apical_dend_rate_temp, a_ind = np.unique(apical_dend_rate, return_index=True)
    apical_dend_rate = apical_dend_rate_temp[np.argsort(a_ind)]

    basal_dend_rate = d_utils.subselect_data_by_idxs(
        basal_dataset.dendrite_activity_rate, basal_present
    )
    basal_dend_rate_temp, b_ind = np.unique(basal_dend_rate, return_index=True)
    basal_dend_rate = basal_dend_rate_temp[np.argsort(b_ind)]

    dend_rate_dict = {
        "Apical": apical_dend_rate,
        "Basal": basal_dend_rate,
    }
    dend_rate_df = pd.DataFrame(
        dict([(key, pd.Series(value)) for key, value in dend_rate_dict.items()])
    )
    dend_rate_df.to_csv("dendrite_event_rate.csv", index=False)

    apical_trace_spines = apical_MRSs
    basal_trace_spines = basal_MRSs
    if MRSs == "nMRS":
        apical_trace_spines = apical_nMRSs
        basal_trace_spines = basal_nMRSs

    # Get mean and sem traces for plotting
    ## Apical GluSnFR
    a_glu_traces = list(compress(apical_glu_traces, apical_trace_spines))
    a_glu_means = [np.nanmean(x, axis=1) for x in a_glu_traces if type(x) == np.ndarray]
    a_glu_means = np.vstack(a_glu_means)
    apical_glu_hmap = a_glu_means.T
    apical_grouped_glu_traces = np.nanmean(a_glu_means, axis=0)
    apical_grouped_glu_sems = stats.sem(a_glu_means, axis=0, nan_policy="omit")
    ## Basal GluSnFR
    b_glu_traces = list(compress(basal_glu_traces, basal_trace_spines))
    b_glu_means = [np.nanmean(x, axis=1) for x in b_glu_traces if type(x) == np.ndarray]
    b_glu_means = np.vstack(b_glu_means)
    basal_glu_hmap = b_glu_means.T
    basal_grouped_glu_traces = np.nanmean(b_glu_means, axis=0)
    basal_grouped_glu_sems = stats.sem(b_glu_means, axis=0, nan_policy="omit")
    ## Apical Calcium
    a_ca_traces = list(compress(apical_ca_traces, apical_trace_spines))
    a_ca_means = [np.nanmean(x, axis=1) for x in a_ca_traces if type(x) == np.ndarray]
    a_ca_means = np.vstack(a_ca_means)
    apical_ca_hmap = a_ca_means.T
    apical_grouped_ca_traces = np.nanmean(a_ca_means, axis=0)
    apical_grouped_ca_sems = stats.sem(a_ca_means, axis=0, nan_policy="omit")
    ## Basal GluSnFR
    b_ca_traces = list(compress(basal_ca_traces, basal_trace_spines))
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
        JKLM.
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
        hmap_range=(0.0, 0.8),
        center=None,
        vline=121,
        sorted="onset",
        onset_color="white",
        normalize=True,
        cmap="plasma",
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
        hmap_range=(0.0, 0.8),
        center=None,
        vline=121,
        sorted="onset",
        onset_color="white",
        normalize=True,
        cmap="plasma",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Apical Calcium heatmap
    plot_activity_heatmap(
        apical_ca_hmap,
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Apical CA",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="YlOrBr",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["L"],
        save=False,
        save_path=None,
    )
    ## Basal Calcium heatmap
    plot_activity_heatmap(
        basal_ca_hmap,
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Basal Ca",
        cbar_label=activity_type,
        hmap_range=(0, 1),
        center=None,
        sorted="peak",
        normalize=True,
        cmap="Greens",
        axis_width=2,
        minor_ticks="x",
        tick_len=3,
        ax=axes["M"],
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
        if MRSs:
            mname = f"_{MRSs}"
        else:
            mname = ""
        fname = os.path.join(save_path, f"Apical_vs_Basal_Mvmt_Activity{mname}")
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
    elif test_type == "mixed-effect":
        rate_t, rate_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_event_rate,
                "Basal": basal_event_rate,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
            slopes_intercept=False,
        )
        amp_t, amp_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_glu_amp,
                "Basal": basal_glu_amp,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
            slopes_intercept=False,
        )
        onset_t, onset_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_onset,
                "Basal": basal_onset,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
            slopes_intercept=False,
        )
        ca_amp_t, ca_amp_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_ca_amp,
                "Basal": basal_ca_amp,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
            slopes_intercept=False,
        )
        test_title = "Mixed-Effects"

    elif test_type == "ART":
        rate_t, rate_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_event_rate,
                "Basal": basal_event_rate,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
        )
        amp_t, amp_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_glu_amp,
                "Basal": basal_glu_amp,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
        )
        onset_t, onset_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_onset,
                "Basal": basal_onset,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
        )
        ca_amp_t, ca_amp_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_ca_amp,
                "Basal": basal_ca_amp,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
        )
        test_title = "Aligned Rank Transform"

    # Perform the fishers exact

    contingency_table = np.array(
        [
            [np.nansum(apical_MRSs), np.nansum(basal_MRSs)],
            [np.nansum(apical_nMRSs), np.nansum(basal_nMRSs)],
        ]
    )
    MRS_ratio, MRS_p = stats.fisher_exact(contingency_table)

    print(f"Apical total N: {len(apical_event_rate[~np.isnan(apical_event_rate)])}")
    print(f"Apical MRS n: {np.nansum(apical_MRSs)}")
    print(f"Basal total N: {len(basal_event_rate[~np.isnan(basal_event_rate)])}")
    print(f"Basal MRS n: {np.nansum(basal_MRSs)}")

    # Organize the results
    results_dict = {
        "test": ["Event Rate", "GluSnFR Amp", "Onset", "Calcium Amp", "Frac MRS"],
        "stat": [rate_t, amp_t, onset_t, ca_amp_t, MRS_ratio],
        "p-val": [rate_p, amp_p, onset_p, ca_amp_p, MRS_p],
    }
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.update(results_df[["stat"]].applymap("{:.4}".format))
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
        if MRSs:
            mname = f"_{MRSs}"
        else:
            mname = ""
        fname = os.path.join(save_path, f"Apical_vs_Basal_Mvmt_Activity{mname}_stats")
        fig2.savefig(fname + ".pdf")


def plot_spine_movement_encoding(
    apical_dataset,
    basal_dataset,
    MRSs=None,
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

    # apical_ids = np.array(d_utils.code_str_to_int(apical_dataset.mouse_id))
    # basal_ids = np.array(d_utils.code_str_to_int(basal_dataset.mouse_id))

    all_ids = apical_dataset.mouse_id + basal_dataset.mouse_id
    all_ids_coded = np.array(d_utils.code_str_to_int(all_ids))
    apical_ids = all_ids_coded[: len(apical_dataset.mouse_id)]
    basal_ids = all_ids_coded[len(apical_dataset.mouse_id) :]

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

    if MRSs == "MRS":
        apical_present = apical_present * apical_dataset.movement_spines
        basal_present = basal_present * basal_dataset.movement_spines
    elif MRSs == "nMRS":
        apical_present = apical_present * apical_dataset.nonmovement_spines
        basal_present = basal_present * basal_dataset.nonmovement_spines

    # Subselect only the present spines
    apical_ids = d_utils.subselect_data_by_idxs(apical_ids, apical_present)
    basal_ids = d_utils.subselect_data_by_idxs(basal_ids, basal_present)
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
        if MRSs:
            mname = f"_{MRSs}"
        else:
            mname = ""
        fname = os.path.join(save_path, f"Apical_vs_Basal_Mvmt_Encoding{mname}")
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
    elif test_type == "mixed-effect":
        LMP_t, LMP_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_LMP,
                "Basal": basal_LMP,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
            slopes_intercept=False,
        )
        stereo_t, stereo_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_sterotypy,
                "Basal": basal_sterotypy,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
            slopes_intercept=False,
        )
        reli_t, reli_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_reliability,
                "Basal": basal_reliability,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
            slopes_intercept=False,
        )
        speci_t, speci_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_specificity,
                "Basal": basal_specificity,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
            slopes_intercept=False,
        )
        rwd_t, rwd_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_fraction_rwd_mvmt,
                "Basal": basal_fraction_rwd_mvmt,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
            slopes_intercept=False,
        )
        test_title = "Mixed-Effects"

    elif test_type == "ART":
        LMP_t, LMP_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_LMP,
                "Basal": basal_LMP,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
        )
        stereo_t, stereo_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_sterotypy,
                "Basal": basal_sterotypy,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
        )
        reli_t, reli_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_reliability,
                "Basal": basal_reliability,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
        )
        speci_t, speci_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_specificity,
                "Basal": basal_specificity,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
        )
        rwd_t, rwd_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_fraction_rwd_mvmt,
                "Basal": basal_fraction_rwd_mvmt,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method=test_method,
        )
        test_title = "Aligned Rank Transform"

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
        if MRSs:
            mname = f"_{MRSs}"
        else:
            mname = ""
        fname = os.path.join(save_path, f"Apical_vs_Basal_Mvmt_Encoding{mname}_stats")
        fig2.savefig(fname + ".pdf")


def plot_local_coactivity(
    apical_dataset,
    basal_dataset,
    partners=None,
    figsize=(10, 8),
    showmeans=False,
    mean_type="mean",
    test_type="nonparametric",
    test_method="fdr_bh",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to plot local coactivity properties of apical and basal spines

    INPUT PARAMETERS
        apical_dataset - Local_Coactivity_Data object for apical data

        basal_dataset - Local_Coactivity_Data object for basal data

        MRSs - str specifying if you wish to examine only MRSs or nMRSs. Accepts
                "MRS" or "nMRS". Default is None to examine all spines

        figsize - tuple specifying the size of the figure

        showmeans - boolean specifying whetehr to show means on box plots

        mean_type - str specifying the type of mean for the line plots

        err_type - str specifying the time of err for the line plots

        hist_bins - int specifying the number of histogram bins

        test_type - str specifying the type of test to perform

        display_stats - boolean specifying whetehr to perform statistics

        save - boolean specifying whether to save the figure

        save_path - str specifying where to save the figure

    """
    COLORS = ["goldenrod", "mediumseagreen"]
    all_ids = apical_dataset.mouse_id + basal_dataset.mouse_id
    all_ids_coded = np.array(d_utils.code_str_to_int(all_ids))
    a_ids = all_ids_coded[: len(apical_dataset.mouse_id)]
    b_ids = all_ids_coded[len(apical_dataset.mouse_id) :]
    # a_ids = np.array(d_utils.code_str_to_int(apical_dataset.mouse_id))
    # b_ids = np.array(d_utils.code_str_to_int(basal_dataset.mouse_id))

    # Find the present spines
    apical_present = find_present_spines(apical_dataset.spine_flags)
    basal_present = find_present_spines(basal_dataset.spine_flags)
    distance_bins = apical_dataset.parameters["position bins"][1:]

    # Pull relevant data
    a_distance_coactivity = apical_dataset.distance_coactivity_rate
    b_distance_coactivity = basal_dataset.distance_coactivity_rate
    a_distance_coactivity_norm = apical_dataset.distance_coactivity_rate_norm
    b_distance_coactivity_norm = basal_dataset.distance_coactivity_rate_norm
    apical_local_coactivity = apical_dataset.avg_local_coactivity_rate
    basal_local_coactivity = basal_dataset.avg_local_coactivity_rate
    apical_local_coactivity_norm = apical_dataset.avg_local_coactivity_rate_norm
    basal_local_coactivity_norm = basal_dataset.avg_local_coactivity_rate_norm
    apical_coactive_spines = apical_dataset.coactive_spines
    basal_coactive_spines = basal_dataset.coactive_spines
    apical_coactive_spines_norm = apical_dataset.coactive_spines_norm
    basal_coactive_spines_norm = basal_dataset.coactive_spines_norm
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
    apical_mvmt_spines = apical_dataset.movement_spines
    basal_mvmt_spines = basal_dataset.movement_spines
    apical_nonmvmt_spines = apical_dataset.nonmovement_spines
    basal_nonmvmt_spines = basal_dataset.nonmovement_spines

    apical_MRS = apical_present * apical_mvmt_spines
    basal_MRS = basal_present * basal_mvmt_spines

    apical_nMRS = apical_present * apical_nonmvmt_spines
    basal_nMRS = basal_present * basal_nonmvmt_spines

    # Subselect for present spines
    apical_ids = d_utils.subselect_data_by_idxs(a_ids, apical_present)
    basal_ids = d_utils.subselect_data_by_idxs(
        b_ids,
        basal_present,
    )
    apical_MRS_ids = d_utils.subselect_data_by_idxs(a_ids, apical_MRS)
    basal_MRS_ids = d_utils.subselect_data_by_idxs(
        b_ids,
        basal_MRS,
    )
    apical_nMRS_ids = d_utils.subselect_data_by_idxs(a_ids, apical_nMRS)
    basal_nMRS_ids = d_utils.subselect_data_by_idxs(
        b_ids,
        basal_nMRS,
    )

    apical_distance_coactivity = d_utils.subselect_data_by_idxs(
        a_distance_coactivity, apical_present
    )
    basal_distance_coactivity = d_utils.subselect_data_by_idxs(
        b_distance_coactivity, basal_present
    )
    apical_MRS_distance_coactivity = d_utils.subselect_data_by_idxs(
        a_distance_coactivity, apical_MRS
    )
    basal_MRS_distance_coactivity = d_utils.subselect_data_by_idxs(
        b_distance_coactivity, basal_MRS
    )
    apical_nMRS_distance_coactivity = d_utils.subselect_data_by_idxs(
        a_distance_coactivity, apical_nMRS
    )
    basal_nMRS_distance_coactivity = d_utils.subselect_data_by_idxs(
        b_distance_coactivity, basal_nMRS
    )
    apical_distance_coactivity_norm = d_utils.subselect_data_by_idxs(
        a_distance_coactivity_norm, apical_present
    )
    basal_distance_coactivity_norm = d_utils.subselect_data_by_idxs(
        b_distance_coactivity_norm, basal_present
    )
    apical_MRS_distance_coactivity_norm = d_utils.subselect_data_by_idxs(
        a_distance_coactivity_norm, apical_MRS
    )
    basal_MRS_distance_coactivity_norm = d_utils.subselect_data_by_idxs(
        b_distance_coactivity_norm, basal_MRS
    )
    apical_nMRS_distance_coactivity_norm = d_utils.subselect_data_by_idxs(
        a_distance_coactivity_norm, apical_nMRS
    )
    basal_nMRS_distance_coactivity_norm = d_utils.subselect_data_by_idxs(
        b_distance_coactivity_norm, basal_nMRS
    )
    apical_local_coactivity = d_utils.subselect_data_by_idxs(
        apical_local_coactivity, apical_present
    )
    basal_local_coactivity = d_utils.subselect_data_by_idxs(
        basal_local_coactivity, basal_present
    )
    apical_local_coactivity_norm = d_utils.subselect_data_by_idxs(
        apical_local_coactivity_norm, apical_present
    )
    basal_local_coactivity_norm = d_utils.subselect_data_by_idxs(
        basal_local_coactivity_norm, basal_present
    )
    apical_coactive_spines = d_utils.subselect_data_by_idxs(
        apical_coactive_spines, apical_present
    )
    basal_coactive_spines = d_utils.subselect_data_by_idxs(
        basal_coactive_spines, basal_present
    )
    apical_coactive_spines_norm = d_utils.subselect_data_by_idxs(
        apical_coactive_spines_norm, apical_present
    )
    basal_coactive_spines_norm = d_utils.subselect_data_by_idxs(
        basal_coactive_spines_norm, basal_present
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

    # Construct the figure
    fig, axes = plt.subplot_mosaic(
        """
        ABaCDE
        FGfHIJ
        ..KLM.
        """,
        figsize=figsize,
        width_ratios=[2, 2, 1, 1, 1, 1],
    )
    fig.suptitle("Apical vs Basal Local Coactivity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    print(f"All apical n: {apical_distance_coactivity.shape[1]}")
    print(f"Apical MRS n: {apical_MRS_distance_coactivity.shape[1]}")
    print(f"Apical nMRS n: {apical_nMRS_distance_coactivity.shape[1]}")
    print(f"All basal n: {basal_distance_coactivity.shape[1]}")
    print(f"Basal MRS n: {basal_MRS_distance_coactivity.shape[1]}")
    print(f"Basal nMRS n: {basal_nMRS_distance_coactivity.shape[1]}")

    #################### Plot data onto axes #######################
    # Distance coactivity rate
    plot_bins = distance_bins - 2.5
    plot_multi_line_plot(
        data_dict={
            "Apical": apical_distance_coactivity,
            "Basal": basal_distance_coactivity,
        },
        x_vals=plot_bins,
        x_labels=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        mean_type=mean_type,
        title="Raw Coactivity",
        ytitle="Coactivity rate (events/min)",
        xtitle="Distance (\u03BCm)",
        ylim=(0.2, 0.8),
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
        mean_type=mean_type,
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

    # Distance real vs shuff
    plot_multi_line_plot(
        data_dict={
            "Apical MRS": apical_MRS_distance_coactivity,
            "Basal MRS": basal_MRS_distance_coactivity,
        },
        x_vals=plot_bins,
        x_labels=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        mean_type=mean_type,
        title="Raw MRS vs nMRS",
        ytitle="Coactivity rate (events/min)",
        xtitle="Distance (\u03BCm)",
        ylim=(0.3, 1.4),
        line_color=COLORS,
        face_color="white",
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["B"],
        legend=True,
        save=False,
        save_path=None,
    )
    plot_multi_line_plot(
        data_dict={
            "Apical nMRS": apical_nMRS_distance_coactivity,
            "Basal nMRS": basal_nMRS_distance_coactivity,
        },
        x_vals=plot_bins,
        x_labels=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        mean_type=mean_type,
        title="Raw MRS vs nMRS",
        ytitle="Coactivity rate (events/min)",
        xtitle="Distance (\u03BCm)",
        ylim=(0.2, 1.0),
        line_color=COLORS,
        face_color=COLORS,
        m_size=6,
        linewidth=1.5,
        linestyle="-",
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["B"],
        legend=True,
        save=False,
        save_path=None,
    )
    # Distance real vs shuff norm
    plot_multi_line_plot(
        data_dict={
            "Apical MRS": apical_MRS_distance_coactivity_norm,
            "Basal MRS": basal_MRS_distance_coactivity_norm,
        },
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        mean_type=mean_type,
        title="Norm. MRS vs nMRS",
        ytitle="Norm. coactivity rate",
        xtitle="Distance (\u03BCm)",
        ylim=(0, None),
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
    plot_multi_line_plot(
        data_dict={
            "Apical nMRS": apical_nMRS_distance_coactivity_norm,
            "Basal nMRS": basal_nMRS_distance_coactivity_norm,
        },
        x_vals=distance_bins,
        plot_ind=False,
        figsize=(5, 5),
        mean_type=mean_type,
        title="Norm. MRS vs nMRS",
        ytitle="Norm. coactivity rate",
        xtitle="Distance (\u03BCm)",
        ylim=(0, None),
        line_color=COLORS,
        face_color=COLORS,
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
    # Fraction Apical coactive
    plot_pie_chart(
        data_dict={
            "Coactive": np.nansum(apical_coactive_spines),
            "Noncoactive": np.nansum(~apical_coactive_spines),
        },
        title="Apical",
        figsize=(5, 5),
        colors=[COLORS[0], "silver"],
        alpha=0.9,
        edgecolor="white",
        txt_color="white",
        txt_size=8,
        legend=True,
        donut=0.6,
        linewidth=1.5,
        ax=axes["C"],
        save=False,
        save_path=None,
    )  # Fraction Basal coactive
    plot_pie_chart(
        data_dict={
            "Coactive": np.nansum(basal_coactive_spines),
            "Noncoactive": np.nansum(~basal_coactive_spines),
        },
        title="Basal",
        figsize=(5, 5),
        colors=[COLORS[1], "silver"],
        alpha=0.9,
        edgecolor="white",
        txt_color="white",
        txt_size=8,
        legend=True,
        donut=0.6,
        linewidth=1.5,
        ax=axes["D"],
        save=False,
        save_path=None,
    )
    # Fraction Apical coactive norm.
    plot_pie_chart(
        data_dict={
            "Coactive": np.nansum(apical_coactive_spines_norm),
            "Noncoactive": np.nansum(~apical_coactive_spines_norm),
        },
        title="Apical",
        figsize=(5, 5),
        colors=[COLORS[0], "silver"],
        alpha=0.9,
        edgecolor="white",
        txt_color="white",
        txt_size=8,
        legend=True,
        donut=0.6,
        linewidth=1.5,
        ax=axes["H"],
        save=False,
        save_path=None,
    )  # Fraction Basal coactive norm.
    plot_pie_chart(
        data_dict={
            "Coactive": np.nansum(basal_coactive_spines_norm),
            "Noncoactive": np.nansum(~basal_coactive_spines_norm),
        },
        title="Basal",
        figsize=(5, 5),
        colors=[COLORS[1], "silver"],
        alpha=0.9,
        edgecolor="white",
        txt_color="white",
        txt_size=8,
        legend=True,
        donut=0.6,
        linewidth=1.5,
        ax=axes["I"],
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
        ax=axes["a"],
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
        ax=axes["f"],
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
        ytitle=f"Relative coactivity",
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
        ytitle=f"Relative norm. coactivity",
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
        ylim=(0, 1),
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
        ylim=(0, 6),
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

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if partners:
            pname = f"_{partners}"
        else:
            pname = ""
        fname = os.path.join(save_path, f"Apical_vs_Basal_Local_Coactivity{pname}")
        fig.savefig(fname + ".pdf")

    ######################### Statistics Section ###########################
    if display_stats == False:
        return

    # Perform tests
    if test_type == "parametric":
        dist_coactivity_table, _, dist_coactivity_posthoc = (
            t_utils.ANOVA_2way_mixed_posthoc(
                data_dict={
                    "Apical": apical_distance_coactivity,
                    "Basal": basal_distance_coactivity,
                },
                method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        dist_coactivity_norm_table, _, dist_coactivity_norm_posthoc = (
            t_utils.ANOVA_2way_mixed_posthoc(
                data_dict={
                    "Apical": apical_distance_coactivity_norm,
                    "Basal": basal_distance_coactivity_norm,
                },
                method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        dist_coactivity_MRS_table, _, dist_coactivity_MRS_posthoc = (
            t_utils.ANOVA_2way_mixed_posthoc(
                data_dict={
                    "Apical MRS": apical_MRS_distance_coactivity,
                    "Basal MRS": basal_MRS_distance_coactivity,
                    "Apical nMRS": apical_nMRS_distance_coactivity,
                    "Basal nMRS": basal_nMRS_distance_coactivity,
                },
                method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        dist_coactivity_MRS_table_norm, _, dist_coactivity_MRS_posthoc_norm = (
            t_utils.ANOVA_2way_mixed_posthoc(
                data_dict={
                    "Apical MRS": apical_MRS_distance_coactivity_norm,
                    "Basal MRS": basal_MRS_distance_coactivity_norm,
                    "Apical nMRS": apical_nMRS_distance_coactivity_norm,
                    "Basal nMRS": basal_nMRS_distance_coactivity_norm,
                },
                method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        local_t, local_p = stats.ttest_ind(
            apical_local_coactivity, basal_local_coactivity, nan_policy="omit"
        )
        local_norm_t, local_norm_p = stats.ttest_ind(
            apical_local_coactivity_norm, basal_local_coactivity_norm, nan_policy="omit"
        )
        near_dist_t, near_dist_p = stats.ttest_ind(
            apical_near_vs_dist, basal_near_vs_dist, nan_policy="omit"
        )
        near_dist_norm_t, near_dist_norm_p = stats.ttest_ind(
            apical_near_vs_dist_norm, basal_near_vs_dist_norm, nan_policy="omit"
        )
        frac_coactive_t, frac_coactive_p = stats.ttest_ind(
            apical_fraction_coactive, basal_fraction_coactive, nan_policy="omit"
        )
        frac_part_t, frac_part_p = stats.ttest_ind(
            apical_fraction_participating,
            basal_fraction_participating,
            nan_policy="omit",
        )
        num_t, num_p = stats.ttest_ind(
            apical_coactive_num, basal_coactive_num, nan_policy="omit"
        )
        test_title = "T-Test"
    elif test_type == "nonparametric":
        dist_coactivity_table, _, dist_coactivity_posthoc = (
            t_utils.ANOVA_2way_mixed_posthoc(
                data_dict={
                    "Apical": apical_distance_coactivity,
                    "Basal": basal_distance_coactivity,
                },
                method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        dist_coactivity_norm_table, _, dist_coactivity_norm_posthoc = (
            t_utils.ANOVA_2way_mixed_posthoc(
                data_dict={
                    "Apical": apical_distance_coactivity_norm,
                    "Basal": basal_distance_coactivity_norm,
                },
                method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        dist_coactivity_MRS_table, _, dist_coactivity_MRS_posthoc = (
            t_utils.ANOVA_2way_mixed_posthoc(
                data_dict={
                    "Apical MRS": apical_MRS_distance_coactivity,
                    "Basal MRS": basal_MRS_distance_coactivity,
                    "Apical nMRS": apical_nMRS_distance_coactivity,
                    "Basal nMRS": basal_nMRS_distance_coactivity,
                },
                method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        dist_coactivity_MRS_table_norm, _, dist_coactivity_MRS_posthoc_norm = (
            t_utils.ANOVA_2way_mixed_posthoc(
                data_dict={
                    "Apical MRS": apical_MRS_distance_coactivity_norm,
                    "Basal MRS": basal_MRS_distance_coactivity_norm,
                    "Apical nMRS": apical_nMRS_distance_coactivity_norm,
                    "Basal nMRS": basal_nMRS_distance_coactivity_norm,
                },
                method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        local_t, local_p = stats.mannwhitneyu(
            apical_local_coactivity[~np.isnan(apical_local_coactivity)],
            basal_local_coactivity[~np.isnan(basal_local_coactivity)],
        )
        local_norm_t, local_norm_p = stats.mannwhitneyu(
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
    elif test_type == "mixed-effect":
        dist_coactivity_table, dist_coactivity_posthoc = (
            t_utils.two_way_RM_mixed_effects_model(
                data_dict={
                    "Apical": apical_distance_coactivity,
                    "Basal": basal_distance_coactivity,
                },
                random_dict={"Apical": apical_ids, "Basal": basal_ids},
                post_method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        dist_coactivity_norm_table, dist_coactivity_norm_posthoc = (
            t_utils.two_way_RM_mixed_effects_model(
                data_dict={
                    "Apical": apical_distance_coactivity_norm,
                    "Basal": basal_distance_coactivity_norm,
                },
                random_dict={"Apical": apical_ids, "Basal": basal_ids},
                post_method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        dist_coactivity_MRS_table, dist_coactivity_MRS_posthoc = (
            t_utils.two_way_RM_mixed_effects_model(
                data_dict={
                    "Apical MRS": apical_MRS_distance_coactivity,
                    "Basal MRS": basal_MRS_distance_coactivity,
                    "Apical nMRS": apical_nMRS_distance_coactivity,
                    "Basal nMRS": basal_nMRS_distance_coactivity,
                },
                random_dict={
                    "Apical MRS": apical_MRS_ids,
                    "Basal MRS": basal_MRS_ids,
                    "Apical nMRS": apical_nMRS_ids,
                    "Basal nMRS": basal_nMRS_ids,
                },
                post_method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        dist_coactivity_MRS_table_norm, dist_coactivity_MRS_posthoc_norm = (
            t_utils.two_way_RM_mixed_effects_model(
                data_dict={
                    "Apical MRS": apical_MRS_distance_coactivity_norm,
                    "Basal MRS": basal_MRS_distance_coactivity_norm,
                    "Apical nMRS": apical_nMRS_distance_coactivity_norm,
                    "Basal nMRS": basal_nMRS_distance_coactivity_norm,
                },
                random_dict={
                    "Apical MRS": apical_MRS_ids,
                    "Basal MRS": basal_MRS_ids,
                    "Apical nMRS": apical_nMRS_ids,
                    "Basal nMRS": basal_nMRS_ids,
                },
                post_method=test_method,
                rm_vals=distance_bins,
                compare_type="between",
            )
        )
        local_t, local_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_local_coactivity,
                "Basal": basal_local_coactivity,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
            slopes_intercept=False,
        )
        local_norm_t, local_norm_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_local_coactivity_norm,
                "Basal": basal_local_coactivity_norm,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
            slopes_intercept=False,
        )
        near_dist_t, near_dist_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={"Apical": apical_near_vs_dist, "Basal": basal_near_vs_dist},
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
            slopes_intercept=False,
        )
        near_dist_norm_t, near_dist_norm_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_near_vs_dist_norm,
                "Basal": basal_near_vs_dist_norm,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
            slopes_intercept=False,
        )
        frac_coactive_t, frac_coactive_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_fraction_coactive,
                "Basal": basal_fraction_coactive,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
            slopes_intercept=False,
        )
        frac_part_t, frac_part_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_fraction_participating,
                "Basal": basal_fraction_participating,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
            slopes_intercept=False,
        )
        num_t, num_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_coactive_num,
                "Basal": basal_coactive_num,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
            slopes_intercept=False,
        )
        test_title = "Mixed-Effects"

    elif test_type == "ART":
        dist_coactivity_table, dist_coactivity_posthoc = t_utils.two_way_RM_ART(
            data_dict={
                "Apical": apical_distance_coactivity,
                "Basal": basal_distance_coactivity,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            rm_vals=distance_bins,
            post_method=test_method,
        )
        dist_coactivity_norm_table, dist_coactivity_norm_posthoc = (
            t_utils.two_way_RM_ART(
                data_dict={
                    "Apical": apical_distance_coactivity_norm,
                    "Basal": basal_distance_coactivity_norm,
                },
                random_dict={"Apical": apical_ids, "Basal": basal_ids},
                rm_vals=distance_bins,
                post_method=test_method,
            )
        )
        dist_coactivity_MRS_table, dist_coactivity_MRS_posthoc = t_utils.two_way_RM_ART(
            data_dict={
                "Apical MRS": apical_MRS_distance_coactivity,
                "Basal MRS": basal_MRS_distance_coactivity,
                "Apical nMRS": apical_nMRS_distance_coactivity,
                "Basal nMRS": basal_nMRS_distance_coactivity,
            },
            random_dict={
                "Apical MRS": apical_MRS_ids,
                "Basal MRS": basal_MRS_ids,
                "Apical nMRS": apical_nMRS_ids,
                "Basal nMRS": basal_nMRS_ids,
            },
            rm_vals=distance_bins,
            post_method=test_method,
        )
        dist_coactivity_MRS_table_norm, dist_coactivity_MRS_posthoc_norm = (
            t_utils.two_way_RM_ART(
                data_dict={
                    "Apical MRS": apical_MRS_distance_coactivity_norm,
                    "Basal MRS": basal_MRS_distance_coactivity_norm,
                    "Apical nMRS": apical_nMRS_distance_coactivity_norm,
                    "Basal nMRS": basal_nMRS_distance_coactivity_norm,
                },
                random_dict={
                    "Apical MRS": apical_MRS_ids,
                    "Basal MRS": basal_MRS_ids,
                    "Apical nMRS": apical_nMRS_ids,
                    "Basal nMRS": basal_nMRS_ids,
                },
                rm_vals=distance_bins,
                post_method=test_method,
            )
        )
        local_t, local_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_local_coactivity,
                "Basal": basal_local_coactivity,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
        )
        local_norm_t, local_norm_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_local_coactivity_norm,
                "Basal": basal_local_coactivity_norm,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
        )
        near_dist_t, near_dist_p, _ = t_utils.one_way_ART(
            data_dict={"Apical": apical_near_vs_dist, "Basal": basal_near_vs_dist},
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
        )
        near_dist_norm_t, near_dist_norm_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_near_vs_dist_norm,
                "Basal": basal_near_vs_dist_norm,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
        )
        frac_coactive_t, frac_coactive_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_fraction_coactive,
                "Basal": basal_fraction_coactive,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
        )
        frac_part_t, frac_part_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_fraction_participating,
                "Basal": basal_fraction_participating,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
        )
        num_t, num_p, _ = t_utils.one_way_ART(
            data_dict={
                "Apical": apical_coactive_num,
                "Basal": basal_coactive_num,
            },
            random_dict={"Apical": apical_ids, "Basal": basal_ids},
            post_method=test_method,
        )
        test_title = "Aligned Rank Transform"

    # Perform fishers exact
    contingency_table = np.array(
        [
            [np.nansum(apical_coactive_spines), np.nansum(basal_coactive_spines)],
            [np.nansum(~apical_coactive_spines), np.nansum(~basal_coactive_spines)],
        ]
    )
    coactive_ratio, coactive_p = stats.fisher_exact(contingency_table)
    contingency_norm_table = np.array(
        [
            [
                np.nansum(apical_coactive_spines_norm),
                np.nansum(basal_coactive_spines_norm),
            ],
            [
                np.nansum(~apical_coactive_spines_norm),
                np.nansum(~basal_coactive_spines_norm),
            ],
        ]
    )
    coactive_norm_ratio, coactive_norm_p = stats.fisher_exact(contingency_norm_table)

    # Organize some of the results
    results_dict = {
        "Comparison": [
            "Local CoA",
            "Local Norm. CoA",
            "Near vs Dist",
            "Near vs Dist Norm",
            "Frac. CoA",
            "Frac. Part.",
            "Coactive Num.",
            "Coactive Spines",
            "Coactive Spines Norm",
        ],
        "stat": [
            local_t,
            local_norm_t,
            near_dist_t,
            near_dist_norm_t,
            frac_coactive_t,
            frac_part_t,
            num_t,
            coactive_ratio,
            coactive_norm_ratio,
        ],
        "p-val": [
            local_p,
            local_norm_p,
            near_dist_p,
            near_dist_norm_p,
            frac_coactive_p,
            frac_part_p,
            num_p,
            coactive_p,
            coactive_norm_p,
        ],
    }
    results_df = pd.DataFrame.from_dict(results_dict)
    print(results_df)
    results_df.update(results_df[["stat"]].applymap("{:.4}".format))
    results_df.update(results_df[["p-val"]].applymap("{:.4E}".format))

    # Display stats
    fig2, axes2 = plt.subplot_mosaic(
        """
        AB
        CD
        EF
        GH
        I.
        """,
        figsize=(15, 45),
        height_ratios=[1, 1, 1, 2, 1],
    )
    # Format the tables
    axes2["A"].axis("off")
    axes2["A"].axis("tight")
    axes2["A"].set_title("Distance Coactivity Table")
    A_table = axes2["A"].table(
        cellText=dist_coactivity_table.values,
        colLabels=dist_coactivity_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    A_table.auto_set_font_size(False)
    A_table.set_fontsize(8)
    axes2["B"].axis("off")
    axes2["B"].axis("tight")
    axes2["B"].set_title("Distance Coactivity Norm. Table")
    B_table = axes2["B"].table(
        cellText=dist_coactivity_norm_table.values,
        colLabels=dist_coactivity_norm_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    B_table.auto_set_font_size(False)
    B_table.set_fontsize(8)
    axes2["C"].axis("off")
    axes2["C"].axis("tight")
    axes2["C"].set_title("Distance Coactivity Posthoc")
    C_table = axes2["C"].table(
        cellText=dist_coactivity_posthoc.values,
        colLabels=dist_coactivity_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    C_table.auto_set_font_size(False)
    C_table.set_fontsize(8)
    axes2["D"].axis("off")
    axes2["D"].axis("tight")
    axes2["D"].set_title("Distance Coactivity Posthoc")
    D_table = axes2["D"].table(
        cellText=dist_coactivity_norm_posthoc.values,
        colLabels=dist_coactivity_norm_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    D_table.auto_set_font_size(False)
    D_table.set_fontsize(8)
    axes2["E"].axis("off")
    axes2["E"].axis("tight")
    axes2["E"].set_title("Coactivity MRS nMRS Table")
    E_table = axes2["E"].table(
        cellText=dist_coactivity_MRS_table.values,
        colLabels=dist_coactivity_MRS_table.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    E_table.auto_set_font_size(False)
    E_table.set_fontsize(8)
    axes2["F"].axis("off")
    axes2["F"].axis("tight")
    axes2["F"].set_title("Coactivity MRS nMRS Norm. Table")
    F_table = axes2["F"].table(
        cellText=dist_coactivity_MRS_table_norm.values,
        colLabels=dist_coactivity_MRS_table_norm.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    F_table.auto_set_font_size(False)
    F_table.set_fontsize(8)
    axes2["G"].axis("off")
    axes2["G"].axis("tight")
    axes2["G"].set_title("Coactivity MRS Posthoc")
    G_table = axes2["G"].table(
        cellText=dist_coactivity_MRS_posthoc.values,
        colLabels=dist_coactivity_MRS_posthoc.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    G_table.auto_set_font_size(False)
    G_table.set_fontsize(8)
    axes2["H"].axis("off")
    axes2["H"].axis("tight")
    axes2["H"].set_title("Coactivity MRS Norm. Posthoc")
    H_table = axes2["H"].table(
        cellText=dist_coactivity_MRS_posthoc_norm.values,
        colLabels=dist_coactivity_MRS_posthoc_norm.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    H_table.auto_set_font_size(False)
    H_table.set_fontsize(8)
    axes2["I"].axis("off")
    axes2["I"].axis("tight")
    axes2["I"].set_title(f"{test_title} results")
    I_table = axes2["I"].table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        loc="center",
        bbox=[0, 0.2, 0.9, 0.5],
    )
    I_table.auto_set_font_size(False)
    I_table.set_fontsize(8)

    fig2.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"

        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"

        if partners:
            pname = f"_{partners}"
        else:
            pname = ""
        fname = os.path.join(
            save_path, f"Apical_vs_Basal_Local_Coactivity{pname}_stats"
        )
        fig2.savefig(fname + ".pdf")


def plot_dendrite_coactivity(
    apical_dataset,
    basal_dataset,
    MRSs=None,
    figsize=(10, 10),
    showmeans=False,
    hist_bins=50,
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function for comparing dendritic coactivity between apical and basal spines

    INPUT PARAMETERS
        apical_dataset - Dendritic_Coactivity_Data object for apical data

        basal_dataset - Dendritic_Coactivity_Data object for basal data

        MRSs - str specifying if you wish to examine only MRSs or nMRSs. Accepts
                "MRS" or "nMRS". Default is None to examine all spines

        figsize - tuple specifying the figure size

        showmeans - boolean specifying whether to show means on the box plots

        hist_bins - int specifying number of bins for histogram

        test_type - str specifying type of statistics to perform

        display_stats - boolean specifying whether to perform the stats

        save - boolean specifying whether to save the figure or not

        save_path - str specifying where to save the figure

    """
    COLORS = ["goldenrod", "mediumseagreen"]

    # apical_ids = np.array(d_utils.code_str_to_int(apical_dataset.mouse_id))
    # basal_ids = np.array(d_utils.code_str_to_int(basal_dataset.mouse_id))

    all_ids = apical_dataset.mouse_id + basal_dataset.mouse_id
    all_ids_coded = np.array(d_utils.code_str_to_int(all_ids))
    apical_ids = all_ids_coded[: len(apical_dataset.mouse_id)]
    basal_ids = all_ids_coded[len(apical_dataset.mouse_id) :]

    # Find present spines
    apical_present = find_present_spines(apical_dataset.spine_flags)
    basal_present = find_present_spines(basal_dataset.spine_flags)
    apical_MRSs = apical_dataset.movement_spines
    apical_nMRSs = apical_dataset.nonmovement_spines
    basal_MRSs = basal_dataset.movement_spines
    basal_nMRSs = basal_dataset.nonmovement_spines

    if MRSs == "MRS":
        apical_present = apical_present * apical_MRSs
        basal_present = basal_present * basal_MRSs
    elif MRSs == "nMRS":
        apical_present = apical_present * apical_nMRSs
        basal_present = basal_present * basal_nMRSs

    # Pull relevant data
    apical_ids = d_utils.subselect_data_by_idxs(apical_ids, apical_present)
    basal_ids = d_utils.subselect_data_by_idxs(basal_ids, basal_present)
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
    apical_above_chance = d_utils.subselect_data_by_idxs(
        apical_dataset.all_above_chance_coactivity, apical_present
    )
    basal_above_chance = d_utils.subselect_data_by_idxs(
        basal_dataset.all_above_chance_coactivity,
        basal_present,
    )
    apical_above_chance_conj = d_utils.subselect_data_by_idxs(
        apical_dataset.conj_above_chance_coactivity, apical_present
    )
    basal_above_chance_conj = d_utils.subselect_data_by_idxs(
        basal_dataset.conj_above_chance_coactivity,
        basal_present,
    )
    apical_above_chance_nonconj = d_utils.subselect_data_by_idxs(
        apical_dataset.nonconj_above_chance_coactivity, apical_present
    )
    basal_above_chance_nonconj = d_utils.subselect_data_by_idxs(
        basal_dataset.nonconj_above_chance_coactivity,
        basal_present,
    )
    apical_frac_spine = d_utils.subselect_data_by_idxs(
        apical_dataset.all_fraction_spine_coactive, apical_present
    )
    basal_frac_spine = d_utils.subselect_data_by_idxs(
        basal_dataset.all_fraction_spine_coactive, basal_present
    )
    apical_frac_dend = d_utils.subselect_data_by_idxs(
        apical_dataset.all_fraction_dendrite_coactive, apical_present
    )
    basal_frac_dend = d_utils.subselect_data_by_idxs(
        basal_dataset.all_fraction_dendrite_coactive, basal_present
    )
    apical_coactive_spines = d_utils.subselect_data_by_idxs(
        apical_dataset.all_coactive_spines, apical_present
    )
    basal_coactive_spines = d_utils.subselect_data_by_idxs(
        basal_dataset.all_coactive_spines, basal_present
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

    # Organize traces for plotting
    a_glu_means = [
        np.nanmean(x, axis=1) for x in apical_glu_traces if type(x) == np.ndarray
    ]
    a_glu_means = np.vstack(a_glu_means)
    apical_glu_hmap = a_glu_means.T
    apical_glu_means = np.nanmean(a_glu_means, axis=0)
    apical_glu_sems = stats.sem(a_glu_means, axis=0, nan_policy="omit")
    b_glu_means = [
        np.nanmean(x, axis=1) for x in basal_glu_traces if type(x) == np.ndarray
    ]
    b_glu_means = np.vstack(b_glu_means)
    basal_glu_hmap = b_glu_means.T
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
        ABCDEF
        GHIJKa
        LLNOPP
        RRTUUW
        """,
        figsize=figsize,
    )
    fig.suptitle("Apical vs Basal Dendritic Coactivity")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ########################## Plot data onto axes ###########################
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
    # All above chance box plot
    plot_box_plot(
        data_dict={"Apical": apical_above_chance, "Basal": basal_above_chance},
        figsize=(5, 5),
        title="All events",
        xtitle=None,
        ytitle=f"Coactivity above chance",
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
    # Nonconj coactivity box plot
    plot_box_plot(
        data_dict={
            "Apical": apical_nonconj_coactivity_rate,
            "Basal": basal_nonconj_coactivity_rate,
        },
        figsize=(5, 5),
        title="Without Local",
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
    # All above chance box plot
    plot_box_plot(
        data_dict={
            "Apical": apical_above_chance_nonconj,
            "Basal": basal_above_chance_nonconj,
        },
        figsize=(5, 5),
        title="Without Local",
        xtitle=None,
        ytitle=f"Coactivity above chance",
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
    # Conj coactivity box plot
    plot_box_plot(
        data_dict={
            "Apical": apical_conj_coactivity_rate,
            "Basal": basal_conj_coactivity_rate,
        },
        figsize=(5, 5),
        title="With Local",
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
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Conj above chance box plot
    plot_box_plot(
        data_dict={
            "Apical": apical_above_chance_conj,
            "Basal": basal_above_chance_conj,
        },
        figsize=(5, 5),
        title="With Local",
        xtitle=None,
        ytitle=f"Coactivity above chance",
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
    ## Fraction Apical Coactive
    plot_pie_chart(
        data_dict={
            "Coactive": np.nansum(apical_coactive_spines),
            "Noncoactive": np.nansum(~apical_coactive_spines),
        },
        title="Apical",
        figsize=(5, 5),
        colors=[COLORS[0], "silver"],
        alpha=0.9,
        edgecolor="white",
        txt_color="black",
        txt_size=9,
        legend=True,
        donut=0.6,
        linewidth=1.5,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    ## Fraction Basal MRSs
    plot_pie_chart(
        data_dict={
            "Coactive": np.nansum(basal_coactive_spines),
            "Noncoactive": np.nansum(~basal_coactive_spines),
        },
        title="Basal",
        figsize=(5, 5),
        colors=[COLORS[1], "silver"],
        alpha=0.9,
        edgecolor="white",
        txt_color="black",
        txt_size=9,
        legend=True,
        donut=0.6,
        linewidth=1.5,
        ax=axes["H"],
        save=False,
        save_path=None,
    )
    # All spine fraction coactive
    plot_box_plot(
        data_dict={
            "Apical": apical_frac_spine,
            "Basal": basal_frac_spine,
        },
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle=f"Fraction of Spines' Activity",
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
    # All dendrite fraction coactive
    plot_box_plot(
        data_dict={
            "Apical": apical_frac_dend,
            "Basal": basal_frac_dend,
        },
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle=f"Fraction of Dendrites' Activity",
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
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    # Fraction with local coactivity
    plot_box_plot(
        data_dict={
            "Apical": apical_frac_conj,
            "Basal": basal_frac_conj,
        },
        figsize=(5, 5),
        title=None,
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
        ax=axes["K"],
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
        ax=axes["L"],
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
        ax=axes["R"],
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
        ax=axes["U"],
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
        ax=axes["a"],
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
        ax=axes["T"],
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
        ax=axes["W"],
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
        ax=axes["N"],
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
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    # Onsets
    plot_histogram(
        data=list((apical_onset, basal_onset)),
        bins=hist_bins,
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
        ax=axes["P"],
        save=False,
        save_path=None,
    )
    ax_p_inset = axes["P"].inset_axes([0.8, 0.25, 0.4, 0.6])
    sns.despine(ax=ax_p_inset)
    plot_box_plot(
        data_dict={"Apical": apical_onset, "Basal": basal_onset},
        figsize=(5, 5),
        title="Onsets",
        xtitle=None,
        ytitle="Realtive onset (s)",
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
        ax=ax_p_inset,
        save=False,
        save_path=None,
    )

    fig.tight_layout()
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if MRSs:
            mname = f"_{MRSs}"
        else:
            mname = ""
        fname = os.path.join(save_path, f"Apical_vs_Basal_Dendritic_Coactivity{mname}")
        fig.savefig(fname + ".pdf")

    ######################### Statistics Section ##########################
    if display_stats == False:
        return

    # Perform the tests
    if test_type == "parametric":
        coactivity_t, coactivity_p = stats.ttest_ind(
            apical_coactivity_rate, basal_coactivity_rate, nan_policy="omit"
        )
        above_t, above_p = stats.ttest_ind(
            apical_above_chance, basal_above_chance, nan_policy="omit"
        )
        conj_t, conj_p = stats.ttest_ind(
            apical_conj_coactivity_rate, basal_conj_coactivity_rate, nan_policy="omit"
        )
        conj_above_t, conj_above_p = stats.ttest_ind(
            apical_above_chance_conj, basal_above_chance_conj, nan_policy="omit"
        )
        nonconj_t, nonconj_p = stats.ttest_ind(
            apical_nonconj_coactivity_rate,
            basal_nonconj_coactivity_rate,
            nan_policy="omit",
        )
        nonconj_above_t, nonconj_above_p = stats.ttest_ind(
            apical_above_chance_nonconj, basal_above_chance_nonconj, nan_policy="omit"
        )
        s_frac_t, s_frac_p = stats.ttest_ind(
            apical_frac_spine, basal_frac_spine, nan_policy="omit"
        )
        d_frac_t, d_frac_p = stats.ttest_ind(
            apical_frac_dend, basal_frac_dend, nan_policy="omit"
        )
        c_frac_t, c_frac_p = stats.ttest_ind(
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
        above_t, above_p = stats.mannwhitneyu(
            apical_above_chance[~np.isnan(apical_above_chance)],
            basal_above_chance[~np.isnan(basal_above_chance)],
        )
        conj_t, conj_p = stats.mannwhitneyu(
            apical_conj_coactivity_rate[~np.isnan(apical_conj_coactivity_rate)],
            basal_conj_coactivity_rate[~np.isnan(basal_conj_coactivity_rate)],
        )
        conj_above_t, conj_above_p = stats.mannwhitneyu(
            apical_above_chance_conj[~np.isnan(apical_above_chance_conj)],
            basal_above_chance_conj[~np.isnan(basal_above_chance_conj)],
        )
        nonconj_t, nonconj_p = stats.mannwhitneyu(
            apical_nonconj_coactivity_rate[~np.isnan(apical_nonconj_coactivity_rate)],
            basal_nonconj_coactivity_rate[~np.isnan(basal_nonconj_coactivity_rate)],
        )
        nonconj_above_t, nonconj_above_p = stats.mannwhitneyu(
            apical_above_chance_nonconj[~np.isnan(apical_above_chance_nonconj)],
            basal_above_chance_nonconj[~np.isnan(basal_above_chance_nonconj)],
        )
        s_frac_t, s_frac_p = stats.mannwhitneyu(
            apical_frac_spine[~np.isnan(apical_frac_spine)],
            basal_frac_spine[~np.isnan(basal_frac_spine)],
        )
        d_frac_t, d_frac_p = stats.mannwhitneyu(
            apical_frac_dend[~np.isnan(apical_frac_dend)],
            basal_frac_dend[~np.isnan(basal_frac_dend)],
        )
        c_frac_t, c_frac_p = stats.mannwhitneyu(
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
    elif test_type == "mixed-effect":
        coactivity_t, coactivity_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_coactivity_rate,
                "Basal": basal_coactivity_rate,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        above_t, above_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_above_chance,
                "Basal": basal_above_chance,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        conj_t, conj_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_conj_coactivity_rate,
                "Basal": basal_conj_coactivity_rate,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        conj_above_t, conj_above_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_above_chance_conj,
                "Basal": basal_above_chance_conj,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        nonconj_t, nonconj_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_nonconj_coactivity_rate,
                "Basal": basal_nonconj_coactivity_rate,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        nonconj_above_t, nonconj_above_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_above_chance_nonconj,
                "Basal": basal_above_chance_nonconj,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        s_frac_t, s_frac_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_frac_spine,
                "Basal": basal_frac_spine,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        d_frac_t, d_frac_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_frac_dend,
                "Basal": basal_frac_dend,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        c_frac_t, c_frac_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_frac_conj,
                "Basal": basal_frac_conj,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        glu_t, glu_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_glu_amp,
                "Basal": basal_glu_amp,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        ca_t, ca_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_ca_amp,
                "Basal": basal_ca_amp,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        dend_t, dend_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_dend_amp,
                "Basal": basal_dend_amp,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        onset_t, onset_p, _ = t_utils.one_way_mixed_effects_model(
            data_dict={
                "Apical": apical_onset,
                "Basal": basal_onset,
            },
            random_dict={
                "Apical": apical_ids,
                "Basal": basal_ids,
            },
            post_method="fdr_bh",
            slopes_intercept=False,
        )
        test_title = "Mixed-Effects"

    contingency_table = np.array(
        [
            [np.nansum(apical_coactive_spines), np.nansum(basal_coactive_spines)],
            [np.nansum(~apical_coactive_spines), np.nansum(~basal_coactive_spines)],
        ]
    )
    coactive_ratio, coactive_p = stats.fisher_exact(contingency_table)

    # Organize the results
    results_dict = {
        "Comparison": [
            "Coactivity rate",
            "Above chance",
            "Conj rate",
            "Conj chance",
            "Nonconj rate",
            "Nonconj chance",
            "Spine frac",
            "Dend frac",
            "Conj frac",
            "Glu amp",
            "Ca amp",
            "Dend amp",
            "Onsets",
            "CoA spines",
        ],
        "Stat": [
            coactivity_t,
            above_t,
            conj_t,
            conj_above_t,
            nonconj_t,
            nonconj_above_t,
            s_frac_t,
            d_frac_t,
            c_frac_t,
            glu_t,
            ca_t,
            dend_t,
            onset_t,
            coactive_ratio,
        ],
        "P-Val": [
            coactivity_p,
            above_p,
            conj_p,
            conj_above_p,
            nonconj_p,
            nonconj_above_p,
            s_frac_p,
            d_frac_p,
            c_frac_p,
            glu_p,
            ca_p,
            dend_p,
            onset_p,
            coactive_p,
        ],
    }
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.update(results_df[["Stat"]].applymap("{:.4}".format))
    results_df.update(results_df[["P-Val"]].applymap("{:.4E}".format))

    fig2, axes2 = plt.subplot_mosaic("""A""", figsize=(6, 6))

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
        if MRSs:
            mname = f"_{MRSs}"
        else:
            mname = ""
        fname = os.path.join(
            save_path, f"Apical_vs_Basal_Dendritic_Coactivity{mname}_stats"
        )
        fig2.savefig(fname + ".pdf")
