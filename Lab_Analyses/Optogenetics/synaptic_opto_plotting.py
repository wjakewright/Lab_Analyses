import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.adjust_axes import adjust_axes
from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Plotting.plot_pie_chart import plot_pie_chart
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Utilities import data_utilities as d_utils

sns.set()
sns.set_style("ticks")


def plot_session_activity(
    dataset,
    identifier={"mouse_id": None, "FOV": None},
    zscore=False,
    figsize=(7, 8),
    save=False,
    save_path=None,
):
    """Function to plot the activity of each spine from a single opto session.

    INPUT PARAMETERS
        dataset - Grouped_Synaptic_Opto_Data dataclass

        identifier - dict specifying the mouse id and FOV you wish to plot

        zscore - boolean specifying whether to use zscore activity or not

        figsize - tuple specifying the size of the figure

        save - boolean specifying whether to save the figure output or not

        save_path - str specifying where to save the figure

    """
    # Organize and select the data
    ## Find idxs where it matches the mouse ID
    mouse_idxs = [
        i for i, x in enumerate(dataset.mouse_id) if x == identifier["mouse_id"]
    ]
    ## Find idxs where it matches the FOV
    FOV_idxs = [i for i, x in enumerate(dataset.FOV) if x == identifier["FOV"]]
    ## Find matching indexs
    data_idxs = np.intersect1d(mouse_idxs, FOV_idxs)

    # Grab the trace data
    if zscore:
        dFoF = dataset.spine_z_dFoF[:, data_idxs]
    else:
        dFoF = dataset.spine_processed_dFoF[:, data_idxs]
    # Get stimulation timestamps
    ## Should be all the same for each spine
    stims = dataset.stim_timestamps[data_idxs[0]]
    stim_len = dataset.stim_len[data_idxs[0]]

    sampling_rate = dataset.parameters["Sampling Rate"]

    # Setup the plot
    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout()
    ## z titles
    if zscore:
        main_title = f"{identifier['mouse_id']}_{identifier['FOV']}_zscore"
        dFoF_title = "Zscore"
    else:
        main_title = f"{identifier['mouse_id']}_{identifier['FOV']}_dFoF"
        dFoF_title = "dFoF"
    ## Add title
    ax.set_title(main_title)

    # Plot the data
    for i in range(dFoF.shape[1]):
        x = np.linspace(0, len(dFoF[:, i]) / sampling_rate, len(dFoF[:, i]))
        ax.plot(x, dFoF[:, i] + i * 5, label=i, linewidth=0.5, color="mediumslateblue")
    # Plot the itis
    for stim in stims:
        ax.axvspan(
            stim / sampling_rate,
            (stim + stim_len) / sampling_rate,
            alpha=0.3,
            color="red",
        )

    # Adjust the axes
    adjust_axes(
        ax,
        minor_ticks="both",
        xtitle="Time (s)",
        ytitle=dFoF_title,
        tick_len=3,
        axis_width=1.5,
    )

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        save_name = os.path.join(save_path, main_title)
        fig.savefig(save_name + ".pdf")


def plot_individual_examples(
    dataset,
    identifiers={"mouse_id": None, "FOV": None},
    hmap_range=(0, 2),
    figsize=(8, 8),
    norm=False,
    save=False,
    save_path=None,
):
    """Function to plot examples of individual responsive and non responsive spines

    INPUT PARAMETERS
        dataset - Grouped_Synaptic_Opto_Data set

        identifiers - dict containing the mouse_id and FOV you want to get examples
                        from. If none, examples will be chosen at random

        hmap_range - tuple specifying the range of the heatmap colorbar

        figsize - tuple specifying the size of the figure

        save - boolean specifying whether to save the figure

        save_path - str specifying where to save the figure

    """
    EXAMPLE_NUM = 3
    activity_window = dataset.parameters["Visual Window"]
    sampling_rate = dataset.parameters["Sampling Rate"]
    # First lets get the example data
    if identifiers["mouse_id"]:
        print("getting mouse data")
        prefix = f"{identifiers['mouse_id']}_{identifiers['FOV']}_"
        mouse_idxs = [
            i for i, x in enumerate(dataset.mouse_id) if x == identifiers["mouse_id"]
        ]
        ## Find idxs where it matches the FOV
        FOV_idxs = [i for i, x in enumerate(dataset.FOV) if x == identifiers["FOV"]]
        ## Find matching indexs
        data_idxs = np.intersect1d(mouse_idxs, FOV_idxs)
        # Get responsive idxs
        responsive_idxs = np.nonzero(dataset.responsive_spines[data_idxs])[0]
        nonresponsive_idxs = np.nonzero(1 - dataset.responsive_spines[data_idxs])[0]
        if len(responsive_idxs) < EXAMPLE_NUM:
            return f"{identifiers['mouse_id']} {identifiers['FOV']} does not have enough responsive spines"
        if len(nonresponsive_idxs) < EXAMPLE_NUM:
            return f"{identifiers['mouse_id']} {identifiers['FOV']} does not have enough nonresponsive spines"
    else:
        print("randomly selecting data")
        data_idxs = np.array(list(range(dataset.spine_z_dFoF.shape[1])))
        responsive_idxs = np.nonzero(dataset.responsive_spines)[0]
        nonresponsive_idxs = np.nonzero(1 - dataset.responsive_spines)[0]

    print("processing")
    stim_len = dataset.stim_len[data_idxs[0]]
    # Randomly sample the examples
    responsive_spines = np.random.choice(
        data_idxs[responsive_idxs], EXAMPLE_NUM, replace=False
    )
    nonresponsive_spines = np.random.choice(
        data_idxs[nonresponsive_idxs], EXAMPLE_NUM, replace=False
    )
    print(
        f"Responsive idxs: {[(i, x) for i, x in enumerate(data_idxs) if x in responsive_spines]}"
    )
    print(
        f"Non-responsive idxs: {[(i, x) for i, x in enumerate(data_idxs) if x in nonresponsive_spines]}"
    )

    # Get the trial activity for the examples
    responsive_activity = [dataset.stim_traces[i] for i in responsive_spines]
    nonresponsive_activity = [dataset.stim_traces[j] for j in nonresponsive_spines]

    print("plotting")
    # Initialize the plot
    fig, axes = plt.subplot_mosaic(
        """
        ABCD
        EFGH
        IJKL
        """,
        figsize=figsize,
    )
    fig.subplots_adjust(hspace=1, wspace=0.5)

    responsive_axes = [("A", "B"), ("E", "F"), ("I", "J")]
    nonresponsive_axes = [("C", "D"), ("G", "H"), ("K", "L")]

    joined_activity = responsive_activity + nonresponsive_activity
    joined_axes = responsive_axes + nonresponsive_axes

    for activity, ax in zip(joined_activity, joined_axes):
        # Prep data for plotting
        if ax in responsive_axes:
            title = "Responsive"
        else:
            title = "Non-responsive"
        ## Mean trace activity
        zeroed_activity = d_utils.zero_window(activity, (0, 2), sampling_rate)
        axes[ax[0]].set_title(title)
        mean_trace = np.nanmean(zeroed_activity, axis=1)
        sem_trace = stats.sem(zeroed_activity, axis=1, nan_policy="omit")
        plot_mean_activity_traces(
            means=mean_trace,
            sems=sem_trace,
            group_names="Spine",
            sampling_rate=sampling_rate,
            activity_window=activity_window,
            avlines=None,
            ahlines=None,
            figsize=(5, 5),
            colors="forestgreen",
            title=title,
            ytitle="Zscored dF/F",
            ylim=None,
            axis_width=1.5,
            minor_ticks="both",
            ax=axes[ax[0]],
            save=False,
            save_path=None,
        )
        axes[ax[0]].axvspan(0, stim_len / sampling_rate, color="red", alpha=0.1)
        ## Heatmap

        ## Plot
        plot_activity_heatmap(
            zeroed_activity,
            figsize=(5, 5),
            sampling_rate=sampling_rate,
            activity_window=activity_window,
            title=None,
            cbar_label="Zscore",
            hmap_range=hmap_range,
            center=None,
            sorted=None,
            normalize=norm,
            cmap="plasma",
            axis_width=1.5,
            minor_ticks=None,
            tick_len=3,
            ax=axes[ax[1]],
            save=False,
            save_path=None,
        )
        axes[ax[1]].axvline(
            np.absolute(activity_window[0]) * sampling_rate,
            color="white",
            linestyle="--",
        )

    fig.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        save_name = os.path.join(save_path, f"{prefix}_opto_example_synapses")
        fig.savefig(save_name + ".pdf")


def plot_responsive_synapse_properties(
    dataset,
    cluster_dist=10,
    hmap_range=(0, 2),
    figsize=(10, 10),
    save=False,
    save_path=None,
):
    """Function to characterize and plot properties of responsive and nonresponsive
    synapses

    INPUT PARAMETERS
         dataset - Grouped_Synaptic_Opto_Data set

         hmap_range - tuple specifying the range of the heatmap colorbar

         figsize - tuple specifying the size of the figure

         save - boolean specifying whether to save the figure

         save_path - str specifying where to save the figure

    """
    activity_window = dataset.parameters["Visual Window"]
    sampling_rate = dataset.parameters["Sampling Rate"]

    # First calculate the percentages of responsive spines
    num_responsive = np.sum(dataset.responsive_spines)
    num_nonresponsive = len(dataset.responsive_spines) - num_responsive

    # Perform analyses that required for individual dendrites
    ## Set up some variables
    dend_percent_responsive = []
    distance_between_responsive = []
    clustered_responsive_spines = []
    ## Get unique dendrites
    dendrites = set(dataset.spine_dendrite)
    for dend in dendrites:
        ## Get dendrite idxs
        dend_idxs = np.nonzero(dataset.spine_dendrite == dend)[0]
        ## Get percent responsive
        dend_responsive = dataset.responsive_spines[dend_idxs]
        dend_percent_responsive.append(np.sum(dend_responsive) / len(dend_responsive))
        ## Get distances between responsive spines
        dend_positions = dataset.spine_positions[dend_idxs]
        ## Get only the responsive positions
        responsive_positions = dend_positions[dend_responsive.astype(bool)]
        if len(responsive_positions) == 0:
            continue
        if len(responsive_positions) == 1:
            distance_between_responsive = distance_between_responsive + [
                np.max(dend_positions)
            ]
            clustered_responsive_spines.append(0)
            continue
        for spine in range(len(responsive_positions)):
            curr_pos = responsive_positions[spine]
            other_pos = [x for i, x in enumerate(responsive_positions) if i != spine]
            rel_pos = np.array(other_pos) - curr_pos
            rel_pos = np.absolute(rel_pos)
            distance_between_responsive = distance_between_responsive + list(rel_pos)
            if any(rel_pos <= cluster_dist):
                clustered_responsive_spines.append(1)
            else:
                clustered_responsive_spines.append(0)

    # Calculate percentage of responsive spines clusterd
    num_clustered_responsive = np.sum(clustered_responsive_spines)
    num_non_clustered_responsive = num_responsive - num_clustered_responsive

    # Perpare traces for the heatmap
    responsive_means = []
    nonresponsive_means = []
    responsive_idxs = np.nonzero(dataset.responsive_spines)[0]
    for i, traces in enumerate(dataset.stim_traces):
        mean_trace = np.nanmean(traces, axis=1)
        if i in responsive_idxs:
            responsive_means.append(mean_trace)
        else:
            nonresponsive_means.append(mean_trace)

    responsive_means = np.vstack(responsive_means)
    nonresponsive_means = np.vstack(nonresponsive_means)
    responsive_num = responsive_means.shape[0]

    combined_means = np.vstack((responsive_means, nonresponsive_means)).T

    # Initialize the plot
    fig, axes = plt.subplot_mosaic(
        """
        ABC
        DEF
        """,
        figsize=figsize,
    )

    fig.subplots_adjust(hspace=1, wspace=0.5)

    # Plot fraction of responsive spines
    plot_pie_chart(
        data_dict={
            "Responsive": num_responsive,
            "Non-responsive": num_nonresponsive,
        },
        title="Percent responsive spines",
        figsize=(5, 5),
        colors=["forestgreen", "silver"],
        alpha=0.7,
        edgecolor="white",
        txt_color="black",
        txt_size=10,
        legend="upper left",
        donut=0.6,
        linewidth=1.5,
        ax=axes["A"],
        save=False,
        save_path=None,
    )

    # Plot fraction across dendrites
    plot_swarm_bar_plot(
        data_dict={
            "Data": np.array(dend_percent_responsive) * 100,
        },
        mean_type="mean",
        err_type="sem",
        figsize=(5, 5),
        title="Percent per dendrite",
        xtitle=None,
        ytitle="Responsive spines (%)",
        ylim=None,
        b_colors="forestgreen",
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.3,
        b_linewidth=1,
        b_alpha=0.5,
        s_colors="forestgreen",
        s_size=5,
        s_alpha=0.9,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )

    # Plot the heatmap of activity
    zeroed_means = d_utils.zero_window(combined_means, (0, 2), sampling_rate)
    plot_activity_heatmap(
        zeroed_means,
        figsize=(5, 5),
        sampling_rate=sampling_rate,
        activity_window=activity_window,
        title="Mean activity",
        cbar_label="Zscore",
        hmap_range=hmap_range,
        center=None,
        sorted=None,
        normalize=False,
        cmap="plasma",
        axis_width=2,
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    # Add line to seperate the responsive from nonresponsive
    axes["C"].axhline(responsive_num, color="red")
    axes["C"].axvline(120, color="white", linestyle="--")

    # Plot histogram of distances between responsive spines
    plot_histogram(
        data=np.array(distance_between_responsive),
        bins=25,
        stat="probability",
        avlines=[cluster_dist],
        title="Distance Between Responsive",
        xtitle="Distance (um)",
        xlim=(0, None),
        figsize=(5, 5),
        color="forestgreen",
        alpha=0.5,
        axis_width=1.5,
        minor_ticks=None,
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )

    # Plot percent clustered responsive spines
    plot_pie_chart(
        data_dict={
            "Clustered responsive": num_clustered_responsive,
            "All others": num_non_clustered_responsive + num_nonresponsive,
        },
        title="Fraction clustered responsive",
        colors=["forestgreen", "silver"],
        alpha=0.8,
        edgecolor="white",
        txt_color="black",
        txt_size=10,
        legend="upper left",
        donut=0.6,
        linewidth=1.5,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    # Plot percent clustered out of only responsive spines
    plot_pie_chart(
        data_dict={
            "Clustered responsive": num_clustered_responsive,
            "Non-clustered responsive": num_non_clustered_responsive,
        },
        title="Fraction clustered responsive",
        colors=["forestgreen", "silver"],
        alpha=0.8,
        edgecolor="white",
        txt_color="black",
        txt_size=10,
        legend="upper left",
        donut=0.6,
        linewidth=1.5,
        ax=axes["F"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        save_name = os.path.join(save_path, "responsive_spine_properties")
        fig.savefig(save_name + ".pdf")
