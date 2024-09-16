import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.adjust_axes import adjust_axes
from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_box_plot import plot_box_plot
from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Plotting.plot_pie_chart import plot_pie_chart
from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import find_nearby_spines
from Lab_Analyses.Spine_Analysis_v2.structural_plasticity import calculate_volume_change
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


def plot_plasticity(
    dataset,
    figsize=(5, 5),
    cluster_dist=10,
    test_type="nonparametric",
    test_method="fdr_tsbh",
    save=False,
    save_path=None,
):
    """Function to plot the plasticity of different types of responsive spines
    against different types of non-responsive spines

    INPUT PARAMETERS
        dataset - Grouped_Synaptic_Opto_Data set

        figisze - tuple specifying the size of the figure

        cluster_dist - int specifying the distance considered to be spatially
                        clustered

        test_type - str specifying what type of statistics to perform

        test_method - str specifying what type of multiple comparisons
                     correction to perform

        save - boolean specifying whether to save the figure or not

        save_path - str specifying the path where to save the figure

    """
    # Pull relevant variables
    responsive_spines = dataset.responsive_spines
    spine_flags = dataset.spine_flags
    spine_positions = dataset.spine_positions
    spine_volumes = dataset.spine_volumes

    followup_flags = dataset.followup_flags
    followup_positions = dataset.followup_flags
    followup_volumes = dataset.followup_volumes

    # Get unique dendrites
    dendrites = dataset.spine_dendrite
    unique_dendrites = set(dendrites)
    print(np.nonzero(responsive_spines)[0])

    # Find clustered responsive spines
    clustered_responsive_spines = np.zeros(len(responsive_spines))

    for dend in unique_dendrites:
        dend_idxs = np.nonzero(dendrites == dend)[0]
        # Get current spines and responsive spines
        curr_responsive = responsive_spines[dend_idxs]
        curr_positions = spine_positions[dend_idxs]
        responsive_idxs = np.nonzero(curr_responsive)[0]
        ## Get only the responsive spine positions
        responsive_positions = curr_positions[curr_responsive.astype(bool)]
        if len(responsive_positions) == 0:
            continue
        if len(responsive_positions) == 1:
            continue
        for spine in range(len(responsive_positions)):
            curr_pos = responsive_positions[spine]
            other_pos = [x for i, x in enumerate(responsive_positions) if i != spine]
            rel_pos = np.array(other_pos) - curr_pos
            rel_pos = np.absolute(rel_pos)
            if any(rel_pos <= cluster_dist):
                clustered_responsive_spines[dend_idxs[responsive_idxs[spine]]] = 1

    # Get non-clustered and non-responsive lists
    nonclustered_responsive_spines = responsive_spines - clustered_responsive_spines
    nonresponsive_spines = np.array([not x for x in responsive_spines]).astype(int)

    # Calculate volume changes
    volumes = [spine_volumes, followup_volumes]
    flags = [spine_flags, followup_flags]

    delta_volume, spine_idxs = calculate_volume_change(
        volumes,
        flags,
        norm=False,
        exclude="Shaft Spine",
    )
    delta_volume = delta_volume[-1]

    # Subset spine positions for present spines
    responsive_spines = d_utils.subselect_data_by_idxs(responsive_spines, spine_idxs)
    clustered_responsive_spines = d_utils.subselect_data_by_idxs(
        clustered_responsive_spines, spine_idxs
    )
    nonclustered_responsive_spines = d_utils.subselect_data_by_idxs(
        nonclustered_responsive_spines, spine_idxs
    )
    nonresponsive_spines = d_utils.subselect_data_by_idxs(
        nonresponsive_spines, spine_idxs
    )
    spine_positions = d_utils.subselect_data_by_idxs(spine_positions, spine_idxs)
    dendrites = d_utils.subselect_data_by_idxs(dendrites, spine_idxs)
    spine_flags = d_utils.subselect_data_by_idxs(spine_flags, spine_idxs)

    # Set up volume dictionary
    volume_dict = {
        "clustered": delta_volume[clustered_responsive_spines.astype(bool)],
        "nonclustered": delta_volume[nonclustered_responsive_spines.astype(bool)],
        "nonresponsive": delta_volume[nonresponsive_spines.astype(bool)],
    }

    print(volume_dict["clustered"])
    print(np.nanmedian(volume_dict["clustered"]))
    print(np.nanmedian(volume_dict["nonresponsive"]))
    print(stats.mannwhitneyu(volume_dict["clustered"], volume_dict["nonresponsive"]))

    # Get responsive spines and their neighbors volume changes
    responsive_volume = []
    nearby_volume = []

    for dend in unique_dendrites:
        dend_idxs = np.nonzero(dendrites == dend)[0]
        # Get current spines and responsive spines
        curr_volume = delta_volume[dend_idxs]
        curr_responsive = clustered_responsive_spines[dend_idxs]
        curr_positions = spine_positions[dend_idxs]
        # Normalize spine ids for each dendrite
        grouping = np.arange(len(curr_volume))

        curr_flags = d_utils.subselect_data_by_idxs(spine_flags, dend_idxs)
        nearby_responsive_spines = find_nearby_spines(
            curr_positions,
            curr_flags,
            grouping,
            partner_list=curr_responsive,
            cluster_dist=cluster_dist,
        )

        for i, clustered in enumerate(curr_responsive):
            if clustered == 0:
                continue
            clust_vol = curr_volume[i]
            clust_nearby = nearby_responsive_spines[i]
            nearby_vol = curr_volume[clust_nearby]
            if len(nearby_vol) == 0:
                continue
            responsive_volume.append([[clust_vol] for j in range(len(nearby_vol))])
            nearby_volume.append(nearby_vol)

    responsive_volume = [x for y in responsive_volume for x in y]
    nearby_volume = [x for y in nearby_volume for x in y]

    # Make the plot
    fig, axes = plt.subplot_mosaic(
        """AB""",
        figsize=figsize,
    )

    fig.subplots_adjust(hspace=1, wspace=0.5)

    # Plot the volume changes
    plot_box_plot(
        data_dict=volume_dict,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Volume Change",
        ylim=None,
        b_colors=["forestgreen", "mediumspringgreen", "silver"],
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1,
        b_width=0.5,
        b_linewidth=1,
        b_alpha=0.9,
        b_err_alpha=1,
        whisker_lim=None,
        whisk_width=1,
        outliers=False,
        showmeans=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )

    # Plot the correlation between clustered spines and their neighbors
    plot_scatter_correlation(
        x_var=np.array(responsive_volume),
        y_var=np.array(nearby_volume),
        CI=95,
        title=None,
        xtitle="Clustered volume change",
        ytitle="Nearby volume change",
        figsize=(5, 5),
        xlim=None,
        ylim=None,
        marker_size=10,
        face_color="forestgreen",
        cmap_color=None,
        edge_color="white",
        edge_width=0.3,
        line_color="forestgreen",
        s_alpha=1,
        line_width=1.5,
        axis_width=1.5,
        tick_len=3,
        unity=True,
        ax=axes["B"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        save_name = os.path.join(save_path, "responsive_spine_plasticity")
        fig.savefig(save_name + ".pdf")
