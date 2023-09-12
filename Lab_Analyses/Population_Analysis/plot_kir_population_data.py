import os
from copy import copy
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.plot_activity_heatmap import plot_activity_heatmap
from Lab_Analyses.Plotting.plot_box_plot import plot_box_plot
from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot

sns.set()
sns.set_style("ticks")


def plot_kir_population_data(
    dataset,
    figsize=(10, 10),
    trace_type="dFoF",
    trace_data=None,
    trace_len=None,
    hmap_data=None,
    hmap_len=None,
    hist_bins=30,
    showmeans=False,
    test_type="nonparametric",
    display_stats=True,
    save=False,
    save_path=None,
):
    """Function to plot the kir population data
    
        INPUT PARAMETERS
            dataset - Grouped_Kir_Activity_Data object

            figsize - tuple specifying the size of the overall figure

            trace_data - optional tuple specifying the mouse and FOV to plot.
                         Default is not to plot all cells in the dataset

            trace_len - optional int specifying how long the traces should be.
                        Default is None, to plot entire length of the traces

            hmap_data - optional tuple specifying the mouse and FOV to plot.
                         Default is not to plot all cells in the dataset

            hmap_len - optional int specifying how long the hmap should be.
                        Default is None, to plot entire length of the hmap

            showmeans - boolean specifying whether to show means on boxplots

            test_type - str specifying whether to perform parametric or nonparametric
                        statistics

            display_stats - boolean specifying whether to show statistics

            save - boolean specifying whether to save the plot or not

            save_path - str specifying the path where to save the figure
    
    """
    COLORS = ["black", "darkviolet"]
    # Pull relevant data
    sampling_rate = dataset.sampling_rate[0]
    kir_ids = dataset.kir_ids
    event_rates = dataset.event_rates
    shuff_event_rates = dataset.shuff_event_rates
    expression_intensity = dataset.expression_intensity
    amplitudes = dataset.amplitudes
    shuff_amplitudes = dataset.shuff_amplitudes

    # Pull appropriate traces
    if trace_type == "dFoF":
        trace_plot_data = dataset.dFoF
        activity_plot_data = dataset.activity
        hmap_plot_data = dataset.zscore_dFoF
        ylabel = "dFoF"
    elif trace_type == "processed_dFoF":
        trace_plot_data = dataset.processed_dFoF
        activity_plot_data = dataset.activity
        hmap_plot_data = dataset.activity
        ylabel = "dFoF"
    elif trace_type == "spikes":
        trace_plot_data = dataset.estimated_spikes
        activity_plot_data = np.ones(trace_plot_data.shape)
        hmap_plot_data = dataset.zscore_spikes
        ylabel = "Estimated spikes"
    elif trace_type == "binned_spikes":
        trace_plot_data = dataset.binned_spikes
        activity_plot_data = np.ones(trace_plot_data.shape)
        hmap_plot_data = dataset.zscore_binned_spikes
        ylabel = "Estimated spikes"
    trace_kir_ids = copy(kir_ids)
    hmap_kir_ids = copy(kir_ids)
    trace_idxs = np.arange(trace_plot_data.shape[1])
    hmap_idxs = np.arange(hmap_plot_data.shape[1])

    print(f"kir postive: {len(np.nonzero(kir_ids)[0])}")
    print(f"kir negative: {len(np.nonzero([not x for x in kir_ids])[0])}")

    # Subselect traces
    if trace_data != None:
        if len(trace_data) != 2:
            return "Must input a tuple to specify the traces to plot"
        trace_idxs = np.array(
            [
                (x == trace_data[0]) * (y == trace_data[1])
                for x, y in zip(dataset.mouse_id, dataset.fov)
            ]
        ).astype(bool)
        trace_plot_data = trace_plot_data[:, trace_idxs]
        activity_plot_data = activity_plot_data[:, trace_idxs]
        trace_kir_ids = list(compress(trace_kir_ids, trace_idxs))
    if hmap_data != None:
        if len(hmap_data) != 2:
            return "Must input a tuple to specify the traces to plot"
        hmap_idxs = np.array(
            [
                (x == hmap_data[0]) * (y == hmap_data[1])
                for x, y in zip(dataset.mouse_id, dataset.fov)
            ]
        ).astype(bool)
        hmap_plot_data = hmap_plot_data[:, hmap_idxs]
        hmap_kir_ids = list(compress(hmap_kir_ids, hmap_idxs))

    # Truncate traces if specified
    if trace_len != None:
        trace_plot_data = trace_plot_data[:trace_len, :]
        activity_plot_data = activity_plot_data[:trace_len, :]
    if hmap_len != None:
        hmap_plot_data = hmap_plot_data[:hmap_len, :]

    # Construct the figure
    fig, axes = plt.subplots(
        2, 4, figsize=figsize, width_ratios=[3, 2, 1, 2], layout="constrained"
    )
    fig.suptitle("Kir population activity")

    ###################### Plot data onto axes #######################
    # Trace and heatmap subfigures
    ## Get the gridspec
    gridspec = axes[0, 0].get_subplotspec().get_gridspec()
    ## Clear columns for the subfigures
    for a in axes[:, 0]:
        a.remove()
    for a in axes[:, 1]:
        a.remove()
    ## Make trace subfigure
    tracefig = fig.add_subfigure(gridspec[:, 0])
    trace_axes = tracefig.subplots(1, 1)
    ### Segregate rois to plot
    NEG_NUM = 40
    POS_NUM = 15
    pos_idxs = np.nonzero(trace_kir_ids)[0]
    neg_idxs = np.nonzero([not x for x in trace_kir_ids])[0]
    pos_idxs = np.random.choice(pos_idxs, POS_NUM)
    neg_idxs = np.random.choice(neg_idxs, NEG_NUM)
    pos_traces = trace_plot_data[:, pos_idxs]
    pos_activity = activity_plot_data[:, pos_idxs]
    neg_traces = trace_plot_data[:, neg_idxs]
    neg_activity = activity_plot_data[:, neg_idxs]
    plot_traces(
        group1_trace=neg_traces,
        group1_activity=neg_activity,
        group2_trace=pos_traces,
        group2_activity=pos_activity,
        title=trace_data,
        ylabel=ylabel,
        ax=trace_axes,
        sampling_rate=sampling_rate,
    )
    ## Make heatmap subfigure
    ## Seperate data first
    pos_hmap = hmap_plot_data[:, hmap_kir_ids]
    neg_hmap = hmap_plot_data[:, [not x for x in hmap_kir_ids]]
    neg_hmap = neg_hmap[:, np.random.permutation(neg_hmap.shape[1])]
    hmapfig = fig.add_subfigure(gridspec[:, 1], in_layout=True)
    hmap_axes = hmapfig.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [neg_hmap.shape[1], pos_hmap.shape[1]]},
    )
    xrange = (0, pos_hmap.shape[0] / sampling_rate)
    print(xrange)
    # plot_activity_heatmap(
    #    data=neg_hmap,
    #    figsize=(4, 5),
    #    sampling_rate=sampling_rate,
    #    activity_window=xrange,
    #    title=hmap_data,
    #    cbar_label=f"Zscore {ylabel}",
    #    hmap_range=None,
    #    center=None,
    #    sorted=None,
    #    normalize=False,
    #    cmap="Greys",
    #    axis_width=2,
    #    minor_ticks="x",
    #    tick_len=3,
    #    ax=hmap_axes[0],
    #    save=False,
    #    save_path=None,
    # )
    # plot_activity_heatmap(
    #    data=pos_hmap,
    #    figsize=(4, 5),
    #    sampling_rate=sampling_rate,
    #    activity_window=xrange,
    #    title=hmap_data,
    #    cbar_label=f"Zscore {ylabel}",
    #    hmap_range=None,
    #    center=None,
    #    sorted=None,
    #    normalize=False,
    #    cmap="Purples",
    #    axis_width=2,
    #    minor_ticks="x",
    #    tick_len=3,
    #    ax=hmap_axes[1],
    #    save=False,
    #    save_path=None,
    # )
    hmap_axes[0].imshow(neg_hmap.T, cmap="Greys", aspect="auto")
    hmap_axes[1].imshow(pos_hmap.T, cmap="Purples", aspect="auto")

    # Plot event rates
    plot_box_plot(
        data_dict={
            "Kir-": event_rates[[not x for x in kir_ids]],
            "Kir+": event_rates[kir_ids],
        },
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Event rate (events/min)",
        ylim=(0, None),
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.7,
        b_err_alpha=1,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes[0, 2],
        save=False,
        save_path=None,
    )

    # Plot event rates
    plot_box_plot(
        data_dict={
            "Kir-": amplitudes[[not x for x in kir_ids]],
            "Kir+": amplitudes[kir_ids],
        },
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Event amplitude",
        ylim=None,
        b_colors=COLORS,
        b_edgecolors="black",
        b_err_colors="black",
        m_color="black",
        m_width=1,
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.7,
        b_err_alpha=1,
        whisker_lim=None,
        whisk_width=1.5,
        outliers=False,
        showmeans=showmeans,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes[1, 2],
        save=False,
        save_path=None,
    )

    # Plot shuffled data
    plot_histogram(
        data=list(
            (
                event_rates[kir_ids],
                shuff_event_rates[:, kir_ids].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="density",
        avlines=None,
        title=None,
        xtitle="Event rate (events/min)",
        xlim=(0, 10),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.4,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes[0, 3],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_inset = axes[0, 3].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_inset)
    plot_swarm_bar_plot(
        data_dict={
            "data": event_rates[kir_ids],
            "shuff": shuff_event_rates[:, kir_ids].flatten().astype(np.float32),
        },
        mean_type="median",
        err_type="CI",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Event rate",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.7,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_inset,
        save=False,
        save_path=None,
    )

    # Plot shuffled amplitude data
    plot_histogram(
        data=list(
            (
                amplitudes[kir_ids],
                shuff_amplitudes[:, kir_ids].flatten().astype(np.float32),
            )
        ),
        bins=hist_bins,
        stat="density",
        avlines=None,
        title=None,
        xtitle="Event amplitude",
        xlim=(0, None),
        figsize=(5, 5),
        color=[COLORS[1], "grey"],
        alpha=0.4,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes[1, 3],
        save=False,
        save_path=None,
    )
    ## Inset bar
    ax_inset1 = axes[1, 3].inset_axes([0.9, 0.4, 0.4, 0.6])
    sns.despine(ax=ax_inset1)
    plot_swarm_bar_plot(
        data_dict={
            "data": amplitudes[kir_ids],
            "shuff": shuff_amplitudes[:, kir_ids].flatten().astype(np.float32),
        },
        mean_type="median",
        err_type="CI",
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Event amplitude",
        ylim=None,
        b_colors=[COLORS[1], "grey"],
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.5,
        b_linewidth=1.5,
        b_alpha=0.7,
        s_colors=[COLORS[1], "grey"],
        s_size=5,
        plot_ind=False,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=ax_inset1,
        save=False,
        save_path=None,
    )

    # fig.tight_layout()
    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, "Kir_Population_Activity_Figure")
        fig.savefig(fname + ".pdf")


############### HELPER FUNCTIONS ################
def plot_traces(
    group1_trace,
    group1_activity,
    group2_trace,
    group2_activity,
    title=None,
    ylabel="dFoF",
    figsize=(10, 10),
    ax=None,
    sampling_rate=30,
):
    """Helper function to plot trace data"""
    COLORS = ["darkviolet", "black"]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()

    ax.set_title(title)
    # Seperate into active and inactive portions
    active_group1 = np.zeros(group1_trace.shape) * np.nan
    inactive_group1 = np.zeros(group1_trace.shape) * np.nan
    active_group2 = np.zeros(group2_trace.shape) * np.nan
    inactive_group2 = np.zeros(group2_trace.shape) * np.nan

    for i in range(group1_activity.shape[1]):
        active1, inactive1 = get_active_inactive(
            group1_trace[:, i], group1_activity[:, i]
        )
        active_group1[:, i] = active1
        inactive_group1[:, i] = inactive1
    for i in range(group2_activity.shape[1]):
        active2, inactive2 = get_active_inactive(
            group2_trace[:, i], group2_activity[:, i]
        )
        active_group2[:, i] = active2
        inactive_group2[:, i] = inactive2

    x = np.arange(active_group1.shape[0]) / sampling_rate
    ## Plot group 2 first
    for y in range(active_group2.shape[1]):
        ax.plot(
            x, inactive_group2[:, y] + y + 5, color=COLORS[0], alpha=0.3, linewidth=1
        )
        ax.plot(x, active_group2[:, y] + y + 5, color=COLORS[0], linewidth=1)
    # Plot group 1
    for z in range(active_group1.shape[1]):
        ax.plot(
            x,
            inactive_group1[:, z] + y + z + 1 + 5,
            color=COLORS[1],
            alpha=0.3,
            linewidth=1,
        )
        ax.plot(x, active_group1[:, z] + y + z + 1 + 5, color=COLORS[1], linewidth=1)
    ax.set_xlabel("Time (s)", labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)


def get_active_inactive(trace, active):
    """Helper function to seperate traces into active and inactive portions"""
    inactive_mask = active == 1
    active_mask = active == 0
    inactive_trace = np.copy(trace)
    active_trace = np.copy(trace)
    inactive_trace[inactive_mask] = np.nan
    active_trace[active_mask] = np.nan

    return active_trace, inactive_trace

