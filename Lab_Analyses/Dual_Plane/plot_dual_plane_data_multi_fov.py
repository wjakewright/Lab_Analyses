import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

from Lab_Analyses.Plotting.adjust_axes import adjust_axes
from Lab_Analyses.Plotting.plot_histogram import plot_histogram
from Lab_Analyses.Plotting.plot_mean_activity_traces import plot_mean_activity_traces
from Lab_Analyses.Plotting.plot_scatter_correlation import plot_scatter_correlation
from Lab_Analyses.Plotting.plot_swarm_bar_plot import plot_swarm_bar_plot

sns.set()
sns.set_style("ticks")


def plot_soma_dend_traces(
    dataset,
    colors=["forestgreen", "black"],
    norm=True,
    subselect=None,
    save=False,
    save_path=None,
):
    """Function to plot overlayed traced of dendritic and somatic traces

    INPUT PARAMETERS
        dataset - Dual_Plane_Data dataclass containing all the relevant data

        colors - list of str speicfying the colors to plot the different traces
                (dendritic and somatic)

        norm - boolean specifying whether to plot the normalized traces or not

        subselect - tuple specifying a specific x axis range you wish to subselect
                    for plotting

        save - boolean specifying whether or not to save the figure

        save_path - str specifying where to save the data
    """
    # pull relevant data
    if norm:
        title = "Normalized Somatic and Dendritic Traces"
        ytitle = "Normalized dF/F"
        apical_traces = dataset.apical_dFoF_norm
        basal_traces = dataset.basal_dFoF_norm
        a_soma_traces = dataset.a_somatic_dFoF_norm
        b_soma_traces = dataset.b_somatic_dFoF_norm
    else:
        title = "Somatic and Dendritic Traces"
        ytitle = "dF/F"
        apical_traces = dataset.apical_dFoF
        basal_traces = dataset.basal_dFoF
        a_soma_traces = dataset.a_somatic_dFoF
        b_soma_traces = dataset.b_somatic_dFoF

    # Make the figure for apical data
    apical_row_num = apical_traces.shape[1]
    fig_size = (10, 4 * apical_row_num)
    apical_fig = plt.figure(figsize=fig_size)
    apical_fig.subplots_adjust(hspace=0.5)
    apical_fig.suptitle(title)
    apical_fig.tight_layout()

    # Add subplots
    for i in range(apical_traces.shape[1]):
        d_trace = apical_traces[:, i]
        s_trace = a_soma_traces[:, i]
        a_ax = apical_fig.add_subplot(apical_row_num, 1, i + 1)
        ## Plot full traces
        if subselect is None:
            x = np.arange(len(d_trace)) / dataset.sampling_rate
            a_ax.plot(x, d_trace, color=colors[0], label="Dendrite")
            a_ax.plot(x, s_trace, color=colors[1], label="Soma")
        ## Plot subselected traces
        else:
            x = (
                np.arange(len(d_trace[subselect[0] : subselect[1]]))
                / dataset.sampling_rate
            )
            a_ax.plot(
                x,
                d_trace[subselect[0] : subselect[1]],
                color=colors[0],
                label="Dendrite",
            )
            a_ax.plot(
                x,
                s_trace[subselect[0] : subselect[1]],
                color=colors[1],
                label="Soma",
            )

        # Set up the axes
        a_ax.set_title(f"Apical Dendrite {i + 1}", fontsize=10)
        adjust_axes(a_ax, None, "Time (s)", ytitle, 3, 1.5)

    # Make the figure for apical data
    basal_row_num = basal_traces.shape[1]
    fig_size = (10, 4 * basal_row_num)
    basal_fig = plt.figure(figsize=fig_size)
    basal_fig.subplots_adjust(hspace=0.5)
    basal_fig.suptitle(title)
    basal_fig.tight_layout()

    # Add subplots
    for i in range(basal_traces.shape[1]):
        d_trace = basal_traces[:, i]
        s_trace = b_soma_traces[:, i]
        b_ax = basal_fig.add_subplot(basal_row_num, 1, i + 1)
        ## Plot full traces
        if subselect is None:
            x = np.arange(len(d_trace)) / dataset.sampling_rate
            b_ax.plot(x, d_trace, color=colors[0], label="Dendrite")
            b_ax.plot(x, s_trace, color=colors[1], label="Soma")
        ## Plot subselected traces
        else:
            x = (
                np.arange(len(d_trace[subselect[0] : subselect[1]]))
                / dataset.sampling_rate
            )
            b_ax.plot(
                x,
                d_trace[subselect[0] : subselect[1]],
                color=colors[0],
                label="Dendrite",
            )
            b_ax.plot(
                x,
                s_trace[subselect[0] : subselect[1]],
                color=colors[1],
                label="Soma",
            )

        # Set up the axes
        b_ax.set_title(f"Basal Dendrite {i + 1}", fontsize=10)
        adjust_axes(a_ax, None, "Time (s)", ytitle, 3, 1.5)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        a_fname = os.path.join(save_path, "Apical " + title)
        apical_fig.savefig(a_fname + ".pdf")
        b_fname = os.path.join(save_path, "Basal " + title)
        basal_fig.savefig(b_fname + ".pdf")


def plot_soma_dend_coactivity(
    dataset,
    colors=["forestgreen", "black"],
    mean_type="mean",
    err_type="sem",
    norm=False,
    trace_avg="all",
    figsize=(5, 5),
    save=False,
    save_path=None,
):
    """Function to plot the fraction of coactivity between soma and dendritic events.
    Also plots the traces during "coactive" and "non-coactive" events

    INPUT PARAMETERS
        dataset - Dual_Plane_Data dataclass containing all the relevant data

        colors - list of str speicfying the colors to plot the different traces
                (dendritic and somatic)

        mean_type - str specifying the mean type for the bar plots

        err_type - str specifying the error type for the bar plots

        norm - boolean specifying whether or not to plot normalized traces

        trace_avg - str specifying how to average the traces. Average them all together
                    or average them based on dendrite averages

        figsize - tuple specifying the size of the figure

        save - boolean specifying whether to save the data or not

        save_path - str specifying where to save the data
    """
    # Set up the coactivity data
    a_soma_points = dataset.fraction_a_somatic_active
    apical_points = dataset.fraction_apical_active
    b_soma_points = dataset.fraction_b_somatic_active
    basal_points = dataset.fraction_basal_active

    # Construct the subplot
    fig, axes = plt.subplot_mosaic(
        [
            ["left", "top left middle", "right middle", "top right right"],
            ["left", "bottom left middle", "right middle", "bottom right right"],
        ],
        constrained_layout=True,
        figsize=figsize,
    )

    fig.suptitle("Somatic and Dendritic Coactivity")

    # Plot data on the axes
    ## First plot fraction coactive
    plot_swarm_bar_plot(
        data_dict={"Soma": a_soma_points, "Apical": apical_points},
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction of events coactive",
        ylim=(0, 1),
        b_colors=colors,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=colors,
        s_size=5,
        s_alpha=0.8,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["left"],
        save=False,
        save_path=None,
    )
    ## First plot fraction coactive
    plot_swarm_bar_plot(
        data_dict={"Soma": b_soma_points, "Basal": basal_points},
        mean_type=mean_type,
        err_type=err_type,
        figsize=(5, 5),
        title=None,
        xtitle=None,
        ytitle="Fraction of events coactive",
        ylim=(0, 1),
        b_colors=colors,
        b_edgecolors="black",
        b_err_colors="black",
        b_width=0.7,
        b_linewidth=0,
        b_alpha=0.5,
        s_colors=colors,
        s_size=5,
        s_alpha=0.8,
        plot_ind=True,
        axis_width=1.5,
        minor_ticks="y",
        tick_len=3,
        ax=axes["right middle"],
        save=False,
        save_path=None,
    )
    ## Plot coactive and non-coactive traces
    ### Get the trace data
    if norm:
        apical_coactive = dataset.coactive_apical_traces_norm
        apical_noncoactive = dataset.noncoactive_apical_traces_norm
        a_soma_coactive = dataset.coactive_a_somatic_traces_norm
        a_soma_noncoactive = dataset.noncoactive_a_somatic_traces_norm
        basal_coactive = dataset.coactive_basal_traces_norm
        basal_noncoactive = dataset.noncoactive_basal_traces_norm
        b_soma_coactive = dataset.coactive_b_somatic_traces_norm
        b_soma_noncoactive = dataset.noncoactive_b_somatic_traces_norm
        trace_ytitle = "Normalized dF/F"
    else:
        apical_coactive = dataset.coactive_apical_traces
        apical_noncoactive = dataset.noncoactive_apical_traces
        a_soma_coactive = dataset.coactive_a_somatic_traces
        a_soma_noncoactive = dataset.noncoactive_a_somatic_traces
        basal_coactive = dataset.coactive_basal_traces
        basal_noncoactive = dataset.noncoactive_basal_traces
        b_soma_coactive = dataset.coactive_b_somatic_traces
        b_soma_noncoactive = dataset.noncoactive_b_somatic_traces
        trace_ytitle = "dF/F"
    ### Get the means and sems
    if trace_avg == "all":
        a_coactive = np.hstack(apical_coactive)
        a_noncoactive = np.hstack(apical_noncoactive)
        as_coactive = np.hstack(a_soma_coactive)
        as_noncoactive = np.hstack(a_soma_noncoactive)
        b_coactive = np.hstack(basal_coactive)
        b_noncoactive = np.hstack(basal_noncoactive)
        bs_coactive = np.hstack(b_soma_coactive)
        bs_noncoactive = np.hstack(b_soma_noncoactive)
    elif trace_avg == "dend":
        a_c_means = []
        a_n_means = []
        as_c_means = []
        as_n_means = []
        b_c_means = []
        b_n_means = []
        bs_c_means = []
        bs_n_means = []
        for dc, dn, sc, sn in zip(
            apical_coactive, apical_noncoactive, a_soma_coactive, a_soma_noncoactive
        ):
            a_c_means.append(np.nanmean(dc, axis=1))
            a_n_means.append(np.nanmean(dn, axis=1))
            as_c_means.append(np.nanmean(sc, axis=1))
            as_n_means.append(np.nanmean(sn, axis=1))
        for dc, dn, sc, sn in zip(
            basal_coactive, basal_noncoactive, b_soma_coactive, b_soma_noncoactive
        ):
            b_c_means.append(np.nanmean(dc, axis=1))
            b_n_means.append(np.nanmean(dn, axis=1))
            bs_c_means.append(np.nanmean(sc, axis=1))
            bs_n_means.append(np.nanmean(sn, axis=1))
        a_coactive = np.vstack(a_c_means).T
        a_noncoactive = np.vstack(a_n_means).T
        as_coactive = np.vstack(as_c_means).T
        as_noncoactive = np.vstack(as_n_means).T
        b_coactive = np.vstack(b_c_means).T
        b_noncoactive = np.vstack(b_n_means).T
        bs_coactive = np.vstack(bs_c_means).T
        bs_noncoactive = np.vstack(bs_n_means).T

    apical_co_mean = np.nanmean(a_coactive, axis=1)
    apical_non_mean = np.nanmean(a_noncoactive, axis=1)
    a_soma_co_mean = np.nanmean(as_coactive, axis=1)
    a_soma_non_mean = np.nanmean(as_noncoactive, axis=1)
    apical_co_sem = stats.sem(a_coactive, axis=1, nan_policy="omit")
    apical_non_sem = stats.sem(a_noncoactive, axis=1, nan_policy="omit")
    a_soma_co_sem = stats.sem(as_coactive, axis=1, nan_policy="omit")
    a_soma_non_sem = stats.sem(as_noncoactive, axis=1, nan_policy="omit")

    basal_co_mean = np.nanmean(b_coactive, axis=1)
    basal_non_mean = np.nanmean(b_noncoactive, axis=1)
    b_soma_co_mean = np.nanmean(bs_coactive, axis=1)
    b_soma_non_mean = np.nanmean(bs_noncoactive, axis=1)
    basal_co_sem = stats.sem(b_coactive, axis=1, nan_policy="omit")
    basal_non_sem = stats.sem(b_noncoactive, axis=1, nan_policy="omit")
    b_soma_co_sem = stats.sem(bs_coactive, axis=1, nan_policy="omit")
    b_soma_non_sem = stats.sem(bs_noncoactive, axis=1, nan_policy="omit")

    ### Plot coactive traces
    plot_mean_activity_traces(
        means=[a_soma_co_mean, apical_co_mean],
        sems=[a_soma_co_sem, apical_co_sem],
        group_names=["Soma", "Apical"],
        sampling_rate=dataset.sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=colors,
        title="Apical Coactive Events",
        ytitle=trace_ytitle,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["top left middle"],
        save=False,
        save_path=None,
    )
    ### Plot noncoactive traces
    ylim = axes["top left middle"].get_ylim()  # Ensure same scale
    plot_mean_activity_traces(
        means=[a_soma_non_mean, apical_non_mean],
        sems=[a_soma_non_sem, apical_non_sem],
        group_names=["Soma", "Apical"],
        sampling_rate=dataset.sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=colors,
        title="Apical Non-coactive Events",
        ytitle=trace_ytitle,
        ylim=ylim,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["bottom left middle"],
        save=False,
        save_path=None,
    )

    ### Plot coactive traces
    plot_mean_activity_traces(
        means=[b_soma_co_mean, basal_co_mean],
        sems=[b_soma_co_sem, basal_co_sem],
        group_names=["Soma", "Basal"],
        sampling_rate=dataset.sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=colors,
        title="Basal Coactive Events",
        ytitle=trace_ytitle,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["top right right"],
        save=False,
        save_path=None,
    )
    ### Plot noncoactive traces
    ylim = axes["top left middle"].get_ylim()  # Ensure same scale
    plot_mean_activity_traces(
        means=[b_soma_non_mean, basal_non_mean],
        sems=[b_soma_non_sem, basal_non_sem],
        group_names=["Soma", "Basal"],
        sampling_rate=dataset.sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=colors,
        title="Basal Non-coactive Events",
        ytitle=trace_ytitle,
        ylim=ylim,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["bottom right right"],
        save=False,
        save_path=None,
    )

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Soma_Apical_Basal_Coactivity_Plot")
        fig.savefig(fname + ".pdf")


def plot_amplitude_correlations(
    dataset,
    norm=True,
    apical_examples=None,
    basal_examples=None,
    soma_examples=None,
    color=["goldenrod", "black", "forestgreen", "deeppink"],
    figsize=(5, 5),
    corr_lim=None,
    s_size=5,
    s_alpha=0.5,
    bins=20,
    save=False,
    save_path=None,
):
    """Function to plot and correlate paired event amplitude between dendrites and
    soma. Plots two examples between sister dendrites as well as dendrites and
    parent somas. Plots across all data points and also the r2 values for each
    individual dendrite pair

    INPUT PARAMETERS
        dataset - Dual_Plane_Data dataclass containing all the relevant data

        norm - boolean specifying whether to plot normalized amplitudes or not

        dend_examples - list of idxs of example dendrite-dendrite pairs to plot

        soma_examples - list of idxs of example soma-dendrite pairs to plot

        color - list of str specifying the color to make the plots

        figsize - tuple specifying the size of the figure

        corr_lim - tuple specifying the limits for all the axes of the corr plots

        s_size - int specifying how large to make the scatter points

        s_alpha - float specifying alpha of the scatter points

        bins - int specifying how many bins for the histograms

        save - boolean specifying whether to save the data or not

        save_path - str specifying where to save the figure

    """
    # Grab the relevant data
    if norm:
        a_soma_amps = dataset.a_somatic_amplitudes_norm
        apical_amps = dataset.apical_amplitudes_norm
        other_apical_amps = dataset.other_apical_amplitudes_norm
        b_soma_amps = dataset.b_somatic_amplitudes_norm
        basal_amps = dataset.basal_amplitudes_norm
        other_basal_amps = dataset.other_basal_amplitudes_norm
        other_apical_across_amps = dataset.other_apical_across_amplitudes_norm
        other_basal_across_amps = dataset.other_basal_across_amplitudes_norm
        title_suffix = "norm. dF/F"
    else:
        a_soma_amps = dataset.a_somatic_amplitudes
        apical_amps = dataset.apical_amplitudes
        other_apical_amps = dataset.other_apical_amplitudes
        b_soma_amps = dataset.b_somatic_amplitudes
        basal_amps = dataset.basal_amplitudes
        other_basal_amps = dataset.other_basal_amplitudes
        other_apical_across_amps = dataset.other_apical_across_amplitudes
        other_basal_across_amps = dataset.other_basal_across_amplitudes
        title_suffix = "dF/F"

    # Combine all data points
    ## Apical
    all_a_soma_amps = np.concatenate(a_soma_amps)
    all_apical_amps = np.concatenate(apical_amps)
    all_other_apical_amps = np.concatenate(
        [np.concatenate(x) for x in other_apical_amps]
    )
    all_apical_apical_amps = []
    for i, x in enumerate(other_apical_amps):
        for j in x:
            all_apical_apical_amps.append(apical_amps[i])
    all_apical_apical_amps = np.concatenate(all_apical_apical_amps)
    ## Basal
    all_b_soma_amps = np.concatenate(b_soma_amps)
    all_basal_amps = np.concatenate(basal_amps)
    all_other_basal_amps = np.concatenate([np.concatenate(x) for x in other_basal_amps])
    all_basal_basal_amps = []
    for i, x in enumerate(other_basal_amps):
        for j in x:
            all_basal_basal_amps.append(basal_amps[i])
    all_basal_basal_amps = np.concatenate(all_basal_basal_amps)
    ## Apical to Basal
    all_other_apical_across_amps = np.concatenate(
        [np.concatenate(x) for x in other_apical_across_amps]
    )
    all_apical_basal_amps = []
    for i, x in enumerate(other_apical_across_amps):
        for j in x:
            all_apical_basal_amps.append(apical_amps[i])
    all_apical_basal_amps = np.concatenate(all_apical_basal_amps)

    # Get all of the R squared values
    apical_apical_r2 = []
    soma_apical_r2 = []
    basal_basal_r2 = []
    soma_basal_r2 = []
    apical_basal_r2 = []
    for soma, dend, other, across in zip(
        a_soma_amps, apical_amps, other_apical_amps, other_apical_across_amps
    ):
        sdc, _ = stats.pearsonr(soma, dend)
        soma_apical_r2.append(sdc**2)
        for o in other:
            ddc, _ = stats.pearsonr(dend, o)
            apical_apical_r2.append(ddc**2)
        for a in across:
            dac, _ = stats.pearsonr(dend, a)
            apical_basal_r2.append(dac**2)
    for soma, dend, other in zip(b_soma_amps, basal_amps, other_basal_amps):
        sdc, _ = stats.pearsonr(soma, dend)
        soma_basal_r2.append(sdc**2)
        for o in other:
            ddc, _ = stats.pearsonr(dend, o)
            basal_basal_r2.append(ddc**2)

    b = np.histogram(
        np.hstack(
            [
                apical_apical_r2,
                soma_apical_r2,
                basal_basal_r2,
                soma_basal_r2,
                apical_basal_r2,
            ]
        ),
        bins=bins,
    )[1]

    # Grab the example data
    ## Get apical-apical examples
    ### Example 1
    apical_apical_ex1 = apical_amps[apical_examples[0]]
    a_a_ex1 = other_apical_amps[apical_examples[0]]
    ### Get random partner
    apical_apical_other_ex1 = a_a_ex1[random.randint(0, len(a_a_ex1) - 1)]
    ### Example 2
    apical_apical_ex2 = apical_amps[apical_examples[1]]
    a_a_ex2 = other_apical_amps[apical_examples[1]]
    ### Get random partner
    apical_apical_other_ex2 = a_a_ex2[random.randint(0, len(a_a_ex2) - 1)]
    ## Get basal-basal examples
    ### Example 1
    basal_basal_ex1 = basal_amps[basal_examples[0]]
    b_b_ex1 = other_basal_amps[basal_examples[0]]
    ### Get random partner
    basal_basal_other_ex1 = b_b_ex1[random.randint(0, len(b_b_ex1) - 1)]
    ### Example 2
    basal_basal_ex2 = basal_amps[basal_examples[1]]
    b_b_ex2 = other_basal_amps[basal_examples[1]]
    ### Get random partner
    basal_basal_other_ex2 = b_b_ex2[random.randint(0, len(b_b_ex2) - 1)]
    ## Get apical-basal examples
    ### Example 1
    apical_basal_ex1 = apical_amps[apical_examples[0]]
    a_b_ex1 = other_apical_across_amps[apical_examples[0]]
    ### Get random partner
    apical_basal_other_ex1 = a_b_ex1[random.randint(0, len(a_b_ex1) - 1)]
    ### Example 2
    apical_basal_ex2 = apical_amps[apical_examples[1]]
    a_b_ex2 = other_apical_across_amps[apical_examples[1]]
    ### Get random partner
    apical_basal_other_ex2 = a_b_ex2[random.randint(0, len(a_b_ex2) - 1)]

    ## Get soma dend examples
    ### Example 1
    soma_apical_ex1 = apical_amps[soma_examples[0]]
    soma_a_soma_ex1 = a_soma_amps[soma_examples[0]]
    ### Example 2
    soma_apical_ex2 = apical_amps[soma_examples[1]]
    soma_a_soma_ex2 = a_soma_amps[soma_examples[1]]
    ### Example 1
    soma_basal_ex1 = basal_amps[soma_examples[0]]
    soma_b_soma_ex1 = b_soma_amps[soma_examples[0]]
    ### Example 2
    soma_basal_ex2 = basal_amps[soma_examples[1]]
    soma_b_soma_ex2 = b_soma_amps[soma_examples[1]]

    print(f"Number of soma-apical pairs: {len(apical_amps)}")
    print(
        f"Number of apical-apical pairs: {np.sum([len(x) for x in other_apical_amps])}"
    )
    print(f"Total soma-apical paired events: {len(all_apical_amps)}")
    print(f"Total apical-apical paired events: {len(all_apical_apical_amps)}")

    print(f"Number of soma-basal pairs: {len(basal_amps)}")
    print(f"Number of basal-basal pairs: {np.sum([len(x) for x in other_basal_amps])}")
    print(f"Total soma-basal paried events: {len(all_basal_amps)}")
    print(f"Total basal-basal paired events: {len(all_basal_basal_amps)}")

    print(
        f"Number of apical-basal pairs: {np.sum([len(x) for x in other_apical_across_amps])}"
    )
    print(f"Total apical-basal pairs: {len(all_apical_basal_amps)}")

    # Construct the subplot
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
    fig.suptitle("Paired Somatic and Apical and Basal Amplitudes")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ####################### Plot the data on the axes ########################
    # Apical-Apical Data
    ## Apical-Apical example 1
    plot_scatter_correlation(
        x_var=apical_apical_ex1,
        y_var=apical_apical_other_ex1,
        CI=None,
        title="Apical Ex.1",
        xtitle=f"Apical 1 {title_suffix}",
        ytitle=f"Apical 2 {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color=color[0],
        edge_color="white",
        line_color=color[0],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["A"],
        save=False,
        save_path=None,
    )
    ## Apical-Apical example 2
    plot_scatter_correlation(
        x_var=apical_apical_ex2,
        y_var=apical_apical_other_ex2,
        CI=None,
        title="Apical-Apical Ex.2",
        xtitle=f"Apical 1 {title_suffix}",
        ytitle=f"Apical 2 {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color=color[0],
        edge_color="white",
        line_color=color[0],
        line_width=1.5,
        s_alpha=s_alpha,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["B"],
        save=False,
        save_path=None,
    )
    ## Apical-Apical all
    plot_scatter_correlation(
        x_var=all_apical_apical_amps,
        y_var=all_other_apical_amps,
        CI=None,
        title="All Apical-Apical Pairs",
        xtitle=f"Dendrite {title_suffix}",
        ytitle=f"Sister Dendrite {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color="cmap",
        edge_color="white",
        line_color=color[0],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["C"],
        save=False,
        save_path=None,
    )
    ## Dendrite-Dendrite r squared histogram
    plot_histogram(
        data=np.array(apical_apical_r2),
        bins=b,
        stat="probability",
        title="Apical-Apical r-squared",
        xtitle="R-squared value",
        xlim=(0, 1),
        color=color[0],
        alpha=0.8,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["D"],
        save=False,
        save_path=None,
    )

    # Soma-Apical Data
    ## Soma-apical example 1
    plot_scatter_correlation(
        x_var=soma_a_soma_ex1,
        y_var=soma_apical_ex1,
        CI=None,
        title="Soma-Apical Ex.1",
        xtitle=f"Soma {title_suffix}",
        ytitle=f"Apical {title_suffix}",
        ylim=corr_lim,
        xlim=corr_lim,
        marker_size=s_size,
        face_color=color[1],
        edge_color="white",
        line_color=color[1],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["E"],
        save=False,
        save_path=None,
    )
    ## Soma-Apical example 2
    plot_scatter_correlation(
        x_var=soma_a_soma_ex2,
        y_var=soma_apical_ex2,
        CI=None,
        title="Soma-Apical Ex.2",
        xtitle=f"Soma {title_suffix}",
        ytitle=f"Apical {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color=color[1],
        edge_color="white",
        line_color=color[1],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["F"],
        save=False,
        save_path=None,
    )
    ## Soma-Apical all
    plot_scatter_correlation(
        x_var=all_a_soma_amps,
        y_var=all_apical_amps,
        CI=None,
        title="All Soma-Apical Pairs",
        xtitle=f"Soma {title_suffix}",
        ytitle=f"Apical {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color="cmap",
        edge_color="white",
        line_color=color[1],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["G"],
        save=False,
        save_path=None,
    )
    ## Soma-Apical r squared histogram
    plot_histogram(
        data=np.array(soma_apical_r2),
        bins=b,
        stat="probability",
        title="Soma-Apical r-squared",
        xtitle="R-squared value",
        xlim=(0, 1),
        color=color[1],
        alpha=0.8,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["H"],
        save=False,
        save_path=None,
    )

    # Basal-Basal Data
    ## Basal-Basal example 1
    plot_scatter_correlation(
        x_var=basal_basal_ex1,
        y_var=basal_basal_other_ex1,
        CI=None,
        title="Basal Ex.1",
        xtitle=f"Basal 1 {title_suffix}",
        ytitle=f"Apical 2 {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color=color[2],
        edge_color="white",
        line_color=color[2],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["I"],
        save=False,
        save_path=None,
    )
    ## Basal-Basal example 2
    plot_scatter_correlation(
        x_var=basal_basal_ex2,
        y_var=basal_basal_other_ex2,
        CI=None,
        title="Basal-Basal Ex.2",
        xtitle=f"Basal 1 {title_suffix}",
        ytitle=f"Basal 2 {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color=color[2],
        edge_color="white",
        line_color=color[2],
        line_width=1.5,
        s_alpha=s_alpha,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["J"],
        save=False,
        save_path=None,
    )
    ## Basal-Basal all
    plot_scatter_correlation(
        x_var=all_basal_basal_amps,
        y_var=all_other_basal_amps,
        CI=None,
        title="All Basal-Basal Pairs",
        xtitle=f"Dendrite {title_suffix}",
        ytitle=f"Sister Dendrite {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color="cmap",
        edge_color="white",
        line_color=color[2],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["K"],
        save=False,
        save_path=None,
    )
    ## Basal-Basal r squared histogram
    plot_histogram(
        data=np.array(basal_basal_r2),
        bins=b,
        stat="probability",
        title="Basal-Basal r-squared",
        xtitle="R-squared value",
        xlim=(0, 1),
        color=color[2],
        alpha=0.8,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["L"],
        save=False,
        save_path=None,
    )

    # Soma-Basal Data
    ## Soma-basal example 1
    plot_scatter_correlation(
        x_var=soma_b_soma_ex1,
        y_var=soma_basal_ex1,
        CI=None,
        title="Soma-Basal Ex.1",
        xtitle=f"Soma {title_suffix}",
        ytitle=f"Basal {title_suffix}",
        ylim=corr_lim,
        xlim=corr_lim,
        marker_size=s_size,
        face_color=color[1],
        edge_color="white",
        line_color=color[1],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["M"],
        save=False,
        save_path=None,
    )
    ## Soma-Basal example 2
    plot_scatter_correlation(
        x_var=soma_b_soma_ex2,
        y_var=soma_basal_ex2,
        CI=None,
        title="Soma-Basal Ex.2",
        xtitle=f"Soma {title_suffix}",
        ytitle=f"Basal {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color=color[1],
        edge_color="white",
        line_color=color[1],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["N"],
        save=False,
        save_path=None,
    )
    ## Soma-Basal all
    plot_scatter_correlation(
        x_var=all_b_soma_amps,
        y_var=all_basal_amps,
        CI=None,
        title="All Soma-Basal Pairs",
        xtitle=f"Soma {title_suffix}",
        ytitle=f"Basal {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color="cmap",
        edge_color="white",
        line_color=color[1],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["O"],
        save=False,
        save_path=None,
    )
    ## Soma-Basal r squared histogram
    plot_histogram(
        data=np.array(soma_basal_r2),
        bins=b,
        stat="probability",
        title="Soma-Basal r-squared",
        xtitle="R-squared value",
        xlim=(0, 1),
        color=color[1],
        alpha=0.8,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["P"],
        save=False,
        save_path=None,
    )

    # Apical-Basal Data
    ## Apical-Basal example 1
    plot_scatter_correlation(
        x_var=apical_basal_ex1,
        y_var=apical_basal_other_ex1,
        CI=None,
        title="Apical-Basal Ex.1",
        xtitle=f"Apical {title_suffix}",
        ytitle=f"Basal {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color=color[3],
        edge_color="white",
        line_color=color[3],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["Q"],
        save=False,
        save_path=None,
    )
    ## Apical-Basal example 2
    plot_scatter_correlation(
        x_var=apical_basal_ex2,
        y_var=apical_basal_other_ex2,
        CI=None,
        title="Apical-Basal Ex.2",
        xtitle=f"Apical {title_suffix}",
        ytitle=f"Basal {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color=color[3],
        edge_color="white",
        line_color=color[3],
        line_width=1.5,
        s_alpha=s_alpha,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["R"],
        save=False,
        save_path=None,
    )
    ## Apical-Basal all
    plot_scatter_correlation(
        x_var=all_apical_basal_amps,
        y_var=all_other_apical_across_amps,
        CI=None,
        title="All Apical-Basal Pairs",
        xtitle=f"Apical {title_suffix}",
        ytitle=f"Basal {title_suffix}",
        xlim=corr_lim,
        ylim=corr_lim,
        marker_size=s_size,
        face_color="cmap",
        edge_color="white",
        line_color=color[3],
        s_alpha=s_alpha,
        line_width=1.5,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["S"],
        save=False,
        save_path=None,
    )
    ## Apical-Basal r squared histogram
    plot_histogram(
        data=np.array(apical_basal_r2),
        bins=b,
        stat="probability",
        title="Apical-Basal r-squared",
        xtitle="R-squared value",
        xlim=(0, 1),
        color=color[3],
        alpha=0.8,
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
        fname = os.path.join(save_path, "Soma_Apical_Basal_Amplitudes_Plot")
        fig.savefig(fname + ".pdf")


def plot_ind_events(
    dataset,
    colors=["forestgreen", "black"],
    norm=False,
    event_type="coactive",
    figsize=(4, 4),
    save=False,
    save_path=None,
):
    """Function to plot the traces of individual events that are either considered
    coactive or non-coactive by event detection

    INPUT PARAMETERS
        dataset - Dual_Plane_Data dataclass containing all the relevant data

        colors - list of str specifying the colors to plot the different traces
                (dendritic and somatic)

        norm - boolean specifying whether to plot normalized or raw traces

        event_type - str specifying whether to plot 'coactve' or
                    'noncoactive' events

        figsize - tuple specifying how large you roughly want each plot
                    to be

        save - boolean specifying whether to save the figure or not

        save_path - str specifying where to save the data
    """
    # Grab the relevant traces
    if norm:
        normalize = " normalized "
        trace_title = "Normalized dF/F"
        if event_type == "coactive":
            dend_traces = dataset.coactive_dendrite_traces_norm
            soma_traces = dataset.coactive_somatic_traces_norm
        elif event_type == "noncoactive":
            dend_traces = dataset.noncoactive_dendrite_traces_norm
            soma_traces = dataset.noncoactive_somatic_traces_norm
    else:
        normalize = " "
        trace_title = "dF/F"
        if event_type == "coactive":
            dend_traces = dataset.coactive_dendrite_traces
            soma_traces = dataset.coactive_somatic_traces
        elif event_type == "noncoactive":
            dend_traces = dataset.noncoactive_dendrite_traces
            soma_traces = dataset.noncoactive_somatic_traces

    # Organize traces
    plot_dend_traces = []
    plot_soma_traces = []
    event_names = []
    for i, (dend, soma) in enumerate(zip(dend_traces, soma_traces)):
        for j in range(dend.shape[1]):
            event_names.append(f"Dendrite {i} event {j}")
            plot_dend_traces.append(dend[:, j])
            plot_soma_traces.append(soma[:, j])

    # Set up the overall title
    title = f"{event_type} event{normalize}traces"

    # Setup the figure
    col_num = 3
    total = len(plot_dend_traces)
    print(total)
    row_num = total // col_num
    row_num += total % col_num
    fig_size = (figsize[0] * col_num, figsize[1] * row_num)
    fig = plt.figure(figsize=fig_size)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(title)

    # Begin plotting
    for i, (dend, soma, name) in enumerate(
        zip(plot_dend_traces, plot_soma_traces, event_names)
    ):
        ## Add subplot
        ax = fig.add_subplot(row_num, col_num, i + 1)
        sems = np.zeros(len(dend))
        ## Make individual plot
        plot_mean_activity_traces(
            means=[soma, dend],
            sems=[sems, sems],
            group_names=["Soma", "Dendrite"],
            sampling_rate=dataset.sampling_rate,
            activity_window=(-1, 2),
            avlines=None,
            ahlines=None,
            figsize=(5, 5),
            colors=colors,
            title=name,
            ytitle=trace_title,
            ylim=None,
            axis_width=1.5,
            minor_ticks="both",
            tick_len=3,
            ax=ax,
            save=False,
            save_path=None,
        )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Soma_Dendrite_Individual_Traces")
        fig.savefig(fname + ".pdf")
