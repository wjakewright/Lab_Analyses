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
        dend_traces = dataset.dendrite_dFoF_norm
        soma_traces = dataset.somatic_dFoF_norm
    else:
        title = "Somatic and Dendritic Traces"
        ytitle = "dF/F"
        dend_traces = dataset.dendrite_dFoF
        soma_traces = dataset.somatic_dFoF

    # Make the figure
    row_num = dend_traces.shape[1]
    fig_size = (10, 4 * row_num)
    fig = plt.figure(figsize=fig_size)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(title)
    fig.tight_layout()

    # Add subplots
    for i in range(dend_traces.shape[1]):
        d_trace = dend_traces[:, i]
        s_trace = soma_traces[:, i]
        ax = fig.add_subplot(row_num, 1, i + 1)
        ## Plot full traces
        if subselect is None:
            x = np.arange(len(d_trace)) / dataset.sampling_rate
            ax.plot(x, d_trace, color=colors[0], label="Dendrite")
            ax.plot(x, s_trace, color=colors[1], label="Soma")
        ## Plot subselected traces
        else:
            x = (
                np.arange(len(d_trace[subselect[0] : subselect[1]]))
                / dataset.sampling_rate
            )
            ax.plot(
                x,
                d_trace[subselect[0] : subselect[1]],
                color=colors[0],
                label="Dendrite",
            )
            ax.plot(
                x, s_trace[subselect[0] : subselect[1]], color=colors[1], label="Soma",
            )

        # Set up the axes
        ax.set_title(f"Dendrite {i + 1}", fontsize=10)
        adjust_axes(ax, None, "Time (s)", ytitle, None, None, 3, 1.5)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")


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
    soma_points = dataset.fraction_somatic_active
    dend_points = dataset.fraction_dendrite_active

    # Construct the subplot
    fig, axes = plt.subplot_mosaic(
        [["left", "top right"], ["left", "bottom right"]],
        constrained_layout=True,
        figsize=figsize,
    )

    fig.suptitle("Somatic and Dendritic Coactivity")

    # Plot data on the axes
    ## First plot fraction coactive
    plot_swarm_bar_plot(
        data_dict={"Soma": soma_points, "Dendrite": dend_points},
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
    ## Plot coactive and non-coactive traces
    ### Get the trace data
    if norm:
        dend_coactive = dataset.coactive_dendrite_traces_norm
        dend_noncoactive = dataset.noncoactive_dendrite_traces_norm
        soma_coactive = dataset.coactive_somatic_traces_norm
        soma_noncoactive = dataset.noncoactive_somatic_traces_norm
        trace_ytitle = "Normalized dF/F"
    else:
        dend_coactive = dataset.coactive_dendrite_traces
        dend_noncoactive = dataset.noncoactive_dendrite_traces
        soma_coactive = dataset.coactive_somatic_traces
        soma_noncoactive = dataset.noncoactive_somatic_traces
        trace_ytitle = "dF/F"
    ### Get the means and sems
    if trace_avg == "all":
        d_coactive = np.hstack(dend_coactive)
        d_noncoactive = np.hstack(dend_noncoactive)
        s_coactive = np.hstack(soma_coactive)
        s_noncoactive = np.hstack(soma_noncoactive)
    elif trace_avg == "dend":
        d_c_means = []
        d_n_means = []
        s_c_means = []
        s_n_means = []
        for dc, dn, sc, sn in zip(
            dend_coactive, dend_noncoactive, soma_coactive, soma_noncoactive
        ):
            d_c_means.append(np.nanmean(dc, axis=1))
            d_n_means.append(np.nanmean(dn, axis=1))
            s_c_means.append(np.nanmean(sc, axis=1))
            s_n_means.append(np.nanmean(sn, axis=1))
        d_coactive = np.vstack(d_c_means).T
        d_noncoactive = np.vstack(d_n_means).T
        s_coactive = np.vstack(s_c_means).T
        s_noncoactive = np.vstack(s_n_means).T

    dend_co_mean = np.nanmean(d_coactive, axis=1)
    dend_non_mean = np.nanmean(d_noncoactive, axis=1)
    soma_co_mean = np.nanmean(s_coactive, axis=1)
    soma_non_mean = np.nanmean(s_noncoactive, axis=1)
    dend_co_sem = stats.sem(d_coactive, axis=1, nan_policy="omit")
    dend_non_sem = stats.sem(d_noncoactive, axis=1, nan_policy="omit")
    soma_co_sem = stats.sem(s_coactive, axis=1, nan_policy="omit")
    soma_non_sem = stats.sem(s_noncoactive, axis=1, nan_policy="omit")

    ### Plot coactive traces
    plot_mean_activity_traces(
        means=[soma_co_mean, dend_co_mean],
        sems=[soma_co_sem, dend_co_sem],
        group_names=["Soma", "Dendrite"],
        sampling_rate=dataset.sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=colors,
        title="Coactive Events",
        ytitle=trace_ytitle,
        ylim=None,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["top right"],
        save=False,
        save_path=None,
    )
    ### Plot noncoactive traces
    ylim = axes["top right"].get_ylim()  # Ensure same scale
    plot_mean_activity_traces(
        means=[soma_non_mean, dend_non_mean],
        sems=[soma_non_sem, dend_non_sem],
        group_names=["Soma", "Dendrite"],
        sampling_rate=dataset.sampling_rate,
        activity_window=(-1, 2),
        avlines=None,
        ahlines=None,
        figsize=(5, 5),
        colors=colors,
        title="Non-coactive Events",
        ytitle=trace_ytitle,
        ylim=ylim,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=3,
        ax=axes["bottom right"],
        save=False,
        save_path=None,
    )

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Soma_Dendrite_Coactivity_Plot")
        fig.savefig(fname + ".pdf")


def plot_amplitude_correlations(
    dataset,
    norm=True,
    dend_examples=None,
    soma_examples=None,
    color=["forestgreen", "black"],
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
        soma_amps = dataset.somatic_amplitudes_norm
        dend_amps = dataset.dendrite_amplitudes_norm
        other_dend_amps = dataset.other_dendrite_amplitudes_norm
        title_suffix = "norm. dF/F"
    else:
        soma_amps = dataset.somatic_amplitudes
        dend_amps = dataset.dendrite_amplitudes
        other_dend_amps = dataset.other_dendrite_amplitudes
        title_suffix = "dF/F"

    # Combine all data points
    all_soma_amps = np.concatenate(soma_amps)
    all_dend_amps = np.concatenate(dend_amps)
    all_other_dend_amps = np.concatenate([np.concatenate(x) for x in other_dend_amps])
    all_dend_dend_amps = []
    for i, x in enumerate(other_dend_amps):
        for j in x:
            all_dend_dend_amps.append(dend_amps[i])
    all_dend_dend_amps = np.concatenate(all_dend_dend_amps)

    # Get all of the R squared values
    dend_dend_r2 = []
    soma_dend_r2 = []
    for soma, dend, other in zip(soma_amps, dend_amps, other_dend_amps):
        sdc, _ = stats.pearsonr(soma, dend)
        soma_dend_r2.append(sdc ** 2)
        for o in other:
            ddc, _ = stats.pearsonr(dend, o)
            dend_dend_r2.append(ddc ** 2)

    b = np.histogram(np.hstack([dend_dend_r2, soma_dend_r2]), bins=bins)[1]

    # Grab the example data
    ## Get other dend examples
    ### Example 1
    dend_dend_ex1 = dend_amps[dend_examples[0]]
    d_o_ex1 = other_dend_amps[dend_examples[0]]
    ### Get random partner
    dend_other_ex1 = d_o_ex1[random.randint(0, len(d_o_ex1) - 1)]
    ### Example 2
    dend_dend_ex2 = dend_amps[dend_examples[1]]
    d_o_ex2 = other_dend_amps[dend_examples[1]]
    ### Get random partner
    dend_other_ex2 = d_o_ex2[random.randint(0, len(d_o_ex2) - 1)]

    ## Get soma dend examples
    ### Example 1
    soma_dend_ex1 = dend_amps[soma_examples[0]]
    soma_soma_ex1 = soma_amps[soma_examples[0]]
    ### Example 2
    soma_dend_ex2 = dend_amps[soma_examples[1]]
    soma_soma_ex2 = soma_amps[soma_examples[1]]

    # Construct the subplot
    fig, axes = plt.subplot_mosaic(
        [["tll", "tlm", "trm", "trr"], ["bll", "blm", "brm", "brr"]], figsize=figsize,
    )
    fig.suptitle("Paired Somatic and Dendritic Amplitudes")
    fig.subplots_adjust(hspace=1, wspace=0.5)

    ####################### Plot the data on the axes ########################
    # Dendrite-Dendrite Data
    ## Dend-Dend example 1
    plot_scatter_correlation(
        x_var=dend_dend_ex1,
        y_var=dend_other_ex1,
        CI=None,
        title="Dendrite-Dendrite Ex.1",
        xtitle=f"Dendrite 1 {title_suffix}",
        ytitle=f"Dendrite 2 {title_suffix}",
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
        tick_len=2,
        ax=axes["tll"],
        save=False,
        save_path=None,
    )
    ## Dend-Dend example 2
    plot_scatter_correlation(
        x_var=dend_dend_ex2,
        y_var=dend_other_ex2,
        CI=None,
        title="Dendrite-Dendrite Ex.2",
        xtitle=f"Dendrite 1 {title_suffix}",
        ytitle=f"Dendrite 2 {title_suffix}",
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
        tick_len=2,
        ax=axes["tlm"],
        save=False,
        save_path=None,
    )
    ## Dend-Dend all
    plot_scatter_correlation(
        x_var=all_dend_dend_amps,
        y_var=all_other_dend_amps,
        CI=None,
        title="All Dendrite-Dendrite Pairs",
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
        tick_len=2,
        ax=axes["trm"],
        save=False,
        save_path=None,
    )
    ## Dendrite-Dendrite r squared histogram
    plot_histogram(
        data=np.array(dend_dend_r2),
        bins=b,
        stat="probability",
        title="Dendrite-Dendrite r-squared",
        xtitle="R-squared value",
        xlim=(0, 1),
        color=color[0],
        alpha=0.8,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=2,
        ax=axes["trr"],
        save=False,
        save_path=None,
    )

    # Soma-Dendrite Data
    ## Soma-Dend example 1
    plot_scatter_correlation(
        x_var=soma_soma_ex1,
        y_var=soma_dend_ex1,
        CI=None,
        title="Soma-Dendrite Ex.1",
        xtitle=f"Soma {title_suffix}",
        ytitle=f"Dendrite {title_suffix}",
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
        tick_len=2,
        ax=axes["bll"],
        save=False,
        save_path=None,
    )
    ## Soma-Dend example 2
    plot_scatter_correlation(
        x_var=soma_soma_ex2,
        y_var=soma_dend_ex2,
        CI=None,
        title="Soma-Dendrite Ex.2",
        xtitle=f"Soma {title_suffix}",
        ytitle=f"Dendrite {title_suffix}",
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
        tick_len=2,
        ax=axes["blm"],
        save=False,
        save_path=None,
    )
    ## Soma-Dend all
    plot_scatter_correlation(
        x_var=all_dend_amps,
        y_var=all_soma_amps,
        CI=None,
        title="All Soma-Dendrite Pairs",
        xtitle=f"Soma {title_suffix}",
        ytitle=f"Dendrite {title_suffix}",
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
        tick_len=2,
        ax=axes["brm"],
        save=False,
        save_path=None,
    )
    ## Soma-Dendrite r squared histogram
    plot_histogram(
        data=np.array(soma_dend_r2),
        bins=b,
        stat="probability",
        title="Soma-Dendrite r-squared",
        xtitle="R-squared value",
        xlim=(0, 1),
        color=color[1],
        alpha=0.8,
        axis_width=1.5,
        minor_ticks="both",
        tick_len=2,
        ax=axes["brr"],
        save=False,
        save_path=None,
    )

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, "Soma_Dendrite_Amplitudes_Plot")
        fig.savefig(fname + ".pdf")

