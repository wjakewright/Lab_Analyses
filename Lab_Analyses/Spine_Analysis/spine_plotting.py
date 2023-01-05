"""Module to perform plotting of spine and coactivity activity data"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from Lab_Analyses.Utilities import data_utilities as d_utils

sns.set()
sns.set_style("ticks")


def plot_sns_scatter_correlation(
    var1,
    var2,
    CI=None,
    title=None,
    xtitle=None,
    ytitle=None,
    figsize=(5, 5),
    xlim=None,
    ylim=None,
    marker_size=3,
    face_color="mediumblue",
    edge_color="mediumblue",
    edge_width=0.3,
    s_alpha=0.5,
    line_color="mediumblue",
    line_width=1,
    save=False,
    save_path=None,
):
    """Function to plot a general correlation between two variables
        Uses seaborn regplot to plot line and confidence interval
        
        INPUT PARAMETERS
            var1 - np.array of the x variable to plot
            
            var2 - np.array of the y variable to plot

            CI - int specifying the confidence interval to use

            title - str specifying the title of the plot

            xtitle - str specifying the title of the x axis

            ytitle - str specifying the title of the y axis
            
            figsize - tuple of the size to set the figure
            
            xlim - tuple specifying the limits (left, right) of the x axis
            
            ylim - tuple specifying the limits (bottom, top) of the y axis
            
            face_color - str specifying what color you wish the scatter points to be

            edge_color - str specifying what color you wish the scatter edges to be

            edge_width - float specifying how thick the edges of the scatter should be

            s_alpha - float specifying how transparent the points should be

            line_color - str specifying what color the regression line should be
            
            line_width - float specifying how thick the regression line should be

            save - boolean specifying if you wish to save the figure
            
            save_path - str specifying the path of where to save the figure
    """

    # Make the figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontweight="bold")

    # Perform correlation of the data
    corr, p = stats.pearsonr(var1, var2)
    # Add correlation as a subtitle
    subtitle = f"r = {corr}    p = {p}"
    plt.title(subtitle, style="italic")

    # Set up some styles for the scatter points
    scatter_kws = {
        "facecolor": face_color,
        "edgecolor": edge_color,
        "alpha": s_alpha,
        "linewidth": edge_width,
        "s": marker_size,
    }
    line_kws = {"linewidth": line_width, "color": line_color}

    # Make the plot
    sns.regplot(x=var1, y=var2, ci=CI, scatter_kws=scatter_kws, line_kws=line_kws)

    plt.xlabel(xtitle, labelpad=15)
    if xlim:
        plt.xlim(left=xlim[0], right=xlim[1])
    plt.ylabel(ytitle, labelpad=15)
    if ylim:
        plt.ylim(bottom=ylim[0], top=ylim[1])

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def plot_swarm_bar_plot(
    data_dict,
    mean_type="mean",
    err_type="sem",
    figsize=(5, 5),
    title=None,
    xtitle=None,
    ytitle=None,
    ylim=None,
    b_colors="mediumblue",
    b_edgecolors="black",
    b_err_colors="black",
    b_width=0.5,
    b_linewidth=0,
    b_alpha=0.3,
    s_colors="mediumblue",
    s_size=5,
    s_alpha=0.8,
    ahlines=None,
    save=False,
    save_path=None,
):
    """General function to plot sns swarm plots with overlying mean and error
        
        INPUT PARAMETERS
            data_dict - dictionary of data to be plotted. Keys will serve as x values for
                        the different groups

            mean_type - str specifying what central point you wish to plot. Accepts
                        'mean' or 'median'
            
            err_type - str specifying what type of error you wish the error bars to 
                        represent. Accepts 'sem', 'std', and 'CI'
            
            marker - str specifying what type of marker you wish to represent the mean
            
            figsize - tuple specifying the figure size
            
            title - str specifying the name of the title
            
            xtitle - str specifying the x axis label
            
            ytitle - str specifying the y axis label
            
            ylim - tuple specifying the limits of the y axis

            m_colors - 
            
            s_colors - str or list of strs specifying the colors of each plot. If only one
                    color is given, all plots will be the same color
                    
            s_alpha - float specifying what level of transparency the scatter points should be
            
            save - boolean specifying wheather to save the figure or not
            
            save_path - str specifying where to save the figure
    """
    # Make list of colors if only one is provided
    if type(m_colors) == str:
        m_colors = [m_colors for i in range(len(list(data_dict.keys())))]
    if type(s_colors) == str:
        s_colors = [s_colors for i in range(len(list(data_dict.keys())))]

    # Make the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.set_title(title)

    # Set up data
    groups = list(data_dict.keys())
    x = list(range(len(groups)))
    data_points = list(data_dict.values())

    # Calculate the appropriate means and errors
    if mean_type == "mean":
        data_mean = [np.nanmean(i) for i in data_points]
    elif mean_type == "median":
        data_mean = [np.nanmedian(i) for i in data_points]
    else:
        return "Only accepts mean and median for mean_type!!!"
    if err_type == "sem":
        data_sems = [stats.sem(i, nan_policy="omit") for i in data_points]
    elif err_type == "std":
        data_sems = [np.nanstd(i) for i in data_points]
    elif err_type == "CI":
        num_p = len(data_points[0])
        data_sems = []
        if num_p <= 30:
            for data in data_points:
                ci = stats.t.interval(
                    alpha=0.95,
                    df=len(data) - 1,
                    loc=np.nanmean(data),
                    scale=stats.sem(data, nan_policy="omit"),
                )

                # ci = sm.DescrStatsW(data).tconfint_mean()
                s = np.array([ci[0], ci[1]]).reshape(-1, 1)
                data_sems.append(s)
        else:
            for data in data_points:
                ci = stats.norm.interval(
                    alpha=0.95,
                    loc=np.nanmean(data),
                    scale=stats.sem(data, nan_policy="omit"),
                )
                s = np.array([ci[0], ci[1]]).reshape(-1, 1)
                data_sems.append(s)

    data_df = pd.DataFrame.from_dict(data_dict, orient="index")
    data_df = data_df.T

    # Plot the points
    sns.stripplot(data=data_df, palette=s_colors, alpha=s_alpha, zorder=0, size=s_size)

    # Plot means
    plt.bar(
        x=x,
        height=data_mean,
        yerr=data_sems,
        width=b_width,
        color=b_colors,
        edgecolor=b_edgecolors,
        ecolor=b_err_colors,
        linewidth=b_linewidth,
        alpha=b_alpha,
    )

    # Format axes
    sns.despine()
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)
    ax.set_xticklabels(labels=groups)

    if ahlines:
        for line in ahlines:
            ax.axhline(y=line, linestyle="--", linewidth=1)

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def plot_grouped_swarm_bar_plot(
    data_dict,
    groups,
    mean_type="mean",
    err_type="sem",
    figsize=(5, 5),
    title=None,
    xtitle=None,
    ytitle="value",
    ylim=None,
    b_colors="mediumblue",
    b_edgecolors="black",
    b_err_colors="black",
    b_width=0.5,
    b_linewidth=0,
    b_alpha=0.3,
    s_colors="mediumblue",
    s_size=5,
    s_alpha=0.3,
    ahlines=None,
    save=False,
    save_path=None,
):
    """Function to plot grouped sns swarm plots with overlaying mean and error
        INPUT PARAMETERS
            data_dict - nested dictionary of data to be plotted. Outer keys represent the
                        subgroups, while the inner keys represent the main groups
            
            groups - list of strings representing the groups. Main group will be first and 
                    subgroups will be listed second

            mean_type - str specifying what central point you wish to plot. Accepts
                        'mean' or 'median'
            
            err_type - str specifying what type of error you wish the error bars to 
                        represent. Accepts 'sem', 'std', and 'CI'
            
            marker - str specifying what type of marker you wish to represent the mean
            
            figsize - tuple specifying the figure size
            
            title - str specifying the name of the title
            
            xtitle - str specifying the x axis label
            
            ytitle - str specifying the y axis label
            
            ylim - tuple specifying the limits of the y axis

            m_colors - 
            
            s_colors - str or list of strs specifying the colors of each plot. If only one
                    color is given, all plots will be the same color
                    
            s_alpha - float specifying what level of transparency the scatter points should be
            
            save - boolean specifying wheather to save the figure or not
            
            save_path - str specifying where to save the figure
    """
    # Check colors
    if type(s_colors) == str:
        s_colors = [s_colors for i in range(len(list(data_dict.keys())))]
    elif (type(s_colors) == list) & (len(s_colors) != len(list(data_dict.keys()))):
        return "Number of scatter colors does not match number of groups"
    if type(m_colors) == str:
        m_colors = [m_colors for i in range(len(list(data_dict.keys())))]
    elif (type(m_colors) == list) & (len(m_colors) != len(list(data_dict.keys()))):
        return "Number of marker colors does not match number of groups"

    # Organize the data to be plotted
    dfs = []
    g1_keys = []
    g2_keys = []
    for key, value in data_dict.items():
        g1_keys = list(value.keys())
        g2_keys.append(key)
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in value.items()]))
        df = pd.melt(df, value_vars=df.columns, var_name=groups[0], value_name=ytitle)
        df[groups[1]] = key
        dfs.append(df)
    plot_df = pd.concat(dfs)

    # Calculate means and errors
    data_mean = []
    data_err = []
    for k1 in g1_keys:
        for k2 in g2_keys:
            data = plot_df[ytitle].loc[
                (plot_df[groups[0]] == k1) & (plot_df[groups[1]] == k2)
            ]
            # Get approriate mean
            if mean_type == "mean":
                data_mean.append(np.nanmean(data))
            elif mean_type == "median":
                data_mean.append(np.nanmedian(data))
            else:
                return "Only accepts mean and median for mean_type!!!"
            # Get appropriate error
            if err_type == "sem":
                data_err.append(stats.sem(data, nan_policy="omit"))
            elif err_type == "std":
                data_err.append(np.nanstd(data))
            elif err_type == "CI":
                num_p = len(data)
                if num_p <= 30:
                    ci = stats.t.interval(
                        alpha=0.95,
                        df=len(data) - 1,
                        loc=np.nanmean(data),
                        scale=stats.sem(data, nan_policy="omit"),
                    )
                    s = np.array([ci[0], ci[1]]).reshape(-1, 1)
                    data_err.append(s)
                else:
                    ci = stats.norm.interval(
                        alpha=0.95,
                        loc=np.nanmean(data),
                        scale=stats.sem(data, nan_policy="omit"),
                    )
                    s = np.array([ci[0], ci[1]]).reshape(-1, 1)
                    data_err.append(s)

    # Make the plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.set_title(title)

    # Plot the scatter points
    strip = sns.stripplot(
        data=plot_df,
        x=groups[0],
        y=ytitle,
        hue=groups[1],
        palette=s_colors,
        dodge=True,
        ax=ax,
        zorder=0,
        size=s_size,
        alpha=s_alpha,
    )
    # Get x positions
    x_coords = []
    for c in strip.collections:
        offsets = c.get_offsets()
        if len(offsets) != 0:
            xs = [x[0] for x in offsets]
            x_coords.append(np.nanmean(xs))

    err_colors = [m_colors for x in g1_keys]
    err_colors = [x for y in err_colors for x in y]
    # Plot the mean and error
    plt.bar(
        x=x_coords,
        height=data_mean,
        yerr=data_err,
        width=b_width,
        color=b_colors,
        edgecolor=b_edgecolors,
        ecolor=b_err_colors,
        linewidth=b_linewidth,
        alpha=b_alpha,
    )

    # Format axes
    sns.despine()
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)

    if ahlines:
        for line in ahlines:
            ax.axhline(y=line, linestyle="--", linewidth=1)

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def mean_and_lines_plot(
    data_dict,
    plot_ind=True,
    figsize=(5, 5),
    title=None,
    xtitle=None,
    ytitle=None,
    m_color="mediumblue",
    l_colors="mediumblue",
    ylim=None,
    l_alpha=0.5,
    save=False,
    save_path=None,
):
    """General function to plot means and individual data points over time, with
        connecting lines
        
        INPUT PARAMETERS
            data_dict - dict of data to be plotted, with each item representing 
                        a different group
            
            plot_ind - boolean of whether to plot individual values or not
            
            figsize - tuple specifying the figure size
            
            title - str specifying the title of the figure
            
            xtitle - str specifying the title of the x axis
            
            ytitle - str specifying the title of the y axis
            
            m_color - str specifying the color of the mean markers
            
            l_colors - str or list of str specifying the colors of the individual
                        data point lines
            
            ylim - tuple specifying the limits of the y axis
            
            l_alpha - float specifying the transparency of the individual lines
            
            save - boolean specifying whather or not to save the figure
            
            save_path - str specifying where to save the figure
    """
    # Make list of colors if only one is provided
    if type(l_colors) == str:
        l_colors = [l_colors for i in range(len(list(data_dict.values())[0]))]

    # Make the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax.set_title(title)
    # Set up data
    groups = list(data_dict.keys())
    x = list(range(len(groups)))
    data_points = list(data_dict.values())
    data_mean = [np.nanmean(i) for i in data_points]
    data_sems = [stats.sem(i, nan_policy="omit") for i in data_points]

    # Plot means
    ax.errorbar(
        x,
        data_mean,
        data_sems,
        color=m_color,
        marker="o",
        markerfacecolor=m_color,
        ecolor=m_color,
    )
    # Plot the individual values
    if plot_ind:
        for i, data in enumerate(list(zip(*data_points))):
            plt.plot(x, data, color=l_colors[i], alpha=l_alpha)

    # Format axes
    sns.despine()
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)
    ax.set_xticks(ticks=x, labels=groups)

    fig.tight_layout()

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def plot_mean_activity_traces(
    mean,
    sem,
    group_names=None,
    sampling_rate=60,
    activity_window=(-2, 3),
    avlines=None,
    figsize=(5, 5),
    colors="mediumblue",
    title=None,
    ytitle=None,
    ylim=None,
    save=False,
    save_path=None,
):
    """Function to plot mean activity traces
        
        INPUT PARAMTERS
            mean - np.array of mean trace or a list of arrays for multiple groups
            
            sem - np.array of sem trace or a list of arrays for multiple groups

            sampling_rate - int specifing the imaging sampling rate
            
            avlines - tuple of x values for vertical lines
            
            color - str to specify the color
            
            title - str for the figure title
            
            ytitle - title to label the y axis
            
            ylim - tuple specifying the limit of the y axis
            
            save - boolean of whether to not save the figure
            
            save_path - str of where to save the figure
            
    """
    fig = plt.figure(figsize=figsize)
    if type(mean) == np.ndarray:
        x = np.linspace(activity_window[0], activity_window[1], len(mean))
        plt.plot(x, mean, color=colors)
        plt.fill_between(x, mean - sem, mean + sem, color=colors, alpha=0.2)
    elif type(mean) == list:
        for m, s, c, n in zip(mean, sem, colors, group_names):
            x = np.linspace(activity_window[0], activity_window[1], len(m))
            plt.plot(x, m, color=c, label=n)
            plt.fill_between(x, m - s, m + s, color=c, alpha=0.2)
    if avlines:
        for line in avlines:
            plt.axvline(x=line / sampling_rate, linestyle="--", color="black")
    if ylim:
        plt.ylim(bottom=ylim[0], top=ylim[1])
    plt.title(title)
    plt.ylabel(ytitle)
    plt.xlabel("Time (s)")
    plt.xticks(
        ticks=[activity_window[0], 0, activity_window[1]],
        labels=[activity_window[0], 0, activity_window[1]],
    )
    plt.tick_params(axis="both", which="both", direction="in", length=4)
    plt.legend(loc="upper left")
    fig.tight_layout()
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def ind_mean_activity_traces(
    mean_list,
    sem_list,
    sampling_rate=60,
    activity_window=(-2, 3),
    avlines=None,
    figsize=(10, 4),
    color="mediumblue",
    title=None,
    ytitle=None,
    ylim=None,
    save=False,
    save_path=None,
):
    """Function to plot all of the mean activity traces across all 
        spiens
        
        INPUT PARAMETERS
            mean_list - list of np.arrays contianing the mean activity
                        trace for each spine
            
            sem_list - list of np.arrays containing the sem of activity
                        traces for each spine
            
            sampling_rate - int or float specifying the sampling rate
            
            avlines - list of tuples of lines to label on the x axis (e.g., onsets)
            
            figsize - tuple specifying the size of the figure
            
            color - str specifying the color of the traces
            
            title - str specifying the title of the plot
            
            ytitle - str specifying the title of the y axis
            
            ylim - tuple specifying the limits of the y axis
            
            save - boolean of whetehr or not to save the graph
            
            save_path - str of the path of where to save the graph
            
    """
    # Set up the subplots
    tot = len(mean_list)
    COL_NUM = 4
    row_num = tot // COL_NUM
    row_num += tot % COL_NUM
    fig_size = (figsize[0], figsize[1] * row_num)
    fig = plt.figure(figsize=fig_size)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(title)

    # Get max amplitude in order to set ylim
    if ylim is None:
        maxes = []
        mins = []
        for mean, sem in zip(mean_list, sem_list):
            maxes.append(np.max(mean) + np.max(sem))
            mins.append(np.min(mean) - np.max(sem))
        ylim = (np.max(maxes), np.min(mins))

    if avlines is None:
        avlines = [None for i in mean_list]

    count = 1
    # Make individual plots
    for mean, sem, avline in zip(mean_list, sem_list, avlines):
        x = np.linspace(activity_window[0], activity_window[1], len(mean))
        ax = fig.add_subplot(row_num, COL_NUM, count)
        ax.plot(x, mean, color=color)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
        if avline:
            for line in avline:
                ax.avline(x=line, linestyle="--", color="black")
        plt.xticks(
            ticks=[activity_window[0], 0, activity_window[1]],
            labels=[activity_window[0], 0, activity_window[1]],
        )
        plt.xlabel("Time (s)")
        plt.ylim(bottom=ylim[0], top=ylim[1])
        plt.ylabel(ytitle)
        plt.tick_params(axis="both", which="both", direction="in", length=4)
        count += 1

    fig.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def plot_multi_mean_activity_traces(
    mean_dict,
    sem_dict,
    trace_types,
    activity_window=(-2, 3),
    avline_dict=None,
    figsize=(8, 4),
    colors=["mediumblue", "firebrick"],
    title=None,
    ytitle=None,
    ylim=None,
    save=False,
    save_path=None,
):
    """Function to plot multiple activity trace plots in a single subplot figure"""
    # Set up subplot
    tot = len(list(mean_dict.values()))
    COL_NUM = 4
    row_num = tot // COL_NUM
    row_num += tot % COL_NUM
    fig_size = (figsize[0], figsize[1] * row_num)
    fig = plt.figure(figsize=fig_size)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(title)

    count = 1
    plot_groups = list(mean_dict.keys())
    for group in plot_groups:
        means = mean_dict[group]
        sems = sem_dict[group]
        try:
            avlines = avline_dict[group]
        except:
            avlines = [None for x in means]
        ax = fig.add_subplot(row_num, COL_NUM, count)
        ax.set_title(group)
        for m, s, c, t, a in zip(means, sems, colors, trace_types, avlines):
            x = np.linspace(activity_window[0], activity_window[1], len(m))
            ax.plot(x, m, color=c)
            ax.fill_between(x, m - s, m + s, color=c, alpha=0.2, label=t)
            ax.axvline(x=a, linestyle="--", color=c)
        plt.xticks(
            ticks=[activity_window[0], 0, activity_window[1]],
            labels=[activity_window[0], 0, activity_window[1]],
        )
        plt.xlabel("Time (s)")
        plt.ylabel(ytitle)
        plt.tick_params(axis="both", which="both", direction="in", length=4)
        if ylim:
            plt.ylim(bottom=ylim[0], top=ylim[1])
        count += 1

    fig.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def plot_histogram(
    data,
    bins,
    avlines=None,
    title=None,
    xtitle=None,
    figsize=(5, 5),
    color="mediumblue",
    alpha=0.4,
    save=False,
    save_path=None,
):
    """Function to plot a single variable as a histogram
        
        INPUT PARAMETERS
            data - np.array of the data to be plotted

            bins - int specifying the number of bins to plot
            
            avline - tuple specifying were to draw vertical lines if desired
            
            title - str specifying the plots title
            
            xtitle - str specifying the title of the x axis
            
            figsize - tuple specifying the size of the figure
            
            color - str specifyin the color to make the histogram

            save - boolean specifying whether to save the figure or not

            save_path - str specifying where to save the figure
            
    """

    plt.figure(figsize=figsize)
    if type(data) == list:
        for d, c in zip(data, color):
            plt.hist(d, bins, color=c, alpha=alpha)
    else:
        plt.hist(data, bins, color=color, alpha=alpha)
    if avlines:
        for line in avlines:
            plt.axvline(line, linestyle="--", color="black")
    plt.title(title)
    plt.xlabel(xtitle)

    plt.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def plot_spine_coactivity_distance(
    data_dict,
    bins,
    colors,
    title_suff=None,
    figsize=(5, 5),
    ylim=None,
    ytitle=None,
    save=False,
    save_path=None,
):
    """Function to plot the distance dependent spine coactivity
    
        INPUT PARAMETERS
            data_dict - dict of 2d np.arrays of the data to be plotted. Each column
                        represents a spine
            
            bins - list or np.array of the distances the data is binned over
            
            colors - list of colors for each group

            title_suff - str specifying additional info for the graph title
            
            figsize - tuple specifying the size of the figure
            
            ylim - tuple specifying the limits of the y axis
            
            save - boolean specifying whether to save the figure of not
            
            save_path - str specifying where to save the figure
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    title = "Distance-depdendent spine coactivity"
    if title_suff:
        title = title + " " + title_suff
    ax.set_title(title)
    x = list(range(len(bins)))

    for i, (key, value) in enumerate(data_dict.items()):
        mean = np.nanmean(value, axis=1)
        sem = stats.sem(value, axis=1, nan_policy="omit")

        ax.errorbar(
            x, mean, yerr=sem, color=colors[i], linestyle="-", label=key,
        )

    plt.xticks(ticks=x, labels=bins)
    plt.xlabel("Distance (um)")
    plt.ylabel(ytitle)
    if ylim:
        plt.ylim(bottom=ylim[0], top=ylim[1])
    sns.despine()
    fig.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def plot_spine_heatmap(
    data_dict,
    figsize=(4, 5),
    sampling_rate=60,
    activity_window=(-2, 2),
    title=None,
    cbar_label=None,
    hmap_range=None,
    center=None,
    sorted=None,
    normalize=False,
    cmap="plasma",
    save=False,
    save_path=None,
):
    """Function to plot trial averaged spine activity as a heatmap.
        Allows for multiple subplots for different groups

        INPUT PARAMETERS
            data_dict - dict of data, with each item containing data for a 
                        specific group (2d np.array, columns=rois)
            
            figsize - tuple specifying the size of the figure

            sampling_rate - int or float specifying the imaging sampling rate

            activity_window - tuple specifying the range of the activity window in sec

            title - str specifying the title of the figure

            cbar_label - str specifying the label for the color bar

            hmap_rate - tuple specifying a min and max of the heatmap

            center - int specifying the center of the color map

            sorted - str specifying if and how to sort the data. 
                    Accepts 'peak' and 'difference' to sort the data
                    based on their peak timing or the difference in their
                    activity before and after the center point

            normalize - boolean specifying if you wish to normalize activity
                        for each roi

            cmap - str specifying the color map to use

            save - boolean specifying whether to save the figure

            save_path - str specifying where to save the figure
        
    """
    # Set up subplots
    tot = len(data_dict.keys())
    COL_NUM = 4
    row_num = tot // COL_NUM
    row_num += tot % COL_NUM
    fig_size = (figsize[0], figsize[1] * row_num)

    fig = plt.figure(figsize=fig_size)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(title)

    if hmap_range is None:
        hmap_range = (None, None)
        cbar_ticks = None
    elif hmap_range is not None and center is None:
        cbar_ticks = (hmap_range[0], hmap_range[1])
    else:
        cbar_ticks = (hmap_range[0], center, hmap_range[1])

    # Plot each group
    for count, (key, value) in enumerate(data_dict.items()):
        # Process some of the data
        data = value
        if sorted == "peak":
            data = d_utils.peak_sorting(data)
        if sorted == "difference":
            data = d_utils.diff_sorting(
                data.T, np.absolute(activity_window), sampling_rate
            ).T
        if normalize:
            data = d_utils.peak_normalize_data(data)
        data_t = data.T

        # Plot
        ax = fig.add_subplot(row_num, COL_NUM, count + 1)
        ax.set_title(key)
        hax = sns.heatmap(
            data_t,
            cmap=cmap,
            center=center,
            vmax=hmap_range[1],
            vmin=hmap_range[0],
            cbar_kws={
                "label": cbar_label,
                "orientation": "vertical",
                "ticks": cbar_ticks,
            },
            yticklabels=False,
            ax=ax,
        )
        x = np.linspace(activity_window[0], activity_window[1], data_t.shape[1])
        xticks = np.unique(x.astype(int))
        plt.xticks(ticks=xticks, labels=xticks)
        plt.xlabel("Time (s)")
        hax.patch.set_edgecolor("black")
        hax.patch.set_linewidth("2.5")

    fig.tight_layout()

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")

