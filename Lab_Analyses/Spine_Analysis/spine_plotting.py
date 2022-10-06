"""Module to perform plotting of spine and coactivity activity data"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sm
from scipy import stats

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
    marker="o",
    figsize=(5, 5),
    title=None,
    xtitle=None,
    ytitle=None,
    ylim=None,
    linestyle="",
    m_colors="mediumblue",
    s_colors="mediumblue",
    s_alpha=0.3,
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
    # if type(colors) == str:
    #    colors = [colors for i in range(len(list(data_dict.keys())))]

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
        sem1 = []
        sem2 = []
        if num_p <= 30:
            print("T")
            for data in data_points:
                ci = stats.t.interval(
                    alpha=0.95,
                    df=len(data) - 1,
                    loc=np.nanmean(data),
                    scale=stats.sem(data),
                )

                # ci = sm.DescrStatsW(data).tconfint_mean()
                sem1.append(ci[0])
                sem2.append(ci[1])
        else:
            for data in data_points:
                ci = stats.norm.interval(
                    alpha=0.95,
                    loc=np.nanmean(data),
                    scale=stats.sem(data, nan_policy="omit"),
                )
                sem1.append(ci[0])
                sem2.append(ci[1])
        data_sems = [np.array(sem1), np.array(sem2)]
        print(data_sems)

    data_df = pd.DataFrame.from_dict(data_dict, orient="index")
    data_df = data_df.T

    # Plot the points
    sns.stripplot(data=data_df, palette=s_colors, alpha=s_alpha, zorder=0)

    # Plot means
    ax.errorbar(
        x,
        data_mean,
        data_sems,
        color=m_colors,
        fmt=marker,
        markerfacecolor=m_colors,
        ecolor=m_colors,
        linestyle=linestyle,
    )

    # Format axes
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


def plot_mean_activity_trace(
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
            plt.fill_between(x, m - s, m + s, c, alpha=0.2)
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
    plt.legend(loc="upper right")
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

