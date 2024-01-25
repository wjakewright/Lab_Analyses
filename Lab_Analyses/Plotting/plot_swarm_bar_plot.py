import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.adjust_axes import adjust_axes, get_axis_limit

sns.set()
sns.set_style("ticks")


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
    plot_ind=True,
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """General function for plotting a scatter bar plots

    INPUT PARAMETERS
        data_dict - dictionary of data to be plotted. Keys will serve as x values for
                    the different groups

        mean_type - str specifying what central point to plot. Accepts 'mean' and
                    'median'.

        err_type - str specifying what type of error bars with. Accepts 'sem',
                    'std', and 'CI', which plots bootstraped 95% confidence interval

        figsize - tuple of the size of the figure to set. Only used if independent figure

        title - str specifying the title of the plot

        xtitle - str specifying the title of the x axis

        ytitle - str specifying the title of the y axis

        ylim - tuple specifying the limits of the y axis

        b_colors - str or list of str specifying the colors of the bar plots

        b_edgecolor - str specifying the edge color of the bar plots

        b_err_color - str specifying the color of the bar plot error bars

        b_width - float specifying the width of the individual bar plots

        b_linewidth - float specifying the bar plot edge widths

        b_alpha - float specifying the alpha of the bar plots

        s_colors - str or list of str specifying the color of the scatter points

        s_size - int specifying the size of the scatter points

        s_alpha - float specifying the alpha of the scatter points

        plot_ind - boolean of whether or not to plot the individual data points

        axis_width - int or float specifying how thick the axis lines should be

        minor_ticks - str specifying if minor ticks should be add to the x and/or y
                      axes. Takes "both", "x", and "y" as inputs.

        tick_len - int or float specifying how long the tick marks should be

        ax - axis object you wish the data to be plotted on. Useful for subplotting

        save - boolean specifying if you wish to save the figure or not

        save_path - str specifying the path of where to save the figure
    """
    # make list of colors if only one is provided
    if type(b_colors) == str:
        b_colors = [b_colors for i in range(len(list(data_dict.keys())))]
    if type(s_colors) == str:
        s_colors = [s_colors for i in range(len(list(data_dict.keys())))]

    # Check if axis was provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()
    else:
        save = False  # Don't wish to save if part of another plot

    # Add the title
    ax.set_title(title)

    # Set up the data for plotting
    groups = list(data_dict.keys())
    x = list(range(len(groups)))
    data_points = list(data_dict.values())

    # Calculate the appropriate means and errors
    ## Means
    if mean_type == "mean":
        data_mean = [np.nanmean(i) for i in data_points]
    elif mean_type == "median":
        data_mean = [np.nanmedian(i) for i in data_points]
    else:
        return "Only accepts mean and median for mean type!!!"
    ## Errors
    if err_type == "sem":
        data_sems = [stats.sem(i, nan_policy="omit") for i in data_points]
    elif err_type == "std":
        data_sems = [np.nanstd(i) for i in data_points]
    elif err_type == "CI":
        data_sems = []
        for i, data in enumerate(data_points):
            d = (data,)
            bootstrap = stats.bootstrap(
                d,
                np.nanmedian,
                confidence_level=0.95,
                method="basic",
                n_resamples=100,
            )
            low = data_mean[i] - bootstrap.confidence_interval.low
            high = bootstrap.confidence_interval.high - data_mean[i]
            sem = np.array([low, high]).reshape(-1, 1)
            data_sems.append(sem)
        data_sems = np.hstack(data_sems)
    else:
        return "Only accepts sem, std and CI for erro type!!!"

    # Plot the points
    if plot_ind == True:
        data_df = pd.DataFrame.from_dict(data_dict, orient="index")
        data_df = data_df.T
        sns.stripplot(
            data=data_df,
            palette=s_colors,
            alpha=s_alpha,
            zorder=0,
            size=s_size,
            ax=ax,
            clip_on=False,
        )

    # Plot means and error
    ax.bar(
        x=x,
        height=data_mean,
        yerr=data_sems,
        width=b_width,
        color=b_colors,
        edgecolor=b_edgecolors,
        ecolor=b_err_colors,
        linewidth=b_linewidth,
        alpha=b_alpha,
        tick_label=list(data_dict.keys()),
    )

    # Format the axes
    adjust_axes(
        ax,
        minor_ticks,
        xtitle,
        ytitle,
        tick_len,
        axis_width,
    )
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax.set_xmargin(0.15)
    ticks = ax.get_yticks()
    bottom, top = get_axis_limit(ylim, ticks)
    ax.set_ylim(bottom=bottom, top=top)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")
