import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.adjust_axes import adjust_axes

sns.set()
sns.set_style("ticks")


def longitudinal_means_plot(
    data_dict,
    mean_type="mean",
    err_type="sem",
    plot_ind=True,
    figsize=(5, 5),
    title=None,
    xtitle=None,
    ytitle=None,
    face_color="white",
    edge_color="mediumblue",
    ind_color=None,
    linewidth=1.5,
    err_width=1,
    ind_width=0.7,
    ylim=None,
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """General function to plot longitudinal mean data and individual data points with
        connecting lines

        INPUT PARAMETERS
            data_dict - dict of data to be plotted, each item representing a different time
                        point
            
            plot_ind - boolean of whether to plot individual values or not

            figsize - tuple specifing the figure size. Only used if independent figure

            title - str specifying the title of the figure

            xtitle - str specifying the title of the x axis

            ytitle - str speciying the title of the y axis

            face_color - str specifying the face color of mean markers

            edge_color - str specifying the color of the edges and lines of mean

            ind_color - str specifying the sns color palette to use for individual lines

            linewidth - float specifying the thickness of the mean line

            err_width - float specifying the thickness of the err bars

            ind_width - float specifying the thickness of the individual lines

            ylim - tuple specifying the limits of the y axis

            axis_width - int or float specifying how thick the axis lines should be

            minor_ticks - str specifying if minor ticks should be add to the x and/or y 
                          axes. Takes "both", "x", and "y" as inputs.

            tick_len - int or float specifying how long the tick marks should be

            ax - axis object you wish the data to be plotted on. Useful for subplotting

            save - boolean specifying if you wish to save the figure or not

            save_path - str specifying the path of where to save the figure
    """
    # Check if axis was passed
    if ax is None:
        fig, ax = plt.subplot(figsize=figsize)
        fig.tight_layout()
    else:
        save = False  # Don't wish to save if part of another plot

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
                d, np.nanmedian, confidence_level=0.95, method="percentile"
            )
            low = data_mean[i] - bootstrap.confidence_interval.low
            high = bootstrap.confidence_interval.high - data_mean[i]
            sem = np.array([low, high]).reshape(-1, 1)
            data_sems.append(sem)
        data_sems = np.hstack(data_sems)
    else:
        return "Only accepts sem, std and CI for erro type!!!"

    # Plot the means
    ax.errorbar(
        x,
        data_mean,
        data_sems,
        color=edge_color,
        marker="o",
        markerfacecolor=face_color,
        markeredgecolor=edge_color,
        linewidth=linewidth,
        elinewidth=err_width,
        ecolor=edge_color,
    )

    # Plot individual lines
    if plot_ind:
        # Set up the graded colors for the lines
        if ind_color is None:
            palette = "Blues"
        else:
            palette = ind_color
        colors = sns.color_palette(palette, as_cmap=True)
        colors = [mcolors.rgb2hex(colors(i)) for i in range(colors.N)]
        counts = np.linspace(
            start=50, stop=len(colors) - 1, num=len(data_points[0])
        ).astype(int)
        for i, data in enumerate(list(zip(*data_points))):
            ax.plot(x, data, color=colors[counts[i]])

    # Format the axes
    ## Add minor tick marks
    adjust_axes(ax, minor_ticks, xtitle, ytitle, None, ylim, tick_len, axis_width)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")

