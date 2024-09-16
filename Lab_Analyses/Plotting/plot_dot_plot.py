import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.adjust_axes import adjust_axes, get_axis_limit

sns.set()
sns.set_style("ticks")


def plot_dot_plot(
    data_dict,
    figsize=(5, 5),
    mean_type="mean",
    title=None,
    ytitle=None,
    xtitle=None,
    ylim=None,
    line_color="mediumblue",
    face_color="white",
    m_size=7,
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """General function to plot a single or multiline plot

    INPUT PARAMETERS
        data_dict - dict of 2 different lists of the data to be plotted.

        figsize - tuple specifying the size of the figure

        title - str specifying the title of the plot

        ytitle - str specifying the title of the y axis

        xtitle - str specifying the title of the x axis

        ylim -  tuple specifying the limits of the y axis

        line_color - str or list of str specifying the color of the mean lines

        face_color - str or list of str specifying the color of the marker faces

        m_size - int specifying the sice of the mean markers

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
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()
    else:
        save = False

    # Set title
    ax.set_title(title)

    # Sort colors
    if type(line_color) == str:
        line_color = [line_color for x in data_dict.keys()]
    if type(face_color) == str:
        face_color = [face_color for x in data_dict.keys()]

    # Organize data and plot
    x = list(range(len(data_dict.keys())))

    for i, (key, value) in enumerate(data_dict.items()):
        # Go through each list in the value
        for j, v in enumerate(value):
            if mean_type == "mean":
                mean = np.nanmean(v)
                data_sems = stats.sem(v, nan_policy="omit")
            elif mean_type == "median":
                mean = np.nanmedian(v)
                d = (v,)
                bootstrap = stats.bootstrap(
                    d,
                    np.nanmedian,
                    confidence_level=0.95,
                    method="percentile",
                    n_resamples=100,
                )
                low = mean - bootstrap.confidence_interval.low
                high = bootstrap.confidence_interval.high - mean
                data_sems = np.array([low, high]).reshape(-1, 1)

            if j == 0:
                curr_fill = face_color[i]
            else:
                curr_fill = line_color[i]
            ax.errorbar(
                i,
                mean,
                yerr=data_sems,
                color=line_color[i],
                marker="o",
                markerfacecolor=curr_fill,
                markeredgecolor=line_color[i],
                markersize=m_size,
                linestyle="",
                label=key,
            )

    # Adjust the axes
    ax.set_xticks(ticks=x)
    ax.set_xticklabels(labels=data_dict.keys())
    adjust_axes(ax, minor_ticks, xtitle, ytitle, tick_len, axis_width)
    ticks = ax.get_yticks()
    bottom, top = get_axis_limit(ylim, ticks)
    ax.set_ylim(bottom=bottom, top=top)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")
