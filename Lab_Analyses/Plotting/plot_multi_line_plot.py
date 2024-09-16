import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.adjust_axes import adjust_axes, get_axis_limit

sns.set()
sns.set_style("ticks")


def plot_multi_line_plot(
    data_dict,
    x_vals,
    x_labels=None,
    plot_ind=False,
    figsize=(5, 5),
    mean_type="mean",
    title=None,
    ytitle=None,
    xtitle=None,
    ylim=None,
    line_color="mediumblue",
    face_color="white",
    m_size=7,
    linewidth=1,
    linestyle="-",
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    legend=True,
    save=False,
    save_path=None,
):
    """General function to plot a single or multiline plot

    INPUT PARAMETERS
        data_dict - dict of 2d np.arrays of the data to be plotted. Each column
                    represents a different data point and rows represent different
                    time points. Keys represent group names

        x_vals - np.array of the values corresponding to the x axis

        plot_ind - boolean specifying whether to plot each individual data point

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
        save = False  # Don't wish to save if part of another plot

    # Set title
    ax.set_title(title)

    if x_labels is None:
        x_labels = x_vals

    # Sort colors
    if type(line_color) == str:
        line_color = [line_color for x in data_dict.keys()]
    if type(face_color) == str:
        face_color = [face_color for x in data_dict.keys()]

    # Plot the data
    # x = list(range(len(x_vals)))
    x = x_vals
    for i, (key, value) in enumerate(data_dict.items()):
        ## Get mean and sem of the group
        if mean_type == "mean":
            mean = np.nanmean(value, axis=1)
            data_sems = stats.sem(value, axis=1, nan_policy="omit")
        elif mean_type == "median":
            mean = np.nanmedian(value, axis=1)
            data_sems = []
            for j in range(value.shape[0]):
                d = (value[j, :],)
                bootstrap = stats.bootstrap(
                    d,
                    np.nanmedian,
                    confidence_level=0.90,
                    method="percentile",
                    n_resamples=100,
                )
                low = mean[j] - bootstrap.confidence_interval.low
                high = bootstrap.confidence_interval.high - mean[j]
                sem = np.array([low, high]).reshape(-1, 1)
                data_sems.append(sem)
                # d = value[j, :]
                # est_MAD = stats.median_abs_deviation(d, scale=1, nan_policy="omit")
                # std = np.nanstd(d)
                # const = std / est_MAD
                # MAD = stats.median_abs_deviation(d, scale=const, nan_policy="omit")
                # data_sems.append(MAD)
            data_sems = np.hstack(data_sems)
        # sem = stats.sem(value, axis=1, nan_policy="omit")
        ## Plot individual values
        if plot_ind:
            for ind in range(value.shape[1]):
                ax.plot(
                    x,
                    value[:, ind],
                    color=line_color[i],
                    linewidth=linewidth / 2,
                    alpha=0.3,
                    zorder=0,
                )
        ## Plot mean values
        if m_size is None:
            ax.errorbar(
                x,
                mean,
                yerr=data_sems,
                color=line_color[i],
                linestyle=linestyle,
                label=key,
            )
        else:
            ax.errorbar(
                x,
                mean,
                yerr=data_sems,
                color=line_color[i],
                marker="o",
                markerfacecolor=face_color[i],
                markeredgecolor=line_color[i],
                markersize=m_size,
                linestyle="-",
                label=key,
            )

    # Adjust the axes
    ax.set_xticks(ticks=x_labels)
    ax.set_xticklabels(labels=x_labels)
    adjust_axes(ax, minor_ticks, xtitle, ytitle, tick_len, axis_width)
    ax.set_xlim(left=None, right=x_labels[-1])
    ticks = ax.get_yticks()
    bottom, top = get_axis_limit(ylim, ticks)
    ax.set_ylim(bottom=bottom, top=top)

    # Make the legend
    if legend:
        ax.legend(
            loc="upper right", borderaxespad=0, fontsize="xx-small", frameon=False
        )

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")
