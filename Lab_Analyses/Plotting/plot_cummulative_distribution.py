import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Lab_Analyses.Plotting.adjust_axes import adjust_axes, get_axis_limit

sns.set()
sns.set_style("ticks")


def plot_cummulative_distribution(
    data,
    plot_ind=False,
    title=None,
    xtitle=None,
    xlim=None,
    figsize=(5, 5),
    color="mediumblue",
    ind_color="lightgrey",
    line_width=1,
    ind_line_width=0.5,
    alpha=0.002,
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """General function to plot cummulative distributions

        INPUT PARAMETERS
            data - np.array or list of arrays of the data to be plotted

            plot_ind - boolean specifying if a 2d array is inputed, whether
                        to plot each individual row (sample)
            
            title - str specifying the name of the plot

            xtitle - str specifying the x-axis title

            xlim - tuple specifying the limit of the x-axis

            figsize - tuple specifying the figure size

            color - str or list of str specifying plot colors

            alpha - float specifying alpha for individual plots

            axis_width - float specifying axis thickness

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
        save = False  # Don't want to save if part of another plot

    # Make the plots
    if type(data) == list:
        for d, c, ic in zip(data, color, ind_color):
            ## Check to plot individual data samples
            if (plot_ind == True) and (len(d.shape) == 2):
                idxs = list(range(d.shape[0]))
                if len(idxs) > 50:
                    idxs = random.sample(idxs, 50)
                for i in idxs:
                    ind = d[i, :]
                    ind = ind[~np.isnan(ind)]
                    sns.ecdfplot(
                        data=ind, color=ic, linewidth=ind_line_width, alpha=alpha, ax=ax
                    )
                plot_d = d.flatten().astype(np.float32)
                plot_d = plot_d[~np.isnan(plot_d)]
            elif (plot_ind == False) and (len(d.shape) == 2):
                plot_d = d.flatten().astype(np.float32)
            else:
                plot_d = d[~np.isnan(d)]
            sns.ecdfplot(data=plot_d, color=c, linewidth=line_width, ax=ax)
    else:
        plot_d = data[~np.isnan(data)]
        sns.ecdfplot(data=plot_d, color=color, linewidth=line_width, ax=ax)

    # Add the title
    ax.set_title(title)

    # Adjust axes
    adjust_axes(ax, minor_ticks, xtitle, "Cumulative probability", tick_len, axis_width)
    ax.set_ylim(0, 1)
    xticks = ax.get_xticks()
    left, right = get_axis_limit(xlim, xticks)
    ax.set_xlim(left, right)

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")

