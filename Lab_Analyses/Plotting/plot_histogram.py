import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Lab_Analyses.Plotting.adjust_axes import adjust_axes

sns.set()
sns.set_style()


def plot_histogram(
    data,
    bins,
    stat="probability",
    avlines=None,
    title=None,
    xtitle=None,
    xlim=None,
    figsize=(5, 5),
    color="mediumblue",
    alpha=0.4,
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """General function to plot histogram of one or more datasets
        
        INPUT PARAMETERS
            data - np.array or list of arrays of the data to be plotted
            
            bins - int specifyig the number of bins to plot

            stat - str specifying the stat plot along the y axis. See sns.histplot
            
            avline - list specifying where to draw vertical lines if desired
            
            title - str specifying the title of the plot
            
            xtitle - str specifying the title of the x axis
            
            figsize - tuple specifying the size of the figure. Only used if independent
            
            color - str or list of str speicfying the colors of the bars
            
            alpha - float specifying the alpha 

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

    # Make the plots
    if type(data) == list:
        data = [d[~np.isnan(d)] for d in data]
        b = np.histogram(np.hstack(data), bins=bins)[1]
        for d, c in zip(data, color):
            sns.histplot(data=d, bins=b, color=c, alpha=alpha, stat=stat, ax=ax)
    else:
        sns.histplot(data=data, bins=bins, color=color, alpha=alpha, stat=stat, ax=ax)

    # Add vertical lines if given
    if avlines:
        for line in avlines:
            ax.axvline(line, linestyle="--", color="black")

    # Add the title
    ax.set_title(title)

    # Adjust axes
    adjust_axes(ax, minor_ticks, xtitle, stat, xlim, None, tick_len, axis_width)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")
