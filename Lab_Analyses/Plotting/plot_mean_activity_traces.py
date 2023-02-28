import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Lab_Analyses.Plotting.adjust_axes import adjust_axes

sns.set()
sns.set_style("ticks")


def plot_mean_activity_traces(
    means,
    sems,
    group_names=None,
    sampling_rate=60,
    activity_window=(-2, 4),
    avlines=None,
    ahlines=None,
    figsize=(5, 5),
    colors="mediumblue",
    title=None,
    ytitle=None,
    ylim=None,
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """General function to plot the mean activity traces. Can plot individual means or 
        means of different groups
        
        INPUT PARAMETERS
            means - np.array of mean trace or a list of arrays for multiple groups
            
            sems - np.array of sem trace of a list of arrays for multiple groups
            
            group_names - list of str specifying the names of multiple groups if provided
            
            sampling_rate - int speciying the imaging sampling rate
            
            activity_window - tuple specifying the window (in sec) the trace is centered around
            
            avlines - list of x values for vertical lines to be plotted 
            
            ahlines - dict of y values to plot horizontal lines (for significance testing)
            
            figsize - tuple specifying figure size. Only used if independent figure
            
            colors - str or list of str specifying the color of the traces
            
            title - str specifying the title of the plot
            
            ytitle - str specifying the title of the y axis

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

    # Start making the plots
    if type(means) == np.ndarray:
        x = np.linspace(activity_window[0], activity_window[1], len(means))
        ax.plot(x, means, color=colors)
        ax.fill_between(x, means - sems, means + sems, color=colors, alpha=0.2)
    elif type(means) == list:
        for m, s, c, n in zip(means, sems, colors, group_names):
            x = np.linspace(activity_window[0], activity_window[1], len(m))
            ax.plot(x, m, color=c, label=n)
            ax.fill_between(x, m - s, m + s, color=c, alpha=0.2)
    # Add vertical lines if provided
    if avlines:
        for line in avlines:
            ax.axvline(x=line / sampling_rate, linestyle="--", color="black")
    # Add horizontal lines if provided
    if ahlines:
        line_colors = sns.color_palette()
        for i, (key, value) in enumerate(ahlines.items()):
            if value is None:
                continue
            h = [a[0] for a in value]
            start = [x[a[1]] for a in value]
            stop = [x[a[2]] for a in value]
            ax.hlines(
                y=h,
                xmin=start,
                xmax=stop,
                linestyle="solid",
                colors=line_colors[i],
                label=key,
            )
    # Make legend and title
    ax.legend(loc="upper right", borderaxespad=0, fontsize="xx-small", frameon=False)
    ax.set_title(title)

    # Adjust the axes
    ax.set_xticks(ticks=[activity_window[0], 0, activity_window[1]],)
    ax.set_xticklabels(labels=[activity_window[0], 0, activity_window[1]])
    adjust_axes(ax, minor_ticks, "Time (s)", ytitle, None, ylim, tick_len, axis_width)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")

