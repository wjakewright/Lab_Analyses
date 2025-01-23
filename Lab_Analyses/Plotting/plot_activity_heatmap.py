import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import scipy.signal as sysignal
import seaborn as sns

from Lab_Analyses.Utilities import data_utilities as d_utils


def plot_activity_heatmap(
    data,
    figsize=(4, 5),
    sampling_rate=60,
    activity_window=(-2, 2),
    title=None,
    cbar_label=None,
    hmap_range=None,
    center=None,
    vline=None,
    sorted=None,
    onset_color="mediumslateblue",
    normalize=False,
    cmap="plasma",
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """General function to plot a heatmap for activity traces

    INPUT PARAMETERS
        data - 2d np.array of the traces to be plotted (colums=rows of the heatmap)

        figsize - tuple specifying the size of the figure. Only if independent

        sampling_rate - int specifying the sampling rate

        activity_window - tuple specifying the window (sec) the activity is centered on

        title - str specifying the title of the plot

        cbar_label - str specifying the label of the color bar

        hmap_range - tuple specifying the min and max of the heatmap range

        center - int specifying the center of the color map

        sorted - str specifying if and how to sort the data. Accepts 'peak'
                and 'difference' to sort the data based on their peak timing
                or difference in their activity before and after center point

        normalize - boolean specifying if you wish to normalize activity for each row

        cmap - str specifying the color map to use

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

    # Check some other parameters
    if hmap_range is None:
        hmap_range = (None, None)
        cbar_ticks = None
    elif hmap_range is not None and center is None:
        cbar_ticks = (hmap_range[0], hmap_range[1])
    else:
        cbar_ticks = (hmap_range[0], center, hmap_range[1])

    # Process data if specified
    if normalize:
        data = d_utils.peak_normalize_data(data)
    if sorted == "peak":
        data = d_utils.peak_sorting(data)
    elif sorted == "difference":
        data = d_utils.diff_sorting(
            data.T, np.absolute(activity_window), sampling_rate
        ).T
    elif sorted == "onset":
        data, onsets = d_utils.onset_sorting(data)

    # Rotate data for plotting
    data_t = data.T

    # Plot the data
    ax.set_title(title)
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

    # Add onset markers for onset sorted
    if sorted == "onset":
        for i in range(data_t.shape[0]):
            ax.scatter(x=onsets[i] - 1, y=i, c=onset_color, s=1)

    # Adjust colorbar
    cbar = hax.collections[0].colorbar
    cbar.set_label(label=cbar_label, rotation=270, labelpad=-1)
    # Setup the axes
    x = np.linspace(activity_window[0], activity_window[1], data_t.shape[1])
    t = np.linspace(0, activity_window[1] - activity_window[0], data_t.shape[1])
    xticks = np.unique(x.astype(int))
    t = np.unique(t.astype(int))
    ax.set_xticks(ticks=t * sampling_rate)
    ax.set_xticklabels(labels=xticks, rotation=0)
    ax.set_xlabel("Time (s)", labelpad=7)
    hax.patch.set_edgecolor("black")
    hax.patch.set_linewidth(axis_width)
    if minor_ticks == "x":
        ax.xaxis.set_minor_locator(mticks.AutoMinorLocator(n=2))
    ax.tick_params(
        axis="x", which="major", direction="out", length=tick_len, width=axis_width
    )
    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        length=tick_len / 1.5,
        width=axis_width,
    )

    if vline is not None:
        ax.axvline(x=vline, color="white", linestyle="--")

    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")
