import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import seaborn as sns

sns.set()
sns.set_style("ticks")


def adjust_axes(ax, minor_ticks, xtitle, ytitle, tick_len, axis_width):
    """Helper function to adjust the axes of plots"""
    # Format the axes
    ## Add minor tick marks
    if minor_ticks == "both":
        ax.xaxis.set_minor_locator(mticks.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(mticks.AutoMinorLocator(n=2))
    elif minor_ticks == "x":
        ax.xaxis.set_minor_locator(mticks.AutoMinorLocator(n=2))
    elif minor_ticks == "y":
        ax.yaxis.set_minor_locator(mticks.AutoMinorLocator(n=2))
    ## Add axes labels and set limits
    ax.set_xlabel(xtitle, labelpad=5)
    ax.set_ylabel(ytitle, labelpad=5)
    ## Adjust tick marks
    ax.tick_params(
        axis="both", which="major", direction="in", length=tick_len, width=axis_width
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=tick_len / 1.5,
        width=axis_width,
        zorder=-1,
    )
    ## Adjust axis width
    for axis in ax.spines.keys():
        ax.spines[axis].set_linewidth(axis_width)

    ax.autoscale_view(tight=False)

    sns.despine(ax=ax)


def get_axis_limit(lim, ticks):
    """Helper function to set axis limits"""
    if lim is None:
        minimum = ticks[0]
        maximum = ticks[-1]
    else:
        if lim[0] is None:
            minimum = ticks[0]
        else:
            minimum = lim[0]
        if lim[1] is None:
            maximum = ticks[-1]
        else:
            maximum = lim[1]

    return minimum, maximum

