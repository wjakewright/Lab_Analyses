import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import seaborn as sns

sns.set()
sns.set_style("ticks")


def adjust_axes(ax, minor_ticks, xtitle, ytitle, xlim, ylim, tick_len, axis_width):
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
    ax.set_xlabel(xtitle, labelpad=15)
    if xlim:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylabel(ytitle, labelpad=15)
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
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
    )
    ## Adjust axis width
    for axis in ax.spines.keys():
        ax.spines[axis].set_linewidth(axis_width)

    sns.despine(ax=ax)
