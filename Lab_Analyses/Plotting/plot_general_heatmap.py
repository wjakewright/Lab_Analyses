import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import seaborn as sns

sns.set()
sns.set_style("ticks")

def plot_general_heatmap(
    data,
    figsize=(7, 6),
    title=None,
    xtitle=None,
    ytitle=None,
    cbar_label=None,
    hmap_range=None,
    center=None,
    cmap="plasma",
    axis_width=2.5,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """General function to plot generic heatmap plots
    
        INPUT PARAMETERS
            data - 2d np.array of the data to be plotted in the heatmap
            
            figsize - tuple specifying the figure size. Only if independent
            
            title - str specifying the figure title
            
            xtitle - str specifying the title of the x axis
            
            ytitle - str specifying the title of the y axis
            
            cbar_label - str specifying the label of the color bar
            
            hmap_range - tuple specifying the limits of the heatmap range

            center - int specifying the center of the color map

            cmap - str specifying the color map

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

    # Check some other parameters
    if hmap_range is None:
        hmap_range = (None, None)
        cbar_ticks = None
    elif hmap_range is not None and center is None:
        cbar_ticks = (hmap_range[0], hmap_range[1])
    else:
        cbar_ticks = (hmap_range[0], center, hmap_range[1])

    ax.set_title(title)
    # Plot the data
    hax = sns.heatmap(
        data,
        cmap=cmap,
        center=center,
        vmax=hmap_range[1],
        vmin=hmap_range[0],
        cbar_kws={"label": cbar_label, "orientation": "vertical", "ticks": cbar_ticks},
        yticklabels=False,
        ax=ax,
    )
    # setup the axes
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    hax.patch.set_edgecolor("black")
    hax.patch.set_linewidth(axis_width)
    if minor_ticks == "both":
        ax.xaxis.set_minor_locator(mticks.AutoMinorLocator(n=2))
        ax.yaxis.set_minor_locator(mticks.AutoMinorLocator(n=2))
    ax.tick_params(
        axis="both", which="major", direction="out", length=tick_len, width=axis_width,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=tick_len / 1.5,
        width=axis_width,
    )

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")
