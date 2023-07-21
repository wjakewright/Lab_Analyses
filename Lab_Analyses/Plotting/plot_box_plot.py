import os

import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Lab_Analyses.Plotting.adjust_axes import adjust_axes, get_axis_limit

sns.set()
sns.set_style("ticks")


def plot_box_plot(
    data_dict,
    figsize=(5, 5),
    title=None,
    xtitle=None,
    ytitle=None,
    ylim=None,
    b_colors="mediumblue",
    b_edgecolors="black",
    b_err_colors="black",
    m_color="white",
    m_width=1,
    b_width=0.5,
    b_linewidth=1,
    b_alpha=0.7,
    b_err_alpha=0.7,
    whisker_lim=None,
    whisk_width=1,
    outliers=False,
    showmeans=False,
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """Function to plot grouped box and whisker plots"""
    # Make sure colors are in a list
    if type(b_colors) == str:
        b_colors = [b_colors for i in range(len(list(data_dict.keys())))]
    if type(b_err_colors) == str:
        b_err_colors = [b_err_colors for i in range(len(list(data_dict.keys())))]

    # Convert colors to incorporate the alpha
    box_colors = [mcolor.to_rgba(x, b_alpha) for x in b_colors]
    box_err_colors = [mcolor.to_rgba(x, b_err_alpha) for x in b_err_colors]

    # Set up the whisker colors
    w_colors = []
    for color in box_err_colors:
        w_colors.append(color)
        w_colors.append(color)

    # Check if axes was provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()
    else:
        save = False  # Don't wish to save if part of another plot

    # Add the title
    ax.set_title(title)

    # Set up some properties
    whiskerprops = {"linewidth": whisk_width, "zorder": 0}
    boxprops = {"linewidth": b_linewidth, "color": b_edgecolors}
    meanprops = {"marker": "+", "markeredgecolor": m_color, "markersize": 4}
    medianprops = {"color": m_color, "linewidth": m_width}
    flierprops = {"marker": "o"}

    if whisker_lim is None:
        whisker_lim = 1.5

    # Clean out nan values
    plot_dict = {}
    for key, value in data_dict.items():
        plot_dict[key] = value[~np.isnan(value)]

    # Plot the data
    bplot = ax.boxplot(
        x=list(plot_dict.values()),
        vert=True,
        whis=whisker_lim,
        widths=b_width,
        labels=list(plot_dict.keys()),
        whiskerprops=whiskerprops,
        boxprops=boxprops,
        meanprops=meanprops,
        medianprops=medianprops,
        flierprops=flierprops,
        showcaps=False,
        showmeans=showmeans,
        showfliers=outliers,
        patch_artist=True,
    )

    # Adjust the colors
    for patch, color in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(color)
    for patch, color in zip(bplot["whiskers"], w_colors):
        patch.set_color(color)
    for patch, color in zip(bplot["fliers"], box_colors):
        patch.set_markeredgecolor(color)
        patch.set_markerfacecolor("white")

    # Format the axes
    adjust_axes(
        ax,
        minor_ticks=minor_ticks,
        xtitle=xtitle,
        ytitle=ytitle,
        tick_len=tick_len,
        axis_width=axis_width,
    )

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    ax.set_xmargin(0.15)
    ticks = ax.get_yticks()
    bottom, top = get_axis_limit(ylim, ticks)
    ax.set_ylim(bottom=bottom, top=top)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")

