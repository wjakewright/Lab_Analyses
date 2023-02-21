import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.adjust_axes import adjust_axes

sns.set()
sns.set_style("ticks")


def plot_grouped_swam_bar_plot(
    data_dict,
    groups,
    mean_type="mean",
    err_type="sem",
    figsize=(5, 5),
    title=None,
    xtitle=None,
    ytitle=None,
    ylim=None,
    b_colors="mediumblue",
    b_edgecolors="black",
    b_err_colors="black",
    b_width=0.5,
    b_linewidth=0,
    b_alpha=0.3,
    s_colors="mediumblue",
    s_size=5,
    s_alpha=0.8,
    plot_ind=True,
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """General function for plotting a scatter bar plots with different subgroups
        
        INPUT PARAMETERS
            data_dict - nested dictionary of data to be plotted. Outer keys represent the
                        subgroups, while the inner keys represent the main groups
            
            groups - list of str representing the groups. Main group will be first and
                     subgroups will be listed second.
            
            mean_type - str specifying what central point to plot. Accepts 'mean' and 
                        'median'.
                        
            err_type - str specifying what type of error bars with. Accepts 'sem', 
                        'std', and 'CI', which plots bootstraped 95% confidence interval
            
            figsize - tuple of the size of the figure to set. Only used if independent figure

            title - str specifying the title of the plot

            xtitle - str specifying the title of the x axis

            ytitle - str specifying the title of the y axis

            ylim - tuple specifying the limits of the y axis

            b_colors - str or list of str specifying the colors of the bar plots

            b_edgecolor - str specifying the edge color of the bar plots

            b_err_color - str specifying the color of the bar plot error bars

            b_width - float specifying the width of the individual bar plots

            b_linewidth - float specifying the bar plot edge widths

            b_alpha - float specifying the alpha of the bar plots

            s_colors - str or list of str specifying the color of the scatter points

            s_size - int specifying the size of the scatter points

            s_alpha - float specifying the alpha of the scatter points

            plot_ind - boolean of whether or not to plot the individual data points

            axis_width - int or float specifying how thick the axis lines should be

            minor_ticks - str specifying if minor ticks should be add to the x and/or y 
                          axes. Takes "both", "x", and "y" as inputs.

            tick_len - int or float specifying how long the tick marks should be

            ax - axis object you wish the data to be plotted on. Useful for subplotting

            save - boolean specifying if you wish to save the figure or not

            save_path - str specifying the path of where to save the figure
    """
    # Sort out the colors
    if type(s_colors) == str:
        s_colors = [s_colors for i in range(len(list(data_dict.keys())))]
    elif (type(s_colors) == list) & (len(s_colors) != len(list(data_dict.keys()))):
        return "Number of scatter colors does not match number of groups"
    if type(b_colors) == str:
        b_colors = [b_colors for i in range(len(list(data_dict.keys())))]
    elif (type(b_colors) == list) & (len(b_colors) != len(list(data_dict.keys()))):
        return "Number of marker colors does not match number of groups"

    # Check if axis was provided
    if ax is None:
        fig, ax = plt.subplot(figsize=figsize)
        fig.tight_layout()
    else:
        save = False  # Don't wish to save if part of another plot

    # Set up the data for plotting
    dfs = []
    g1_keys = []
    g2_keys = []
    for key, value in data_dict.items():
        g1_keys = list(value.keys())
        g2_keys.append(key)
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in value.items()]))
        df = pd.melt(df, value_vars=df.columns, var_namd=groups[0], value_name=ytitle)
        df[groups[1]] = key
        dfs.append(df)
    plot_df = pd.concat(dfs)

    # Calculate the means and errors
    data_mean = []
    data_err = []
    for k1 in g1_keys:
        for k2 in g2_keys:
            ## Grab the data corresponding to the groups
            data = plot_df[ytitle].loc[
                (plot_df[groups[0]] == k1) & (plot_df[groups[1]] == k2)
            ]
            ## Get the appropriate mean
            if mean_type == "mean":
                data_mean.append(np.nanmean(data))
            elif mean_type == "median":
                data_mean.append(np.nanmedian(data))
            else:
                return "Only accepts mean and median for mean type!!!"
            ## Get the appropriate error
            if err_type == "sem":
                data_err.append(stats.sem(data, nan_policy="omit"))
            elif err_type == "std":
                data_err.append(np.nanstd(data))
            elif err_type == "CI":
                d = (data,)
                bootstrap = stats.bootstrap(
                    d, np.nanmedian, confidence_level=0.95, method="percentile"
                )
                low = np.nanmedian(data) - bootstrap.confidence_interval.low
                high = bootstrap.confidence_interval.high = np.nanmedian(data)
                sem = np.array([low, high]).reshape(-1, 1)
                data_err.append(sem)
            else:
                return "Only accepts sem, std, and CI for err type!!!"

    if err_type == "CI":
        data_err = np.hstack(data_err)

    # Plot the scatter points
    strip = sns.stripplot(
        data=plot_df,
        x=groups[0],
        y=ytitle,
        hue=groups[1],
        palette=s_colors,
        dodge=True,
        ax=ax,
        zorder=0,
        size=s_size,
        alpha=s_alpha,
    )
    ## Get x positions
    x_coords = []
    for c in strip.collections:
        offsets = c.get_offsets()
        if len(offsets) != 0:
            xs = [x[0] for x in offsets]
            x_coords.append(np.nanmean(xs))

    # Remove points if you don't want to plot them
    if plot_ind is False:
        ax.cla()

    # Add the title
    ax.set_title(title)

    # Plot the mean and error
    ax.bar(
        x=x_coords,
        height=data_mean,
        yerr=data_err,
        width=b_width,
        color=b_colors,
        edgecolor=b_edgecolors,
        ecolor=b_err_colors,
        linewidth=b_linewidth,
        alpha=b_alpha,
    )

    # Make the legend
    ax.legend(loc="upper right", borderaxespad=0, fontsize="xx-small", frameon=False)

    # Format the axes
    adjust_axes(ax, minor_ticks, xtitle, ytitle, None, ylim, tick_len, axis_width)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname, +".pdf")
