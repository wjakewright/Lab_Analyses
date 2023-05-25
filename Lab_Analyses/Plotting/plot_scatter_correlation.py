import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import numpy as np
import seaborn as sns
from scipy import stats

from Lab_Analyses.Plotting.adjust_axes import adjust_axes, get_axis_limit

sns.set()
sns.set_style("ticks")


def plot_scatter_correlation(
    x_var,
    y_var,
    CI=None,
    title=None,
    xtitle=None,
    ytitle=None,
    figsize=(5, 5),
    xlim=None,
    ylim=None,
    marker_size=5,
    face_color="mediumblue",
    edge_color="white",
    edge_width=0.3,
    line_color="mediumblue",
    s_alpha=0.5,
    line_width=1,
    axis_width=1,
    minor_ticks=None,
    tick_len=3,
    ax=None,
    save=False,
    save_path=None,
):
    """Function to plot general correlation between two variables with a regression line.
    
        INPUT PARAMETERS
            x_var - np.array of the x variable to plot
            
            y_var - np.array of the y variable to plot
            
            CI - int specifying the confidence interval to use around the regression line
            
            title - str specifying the title of the plot
            
            xtitle - str specifying the title of the x axis
            
            ytitle - str specifying the title of the y axis
            
            figsize - tuple of the size of the figure to set. Only used if independent
                      figure
                      
            xlim - tuple specifying the limits of the x axis
            
            ylim - tuple specifying the limits of the y axis

            marker_size - int specifing the size of the scatter points

            face_color - str specifying what color the face of the points should be. Note
                        can enter "cmap" for a density-based heatmap instead

            edge_color - str specifying what color the edge of the points should be

            edge_width - float specifying how think the edges of the points should be

            line_color - str specifying what color the regression line should be

            s_alpha - float specifying the alhpa of the points

            line_width - int or float specifying the thickness of the regression line

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

    # First remove any nan values in the data
    x_non_nan = np.nonzero(~np.isnan(x_var))[0]
    y_non_nan = np.nonzero(~np.isnan(y_var))[0]
    non_nan = set(x_non_nan).intersection(y_non_nan)
    non_nan = np.array(list(non_nan))
    x_var = x_var[non_nan]
    y_var = y_var[non_nan]

    # Perform correlatino of the data
    corr, p = stats.pearsonr(x_var, y_var)
    # Add correlation to the title
    fig_title = f"{title}\nr = {corr:.3}  p = {p:.3E}"
    ax.set_title(fig_title)

    # Set up some styles
    line_kws = {"linewidth": line_width, "color": line_color}
    ## uniform scatter points
    if face_color != "cmap":
        scatter_kws = {
            "facecolor": face_color,
            "edgecolor": edge_color,
            "alpha": s_alpha,
            "linewidth": edge_width,
            "s": marker_size,
            "clip_on": False,
        }
        ## Make the scatter plot
        sns.regplot(
            x=x_var,
            y=y_var,
            ci=CI,
            scatter_kws=scatter_kws,
            line_kws=line_kws,
            ax=ax,
            truncate=False,
        )

    ## density-based scatter
    else:
        x_var, y_var, face_colors = generate_scatter_cmap(x_var, y_var)
        scatter_kws = {
            "facecolor": face_colors,
            "edgecolor": None,
            "cmap": "plasma",
            "s": marker_size,
            "clip_on": False,
        }
        ax.scatter(x=x_var, y=y_var, c=face_colors, s=marker_size, cmap="plasma")
        sns.regplot(
            x=x_var,
            y=y_var,
            ci=CI,
            scatter=False,
            line_kws=line_kws,
            ax=ax,
            truncate=False,
        )

    # Adjust the axes
    adjust_axes(ax, minor_ticks, xtitle, ytitle, tick_len, axis_width)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    bottom, top = get_axis_limit(ylim, yticks)
    left, right = get_axis_limit(xlim, xticks)
    ax.set_ylim(bottom, top)
    ax.set_xlim(left, right)

    # Save section
    if save:
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Figures"
        fname = os.path.join(save_path, title)
        fig.savefig(fname + ".pdf")


#################### HELPER FUNCTIONS ######################
def generate_scatter_cmap(x, y):
    """Helper function to generate a color map for a density based scatter plot"""
    # Calculate point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    # Sort the points by density, so densest points are plotted last
    idx = z.argsort()
    x = x[idx]
    y = y[idx]
    z = z[idx]

    return x, y, z
