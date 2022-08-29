"""Module to perform plotting of spine and coactivity activity data"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

sns.set()
sns.set_style("ticks")


def plot_sns_scatter_correlation(
    var1,
    var2,
    CI=None,
    title=None,
    xtitle=None,
    ytitle=None,
    figsize=(5, 5),
    xlim=None,
    ylim=None,
    face_color="mediumblue",
    edge_color="mediumblue",
    edge_width=0.3,
    s_alpha=0.5,
    line_color="mediumblue",
    line_width=1,
    save=False,
    save_path=None,
):
    """Function to plot a general correlation between two variables
        Uses seaborn regplot to plot line and confidence interval
        
        INPUT PARAMETERS
            var1 - np.array of the x variable to plot
            
            var2 - np.array of the y variable to plot

            CI - int specifying the confidence interval to use

            title - str specifying the title of the plot

            xtitle - str specifying the title of the x axis

            ytitle - str specifying the title of the y axis
            
            figsize - tuple of the size to set the figure
            
            xlim - tuple specifying the limits (left, right) of the x axis
            
            ylim - tuple specifying the limits (bottom, top) of the y axis
            
            face_color - str specifying what color you wish the scatter points to be

            edge_color - str specifying what color you wish the scatter edges to be

            edge_width - float specifying how thick the edges of the scatter should be

            s_alpha - float specifying how transparent the points should be

            line_color - str specifying what color the regression line should be
            
            line_width - float specifying how thick the regression line should be

            save - boolean specifying if you wish to save the figure
            
            save_path - str specifying the path of where to save the figure
    """

    # Make the figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontweight="bold")

    # Perform correlation of the data
    corr, p = stats.pearsonr(var1, var2)
    # Add correlation as a subtitle
    subtitle = f"r = {corr}    p = {p}"
    fig.title(subtitle, style="italic")

    # Set up some styles for the scatter points
    scatter_kws = {
        "facecolor": face_color,
        "edgecolor": edge_color,
        "alpha": s_alpha,
        "linewidth": edge_width,
    }
    line_kws = {"linewidth": line_width}

