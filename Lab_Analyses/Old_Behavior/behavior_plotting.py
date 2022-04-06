"""Module to perform the plotting of behavioral data
    
    CREATOR
        William (Jake) Wright 02/08/2022
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Lab_Analyses.Utilities import utilities as utils

sns.set()
sns.set_style("ticks")


def plot_session_rewarded_lever_presses(
    mouse_lever_data, session, figsize=(8, 5), x_lim=None, y_lim=None, save=False
):
    """Function to plot each rewarded lever press as well as the
    average rewarded lever press

    INPUT PARAMETERS
        mosue_lever_data - Mouse_Lever_Data dataclass object containing all the
                            lever press data for a single mouse

        session - int specifying which session you wish to plot

        figsize - tuple specifying the size to make the figure. Default set to 8 x 5

        x_lim - tuple specifying what to make the xlim (left, right). Optional

        y_lim - tuple specifying what to make the ylim (bottom, top). Optional
    """

    # Set title
    title = f"{mouse_lever_data.mouse_id} from session {session}"

    # Make figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # Plot individual movments
    move_mat = mouse_lever_data.all_movements[session - 1]
    move_mat = utils.zero_window(move_mat.T, (0, 0.3), 1000)
    move_mat = move_mat.to_numpy()
    move_mat = move_mat.T
    for row, _ in enumerate(move_mat[:, 0]):
        plt.plot(move_mat[row, :], color="gray", linewidth=0.5, alpha=0.3)

    # Plot average movement
    avg_move = mouse_lever_data.average_movements[session - 1]
    avg_move = utils.zero_window(avg_move, (0, 0.3), 1000)
    avg_move = avg_move.to_numpy()
    plt.plot(avg_move, color="black", linewidth=1)

    if x_lim is not None:
        plt.xlim(left=x_lim[0], right=x_lim[1])
    if y_lim is not None:
        plt.ylim(bottom=y_lim[0], top=y_lim[1])

    fig.tight_layout()
    save_directory = r"C:\Users\Jake\Desktop\Figures"
    if save is True:
        fname = os.path.join(save_directory, title)
        plt.savefig(fname + ".pdf")


def plot_movement_corr_matrix(
    correlation_matrix, title=None, cmap=None, figsize=(7, 6), save=False
):
    """Function to plot a heatmap of the average movement correlations across sessions

    INPUT PARAMETERS
        correlation_matrix - 2d np.array containing the pairwise correlations
                            for all behavioral sessons

        title - string specifying what to name the figure. Optional.

        cmap - string specifying a built in matplotlib color map, or a custom
                colormap object. Optional with default set to 'hot'

        figsize - tuple specifying the size to make the figure. Default 6x6

    """
    # Set the title
    if title is None:
        title = "Movement Correlation Matrix"
    else:
        title = title

    if cmap is None:
        cmap = "hot"
    else:
        cmap = cmap

    # Plot the heatmap
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    ax = sns.heatmap(
        correlation_matrix,
        cmap=cmap,
        cbar_kws={"label": "Correlation", "orientation": "vertical"},
        xticklabels=2,
        yticklabels=2,
    )
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth("2.5")

    fig.tight_layout()
    save_directory = r"C:\Users\Jake\Desktop\Figures\test_lever"
    if save is True:
        fname = os.path.join(save_directory, title)
        plt.savefig(fname + ".pdf")


def plot_within_session_corr(
    sessions,
    mean,
    sem,
    individual,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
):
    """Function to plot within session movement correlations. Utilizes the plot_mean_sem_line_plot function
        to carry out the plotting.
    """
    # Set up specific parameters
    title = "Within Session Movement Correlation"
    xtitle = "Session"
    ytitle = "Movement Correlation"
    if ylim is None:
        ylim = (0, None)

    plot_mean_sem_line_plot(
        sessions,
        mean,
        sem,
        individual,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        ylim=ylim,
        xlim=None,
        figsize=figsize,
        color=color,
        save=save,
    )


def plot_across_session_corr(
    sessions,
    mean,
    sem,
    individual,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
):
    """Function to plot across session movement correlations. Utilizes the plot_mean_sem_line_plot function
        to carry out the plotting
    """
    # Set up specific parameters
    title = "Across Session Movement Correlation"
    xtitle = "Session"
    ytitle = "Movement Correlation"
    if ylim is None:
        ylim = (0, None)
    xlim = (0, None)

    plot_mean_sem_line_plot(
        sessions,
        mean,
        sem,
        individual,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        ylim=ylim,
        xlim=xlim,
        figsize=figsize,
        color=color,
        save=save,
    )


def plot_success_rate(
    sessions,
    mean,
    sem,
    individual,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
):
    """Function to plot the success rate across sessions. Utilizes the plot_mean_sem_line_plot function
        to carry out the plotting"""
    # Set up specific parameters
    title = "Success Rate"
    xtitle = "Session"
    ytitle = "Successful Trials (%)"
    if ylim is None:
        ylim = (0, 100)

    plot_mean_sem_line_plot(
        sessions,
        mean,
        sem,
        individual,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        ylim=ylim,
        xlim=None,
        figsize=figsize,
        color=color,
        save=save,
    )


def plot_cue_to_reward(
    sessions,
    mean,
    sem,
    individual,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
):
    """Function to plot the cut_to_reward across sessions. Utilizes the plot_mean_sem_line_plot function 
        to carry out the plotting
    """
    # Set up specific parameters
    title = "Cue to Reward"
    xtitle = "Session"
    ytitle = "Cue to Reward Time (s)"
    if ylim is None:
        ylim = (0, None)

    plot_mean_sem_line_plot(
        sessions,
        mean,
        sem,
        individual,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        ylim=ylim,
        xlim=None,
        figsize=figsize,
        color=color,
        save=save,
    )


def plot_movement_reaction_time(
    sessions,
    mean,
    sem,
    individual,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
):
    """Function to plot the movment reaction time across sessions. Utilizes the plot_mean_sem_line_plot function
        to carry out the plotting
    """
    # Set up specific parameters
    title = "Movement Reaction Time"
    xtitle = "Session"
    ytitle = "Movement Reaction Time (s)"
    if ylim is None:
        ylim = (0, None)

    plot_mean_sem_line_plot(
        sessions,
        mean,
        sem,
        individual,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        ylim=ylim,
        xlim=None,
        figsize=figsize,
        color=color,
        save=save,
    )


def plot_mean_sem_line_plot(
    sessions,
    mean,
    sem,
    individual,
    title=None,
    xtitle=None,
    ytitle=None,
    ylim=None,
    xlim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
):
    """Function to plot data across sessions as a line. Plots mean, sem, and individual values

    INPUT PARAMETERS
        sessions - list or np.array of session numbers to be plotted

        mean - 1d np.array containing the mean for each session

        sem - 1d np.array containing the sem for each session

        individual - 2d np.array containing values for each mouse in rows

        title - string specifying the name of the title

        xtitle - string specifying the name of the x axis

        ytitle - string specifying the name of the y axis (bottom,top)

        ylim - tuple specifying the limits of the y axis

        figsize - tuple specifying the desired figure size, Default is 8x5

        color - string specifying what color you wish the lines and error
                bars to be. default is "medium blue"

    """

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # Put individual data in dataframe for easier plotting
    ind = pd.DataFrame(individual)

    # Plot individual data
    for col in ind.columns:
        plt.plot(sessions, ind[col], color=color, linewidth=0.5, alpha=0.2)

    # Plot mean and sem
    plt.errorbar(
        sessions,
        mean,
        yerr=sem,
        color=color,
        marker="o",
        markerfacecolor="white",
        markeredgecolor=color,
        linewidth=1.5,
        elinewidth=0.8,
        ecolor=color,
    )

    plt.xlabel(xtitle, labelpad=15)
    plt.xticks(sessions, sessions)
    if xlim is not None:
        plt.xlim(left=xlim[0], right=xlim[1])
    plt.ylabel(ytitle, labelpad=15)
    if ylim is not None:
        plt.ylim(bottom=ylim[0], top=ylim[1])

    fig.tight_layout()
    save_directory = r"C:\Users\Jake\Desktop\Figures\test_lever"
    if save is True:
        fname = os.path.join(save_directory, title)
        plt.savefig(fname + ".pdf")

