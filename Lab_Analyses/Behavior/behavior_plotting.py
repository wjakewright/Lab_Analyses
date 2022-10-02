"""Module to perform the plotting of behavioral data

    CREATOR
        William (Jake) Wright - 02/08/2022

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Lab_Analyses.Utilities import data_utilities as utils

sns.set()
sns.set_style("ticks")


def plot_session_rewarded_lever_presses(
    mouse_lever_data,
    session,
    figsize=(4, 5),
    x_lim=None,
    y_lim=None,
    save=False,
    save_path=None,
):
    """Function to plot each rewarded lever press as well as the mean average
        rewarded lever press
        
        INPUT PARAMETERS
            mouse_lever_data - Mouse_Lever_Data dataclass object containing all the 
                                lever press data for a single mouse
                                
            session - int specifying which session you wish to plot
            
            figsize - tuple specifying the size to make the figure. Optional. 
                      Default is set to 4 x 5
            
            x_lim - tuple specifying what to make the xlim (left, right). Optional
            
            y_lim - tuple specifying what to make the ylim (bottom, top). Optional
            
            save - boolean specifying if you wish to save the figure
    
    """
    if save is True and save_path is None:
        raise Exception("Must specify the save path in order to save the figures")

    # Set title
    title = f"{mouse_lever_data.mouse_id} press from session {session}"

    # Make the figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # Plot individual movements
    move_mat = mouse_lever_data.all_movements[session - 1]
    move_mat = utils.zero_window(data=move_mat.T, base_win=(0, 0.3), sampling_rate=1000)
    move_mat = move_mat.to_numpy()  # Output is a dataframe
    move_mat = move_mat.T
    for row, _ in enumerate(move_mat[:, 0]):
        plt.plot(move_mat[row, :], color="gray", linewidth=0.5, alpha=0.3)

    # Plot average movement
    avg_move = mouse_lever_data.average_movements[session - 1]
    avg_move = utils.zero_window(data=avg_move, base_win=(0, 0.3), sampling_rate=1000)
    avg_move = avg_move.to_numpy()
    plt.plot(avg_move, color="black", linewidth=1)

    # Format axes
    if x_lim is not None:
        plt.xlim(left=x_lim[0], right=x_lim[1])
    if y_lim is not None:
        plt.ylim(bottom=y_lim[0], top=y_lim[1])

    fig.tight_layout()

    # Save section
    if save is True:
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def plot_movement_corr_matrix(
    correlation_matrix,
    title=None,
    cmap=None,
    figsize=(7, 6),
    save=False,
    limits=None,
    save_path=None,
):
    """Function to plot a heatmap of the average movement correlations 
        across sessions for a group of mice
        
        INPUT PARAMETERS
            correlation_matrix - 2d np.array containing the pairwise correlations
                                 for all behavioral sessions
                                 
            title - string specifying what to name the figure. Optional.
            
            cmap - string specifying a built in matplotlib colormap, or a custom
                    colormap object. Optional with default set to 'hot'
            
            figsize - tuple specifying the size to make the figure. Optional
                      with default set to 7 x 6 (Makes square heatmap)
                      
            save - boolean specifying whether to save the figure or not 
    
    """
    if save is True and save_path is None:
        raise Exception("Must specify the save path in order to save the figures")

    # Set title
    if title is None:
        title = "Movement Correlation Matrix"
    # Set cmap
    if cmap is None:
        cmap = "hot"
    if limits is None:
        limits = (None, None)
    else:
        limits = limits
    # Plot the heatmap
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    ax = sns.heatmap(
        correlation_matrix,
        cmap=cmap,
        cbar_kws={"label": "Correlation", "orientation": "vertical"},
        xticklabels=2,
        yticklabels=2,
        vmin=limits[0],
        vmax=limits[1],
    )

    # Set plot edges
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth("2.5")

    fig.tight_layout()

    # Save section
    if save is True:
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def plot_mean_sem_line_plot(
    sessions,
    mean,
    sem,
    individual,
    plot_ind=True,
    title=None,
    xtitle=None,
    ytitle=None,
    xlim=None,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
    save_path=None,
):
    """General function to plot data across sessions as a line. 
        Plots mean, sem, and individual values
        
        INPUT PARAMETERS
            sessions - list or np.array of session numbers to be plotted
            
            mean - 1d np.array containing the mean for each sesson
            
            sem - 1d np.array containing the sem for each session
            
            individual - 2d np.array containing values for each mouse in rows
            
            plot_ind - boolean of whether or not to plot individual mice

            title - string specifying the name of the title
            
            xtitle - string specifying the name of the x axis
            
            ytitle - string specifying the name of the y axis

            xlim - tuple specifying the limits (left, right) of the x axis

            ylim - tuple specifying the limits (bottom, top) of the y axis
            
            figsize - tuple specifying the desired figure size. 
                      Default is set to 6 x 5
                      
            color - string specifying what color you wish the lines and error
                    bars to be. Default is 'mediumblue' 
    
    """
    if save is True and save_path is None:
        raise Exception("Must specify the save path in order to save the figures")

    # Make the figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # Put individual data in dataframe for easier plotting
    ind = pd.DataFrame(individual)

    # Plot individual data
    if plot_ind is True:
        for col in ind.columns:
            plt.plot(sessions, ind[col], color=color, linewidth=0.5, alpha=0.3)

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
        elinewidth=1,
        ecolor=color,
    )

    # Set axes
    plt.xlabel(xtitle, labelpad=15)
    plt.xticks(sessions, sessions)
    if xlim is not None:
        plt.xlim(left=xlim[0], right=xlim[1])
    plt.ylabel(ytitle, labelpad=15)
    if ylim is not None:
        plt.ylim(bottom=ylim[0], top=ylim[1])

    fig.tight_layout()

    # Save section
    if save is True:
        fname = os.path.join(save_path, title)
        plt.savefig(fname + ".pdf")


def plot_multi_line_plot(
    group_names,
    data_list,
    to_plot,
    colors=None,
    ylims=None,
    sizes=None,
    save=False,
    save_path=None,
):
    """Function to plot multiple groups of mice behavioral data against each other
        on the same plot
        
        INPUT PARAMETERS
            group_names - list of str specifying the group names
            
            data_list - list of grouped_lever_press dataclasses for each group
            
            to_plot - list of string specifying what to plot. Accepts:
                     'success rate', 'cue to reward', 'reaction time', 
                     'within correlation', 'across correlation'
                     
            colors - list of str specifying what color to give the different groups.
                    Accepts success", "cue_to_reward", "reaction_time", "within", "across"
                    Optional with default set to None. Can specify on the ones you wish to change. 
            
            ylims - dic specifying what ylim to set each different plot to, with each
                    key corresponding to the to_plot list
                    
            save - boolean of whether or not to save the output figures
            
            save_path - str specifying where to save the figures
            
    """
    if save is True and save_path is None:
        raise Exception("Must specify the save path in order to save the figure")

    # Set up some variables
    if ylims is None:
        ylims = {
            "success": None,
            "cue_to_reward": None,
            "reaction_time": None,
            "within": None,
            "across": None,
        }
    if sizes is None:
        sizes = {
            "success": (6, 5),
            "cue_to_reward": (6, 5),
            "reaction_time": (6, 5),
            "within": (6, 5),
            "across": (6, 5),
        }

    # Make each plot specified
    for plot in zip(to_plot):
        # Get the relevant data
        if plot == "success rate":
            data = [x.success_rate for x in data_list]
            ylim = ylims["success"]
            size = sizes["success"]
        elif plot == "reaction time":
            data = [x.avg_reaction_time for x in data_list]
            ylim = ylims["reaction_time"]
            size = sizes["reaction_time"]
        elif plot == "cue to reward":
            data = [x.avg_cue_to_reward for x in data_list]
            ylim = ylims["cue_to_reward"]
            size = sizes["cue_to_reward"]
        elif plot == "within correlation":
            data = [x.within_sess_corr for x in data_list]
            ylim = ylims["within"]
            size = sizes["within"]
        elif plot == "across correlation":
            data = [x.across_sess_corr for x in data_list]
            ylim = ylims["across"]
            size = sizes["across"]

        fig = plt.figure(figsize=size)
        fig.subtitle(plot)
        for d, c, n in zip(data, colors, group_names):
            plt.errorbar(
                d["session"],
                d["mean"],
                yerr=d["sem"],
                color=c,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=c,
                linewidth=1.5,
                elinewidth=1,
                ecolor=c,
                label=n,
            )
        plt.xlabel("Session", labelpad=15)
        plt.xticks(d["session"], d["session"])
        plt.ylabel(plot, labelpad=15)
        if ylim is not None:
            plt.ylim(bottom=ylim[0], top=ylim[1])
        plt.legend(bbox_to_anchor=(1, 1), loc="upper right", ncol=1)

        fig.tight_layout()

        if save is True:
            fname = os.path.join(save_path, plot)
            plt.savefig(fname + ".pdf")


# --------------------------------------------------------------------------
# ------------------------SPECIFIC LINE PLOTS-------------------------------
# --------------------------------------------------------------------------


def plot_within_session_corr(
    sessions,
    mean,
    sem,
    individual,
    plot_ind=True,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
    save_path=None,
):
    """Function to plot within session movement correlations.
        Uses the plot_mean_sem_line_plot
    """
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
        plot_ind=plot_ind,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        ylim=ylim,
        xlim=None,
        figsize=figsize,
        color=color,
        save=save,
        save_path=save_path,
    )


def plot_across_session_corr(
    sessions,
    mean,
    sem,
    individual,
    plot_ind=True,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
    save_path=None,
):
    """Function to plot across session movement correlations.
        Uses the plot_mean_sem_line_plot
    """
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
        plot_ind=plot_ind,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        ylim=ylim,
        xlim=xlim,
        figsize=figsize,
        color=color,
        save=save,
        save_path=save_path,
    )


def plot_success_rate(
    sessions,
    mean,
    sem,
    individual,
    plot_ind=True,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
    save_path=None,
):
    """Function to plot the success rate across sessions. 
       Uses the plot_mean_sem_line_plot 
    """
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
        plot_ind=plot_ind,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        ylim=ylim,
        xlim=None,
        figsize=figsize,
        color=color,
        save=save,
        save_path=save_path,
    )


def plot_cue_to_reward(
    sessions,
    mean,
    sem,
    individual,
    plot_ind=True,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
    save_path=None,
):
    """Function to plot the cut_to_reward across sessions. 
        Uses the plot_mean_sem_line_plot 
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
        plot_ind=plot_ind,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        ylim=ylim,
        xlim=None,
        figsize=figsize,
        color=color,
        save=save,
        save_path=save_path,
    )


def plot_movement_reaction_time(
    sessions,
    mean,
    sem,
    individual,
    plot_ind=True,
    ylim=None,
    figsize=(6, 5),
    color="mediumblue",
    save=False,
    save_path=None,
):
    """Function to plot the movment reaction time across sessions. 
        Uses plot_mean_sem_line_plot 
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
        plot_ind=plot_ind,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        ylim=ylim,
        xlim=None,
        figsize=figsize,
        color=color,
        save=save,
        save_path=save_path,
    )

