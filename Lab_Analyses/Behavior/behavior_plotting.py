"""Module to perform the plotting of behavioral data
    
    CREATOR
        William (Jake) Wright 02/08/2022
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()
sns.set_style("ticks")


def plot_session_rewarded_lever_presses(mouse_lever_data, session, figsize=(8,5), x_lim=None, y_lim=None):
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
    move_mat = mouse_lever_data.all_movements[session-1]
    for row, _ in enumerate(move_mat[:,0]):
        plt.plot(move_mat[row,:], color="gray", linewidth=0.5, alpha=0.3)
    
    # Plot average movement
    avg_move = mouse_lever_data.average_movements[session-1]
    plt.plot(avg_move, color="black", linewidth=1)

    if x_lim is not None:
        plt.xlim(left=x_lim[0], right=x_lim[1])
    if y_lim is not None:
        plt.ylim(bottom=y_lim[0], top=y_lim[1])
    


    


