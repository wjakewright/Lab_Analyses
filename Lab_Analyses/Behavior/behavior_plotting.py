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


def plot_session_rewarded_lever_presses(movement_matrix, movement_avg):
    """Function to plot each rewarded lever press as well as the 
        average rewarded lever press"""

