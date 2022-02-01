"""Module to analyze lever press behavior. Gets lever press traces and 
    correlates lever pressing within and across sessions. 
    
    Takes pickle files output from process_lever_behavior.py
    
    CREATOR
        William (Jake) Wright - 2/1/2022
"""

from dataclasses import dataclass

import numpy as np
import scipy.signal as sysignal


def summarize_lever_behavior(file):
    """Function to analyze lever press behavior of a single mouse across all sessions
    
        INPUT PARAMETERS
            file - An object (Processed_Lever_Data dataclass) containing the processed
                    lever behavior for a single session for a single mouse
    """
    rewards = 0
    movestartfault = 0
    maxtrialnum = 110

    # Smooth lick data if any is present
    if "Lick" in file.xsg_data.channels.keys():
        lick_data_resample = sysignal.resample_poly(
            file.xsg_data.channels["Lick"], up=1, down=10
        )
        butter = sysignal.butter(4, (5 / 500), "low")
        lick_data_smooth = sysignal.filtfilt(
            butter[0],
            butter[1],
            lick_data_resample,
            axis=0,
            padtype="odd",
            padlen=3 * (max(len(butter[1]), len(butter[0])) - 1),
        )
        file.lick_data_smooth = lick_data_smooth
    else:
        file.lick_data_smooth = []

    if not file.behavior_frames.size == 0:
        output = summarize_image_lever_behavior(file)


def summarize_image_lever_behavior(file):
    """Function to summarize lever press behavior for sessions that were imaged"""

    rewards = 0
    trials = len(file.behavior_frames)
    movements_only = file.lever_force_smooth * file.lever_active
    boundary_frames = np.nonzero(
        np.diff(
            np.insert(file.lever_active, [0, len(file.lever_active)], np.Inf, axis=0)
            != 0
        )
    )[0]

    for num, trial in file.behavior_frames:
        if not trial.states.reward == 0:
            a = "stopped here"


@dataclass
class Session_Lever_Data:
    """Dataclass for storing the analyzed lever press data of a single sesson for 
        a single mouse"""

    trials: int
    rewards: int

