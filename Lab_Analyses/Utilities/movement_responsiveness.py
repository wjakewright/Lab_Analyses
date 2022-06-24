"""Module for determining if an ROI displays movement-related activity"""

import random

import numpy as np


def movement_responsiveness(dFoF, active_lever, permutations=10000):
    """Function to determine if ROIs display movement related activity
        
        INPUT PARAMTERS
            dFoF - 2d np.array of the dFoF activity, with each ROI in a column
            
            active_leer - 1d np.array of the binarized lever trace 
            
            permutations - int specifying how many shuffles of the data to perform
                            Default is set to 10,000 shuffles
        OUTPUT PARAMETERS
    
    """

    # Setup outputs
    movement_related = []
    movement_activities = {"Real": [], "Shuffled": []}

    # Get the movement durations and inter-movement durations for shuffling
    ## Get the starts and stops of each movement
    pad = np.zeros(1)
    boundary_frames = np.diff(np.concatenate((pad, active_lever, pad))) != 0
    boundary_frames = np.nonzero(boundary_frames)[0]
    active_lever_splits = []
    for i in boundary_frames[1:]:
        start = boundary_frames[i - 1]
        stop = boundary_frames[i]
        split = active_lever[start : stop + 1]
        active_lever_splits.append(split)

    # Analyze each ROI
    for i in range(dFoF.shape[1]):
        activity = dFoF[:, i].reshape(-1, 1)  # Make it a column vector
        movement_activity = np.dot(activity, active_lever)

        # Perform shuffles
        shuffled_activity = []
        for j in range(permutations):
            # Shuffle the movement epochs
            shuffled_splits = random.shuffle(active_lever_splits)
            shuffled_active_lever = np.concatenate((shuffled_splits))
            shuffled_activity.append(np.dot(activity, shuffled_active_lever))

        # Assess significance
