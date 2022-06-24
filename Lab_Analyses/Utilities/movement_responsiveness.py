"""Module for determining if an ROI displays movement-related activity"""

import random

import numpy as np


def movement_responsiveness(dFoF, active_lever, permutations=10000, percentile=97.5):
    """Function to determine if ROIs display movement related activity
        
        INPUT PARAMTERS
            dFoF - 2d np.array of the dFoF activity, with each ROI in a column
            
            active_leer - 1d np.array of the binarized lever trace 
            
            permutations - int specifying how many shuffles of the data to perform
                            Default is set to 10,000 shuffles

            percentile - float specifying what percentile you are using as a cutoff
                        Default is 97.5     

        OUTPUT PARAMETERS
            movement_rois - boolean list indicating if an roi is movement responsive

            silent_rois - boolean list indicatinf if an roi is movement silent

            movement_activities - dict with lists containing the Real and Shuffled
                                activities of each ROI as well as the Percentile
                                bounds
    
    """

    # Setup outputs
    movement_rois = []
    silent_rois = []
    movement_activities = {"Real": [], "Shuffled": [], "Percentile": []}

    # Get the movement durations and inter-movement durations for shuffling
    ## Get the starts and stops of each movement
    pad = np.zeros(1)
    boundary_frames = np.diff(np.concatenate((pad, active_lever, pad))) != 0
    boundary_frames = np.nonzero(boundary_frames)[0]
    active_lever_splits = []
    active_lever_splits.append(active_lever[: boundary_frames[0]])
    for i, _ in enumerate(boundary_frames[1:]):

        start = boundary_frames[i]
        stop = boundary_frames[i + 1]
        split = active_lever[start:stop]
        print(len(split))
        active_lever_splits.append(np.array(split))
    active_lever_splits.append(active_lever[boundary_frames[-1] :])
    print(np.sum([len(x) for x in active_lever_splits]))

    # Analyze each ROI
    for i in range(dFoF.shape[1]):
        activity = dFoF[:, i]
        active_lever = active_lever
        movement_activity = np.dot(activity, active_lever)

        # Perform shuffles
        shuffled_activity = []
        for j in range(permutations):
            # Shuffle the movement epochs
            shuffled_splits = random.sample(
                active_lever_splits, len(active_lever_splits)
            )
            shuffled_active_lever = np.concatenate((shuffled_splits))
            shuffled_active_lever = shuffled_active_lever
            shuffled_activity.append(np.dot(activity, shuffled_active_lever))

        # Assess significance
        upper = np.percentile(shuffled_activity, percentile)
        lower = np.percentile(shuffled_activity, 100 - percentile)

        if movement_activity > upper:
            move_roi = True
            quiet_roi = False
        elif movement_activity < lower:
            move_roi = False
            quite_roi = True
        else:
            move_roi = False
            quiet_roi = False

        movement_rois.append(move_roi)
        silent_rois.append(quiet_roi)
        movement_activities["Real"].append(movement_activity)
        movement_activities["Shuffled"].append(shuffled_activity)
        movement_activities["Percentile"].append((lower, upper))

    return movement_rois, silent_rois, movement_activities
