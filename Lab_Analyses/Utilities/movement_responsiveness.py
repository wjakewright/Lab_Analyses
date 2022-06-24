"""Module for determining if an ROI displays movement-related activity"""

import numpy as np


def movement_responsiveness(dFoF, active_lever, permutations=10000):
    """Function to determine if ROIs display movement related activity
        
        INPUT PARAMTERS
            dFoF - 2d np.array of the dFoF activity, with each ROI in a column
            
            active_lever - 1d np.array of the binarized lever trace 
            
            permutations - int specifying how many shuffles of the data to perform
                            Default is set to 10,000 shuffles
        OUTPUT PARAMETERS
    
    """

    # Setup outputs
    movement_related = []
    movement_activities = {"Real": [], "Shuffled": []}

    # Analyze each ROI
    for i in range(dFoF.shape[1]):
        activity = dFoF[:, i].reshape(-1, 1)  # Make it a column vector
        movement_activity = np.dot(activity, active_lever)

        # Perform shuffles
        shuffled_activity = []
        for j in range(permutations):
            # Get the movement durations and inter-movement durations
            ## Get the starts and stops of each movement
            pad = np.zeros(1)
            move_starts = np.diff(np.concatenate((pad, active_lever, pad))) == 1
            move_starts = np.nonzero(move_starts)[0]
            move_stops = np.diff(np.concatenate((pad, active_lever, pad))) == -1
            move_stops = np.nonzero(move_stops)[0]
