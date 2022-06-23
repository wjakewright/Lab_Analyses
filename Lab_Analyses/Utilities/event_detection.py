"""Module to find periods of activity in the dFoF traces. 
    Andy and Nathan's matlab codes"""

import numpy as np
import scipy.signal as sysignal


def event_detection(dFoF, threshold, sampling_rate):
    """Function to indentify periods of activity. Used a threshold multiplier
        to find periods above the estimated noise of the trace
    
        INPUT PARAMETERS
            dFoF - np.array of the activity where each column represents
                    a single ROI
            
            theshold - float specifying the threshold multiplier
            
            sampling_rate - int specifying the imaging rate
            
        OUTPUT PARAMETERS
            events - np.array of the binarized active periods where each column
                    represents a single ROI
            
            thesh_values = dict containing a list of the thresholds used
                            for each of the ROIs
                    
    """
    # Set the lower limit of the traces to consider
    #### Important for silent ROIs
    LOWER_TRHESH = 1
    LOWER_LIMIT = 0.2
    SEC_TO_SMOOTH = 1

    smooth_window = int(sampling_rate * SEC_TO_SMOOTH)

    # initialize the output array
    events = np.zeros(np.shape(dFoF))
    thresh_values = {"Upper Threshold": [], "Lower Threshold" [], 
                    "Artifact Limit": []}

    # Analyze each ROI
    for roi in range(dFoF.shape[0]):

        # Estimate the noise of the traces using the mirrored below-zero trace
        below_zero = roi[roi < 0]
        noise_est = np.nanstd(np.concatenate((below_zero, -below_zero)))

        # Set threshold values
        high_thresh = noise_est * threshold
        low_thresh = noise_est * LOWER_THRESH
        # Account for movement artifacts by using largest negative deflections
        artifact_limit = np.absolute(np.percentile(below_trace, 5))
        if high_thresh < artifact_limit:
            high_thresh = artifact_limit
        if high_thresh < LOWER_LIMIT:
            high_thresh = LOWER_LIMIT
        
        thresh_values["Upper Threshold"].append(high_thresh)
        thresh_values["Lower Threshold"].append(low_thresh)
        thresh_values["Artifact Limit"].append(artifact_limit)

        # Generate a smoothed trace
        temp_smooth = sysignal.savgol_filter(roi, smooth_window, 2)
        # Find periods above the thrsholds
        above_low = temp_smooth > low_thresh
        above_high = temp_smooth > high_thresh

        # Fill in high portions where low threshold is not crossed
        ## E.g., dips down but not to baseline, so continuously active

        # Find edges of long-smooth above_thesh periods
        pad = np.zero(1)
        thresh_low_start = np.diff(np.concatenate((pad, above_low, pad))) == 1
        thresh_low_stop =  np.diff(np.concatenate((pad, above_low, pad))) == -1
        thresh_high_start = np.diff(np.concatenate((pad, above_high, pad))) == 1
        thresh_high_stop = np.diff(np.concatenate((pad, above_high, pad))) == -1

        thresh_high_start_idx = np.nonzero(thresh_high_start)[0]
        thresh_high_stop_idx = np.nonzero(thresh_high_stop)[0]

        # Locate transitions from low threshold to high threshold
        thresh_low_high_smooth_idx = []
        for start, stop in zip(thresh_high_start_idx, thresh_high_stop_idx):
            transition = find_low_high_transitions(start, stop, thresh_low_start)
            thresh_low_high_smooth_idx.append(transition)




def find_low_high_transitions(start_idx, stop_idx, thresh_low_start):
    """Helper function to find transitions from low threshold to high threshold"""
    rev_low_start = thresh_low_start[start_idx:0:-1]
    new_start = start_idx - np.nonzero(rev_low_start)[0][0] + 1
    low_high_idx = np.arange(new_start, stop_idx)

    return low_high_idx




