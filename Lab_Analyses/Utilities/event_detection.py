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
            active_traces - np.array of the binarized active periods where each column
                            represents a single ROI
            
            floored_traces - np.array of the traces, which have been floored to zero during
                            inactive periods
            
            thesh_values = dict containing a list of the thresholds used
                            for each of the ROIs
                    
    """
    # Set the lower limit of the traces to consider
    #### Important for silent ROIs
    LOWER_THRESH = 1
    LOWER_LIMIT = 0.2
    SEC_TO_SMOOTH = 0.5

    smooth_window = int(sampling_rate * SEC_TO_SMOOTH)
    # Make sure smooth window is odd
    if not smooth_window % 2:
        smooth_window = smooth_window + 1

    # initialize the output array
    active_traces = np.zeros(np.shape(dFoF))
    floored_traces = np.zeros(np.shape(dFoF))
    thresh_values = {"Upper Threshold": [], "Lower Threshold": [], "Artifact Limit": []}

    # Analyze each ROI
    for i in range(dFoF.shape[1]):
        roi = dFoF[:, i]
        # Estimate the noise of the traces using the mirrored below-zero trace
        below_zero = roi[roi < 0]
        noise_est = np.nanstd(np.concatenate((below_zero, -below_zero)))

        # Set threshold values
        high_thresh = noise_est * threshold
        low_thresh = noise_est * LOWER_THRESH
        # Account for movement artifacts by using largest negative deflections
        artifact_limit = np.absolute(np.percentile(below_zero, 5))
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
        pad = np.zeros(1)
        thresh_low_start = np.diff(np.concatenate((pad, above_low, pad))) == 1
        thresh_low_stop = np.diff(np.concatenate((pad, above_low, pad))) == -1
        thresh_high_start = np.diff(np.concatenate((pad, above_high, pad))) == 1
        thresh_high_stop = np.diff(np.concatenate((pad, above_high, pad))) == -1
        thresh_high_start_idx = np.nonzero(thresh_high_start)[0]
        thresh_high_stop_idx = np.nonzero(thresh_high_stop)[0]

        # Locate transitions from low threshold to high threshold
        thresh_low_high_smooth_idx = []
        for start, stop in zip(thresh_high_start_idx, thresh_high_stop_idx):
            transition = find_low_high_transitions(start, stop, thresh_low_start)
            thresh_low_high_smooth_idx.append(transition)

        # Exclude periods before and after the imaging session
        to_exclude = []
        for x in thresh_low_high_smooth_idx:
            to_exclude.append(any(x <= 0) or any(x > len(roi)))

        # Refine start times of activity when dFoF goes above high thresh
        thresh_low_high_smooth_idx = np.array(thresh_low_high_smooth_idx, dtype=object)
        thresh_low_high_raw_idx = []
        for idx in thresh_low_high_smooth_idx[[not x for x in to_exclude]]:
            thresh_low_high_raw_idx.append(refine_start_times(idx, roi, high_thresh))

        # Exlude periods before and after the imaging session
        to_exclude_2 = []
        for x in thresh_low_high_raw_idx:
            to_exclude_2.append(any(x <= 0) or any(x > len(roi)))
        for exclude in to_exclude_2:
            thresh_low_high_raw_idx[exclude] = np.array([])
        try:
            thresh_low_high_raw_idx = np.concatenate(thresh_low_high_raw_idx).astype(
                int
            )
        except ValueError:
            thresh_low_high_raw_idx = []

        # Find continuous active portions
        active_trace = np.zeros(len(roi))

        active_trace[thresh_low_high_raw_idx] = 1

        # Floor activity trace during inactive portions
        inactive_idxs = np.nonzero(active_trace == 0)[0]
        floored_trace = np.copy(roi)
        floored_trace[inactive_idxs] = 0

        active_traces[:, i] = active_trace
        floored_traces[:, i] = floored_trace

    return active_traces, floored_traces, thresh_values


def find_low_high_transitions(start_idx, stop_idx, thresh_low_start):
    """Helper function to find transitions from low threshold to high threshold"""
    rev_low_start = thresh_low_start[start_idx:0:-1]
    try:
        new_start = start_idx - np.nonzero(rev_low_start)[0][0] + 1
    except IndexError:
        new_start = start_idx
    low_high_idx = np.arange(new_start, stop_idx)

    return low_high_idx


def refine_start_times(idx, trace, high_thresh):
    """Helper function to help refine start times when dFoF goes above high thresh"""
    start = idx[0]
    try:
        u1 = np.nonzero(trace[idx[0] :] > high_thresh)[0][0]
    except IndexError:
        u1 = 0
    try:
        u2 = np.nonzero(trace[start + u1 : 0 : -1] < high_thresh)[0][0]
    except IndexError:
        u2 = 0
    new_idx = np.arange(start + u1 - u2, idx[-1])

    return new_idx

