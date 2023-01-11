from itertools import compress

import numpy as np


def get_activity_timestamps(activity):
    """Function to get the timestamps for activity onsets
    
        INPUT PARAMETERS
            activity - np.array of binarized activity trace
        
        OUTPUT PARAMETERS
            timestamps - list of timestamps
    """
    diff = np.insert(np.diff(activity), 0, 0)
    onsets = np.nonzero(diff == 1)[0]
    offsets = np.nonzero(diff == -1)[0]
    # Check onset offset order
    if onsets[0] > offsets[0]:
        offsets = offsets[1:]
    # Check onsets and offsets are same length
    if len(onsets) > len(offsets):
        onsets = onsets[:-1]
    # Get timestamps
    timestamps = []
    for on, off in zip(onsets, offsets):
        timestamps.append((on, off))

    return timestamps


def refine_activity_timestamps(timestamps, window, max_len, sampling_rate=60):
    """Function to refine the timestamps to makes sure they fit in the dataset and 
        activity window
        
        INPUT PARAMETERS
            timestamps - list of activity timestamps

            window - tuple specifying the before and after window around the timestamp
                    in seconds

            max_len - int specifying the len of activity trace

            sampling_rate - int specifying the sampling rate
        
        OUTPUT PARAMETERS
            refined_stamps - list of the refined timestamps
    """
    refined_idxs = []
    before = np.absolute(window[0] * sampling_rate)
    after = np.absolute(window[1] * sampling_rate)
    for stamp in timestamps:
        # Remove idx if before goes before start of trace
        if stamp - before < 0:
            refined_idxs.append(False)
            continue
        # Remove idx if after goes beyond the end of the trace
        if stamp + after > max_len - 1:
            refined_idxs.append(False)
            continue

        refined_idxs.append(True)

    refined_stamps = list(compress(timestamps, refined_idxs))

    return refined_stamps


def timestamp_onset_correction(timestamps, activity_window, onset, sampling_rate):
    """Function to correct timestamps to be at activity onset
        
        INPUT PARAMETERS
             timestamps - list of timestamps
            
            activity_window - tuple specifying the activity window the activity is 
                              referenced to
            
            onset - int specifying the onset frame in reference to the activity
                    window
        
        OUTPUT PARAMETERS
            corrected_timestamps - list of the ocrrected timestamps
    """
    if np.isnan(onset):
        return timestamps

    center_point = int(np.absolute(activity_window[0] * sampling_rate))
    offset = center_point - onset
    if type(timestamps[0]) == tuple:
        corrected_timestamps = [
            (int(x[0] - offset), int(x[1] - offset)) for x in timestamps
        ]
    else:
        corrected_timestamps = [int(x - offset) for x in timestamps]

    return corrected_timestamps

