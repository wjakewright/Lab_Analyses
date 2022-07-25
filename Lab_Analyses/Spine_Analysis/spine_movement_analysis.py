"""Module to handle some of the movement analysis for the spine activity data"""

from itertools import compress

import numpy as np
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    load_spine_datasets,
)
from Lab_Analyses.Utilities import data_utilities as d_utils


def spine_movement_activity(
    data,
    activity_type="spine_GluSnFr_processed_dFoF",
    exclude="Eliminated",
    sampling_rate=60,
    rewarded=False,
):
    """Function to get the spine activity during movement epochs. 
        Gets the mean before and during movement, as well as the each trace and mean±sem trace
        
        INPUT PARAMETERS
            data - spind_data object. (e.g., Dual_Channel_Spine_Data
            
            activity_type - str specifying what type of activity you wish to use. Must match the 
                            field name of the data object
            
            exclude - string specifying types of spines you wish to exlude from analysis
            
            sampling_rate - float or int specifying the imaging rate the data was collected with

            rewarded - boolean specifying whether to use only rewarded movements or not
            
        OUTPUT PARAMETERS
            all_befores - 
            
            all_durings - 
            
            movement_epochs - 
            
            movement_mean_sems - 
    """

    before_window = int(2 * sampling_rate)

    # Get the activity and behavior out of the object
    activity = getattr(data, activity_type)
    spine_ids = data.spine_ids
    if rewarded:
        movement = data.rewarded_movement_binary
    else:
        movement = data.lever_active

    # Get indexes of spines to analzyed
    if exclude:
        exclude_spines = find_spine_classes(data.spine_flags, exclude)
        exclude_spines = np.array([not x for x in exclude_spines])
        spine_ids = list(compress(spine_ids, exclude_spines))
        activity = activity[:, exclude_spines]

    # Get the movement onset and offset timestamps
    movement_diff = np.insert(np.diff(movement), 0, 0, axis=0)
    movement_onsets = np.nonzero(movement_diff == 1)[0]
    movement_offsets = np.nonzero(movement_diff == -1)[0]

    ## made sure onsets and offsets are the same length
    if len(movement_onsets) > len(movement_offsets):
        # Drop last onset if there is no offset
        movement_onsets = movement_onsets[:-1]
    elif len(movement_onsets) < len(movement_offsets):
        # Drop first offest if there is no onset for it
        movement_offsets = movement_offsets[1:]

    timestamps = []
    for onset, offset in zip(movement_onsets, movement_offsets):
        stamp = (onset, offset)
        timestamps.append(stamp)

    # Refine the timestamps
    refined_idxs = []
    for i, stamp in enumerate(timestamps):
        # remove first movement if to early
        if i == 0:
            if stamp[0] - before_window < 0:
                refined_idxs.append(False)
            else:
                refined_idxs.append(True)
            continue
        # remove movements that go beyond activity window at end
        if i == len(timestamps) - 1:
            if stamp[0] + before_window >= len(activity[:, 0]):
                refined_idxs.append(False)
            else:
                refined_idxs.append(True)
            continue

        # remove movements with another movement 1s before
        if stamp[0] - before_window <= timestamps[i - 1][1]:
            refined_idxs.append(False)
        else:
            refined_idxs.append(True)

    timestamps = list(compress(timestamps, refined_idxs))
    epoch_timestamps = [x[0] for x in timestamps]

    # Get all befores, and durings
    all_befores, all_durings = d_utils.get_before_during_means(
        activity, timestamps, window=1, sampling_rate=sampling_rate
    )
    # Get the traces and mean±sem trace
    movement_epochs, movement_mean_sems = d_utils.get_trace_mean_sem(
        activity,
        spine_ids,
        epoch_timestamps,
        window=(-2, 2),
        sampling_rate=sampling_rate,
    )

    return all_befores, all_durings, movement_epochs, movement_mean_sems
