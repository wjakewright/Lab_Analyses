import numpy as np

from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.coactivity_functions import (
    calculate_coactivity,
    calculate_relative_onset,
)


def calculate_dendrite_coactivity_rate(
    spine_activity,
    dendrite_activity,
    duration,
    sampling_rate=60,
    activity_window=(-2, 4),
    norm_method="mean",
    iterations=10000,
):
    """Function to calculate the coactivity rates between a given spine and its parent 
        dendrite
        
        INPUT PARAMETERS
            spine_activity - np.array of the spine's binarized activity
            
            dendrite_activity - np.array of the spine's parent dendrite binarized activity
            
            duration - int specifying how many frames the activity is taken from.
                        For instances when it has been constrained

            sampling_rate - int specifying the imaging sampling rate

            activity_window - tuple specifying the window to analyze for onset analysis
            
            norm_method - str specifying how to normalize the coactivity rates
            
            iterations - int specifying how many iterations to perform to calculate chance

        OUTPUT PARAMETERS
            coactive_trace - np.array of the binarized coactivity

            coactive_event_num - int specifying the number of coactivity events

            coactivity_rate - float specifying the coactivity rate

            coactivity_rate_norm - float specifying the normalized coactivity rate

            shuff_coactivity_rate - np.array of the shuffled coactivity rate

            shuff_coactivity_rate_norm - np.array of the shuffled normalized coactivity rate

            relative_diff - float of the relative difference between real and shuffled rates

            relative_diff_norm - float of the relative difference between real and shuff norm rates

            fraction_dend_coactive - float of the fraction of dendritic activity that is coactive

            fraction_spine_coactive - float of the fraction of spine activity that is coactive

            relative_spine_onset - float of the avg spine onset relative to dendrite

            spine_onset_jitter - float of the relative onset deviation
            
    """
    # Set up some constants
    SMALLEST_SHIFT = sampling_rate
    BIGGEST_SHIFT = 600 * sampling_rate
    SHIFT_RANGE = (SMALLEST_SHIFT, BIGGEST_SHIFT)

    # Calulate the real coactivity rates
    (
        coactivity_rate,
        coactivity_rate_norm,
        fraction_spine_coactive,
        _,
        coactive_trace,
    ) = calculate_coactivity(
        spine_activity,
        dendrite_activity,
        norm_method=norm_method,
        duration=duration,
        sampling_rate=sampling_rate,
    )
    (_, _, fraction_dend_coactive, _, _,) = calculate_coactivity(
        dendrite_activity,
        spine_activity,
        norm_method=norm_method,
        duration=duration,
        sampling_rate=sampling_rate,
    )

    # Get event num
    coactive_event_num = len(np.nonzero(np.diff(coactive_trace) == 1)[0])

    # Calculate the relative onset
    relative_spine_onset, spine_onset_jitter = calculate_relative_onset(
        spine_activity, dendrite_activity, coactive_trace, sampling_rate, (-2, 2),
    )

    # Calculate the shuffled coactivity
    shuff_coactivity_rate = np.zeros(iterations)
    shuff_coactivity_rate_norm = np.zeros(iterations)

    ## Iterate through each iteration
    for i in range(iterations):
        ## Shuffle the activity
        shuff_activity = d_utils.roll_2d_array(
            spine_activity.reshape(-1, 1), SHIFT_RANGE, axis=1
        ).flatten()
        ## Get the shuffled coactivity rates
        (shuff_rate, shuff_rate_norm, _, _, _,) = calculate_coactivity(
            shuff_activity,
            dendrite_activity,
            norm_method=norm_method,
            duration=duration,
            sampling_rate=sampling_rate,
        )
        shuff_coactivity_rate[i] = shuff_rate
        shuff_coactivity_rate_norm[i] = shuff_rate_norm

    # Calculate the relative differences
    if (np.nanmedian(shuff_coactivity_rate) == 0) and (coactivity_rate == 0):
        relative_diff = 0
    else:
        relative_diff = (coactivity_rate - np.nanmedian(shuff_coactivity_rate)) / (
            coactivity_rate + np.nanmedian(shuff_coactivity_rate)
        )

    if (np.nanmedian(shuff_coactivity_rate_norm) == 0) and (coactivity_rate_norm == 0):
        relative_diff_norm = 0
    else:
        relative_diff_norm = (
            coactivity_rate_norm - np.nanmedian(shuff_coactivity_rate_norm)
        ) / (coactivity_rate_norm + np.nanmedian(shuff_coactivity_rate_norm))

    return (
        coactive_trace,
        coactive_event_num,
        coactivity_rate,
        coactivity_rate_norm,
        shuff_coactivity_rate,
        shuff_coactivity_rate_norm,
        relative_diff,
        relative_diff_norm,
        fraction_spine_coactive,
        fraction_dend_coactive,
        relative_spine_onset,
        spine_onset_jitter,
    )
