import numpy as np

from Lab_Analyses.Spine_Analysis_v2.calculate_distance_coactivity_rate import (
    calculate_distance_coactivity_rate,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.test_utilities import significant_vs_shuffle


def calculate_chance_local_coactivity(
    spine_activity,
    spine_positions,
    spine_flags,
    spine_groupings,
    bin_size=5,
    cluster_dist=5,
    constrain_matrix=None,
    partner_list=None,
    sampling_rate=60,
    norm_method="mean",
    alpha=0.05,
    iterations=10000,
):
    """Function to determine the chance levels of local coactivity based
    on spines activity rates

    INPUT PARAMETERS
        spine_activity - 2d np.array of the spine binary activity traces

        spine_positions - np.array corresponding to each spine's position on dendrite

        spine_flags - list of the spine flags

        spiine_groupings - list of the corresponding groupings of spines on the dendrites

        cluster_dist - int or float specifying the distance to be considered local

        constrain_matrix - np.array of the binarized events to constrain
                          the coactivity to. (e.g., dendritic events, movement
                          periods)

        partner_list - boolean list specifying a subset of spines to anlyze coactivity
                        rates for (e.g., movement spines)

        sampling_rate - int specifying the imaging sampling rate

        norm_method - str specifying how you want to normalize the coactivity rate
                        Accepts "mean" to normalize by the geo mean, or "freq" to
                        normalize by the target spine frequency

        alpah - float speicifying the significance level

        iterations - int specifying how many shuffles to perform

    OUTPUT PARAMETERS
        shuffled_coactivity_rate - 2d np.array of the shuffled coactivity rates. Each
                                    row represents a shuffle and each col a spine

        shuffled_coactivity_rate_norm - 2d np.array of the shuffled norm coactivity rates

        relative_diff - np.array of the relative difference between real and median
                        shuffled coactivity rate

        relative_diff_norm - np.array of the relative differences between real and
                             median shuffled norm coactivity rates

        coactive_spines - boolean array specifying whether each spine's coactivity
                          is above chance

        coactive_norm_spines - boolean array specifying whether each spine's
                                norm. coactivity is above chance
    """
    # setup some constants
    SMALLEST_SHIFT = sampling_rate  # smallest shift of 1 second
    BIGGEST_SHIFT = 600 * sampling_rate  # 10 min shift
    SHIFT_RANGE = (SMALLEST_SHIFT, BIGGEST_SHIFT)
    CLUST_IDX = int(cluster_dist / bin_size) - 1
    # Sort out spine groupings to ensure it is iterable
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # Setup shuffled outputs
    full_shuffled_coactivity_rate = []
    full_shuffled_coactivity_rate_norm = []
    shuffled_coactivity_rate = np.zeros((iterations, spine_activity.shape[1])) * np.nan
    shuffled_coactivity_rate_norm = np.zeros(shuffled_coactivity_rate.shape) * np.nan

    # Get the real coactivity rates
    real_coactivity, _, real_coactivity_norm, _, _ = calculate_distance_coactivity_rate(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix,
        partner_list,
        bin_size,
        sampling_rate,
        norm_method,
    )
    real_coactivity = real_coactivity[CLUST_IDX, :]
    real_coactivity_norm = real_coactivity_norm[CLUST_IDX, :]

    # Iterate through each shuffle
    for i in range(iterations):
        # Shuffle the activity
        ## Switching the chunk shuffling
        # shuffled_activity = d_utils.roll_2d_array(spine_activity, SHIFT_RANGE, axis=1)
        shuffled_activity = d_utils.roll_2d_array(spine_activity, SHIFT_RANGE, axis=1)
        # Get the shuffled coactivity
        (
            shuff_coactivity,
            _,
            shuff_coactivity_norm,
            _,
            _,
        ) = calculate_distance_coactivity_rate(
            shuffled_activity,
            spine_positions,
            spine_flags,
            spine_groupings,
            constrain_matrix,
            partner_list,
            bin_size,
            sampling_rate,
            norm_method,
        )
        full_shuffled_coactivity_rate.append(shuff_coactivity)
        full_shuffled_coactivity_rate_norm.append(shuff_coactivity_norm)
        shuff_coactivity = shuff_coactivity[CLUST_IDX, :]
        shuff_coactivity_norm = shuff_coactivity_norm[CLUST_IDX, :]
        # Store values
        shuffled_coactivity_rate[i, :] = shuff_coactivity
        shuffled_coactivity_rate_norm[i, :] = shuff_coactivity_norm

    # Reorganize the shuffled distance coactivity
    shuff_stacked = np.dstack(full_shuffled_coactivity_rate)
    full_shuffled_coactivity_rate = [
        shuff_stacked[:, i, :] for i in range(shuff_stacked.shape[1])
    ]
    shuff_norm_stacked = np.dstack(full_shuffled_coactivity_rate_norm)
    full_shuffled_coactivity_rate_norm = [
        shuff_norm_stacked[:, i, :] for i in range(shuff_norm_stacked.shape[1])
    ]

    # Calculate the relative differences
    ## Get median of shuffles for each spine
    shuff_coactivity_median = np.nanmedian(shuffled_coactivity_rate, axis=0)
    shuff_coactivity_norm_median = np.nanmedian(shuffled_coactivity_rate_norm, axis=0)
    ## Calculate difference
    # relative_diff = d_utils.normalized_relative_difference(
    #    shuff_coactivity_median, real_coactivity
    # )
    # relative_diff_norm = d_utils.normalized_relative_difference(
    #    shuff_coactivity_norm_median, real_coactivity_norm
    # )

    # Determine significance of coactivity
    _, sig = significant_vs_shuffle(
        real_coactivity, shuffled_coactivity_rate, alpha, nan_policy="omit"
    )
    _, sig_norm = significant_vs_shuffle(
        real_coactivity_norm, shuffled_coactivity_rate_norm, alpha, nan_policy="omit"
    )
    sig[sig < 1] = 0
    sig_norm[sig_norm < 1] = 0
    coactive_spines = sig.astype(bool)
    coactive_norm_spines = sig_norm.astype(bool)

    return (
        real_coactivity,
        real_coactivity_norm,
        shuffled_coactivity_rate,
        shuffled_coactivity_rate_norm,
        full_shuffled_coactivity_rate,
        full_shuffled_coactivity_rate_norm,
        coactive_spines,
        coactive_norm_spines,
    )
