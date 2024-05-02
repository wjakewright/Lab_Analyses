import numpy as np

from Lab_Analyses.Spine_Analysis_v2.calculate_dendrite_coactivity_rate import (
    calculate_dendrite_coactivity_rate,
)
from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    find_nearby_spines,
    find_present_spines,
)
from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities.coactivity_functions import (
    calculate_coactivity,
    calculate_relative_onset,
    get_conservative_coactive_binary,
)
from Lab_Analyses.Utilities.mean_trace_functions import analyze_event_activity


def spine_dendrite_event_analysis(
    spine_activity,
    spine_dFoF,
    spine_calcium,
    dendrite_activity,
    dendrite_dFoF,
    spine_flags,
    spine_positions,
    spine_groupings,
    activity_window=(-2, 4),
    cluster_dist=5,
    constrain_matrix=None,
    sampling_rate=60,
    volume_norm=None,
    extend=None,
    activity_type="all",
    iterations=1000,
):
    """Function to analyze spine-dendrite coactivity and assess how their activity
    during these events

    INPUT PARAMETERS
        spine_activity - 2d np.array of the binarized spine activity traces.
                         Each column represents a spine

        spine_dFoF - 2d np.array of the GluSnFr dFoF traces for each spine

        spine_calcium - 2d np.array of the Calcium dFoF traces for each spine

        dendrite_activity - 2d np.array of the binarized dendrite activity trac3es
                            Each column is the activity of the partent dendrite
                            for each spine

        dendrite_dFoF - 2d np.array of the dendrite calcium dFoF

        spine_flags - list of the spine flags

        spine_positions - np.array of the corresponding spine positions

        spine_groupings - list of the groupings of spines on the same dendrite

        activity_window - tuple specifying the activity window in sec to analyze

        cluster_dist - int or float specifying distance (um) that is considered local

        constrain_matrix - np.array of binarized events to constrain the coactivity to
                            (e.g., movement periods)

        sampling_rate - int specifying the imaging sampling rate

        volume_norm - tuple of lists constaining the constants to normalize GluSnFr
                      and calcium by

        extend - float specifying the duration in seconds to extend the window for
                 considering coactivity. Default is none for no extension

        activity_type - str specifying whether to consider 'all' events or events with
                        'no local' coactivity or with 'local' coactivity

        iterations - int specifying how many iterations to perform when testing against
                      chance

    OUTPUT PARAMETERS
        nearby_spine_idxs - list containing the idxs of nearby spines for each spine

        coactive_binary - 2d np.array of the when each spine is coactive with dendrite

        dendrite_coactive_event_num - np.array of the number of events each spine is
                                      coactive with the dendrite

        dendrite_coactivity_rate - np.array of the coactivity rate of each spine with
                                    the dendrite

        dendrite_coactivity_rate_norm - np.array of the coactivity rate of each spine
                                        with the dendrite normalized

        shuff_dendrite_coactivity_rate - 2d np.array of the shuffled coactivity rates
                                        Each row represents a shuffle and each col a spine

        shuff_dendrite_coactivity_rate_norm - 2d np.array of the shuffled normalized
                                              coactivity rates

        above_chance_caoctivity - np.array of the relative difference between real and
                                  shuffled coactivity rates for each spine

        above_chance_coactivity_norm - np.array of the relative difference between
                                        real and shuffled norm coactivity rates

        fraction_dend_coactive - np.array of the fraction of dendritic events that are
                                coactive with each spine

        fraction_spine_coactive - np.array of the fraction of spine activity that is
                                  coactive with the dendrite for each spine

        spine_coactive_amplitude - np.array of each spines' mean peak amplitude when
                                    coactive with dendrite

        spine_coactive_calcium_amplitude - np.array of each spines' mean peak calcium
                                            when coactive with dendrite

        dendrite_coactive_amplitude - np.array of the dendrites' mean peak amplitude
                                      when coactive with each spine

        relative_onset - np.array of spine GluSnFr onsets relative to dendrite onset
                        during coactive events

        spine_coactive_traces - list of 2d np.array of each spines' GluSnFr traces
                                during coactivity events

        spine_coactive_calcium_traces - list of 2d np.array of each spines' calcium
                                        traces during coactivity events

        dendrite_coactive_traces - list of 2d np.array of the dendrites activity when
                                    coactive with each spine
    """
    # Sort out the spine groupings
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # Get the nearby spine idxs
    nearby_spine_idxs = find_nearby_spines(
        spine_positions, spine_flags, spine_groupings, None, cluster_dist
    )
    present_spines = find_present_spines(spine_flags)

    # Constrain activity if necessary
    if constrain_matrix is not None:
        if len(constrain_matrix.shape) == 1:
            constrain_matrix = constrain_matrix.reshape(-1, 1)
            duration = np.nansum(constrain_matrix)
        else:
            duration = [
                np.nansum(constrain_matrix[:, i])
                for i in range(constrain_matrix.shape[1])
            ]
        spine_activity = spine_activity * constrain_matrix
    else:
        duration = None

    # Extend dendrite activity if specified
    if extend is not None:
        ref_dend_activity = extend_dendrite_activity(
            np.copy(dendrite_activity), extend, sampling_rate
        )
    else:
        ref_dend_activity = dendrite_activity

    # Get normalization constants if specified
    if volume_norm is not None:
        glu_norm_constants = volume_norm[0]
        ca_norm_constants = volume_norm[1]
    else:
        glu_norm_constants = None
        ca_norm_constants = None

    # Set up some outputs
    coactive_binary = np.zeros(spine_activity.shape)
    dendrite_coactive_event_num = np.zeros(spine_activity.shape[1])
    dendrite_coactivity_rate = np.zeros(spine_activity.shape[1])
    dendrite_coactivity_rate_norm = np.zeros(spine_activity.shape[1])
    shuff_dendrite_coactivity_rate = np.zeros((iterations, spine_activity.shape[1]))
    shuff_dendrite_coactivity_rate_norm = np.zeros(
        (iterations, spine_activity.shape[1])
    )
    above_chance_coactivity = np.zeros(spine_activity.shape[1])
    above_chance_coactivity_norm = np.zeros(spine_activity.shape[1])
    fraction_dend_coactive = np.zeros(spine_activity.shape[1])
    fraction_spine_coactive = np.zeros(spine_activity.shape[1])
    coactive_spines = np.zeros(spine_activity.shape[1]).astype(bool)
    coactive_spines_norm = np.zeros(spine_activity.shape[1]).astype(bool)
    relative_onsets = np.zeros(spine_activity.shape[1]) * np.nan

    # coactive_onsets = [[] for i in range(spine_activity.shape[1])]
    dend_onsets = [[] for i in range(spine_activity.shape[1])]

    # Iterate through each spine
    for spine in range(spine_activity.shape[1]):
        # Skip absent spines
        if present_spines[spine] == False:
            continue
        # Constrain spine activity for local or without local activity
        if activity_type != "all":
            n_activity = spine_activity[:, nearby_spine_idxs[spine]]
            combined_activity = np.nansum(n_activity, axis=1)
            combined_activity[combined_activity > 1] = 1
            if activity_type == "local":
                # Old method
                _, _, _, _, s_activity = calculate_coactivity(
                    spine_activity[:, spine],
                    combined_activity,
                    sampling_rate=sampling_rate,
                )
                # s_activity, _ = get_conservative_coactive_binary(spine_activity[:, spine], combined_activity)
            elif activity_type == "no local":
                # Old method
                combined_inactivity = 1 - combined_activity
                _, _, _, _, s_activity = calculate_coactivity(
                    spine_activity[:, spine],
                    combined_inactivity,
                    sampling_rate=sampling_rate,
                )
                # _, s_activity = get_conservative_coactive_binary(spine_activity[:, spine], combined_activity)

        else:
            s_activity = spine_activity[:, spine]

        d_activity = ref_dend_activity[:, spine]

        # Calulate coactivity rates
        (
            coactive_trace,
            event_num,
            event_rate,
            event_rate_norm,
            shuff_event_rate,
            shuff_event_rate_norm,
            relative_diff,
            relative_diff_norm,
            fraction_dend,
            fraction_spine,
            _,
            _,
            coactive,
            coactive_norm,
        ) = calculate_dendrite_coactivity_rate(
            s_activity,
            d_activity,
            duration,
            sampling_rate,
            norm_method="mean",
            iterations=1,
        )

        # Store values
        coactive_binary[:, spine] = coactive_trace
        dendrite_coactive_event_num[spine] = event_num
        dendrite_coactivity_rate[spine] = event_rate
        dendrite_coactivity_rate_norm[spine] = event_rate_norm
        shuff_dendrite_coactivity_rate[:, spine] = shuff_event_rate
        shuff_dendrite_coactivity_rate_norm[:, spine] = shuff_event_rate_norm
        above_chance_coactivity[spine] = relative_diff
        above_chance_coactivity_norm[spine] = relative_diff_norm
        fraction_dend_coactive[spine] = fraction_dend
        fraction_spine_coactive[spine] = fraction_spine
        coactive_spines[spine] = coactive
        coactive_spines_norm[spine] = coactive_norm

        # Get timestamps of coactivity
        # coactive_stamps = t_stamps.get_activity_timestamps(coactive_trace)
        # if len(coactive_stamps) == 0:
        #    coactive_onsets[spine] = coactive_stamps
        # else:
        #    coactive_stamps = [x[0] for x in coactive_stamps]
        #    refine_coactive_stamps = t_stamps.refine_activity_timestamps(
        #        coactive_stamps,
        #        window=activity_window,
        #        max_len=len(d_activity),
        #        sampling_rate=sampling_rate,
        #    )
        #    coactive_onsets[spine] = refine_coactive_stamps

        # Find dendrite onsets for each coactivity event
        # _, d_onsets = find_individual_onsets(
        #     s_activity,
        #     d_activity,
        #     coactivity=coactive_trace,
        #     sampling_rate=sampling_rate,
        #     activity_window=activity_window,
        # )
        dend_coactive, _ = get_conservative_coactive_binary(
            d_activity,
            coactive_trace,
        )
        d_onsets = t_stamps.get_activity_timestamps(dend_coactive)
        d_onsets = [x[0] for x in d_onsets]

        if len(d_onsets) == 0:
            dend_onsets[spine] = d_onsets
        else:
            refine_d_onsets = t_stamps.refine_activity_timestamps(
                d_onsets,
                window=activity_window,
                max_len=len(d_activity),
                sampling_rate=sampling_rate,
            )
            dend_onsets[spine] = refine_d_onsets

    # Analyze the activity during events
    ## Dendrite
    (
        dendrite_coactive_traces,
        dendrite_coactive_amplitude,
        _,
    ) = analyze_event_activity(
        dendrite_dFoF,
        dend_onsets,
        activity_window=activity_window,
        center_onset=True,
        smooth=True,
        avg_window=None,
        norm_constant=None,
        sampling_rate=sampling_rate,
    )

    ## Spine GluSnFr
    (
        spine_coactive_traces,
        spine_coactive_amplitude,
        spine_onsets,
    ) = analyze_event_activity(
        spine_dFoF,
        dend_onsets,
        activity_window=activity_window,
        center_onset=False,
        smooth=True,
        avg_window=None,
        norm_constant=glu_norm_constants,
        sampling_rate=sampling_rate,
    )
    ## Spine calcium
    (
        spine_coactive_calcium_traces,
        spine_coactive_calcium_amplitude,
        _,
    ) = analyze_event_activity(
        spine_calcium,
        dend_onsets,
        activity_window=activity_window,
        center_onset=False,
        smooth=True,
        avg_window=None,
        norm_constant=ca_norm_constants,
        sampling_rate=sampling_rate,
    )

    # Get avg relative onset
    center_point = int(np.absolute(activity_window[0] * sampling_rate))
    for i, onset in enumerate(spine_onsets):
        try:
            rel_onset = (int(onset) - center_point) / sampling_rate
        except ValueError:
            rel_onset = np.nan
        relative_onsets[i] = rel_onset

    return (
        nearby_spine_idxs,
        coactive_binary,
        dendrite_coactive_event_num,
        dendrite_coactivity_rate,
        dendrite_coactivity_rate_norm,
        shuff_dendrite_coactivity_rate,
        shuff_dendrite_coactivity_rate_norm,
        above_chance_coactivity,
        above_chance_coactivity_norm,
        fraction_dend_coactive,
        fraction_spine_coactive,
        spine_coactive_amplitude,
        spine_coactive_calcium_amplitude,
        dendrite_coactive_amplitude,
        relative_onsets,
        spine_coactive_traces,
        spine_coactive_calcium_traces,
        dendrite_coactive_traces,
        coactive_spines,
        coactive_spines_norm,
    )


def extend_dendrite_activity(dend_activity, duration, sampling_rate):
    """Helper function to extend dendrite_activity by a certain amount"""
    expansion = int(duration * sampling_rate)
    exp_constant = np.ones(expansion, dtype=int)
    npad = len(exp_constant) - 1
    extended_activity = np.zeros(dend_activity.shape)

    for i in range(dend_activity.shape[1]):
        # Skip all nan traces
        if not np.nansum(dend_activity[:, i]):
            extended_activity[:, i] = dend_activity[:, i]
            continue
        d_pad = np.pad(
            dend_activity[:, i],
            (npad // 2, npad - npad // 2),
            mode="constant",
        )
        d_extend = np.convolve(d_pad, exp_constant, "valid").astype(bool).astype(int)
        extended_activity[:, i] = d_extend

    return extended_activity
