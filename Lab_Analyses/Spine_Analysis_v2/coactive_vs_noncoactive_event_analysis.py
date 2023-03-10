import numpy as np

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import find_nearby_spines
from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities.coactivity_functions import calculate_coactivity
from Lab_Analyses.Utilities.mean_trace_functions import analyze_event_activity


def coactive_vs_noncoactive_event_analysis(
    spine_activity,
    spine_dFoF,
    spine_calcium,
    spine_flags,
    spine_positions,
    spine_groupings,
    activity_window=(-2, 4),
    cluster_dist=5,
    constrain_matrix=None,
    partner_list=None,
    sampling_rate=60,
    volume_norm=None,
):
    """Function to analyze spine activity events when it is coactive with other local spine
        or when not. 
        
        INPUT PARAMETERS
            spine_activity- 2d np.array of the binarized spine activity traces
                            Each column represents a spine
            
            spine_dFoF - 2d np.array of the GluSnFr dFoF traces for each spine (columns)
            
            spine_calcium - 2d np.array of the calcium dFoF traces for each spine (columns)
            
            spine_flags - list of the spine flags
            
            spine_positions - np.array of the corresponding spine positions along the dendrite
            
            spine_groupings - list of the groupings of spines on the same dendrites
            
            activity_window - tuple specifying the activity window in sec to analyze
            
            cluster_dist - int or float specifying the distance (um) that is considered local
            
            constrain_matrix - np.array of binarized events to contrain the coactivity to
                                (e.g., dendriteic events, movement periods)
            
            partner_list - boolean list specifying a subset of spines to analyze coactivity
                            with
                            
            sampling_rate - int specifying the imaging sampling rate
            
            volume_nom - tuple of lists containing the constants to normalize GluSnFr and 
                        calcium by
        
        OUTPUT PARAMETERS
            nearby_spine_idxs - list of the nearby spine idxs for each spine

            coactive_binary - 2d binary array of when each spine is coactive (columns=spines)

            noncoactive_binary - 2d binary array of when each spine is not coactive

            spine_coactive_traces - list of 2d np.array of each spines (list items) GluSnFr traces
                                    during each coactivity event (column)
            
            spine_noncoactive_traces - list of 2d np.array of each spines' GluSnFr traces during
                                        noncoactivity events
            
            spine_coactive_calcium_traces - list of 2d np.array of each spines' calcium
                                            traces during each coactivity event
            
            spine_noncoactive_calcium_traces - list of 2d np.array of each spine's calcium traces
                                                during each non coactivity event
            
            spine_coactive_amplitude - np.array of each spines' mean peak amplitude GluSnFr 
                                        coactive event
            
            spine_noncoactive_amplitude - np.array of each spines' mean peak amplitude
                                          GluSnFr noncoactive event
                    
            spine_coactive_calcium_amplitude - np.array of each spine's mean peak amplitude
                                                calcium coactive event
                    
            spine_noncoactive_calcium_amplitude - np.array of each spine's mean peak amplitude
                                                  calcium noncoactive event
            
            spine_coactive_onsets - np.array of the spine GluSnFr onsets during coactive events

            spine_noncoactive_onsets - np.array of the spine GluSnFr onsets during noncoactive events

            fraction_spine_coactive - np.array of the fraction of activity that is coactive for each spine

            fraction_coactivity_participation - np.array of the fraction of local coactivity a spine
                                                participates in
    """
    # Sort out the spine groupings to make sure it is iterable
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # Get the nearby spine idxs
    nearby_spine_idxs = find_nearby_spines(
        spine_positions, spine_flags, spine_groupings, partner_list, cluster_dist
    )

    # Constrain activity if necessary
    if constrain_matrix is not None:
        if len(constrain_matrix.shape) == 1:
            constrain_matrix = constrain_matrix.reshape(-1, 1)
        activity_matrix = spine_activity * constrain_matrix
    else:
        activity_matrix = spine_activity

    if volume_norm is not None:
        glu_norm_constants = volume_norm[0]
        ca_norm_constants = volume_norm[1]
    else:
        glu_norm_constants = None
        ca_norm_constants = None

    # Generate coactive and noncoactive traces and timestamps for each spine
    ## Initialize some variables
    coactive_binary = np.zeros(spine_activity.shape)
    noncoactive_binary = np.zeros(spine_activity.shape)
    coactive_onsets = [None for i in range(spine_activity.shape[1])]
    noncoactive_onsets = [None for i in range(spine_activity.shape[1])]
    spine_coactive_event_num = np.zeros(spine_activity.shape[1])
    fraction_spine_coactive = np.zeros(spine_activity.shape[1])
    fraction_coactivity_participation = np.zeros(spine_activity.shape[1])

    ## Iterate through each spine
    ### Don't need to go through groupings since that has been delt with in
    ### the nearby spine idx function
    for spine in range(spine_activity.shape[1]):
        ## Generate combined nearby activity trace
        nearby_spine_activity = activity_matrix[:, nearby_spine_idxs[spine]]
        combined_nearby_activity = np.sum(nearby_spine_activity, axis=1)
        combined_nearby_activity[combined_nearby_activity > 1] = 1
        ## Get coactivity trace
        _, _, frac_coactive, _, coactive = calculate_coactivity(
            activity_matrix[:, spine],
            combined_nearby_activity,
            sampling_rate=sampling_rate,
        )
        ## Generate nearby inactive trace
        combined_nearby_inactivity = 1 - combined_nearby_activity
        _, _, _, _, noncoactive = calculate_coactivity(
            activity_matrix[:, spine],
            combined_nearby_inactivity,
            sampling_rate=sampling_rate,
        )
        ## Particpating coactivity
        participating_coactivity = combined_nearby_activity * activity_matrix[:, spine]
        frac_participating = np.nansum(participating_coactivity) / np.nansum(
            combined_nearby_activity
        )
        ## Get the coactive and noncoactive onset timestamps
        coactive_stamps = t_stamps.get_activity_timestamps(coactive)
        coactive_stamps = [x[0] for x in coactive_stamps]
        refined_coactive_stamps = t_stamps.refine_activity_timestamps(
            coactive_stamps,
            window=activity_window,
            max_len=len(activity_matrix[:, spine]),
            sampling_rate=sampling_rate,
        )
        noncoactive_stamps = t_stamps.get_activity_timestamps(noncoactive)
        noncoactive_stamps = [x[0] for x in noncoactive_stamps]
        refined_noncoactive_stamps = t_stamps.refine_activity_timestamps(
            noncoactive_stamps,
            window=activity_window,
            max_len=len(activity_matrix[:, spine]),
            sampling_rate=sampling_rate,
        )
        coactive_binary[:, spine] = coactive
        noncoactive_binary[:, spine] = noncoactive
        coactive_onsets[spine] = refined_coactive_stamps
        noncoactive_onsets[spine] = refined_noncoactive_stamps
        spine_coactive_event_num[spine] = len(refined_coactive_stamps)
        fraction_spine_coactive[spine] = frac_coactive
        fraction_coactivity_participation[spine] = frac_participating

    # Analyze spine coactive events
    ## GluSnFr
    (
        spine_coactive_traces,
        spine_coactive_amplitude,
        spine_coactive_onset,
    ) = analyze_event_activity(
        spine_dFoF,
        coactive_onsets,
        activity_window=activity_window,
        center_onset=True,
        smooth=True,
        avg_window=None,
        norm_constant=glu_norm_constants,
        sampling_rate=sampling_rate,
    )
    ## Calcium
    (
        spine_coactive_calcium_traces,
        spine_coactive_calcium_amplitude,
        _,
    ) = analyze_event_activity(
        spine_calcium,
        coactive_onsets,
        activity_window=activity_window,
        center_onset=True,
        avg_window=None,
        norm_constant=ca_norm_constants,
        sampling_rate=sampling_rate,
    )

    # Analyze spine noncoactive events
    ## GluSnFr
    (
        spine_noncoactive_traces,
        spine_noncoactive_amplitude,
        spine_noncoactive_onset,
    ) = analyze_event_activity(
        spine_dFoF,
        noncoactive_onsets,
        activity_window=activity_window,
        center_onset=True,
        smooth=True,
        avg_window=None,
        norm_constant=glu_norm_constants,
        sampling_rate=sampling_rate,
    )
    ## Calcium
    (
        spine_noncoactive_calcium_traces,
        spine_noncoactive_calcium_amplitude,
        _,
    ) = analyze_event_activity(
        spine_calcium,
        noncoactive_onsets,
        activity_window=activity_window,
        center_onset=True,
        avg_window=None,
        norm_constant=ca_norm_constants,
        sampling_rate=sampling_rate,
    )

    return (
        nearby_spine_idxs,
        coactive_binary,
        noncoactive_binary,
        spine_coactive_event_num,
        spine_coactive_traces,
        spine_noncoactive_traces,
        spine_coactive_calcium_traces,
        spine_noncoactive_calcium_traces,
        spine_coactive_amplitude,
        spine_noncoactive_amplitude,
        spine_coactive_calcium_amplitude,
        spine_noncoactive_calcium_amplitude,
        spine_coactive_onset,
        spine_noncoactive_onset,
        fraction_spine_coactive,
        fraction_coactivity_participation,
    )

