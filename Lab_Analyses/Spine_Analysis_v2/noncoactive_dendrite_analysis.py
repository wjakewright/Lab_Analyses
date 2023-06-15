import numpy as np

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import find_present_spines
from Lab_Analyses.Utilities import activity_timestamps as t_stamps
from Lab_Analyses.Utilities.coactivity_functions import calculate_coactivity
from Lab_Analyses.Utilities.mean_trace_functions import analyze_event_activity


def noncoactive_dendrite_analysis(
    coactive_matrix,
    spine_activity,
    spine_calcium,
    dendrite_activity,
    nearby_spine_idxs,
    spine_flags,
    activity_window=(-2, 4),
    constrain_matrix=None,
    sampling_rate=60,
    volume_norm=None,
):
    """Function to analyze the spine activity when the dendrite is active and coactive with
        nearby spines, but target spine is inactive
        
        INPUT PARAMETERS
            coactive_matrix - 2d np.array of the binarized coactivity for each spine (columns)
            
            spine_activity - 2d np.array of the binarized spine activity for each spine 
            
            spine_calcium - 2d np.array of the spine calcium dF/F traces
            
            dendrite_activity - 2d np.array of the binarized dendrite activity
            
            nearby_spine_idxs - list containing the nearby spine idxs for each spine
            
            spine_flags - list of the spine flags
            
            activity_window - tuple specifying the window to analyze the activity around
            
            constrain_matrix - binary np.array that is used to constrain the activity 
                                of the spines and dendrite (e.g., active lever)
            
            sampling_rate - int specifying the imaging sampling rate
            
            volume_norm - tuple of lists containing the constants to normalize GluSnFr and 
                            calcium by the spine volumes

        OUTPUT PARAMETERS
            noncoactive_spine_calcium_amplitude - np.array of the spine calcium amplitude
                                                during noncoactive dendritic events for 
                                                each spine
            
            noncoactive_spine_calcium_traces - list of np.arrays of the spine calcium traces
                                                during each noncoactive dendritic event. 
                                                columns = events, items = spine
            
            conj_fraction_participating - np.array of the fraction of conj coactivity events
                                         a spine participates in
            
            nonparticipating_calcium_amplitude - np.array of the spine calcium amplitude
                                                during conj events a spine is not participating in
                                
            nonparticpating_calcium_traces - np.array of the spine calcium traces during each
                                            conj event it doesn't particpate in
    """
    if volume_norm is not None:
        ca_norm_constants = volume_norm[1]

    # constrain dendrite activity if specified
    if constrain_matrix is not None:
        if len(constrain_matrix.shape) == 1:
            constrain_matrix = constrain_matrix.reshape(-1, 1)
        dend_activity = dendrite_activity * constrain_matrix
    else:
        dend_activity = dendrite_activity

    present_spines = find_present_spines(spine_flags)

    # Set up some outputs
    conj_fraction_participating = np.zeros(coactive_matrix.shape[1])

    # Temporary variables
    noncoactive_dend_stamps = [[] for i in range(coactive_matrix.shape[1])]
    nonparticipating_stamps = [[] for i in range(coactive_matrix.shape[1])]

    # Iterate through each spine
    for spine in range(coactive_matrix.shape[1]):
        # Skip of spine is not present
        if present_spines[spine] is False:
            continue
        # Generate a noncoactive dendrite trace
        noncoactive_trace = 1 - coactive_matrix[:, spine]
        _, _, _, _, dend_noncoactive = calculate_coactivity(
            dend_activity[:, spine], noncoactive_trace, sampling_rate=sampling_rate,
        )
        # Get event timestamps
        if np.nansum(dend_noncoactive):
            noncoactive_stamps = t_stamps.get_activity_timestamps(dend_noncoactive)
            noncoactive_stamps = [x[0] for x in noncoactive_stamps]
            noncoactive_stamps = t_stamps.refine_activity_timestamps(
                noncoactive_stamps,
                window=activity_window,
                max_len=len(dend_activity[:, spine]),
                sampling_rate=sampling_rate,
            )
        else:
            noncoactive_stamps = []

        noncoactive_dend_stamps[spine] = noncoactive_stamps

        # Calculate fraction participating to conj coactivity
        if (nearby_spine_idxs[spine] is None) or (len(nearby_spine_idxs[spine]) == 0):
            continue
        nearby_spine_activity = spine_activity[:, nearby_spine_idxs[spine]]
        combined_nearby_activity = np.sum(nearby_spine_activity, axis=1)
        combined_nearby_activity[combined_nearby_activity > 1] = 1
        dend_combined_nearby_activity = (
            combined_nearby_activity * dend_activity[:, spine]
        )
        participating_coactivity = (
            dend_combined_nearby_activity * spine_activity[:, spine]
        )
        frac_participating = np.nansum(participating_coactivity) / np.nansum(
            dend_combined_nearby_activity
        )

        # Get nonparticipating event stamps
        non_part_stamps = []
        ## Iterate through each nearby spine
        for nearby in nearby_spine_idxs[spine]:
            nearby_activity = spine_activity[:, nearby]
            spine_inactivity = 1 - spine_activity[:, spine]
            nearby_isolated = nearby_activity * spine_inactivity
            _, _, _, _, non_part_coactivity = calculate_coactivity(
                dend_activity[:, spine], nearby_isolated, sampling_rate=sampling_rate,
            )
            ## Get the stamps
            if np.nansum(non_part_coactivity):
                np_stamps = t_stamps.get_activity_timestamps(non_part_coactivity)
                np_stamps = [x[0] for x in np_stamps]
                np_stamps = t_stamps.refine_activity_timestamps(
                    np_stamps,
                    window=activity_window,
                    max_len=len(dend_activity[:, spine]),
                    sampling_rate=sampling_rate,
                )
            else:
                np_stamps = []
            non_part_stamps.append(np_stamps)
        ## Unnest the list
        non_part_stamps = [y for x in non_part_stamps for y in x]

        nonparticipating_stamps[spine] = non_part_stamps
        conj_fraction_participating[spine] = frac_participating

    # Analyze the traces
    ## Noncoactive dendrite events
    (
        noncoactive_spine_calcium_traces,
        noncoactive_spine_calcium_amplitude,
        _,
    ) = analyze_event_activity(
        spine_calcium,
        noncoactive_dend_stamps,
        activity_window=activity_window,
        center_onset=False,
        smooth=True,
        avg_window=None,
        norm_constant=ca_norm_constants,
        sampling_rate=sampling_rate,
        peak_required=False,
    )
    ## Nonparticipating coactive events
    (
        nonparticipating_calcium_traces,
        nonparticipating_calcium_amplitude,
        _,
    ) = analyze_event_activity(
        spine_calcium,
        nonparticipating_stamps,
        activity_window=activity_window,
        center_onset=False,
        smooth=True,
        avg_window=None,
        norm_constant=ca_norm_constants,
        sampling_rate=sampling_rate,
        peak_required=False,
    )

    return (
        noncoactive_spine_calcium_amplitude,
        noncoactive_spine_calcium_traces,
        conj_fraction_participating,
        nonparticipating_calcium_amplitude,
        nonparticipating_calcium_traces,
    )
