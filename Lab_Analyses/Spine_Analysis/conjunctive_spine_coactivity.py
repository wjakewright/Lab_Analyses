import numpy as np
from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities import (
    get_activity_timestamps,
    get_coactivity_rate,
    get_dend_spine_traces_and_onsets,
    nearby_spine_conjunctive_events,
)
from Lab_Analyses.Spine_Analysis.spine_movement_analysis import (
    quantify_movement_quality,
)
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    spine_volume_norm_constant,
)
from Lab_Analyses.Utilities import data_utilities as d_utils


def conjunctive_coactivity_analysis(
    data,
    activity_window=(-2, 3),
    movement_epoch=None,
    cluster_dist=10,
    sampling_rate=60,
    zscore=False,
    volume_norm=False,
):
    """Function to analyze spine coactivity with global dendritic activity and the activity
        of nearby spines
        
        INPUT PARAMETERS
            data - spine_data object (e.g., Dual_Channel_Spine_Data

            activity_window - tuple specifying the time window around each event you want to 
                              analyze the activity from in terms of seconds
            
            movement_epoch - str specifying if you want to analyze only during specific
                            types of movements. Accepts - 'movement', 'rewarded', 'unrewarded',
                            'learned', and 'nonmovement'. Default is None, analyzing the entire
                            imaging session
                            
            cluster_dist - int or float specifying the distance from target spine you will consider
                            to be nearby spines
            
            sampling_rate - int or float specifying what the imaging rate is
            
            zscore - boolean of whether to zscore dFoF traces for analysis

            volume_norm - boolean of whether or not to normalize activity by spine volume
            
        OUTPUT PARAMTERS
            local_correlation - np.array of the average correlation of each spine with nearby
                                coactive spines during conjunctive events

            coactivity_event_num - np.array of the number of conjnctive coactive events for each
                                    spine

            coactivity_event_rate - np.array of the event rate of the conjunctive coactive events
                                    for each spine

            spine_fraction_coactive - np.array of the fraction of spine activity events that are
                                      also conjunctive coactivity events for each spine

            dend_fraction_coactive - np.array of the fraction of dendritic activity events that
                                     are also conjunctive coactivity events with each spine

            coactive_spine_num - np.array of the average number of coactive spines across 
                                conjunctive coactivity events for each spine

            coactive_spine_volumes - np.array of the average volume of spines coactive during
                                     conjunctive events for each spine

            spine_coactive_amplitude - np.array of the average peak amplitude of activity during
                                        conjuntive events for each spine

            nearby_coactive_amplitude_sum - np.array of the average summed activity of neaby coactive
                                            spines during conjunctive events for each spine

            spine_coactive_calcium - np.array of the average peak calcium amplitude during conjunctive 
                                    events for each spine

            nearby_coactive_calcium_sum - np.array of the average summed calcium of nearby coactive
                                         spines during conjunctive events for each spine

            dend_coactive_amplitude - np.array of the average peak amplitude of dendrite activity 
                                     during conjunctive events for each spine
                                     
            spine_coactive_std - np.array of the std around the peak activity amplitude during conjunctive
                                 events for each spine

            nearby_coactive_std - np.array of the std around the summed peak activity amplitude of 
                                  nearby coactive spines during conjunctive events for each spine

            spine_coactive_calcium_std - np.array of the std around the peak calcium amplitude during
                                         conjunctive events for each spine

            nearby_coactive_calcium_std - np.array of the std around the summed peak calcium amplitude
                                          of nearby coactive spines during conjunctive events for each spine

            dend_coactive_std - np.array of the std around the peak dendritic activity during
                                conjunctive events for each spine

            spine_coactive_calcium_auc - np.array of the auc of the average calcium trace during
                                        conjunctive events for each spine

            nearby_coactive_calcium_auc_sum - np.array of the auc of the averaged summed calcium
                                                trace of nearby coactive spines during conjunctive
                                                events for each spine

            dend_coactive_auc - np.arry of the auc of the average dendrite trace during conjunctive
                                events for each spine

            relative_spine_dend_onsets - np.array of the relative onset of spine activity in relation
                                        to the onset of dendritic activity during conjunctive events

            coactive_spine_traces - list of 2d np.arrays of activity traces for each conjunctive event
                                    for each spine (columns=events)

            coactive_nearby_traces -list of 2d np.arrays of the summed activity traces of nearby coactive
                                    spines for each conjunctive event for each spine (columns=events)

            coactive_spine_calcium_traces - list of 2d np.arrays of calcium traces for each conjunctive
                                            event for each spine (columns = events)

            coactive_nearby_calcium_traces - list of 2d np.arrays of the summed calcium traces of nearby
                                             coactive spines for each conjunctive event for each spine
                                             (columns=events)

            coactive_dend_traces - list of 2d of np.arrays of the dendrite activity traces for each each
                                    conjunctive event for each spine (columns=events)

            conjunctive_coactivity_matrix - 2d np.array of binarized conjunctive coactivity 
                                            (columns=spines, rows=time) 
        
    """

    spine_groupings = data.spine_grouping
    spine_flags = data.spine_flags
    spine_volumes = np.array(data.corrected_spine_volume)
    spine_positions = np.array(data.spine_positions)
    spine_dFoF = data.spine_GluSnFr_processed_dFoF
    spine_calcium = data.spine_calcium_processed_dFoF
    spine_activity = data.spine_GluSnFr_activity
    dendrite_dFoF = data.dendrite_calcium_processed_dFoF
    dendrite_activity = data.dendrite_calcium_activity

    if zscore:
        spine_dFoF = d_utils.z_score(spine_dFoF)
        spine_calcium = d_utils.z_score(spine_calcium)
        dendrite_dFoF = d_utils.z_score(dendrite_dFoF)

    if volume_norm:
        glu_norm_constants = spine_volume_norm_constant(
            spine_activity,
            spine_dFoF,
            data.corrected_spine_volume,
            data.imaging_parameters["Zoom"],
            sampling_rate=sampling_rate,
            iterations=1000,
        )
        ca_norm_constants = spine_volume_norm_constant(
            spine_activity,
            spine_calcium,
            data.corrected_spine_volume,
            data.imaging_parameters["Zoom"],
            sampling_rate=sampling_rate,
            iterations=1000,
        )
    else:
        glu_norm_constants = np.array([None for x in range(spine_activity.shape[1])])
        ca_norm_constants = np.array([None for x in range(spine_activity.shape[1])])

    # Get specific movement periods if specified
    if movement_epoch == "movement":
        movement = data.lever_active
    elif movement_epoch == "rewarded":
        movement = data.rewarded_movement_binary
    elif movement_epoch == "unrewarded":
        movement = data.lever_active - data.rewarded_movement_binary
    elif movement_epoch == "nonmovement":
        movement = np.absolute(data.lever_active - 1)
    elif movement_epoch == "learned":
        movement, _, _, _, _ = quantify_movement_quality(
            data.mouse_id,
            spine_activity,
            data.lever_active,
            threshold=0.5,
            sampling_rate=sampling_rate,
        )
    else:
        movement = None

    # Set up output variables
    local_correlation = np.zeros(spine_activity.shape[1]) * np.nan
    coactivity_event_num = np.zeros(spine_activity.shape[1])
    coactivity_event_rate = np.zeros(spine_activity.shape[1])
    spine_fraction_coactive = np.zeros(spine_activity.shape[1])
    dend_fraction_coactive = np.zeros(spine_activity.shape[1])
    coactive_spine_num = np.zeros(spine_activity.shape[1])
    coactive_spine_volumes = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    nearby_coactive_amplitude_sum = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium = np.zeros(spine_activity.shape[1]) * np.nan
    nearby_coactive_calcium_sum = np.zeros(spine_activity.shape[1]) * np.nan
    dend_coactive_amplitude = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_std = np.zeros(spine_activity.shape[1]) * np.nan
    nearby_coactive_std = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium_std = np.zeros(spine_activity.shape[1]) * np.nan
    nearby_coactive_calcium_std = np.zeros(spine_activity.shape[1]) * np.nan
    dend_coactive_std = np.zeros(spine_activity.shape[1]) * np.nan
    spine_coactive_calcium_auc = np.zeros(spine_activity.shape[1]) * np.nan
    nearby_coactive_calcium_auc_sum = np.zeros(spine_activity.shape[1]) * np.nan
    dend_coactive_auc = np.zeros(spine_activity.shape[1]) * np.nan
    relative_spine_dend_onsets = np.zeros(spine_activity.shape[1]) * np.nan
    coactive_spine_traces = [None for i in local_correlation]
    coactive_nearby_traces = [None for i in local_correlation]
    coactive_spine_calcium_traces = [None for i in local_correlation]
    coactive_nearby_calcium_traces = [None for i in local_correlation]
    coactive_dend_traces = [None for i in local_correlation]
    conjunctive_coactivity_matrix = np.zeros(spine_activity.shape)

    # Process spines for each parent dendrite
    for dendrite in range(dendrite_activity.shape[1]):
        # Get spines on this dendrite
        if type(spine_groupings[dendrite]) == list:
            spines = spine_groupings[dendrite]
        else:
            spines = spine_groupings

        s_dFoF = spine_dFoF[:, spines]
        s_activity = spine_activity[:, spines]
        s_calcium = spine_calcium[:, spines]
        d_dFoF = dendrite_dFoF[:, dendrite]
        d_activity = dendrite_activity[:, dendrite]
        curr_positions = spine_positions[spines]
        curr_flags = [x for i, x in enumerate(spine_flags) if i in spines]
        curr_volumes = spine_volumes[spines]
        curr_glu_norm_constants = glu_norm_constants[spines]
        curr_ca_norm_constants = ca_norm_constants[spines]

        # Refine activity matrices for only movement epochs if specified
        if movement is not None:
            s_activity = (s_activity.T * movement).T
            d_activity = d_activity * movement

        # Analyze each spine individually
        for spine in range(s_dFoF.shape[1]):
            # Find its neighboring spines
            ## Find spine positions
            curr_el_spines = find_spine_classes(curr_flags, "Eliminated Spine")
            curr_el_spines = np.array(curr_el_spines)
            target_position = curr_positions[spine]
            other_positions = [
                x for idx, x in enumerate(curr_positions) if idx != spine
            ]
            relative_positions = np.array(other_positions) - target_position
            relative_positions = np.absolute(relative_positions)
            ## Find spines within cluster distance
            nearby_spines = np.nonzero(relative_positions <= cluster_dist)[0]
            ## Remove the eliminated spines. Dan't want to consider their activity here
            nearby_spines = [
                i for i in nearby_spines if not curr_el_spines[i] and i != spine
            ]

            # Get the relevant spine activity data
            curr_s_dFoF = s_dFoF[:, spine]
            curr_s_activity = s_activity[:, spine]
            curr_s_calcium = s_calcium[:, spine]
            nearby_s_dFoF = s_dFoF[:, nearby_spines]
            nearby_s_activity = s_activity[:, nearby_spines]
            nearby_s_calcium = s_calcium[:, nearby_spines]
            nearby_volumes = curr_volumes[nearby_spines]
            glu_constant = curr_glu_norm_constants[spine]
            ca_constant = curr_ca_norm_constants[spine]
            nearby_glu_constants = curr_glu_norm_constants[nearby_spines]
            nearby_ca_constants = curr_ca_norm_constants[nearby_spines]

            # Get spine-dendrite coactivity trace
            curr_coactivity = curr_s_activity * d_activity
            # Get a conjunctive coactivity trace, where at least one other nearby spine is coactive
            combined_nearby_activity = np.sum(nearby_s_activity, axis=1)
            combined_nearby_activity[combined_nearby_activity > 1] = 1
            curr_conj_coactivity = combined_nearby_activity * curr_coactivity
            conjunctive_coactivity_matrix[:, spines[spine]] = curr_conj_coactivity

            # Skip further anlaysis if no conjunctive coactivity for current spine
            if not np.sum(curr_conj_coactivity):
                continue

            # Get conjunctive coactivity timestamps
            conj_timestamps = get_activity_timestamps(curr_conj_coactivity)
            if not conj_timestamps:
                continue

            # Start analyzing the conjunctive coactivity
            event_num, event_rate, spine_frac, dend_frac = get_coactivity_rate(
                curr_s_activity,
                d_activity,
                curr_conj_coactivity,
                sampling_rate=sampling_rate,
            )
            coactivity_event_num[spines[spine]] = event_num
            coactivity_event_rate[spines[spine]] = event_rate
            spine_fraction_coactive[spines[spine]] = spine_frac
            dend_fraction_coactive[spines[spine]] = dend_frac

            (
                s_traces,
                d_traces,
                s_amp,
                _,
                s_std,
                d_amp,
                d_auc,
                d_std,
                rel_onset,
            ) = get_dend_spine_traces_and_onsets(
                curr_s_activity.reshape(-1, 1),
                d_dFoF,
                curr_s_dFoF.reshape(-1, 1),
                curr_conj_coactivity,
                norm_constants=[glu_constant],
                activity_window=activity_window,
                sampling_rate=sampling_rate,
            )
            (
                s_ca_traces,
                _,
                s_ca_amp,
                s_ca_auc,
                s_ca_std,
                _,
                _,
                _,
                _,
            ) = get_dend_spine_traces_and_onsets(
                curr_s_activity.reshape(-1, 1),
                d_dFoF,
                curr_s_calcium.reshape(-1, 1),
                curr_conj_coactivity,
                norm_constants=[ca_constant],
                activity_window=activity_window,
                sampling_rate=sampling_rate,
            )
            spine_coactive_amplitude[spines[spine]] = s_amp[0]
            spine_coactive_calcium[spines[spine]] = s_ca_amp[0]
            dend_coactive_amplitude[spines[spine]] = d_amp[0]
            spine_coactive_std[spines[spine]] = s_std[0]
            spine_coactive_calcium_std[spines[spine]] = s_ca_std[0]
            dend_coactive_std[spines[spine]] = d_std[0]
            spine_coactive_calcium_auc[spines[spine]] = s_ca_auc[0]
            dend_coactive_auc[spines[spine]] = d_auc[0]
            relative_spine_dend_onsets[spines[spine]] = rel_onset[0]
            coactive_spine_traces[spines[spine]] = s_traces[0]
            coactive_spine_calcium_traces[spines[spine]] = s_ca_traces[0]
            coactive_dend_traces[spines[spine]] = d_traces[0]

            # Analyze the activity of nearby coactive spines
            (
                local_corr,
                coactive_num,
                coactive_vol,
                activity_amp,
                ca_activity_amp,
                activity_std,
                ca_activity_std,
                ca_activity_auc,
                coactive_s_traces,
                coactive_s_ca_traces,
            ) = nearby_spine_conjunctive_events(
                timestamps=conj_timestamps,
                spine_dFoF=curr_s_dFoF,
                nearby_dFoF=nearby_s_dFoF,
                nearby_calcium=nearby_s_calcium,
                nearby_activity=nearby_s_activity,
                dendrite_dFoF=d_dFoF,
                nearby_spine_volumes=nearby_volumes,
                target_constant=glu_constant,
                glu_constants=nearby_glu_constants,
                ca_constants=nearby_ca_constants,
                activity_window=activity_window,
                sampling_rate=sampling_rate,
            )
            local_correlation[spines[spine]] = local_corr
            coactive_spine_num[spines[spine]] = coactive_num
            coactive_spine_volumes[spines[spine]] = coactive_vol
            nearby_coactive_amplitude_sum[spines[spine]] = activity_amp
            nearby_coactive_calcium_sum[spines[spine]] = ca_activity_amp
            nearby_coactive_std[spines[spine]] = activity_std
            nearby_coactive_calcium_std[spines[spine]] = ca_activity_std
            nearby_coactive_calcium_auc_sum[spines[spine]] = ca_activity_auc
            coactive_nearby_traces[spines[spine]] = coactive_s_traces
            coactive_nearby_calcium_traces[spines[spine]] = coactive_s_ca_traces

    return (
        local_correlation,
        coactivity_event_num,
        coactivity_event_rate,
        spine_fraction_coactive,
        dend_fraction_coactive,
        coactive_spine_num,
        coactive_spine_volumes,
        spine_coactive_amplitude,
        nearby_coactive_amplitude_sum,
        spine_coactive_calcium,
        nearby_coactive_calcium_sum,
        dend_coactive_amplitude,
        spine_coactive_std,
        nearby_coactive_std,
        spine_coactive_calcium_std,
        nearby_coactive_calcium_std,
        dend_coactive_std,
        spine_coactive_calcium_auc,
        nearby_coactive_calcium_auc_sum,
        dend_coactive_auc,
        relative_spine_dend_onsets,
        coactive_spine_traces,
        coactive_nearby_traces,
        coactive_spine_calcium_traces,
        coactive_nearby_calcium_traces,
        coactive_dend_traces,
        conjunctive_coactivity_matrix,
    )

