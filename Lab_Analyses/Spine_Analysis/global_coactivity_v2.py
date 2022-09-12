import numpy as np
from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities import (
    find_activity_onset,
    get_activity_timestamps,
    get_coactivity_rate,
    get_dend_spine_traces_and_onsets,
)
from Lab_Analyses.Spine_Analysis.spine_movement_analysis import quantify_movment_quality
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    spine_volume_norm_constant,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from scipy import stats


def total_coactivity_analysis(
    data, movement_epoch=None, sampling_rate=60, zscore=False, volume_norm=False,
):
    """Function to analyze spine co-activity with global dendritic activity
    
        INPUT PARAMETERS
            data - spine_data object. (e.g. Dual_Channel_Spine_Data)
            
            movement_epoch - str specifying if you want to analyze only during specific
                            types of movements. Accepts - 'movement', 'rewarded', 
                            'unrewarded', learned', and 'nonmovement'. Default is None, 
                            analyzing the entire imaging session
            
            sampling_rate - int or float specifying what the imaging rate

            zscore - boolean of whether to zscore dFoF traces for and analysis

            volume_norm - boolean of whether or not to normalize activity traces by spine volume
            
        OUTPUT PARAMETERS
            global_correlation - np.array of the correlation coefficient between spine
                                and dendrite fluorescence traces
                                
            coactivity_event_num - np.array of the number of coactive events for each spine
            
            coactivity_event_rate - np.array of the normalized coactivity rate for each spine
            
            spine_fraction_coactive - np.array of the fraction of spine events that were also
                                      coactive
            
            dend_fraction_coactive - np.array of the fraction of dendritic events that were also
                                     coactive with a given spine
            
            spine_coactive_amplitude - np.array of the peak mean response of each spine during coactive
                                       events

            dend_coactive_amplitude - np.array of the peak mean response of dendritic activity during
                                      coactive events of a given spine
            
            spine_coactive_calcium - np.array of the peak mean responsse of spine calcium during
                                       coactive events of a given spine

            spine_coactive_std - np.array of the variance in the peak activity of spine activity during
                                 coactive events of a given spine
            
            dend_coactive_std - np.array of the variance in peak activity of dendritic activity during 
                                coactive events of a given spine
            

            spine_coactive_calcium_std - np.array of the variance in the peak calcium of spine activity during
                                        coactive events of a given spine

            dend_coactive_auc - np.array of the area under the dendrite activity curve during coactive events
                                of a given spine
            
            spine_coactive_calcium_auc - np.array of the area under the spine calcium curve during coative
                                        events of a given spine
            
            relative_spine_coactive_amplitude - np.array of the peak mean responses of spine coactivity
                                                during coactive events of a given spine normalized to 
                                                it mean activity across all dendritic events

            relative_dend_coactive_amplitude - np.array of the peak mean responses of dendritic coactivity
                                                during coactive events of a given spine normalized to 
                                                it mean activity across all dendritic events
            
            relative_spine_coactive_calcium - np.array of the peak mean responses of spine calcium during
                                              coactive events of a given spine normalized to its mean activity
                                              across all dendritic events
            
            relative_spine_onsets - np.array of the mean onset of spine activity relative to dendritic 
                                    activity for coactive events
            
            dend_triggered_spine_traces - list of 2d np.arrays of spine activity around each
                                          dendritic event. Centered around dendrite onset. 
                                          columns = each event, rows = time (in frames)
            
            dend_triggered_dend_traces - list of 2d np.array of dendrite activty around each
                                         each dendritic event. Centered around dendrite onsets
            
            dend_triggered_spine_calcium_traces - list of 2d np.array of spine calcium acround 
                                                  each dendritic event. Centered around dendrite onset
                                                  columns = each event, rows = time (in frames)

            coactive_spine_traces - list of 2d np.arrays of spine activity around each coactive
                                    event. Centered around corresponding dendrite onset.
                                    column = each event, rows = time (in frames)
            
            coactive_dend_traces - list of 2d np.arrays of dendrite activity around each coactive
                                    event. Centered arorund dendrite onsets. 
                                    column = each event, rows = time (in frames)
            
            coactive_spine_calcium_traces - list of 2d np.arrays of dendrite activity around each
                                            coactive event. Centered around dnedrite onsets.
                                            column = each event, rows = time (in frames)
            
            coactivity_matrix - 2d np.array of the coactivity trace for each spine (columns)
    """
    # Pull some important information from data
    spine_groupings = data.spine_grouping
    spine_flags = data.spine_flags
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
            data.spine_volume,
            data.imaging_parameters["Zoom"],
            sampling_rate=sampling_rate,
            iterations=1000,
        )
        ca_norm_constants = spine_volume_norm_constant(
            spine_activity,
            spine_calcium,
            data.spine_volume,
            data.imaging_parameters["Zoom"],
            sampling_rate=sampling_rate,
            iterations=1000,
        )
    else:
        glu_norm_constants = None
        ca_norm_constants = None

    ## Get specific movement periods
    if movement_epoch == "movement":
        movement = data.lever_active
    elif movement_epoch == "rewarded":
        movement = data.rewarded_movement_binary
    elif movement_epoch == "unrewarded":
        movement = data.lever_active - data.rewarded_movement_binary
    elif movement_epoch == "nonmovement":
        movement = np.absolute(data.lever_active - 1)
    elif movement_epoch == "learned":
        movement, _, _, _, _ = quantify_movment_quality(
            data.mouse_id,
            spine_activity,
            data.lever_active,
            data.lever_force_smooth,
            threshold=0.5,
            sampling_rate=sampling_rate,
        )
    else:
        movement = None

    # Set up output variables
    global_correlation = np.zeros(spine_activity.shape[1])
    coactivity_event_num = np.zeros(spine_activity.shape[1])
    coactivity_event_rate = np.zeros(spine_activity.shape[1])
    spine_fraction_coactive = np.zeros(spine_activity.shape[1])
    dend_fraction_coactive = np.zeros(spine_activity.shape[1])
    spine_coactive_amplitude = np.zeros(spine_activity.shape[1])
    spine_coactive_calcium = np.zeros(spine_activity.shape[1])
    dend_coactive_amplitude = np.zeros(spine_activity.shape[1])
    spine_coactive_std = np.zeros(spine_activity.shape[1])
    spine_coactive_calcium_std = np.zeros(spine_activity.shape[1])
    dend_coactive_std = np.zeros(spine_activity.shape[1])
    spine_coactive_calcium_auc = np.zeros(spine_activity.shape[1])
    dend_coactive_auc = np.zeros(spine_activity.shape[1])
    relative_dend_coactive_amplitude = np.zeros(spine_activity.shape[1])
    relative_spine_coactive_calcium = np.zeros(spine_activity.shape[1])
    relative_spine_coactive_amplitude = np.zeros(spine_activity.shape[1])
    relative_spine_onsets = np.zeros(spine_activity.shape[1])
    dend_triggered_spine_traces = [None for i in global_correlation]
    dend_triggered_spine_calcium_traces = [None for i in global_correlation]
    dend_triggered_dend_traces = [None for i in global_correlation]
    coactive_spine_traces = [None for i in global_correlation]
    coactive_spine_calcium_traces = [None for i in global_correlation]
    coactive_dend_traces = [None for i in global_correlation]
    coactivity_matrix = np.zeros(spine_activity.shape)

    # Process spines for each parent dendrite
    for dendrite in range(dendrite_activity.shape[1]):
        # Get the spines on this dendrite
        if type(spine_groupings[dendrite]) == list:
            spines = spine_groupings[dendrite]
        else:
            spines = spine_groupings
        s_dFoF = spine_dFoF[:, spines]
        s_activity = spine_activity[:, spines]
        d_dFoF = dendrite_dFoF[:, dendrite]
        d_activity = dendrite_activity[:, dendrite]
        s_calcium = spine_calcium[:, spines]
        curr_coactivity_matrix = np.zeros(s_calcium.shape[1])
        curr_flags = spine_flags[spines]
        curr_el_spines = find_spine_classes(curr_flags, "Eliminated Spine")

        # Refine activity matrices for only movement epochs if specified
        if movement is not None:
            s_activity = (s_activity.T * movement).T
            d_activity = d_activity * movement

        # Analyze coactivity rates for each spine
        for spine in range(s_dFoF.shape[1]):
            # Go ahead and skip over eliminated spines
            if curr_el_spines[spine]:
                coactivity_event_num[spines[spine]] = np.nan
                coactivity_event_rate[spines[spine]] = np.nan
                spine_fraction_coactive[spines[spine]] = np.nan
                dend_fraction_coactive[spines[spine]] = np.nan
                coactivity_matrix[:, spines[spine]] = np.zeros(len(s_dFoF[:, spine]))
                continue
            # Perform correlation
            if movement is not None:
                # Correlation only during specified movements
                move_idxs = np.where(movement == 1)[0]
                corr, _ = stats.pearsonr(s_dFoF[move_idxs, spine], d_dFoF[move_idxs])
            else:
                corr, _ = stats.pearsonr(s_dFoF[:, spine], d_dFoF)
            global_correlation[spines[spine]] = corr

            # Calculate coactivity rate
            curr_coactivity = d_activity * s_activity[:, spine]
            event_num, event_rate, spine_frac, dend_frac = get_coactivity_rate(
                s_activity[:, spine],
                d_activity,
                curr_coactivity,
                sampling_rate=sampling_rate,
            )
            coactivity_event_num[spines[spine]] = event_num
            coactivity_event_rate[spines[spine]] = event_rate
            spine_fraction_coactive[spines[spine]] = spine_frac
            dend_fraction_coactive[spines[spine]] = dend_frac
            coactivity_matrix[:, spines[spine]] = curr_coactivity
            curr_coactivity_matrix[:, spine] = curr_coactivity

        # Get amplitudes, relative_onsets and activity traces
        ### First get for all dendritic events
        (
            dt_spine_traces,
            dt_dendrite_traces,
            dt_spine_amps,
            _,
            _,
            dt_dendrite_amps,
            _,
            _,
            _,
        ) = get_dend_spine_traces_and_onsets(
            d_activity,
            s_activity,
            d_dFoF,
            s_dFoF,
            refrence_trace=d_activity,
            norm_constants=glu_norm_constants,
            activity_window=(-2, 2),
            sampling_rate=sampling_rate,
        )
        (
            dt_spine_calcium_traces,
            _,
            dt_spine_calcium_amps,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = get_dend_spine_traces_and_onsets(
            d_activity,
            s_activity,
            d_dFoF,
            s_calcium,
            reference_trace=d_activity,
            coactivity=False,
            activity_window=(2, 2),
            sampling_rate=sampling_rate,
        )
        ### Get for coactive events only
        (
            co_spine_traces,
            co_dendrite_traces,
            co_spine_amps,
            _,
            co_spine_var,
            co_dendrite_amps,
            co_dendrite_auc,
            co_dendrite_var,
            rel_onsets,
        ) = get_dend_spine_traces_and_onsets(
            d_activity,
            s_activity,
            d_dFoF,
            s_dFoF,
            norm_constants=glu_norm_constants,
            reference_trace=curr_coactivity_matrix,
            activity_window=(-2, 2),
            sampling_rate=sampling_rate,
        )
        (
            co_spine_calcium_traces,
            _,
            co_spine_calcium_amps,
            co_spine_calcium_var,
            co_spine_calcium_auc,
            _,
            _,
            _,
            _,
        ) = get_dend_spine_traces_and_onsets(
            d_activity,
            s_activity,
            d_dFoF,
            s_calcium,
            norm_constants=ca_norm_constants,
            reference_trace=curr_coactivity_matrix,
            activity_window=(2, 2),
            sampling_rate=sampling_rate,
        )
        rel_dend_amps = dt_dendrite_amps - co_dendrite_amps
        rel_spine_amps = dt_spine_amps - co_spine_amps
        rel_spine_calcium_amps = dt_spine_calcium_amps - co_spine_calcium_amps
        # Store values
        for i in range(len(dt_spine_traces)):
            if curr_el_spines[i] is True:
                continue
            spine_coactive_amplitude[spines[i]] = co_spine_amps[i]
            dend_coactive_amplitude[spines[i]] = co_dendrite_amps[i]
            spine_coactive_calcium[spines[i]] = co_spine_calcium_amps[i]
            spine_coactive_std[spines[i]] = co_spine_var[i]
            spine_coactive_calcium_std[spines[i]] = co_spine_calcium_var[i]
            dend_coactive_std[spines[i]] = co_dendrite_var[i]
            spine_coactive_calcium_auc[spines[i]] = co_spine_calcium_auc[i]
            dend_coactive_auc[spines[i]] = co_dendrite_auc[i]
            relative_dend_coactive_amplitude[spines[i]] = rel_dend_amps[i]
            relative_spine_coactive_amplitude[spines[i]] = rel_spine_amps[i]
            relative_spine_coactive_calcium[spines[i]] = rel_spine_calcium_amps[i]
            relative_spine_onsets[spines[i]] = rel_onsets[i]
            dend_triggered_spine_traces[spines[i]] = dt_spine_traces[i]
            dend_triggered_spine_calcium_traces[spines[i]] = dt_spine_calcium_traces[i]
            dend_triggered_dend_traces[spines[i]] = dt_dendrite_traces[i]
            coactive_spine_traces[spines[i]] = co_spine_traces[i]
            coactive_spine_calcium_traces[spines[i]] = co_spine_calcium_traces[i]
            coactive_dend_traces[spines[i]] = co_dendrite_traces[i]

    # Return the output
    return (
        global_correlation,
        coactivity_event_num,
        coactivity_event_rate,
        spine_fraction_coactive,
        dend_fraction_coactive,
        spine_coactive_amplitude,
        dend_coactive_amplitude,
        spine_coactive_calcium,
        spine_coactive_std,
        dend_coactive_std,
        spine_coactive_calcium_std,
        dend_coactive_auc,
        spine_coactive_calcium_auc,
        relative_spine_coactive_amplitude,
        relative_dend_coactive_amplitude,
        relative_spine_coactive_calcium,
        relative_spine_onsets,
        dend_triggered_spine_traces,
        dend_triggered_dend_traces,
        dend_triggered_spine_calcium_traces,
        coactive_spine_traces,
        coactive_dend_traces,
        coactive_spine_calcium_traces,
        coactivity_matrix,
    )


def conjunctive_coactivity_analysis(
    data,
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
        
    """
    # Pull some important information from data
    pix_to_um = data.imaging_parameters["Zoom"] / 2

    spine_groupings = np.array(data.spine_grouping)
    spine_flags = data.spine_flags
    spine_volumes = np.array(data.spine_volume)
    spine_positions = data.spine_positions
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
            data.spine_volume,
            data.imaging_parameters["Zoom"],
            sampling_rate=sampling_rate,
            iterations=1000,
        )
        ca_norm_constants = spine_volume_norm_constant(
            spine_activity,
            spine_calcium,
            data.spine_volume,
            data.imaging_parameters["Zoom"],
            sampling_rate=sampling_rate,
            iterations=1000,
        )
    else:
        glu_norm_constants = np.array([None for x in spine_activity.shape[1]])
        ca_norm_constants = np.array([None for x in spine_activity.shape[1]])

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
        movement, _, _, _, _ = quantify_movment_quality(
            data.mouse_id,
            spine_activity,
            data.lever_active,
            threshold=0.5,
            sampling_rate=sampling_rate,
        )
    else:
        movement = None

    # Set up output variables
    local_correlation = np.zeros(spine_activity.shape[1])
    coactivity_event_num = np.zeros(spine_activity.shape[1])
    coactivity_event_rate = np.zeros(spine_activity.shape[1])
    spine_fraction_coactive = np.zeros(spine_activity.shape[1])
    dend_fraction_coactive = np.zeros(spine_activity.shape[1])
    coactive_spine_num = np.zeros(spine_activity.shape[1])
    nearby_spine_volumes = np.zeros(spine_activity.shape[1])
    spine_coactive_amplitude = np.zeros(spine_activity.shape[1])
    nearby_coactive_amplitude_sum = np.zeros(spine_activity.shape[1])
    spine_coactive_calcium = np.zeros(spine_activity.shape[1])
    nearby_coactive_calcium_sum = np.zeros(spine_activity.shape[1])
    dend_coactive_amplitude = np.zeros(spine_activity.shape[1])
    spine_coactive_std = np.zeros(spine_activity.shape[1])
    nearby_coactive_std = np.zeros(spine_activity.shape[1])
    spine_coactive_calcium_std = np.zeros(spine_activity.shape[1])
    nearby_coactive_calcium_std = np.zeros(spine_activity.shape[1])
    dend_coactive_std = np.zeros(spine_activity.shape[1])
    spine_coactive_calcium_auc = np.zeros(spine_activity.shape[1])
    nearby_coactive_calcium_auc_sum = np.zeros(spine_activity.shape[1])
    dend_coactive_auc = np.zeros(spine_activity.shape[1])
    relative_spine_dend_onsets = np.zeros(spine_activity.shape[1])
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
            ## Remove the eliminated spines. Don't want to consider their activity here
            nearby_spines = nearby_spines * curr_el_spines

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
            # Get conjunctive coactivity timestamps
            conj_timestamps = get_activity_timestamps(curr_conj_coactivity)

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
                d_activity,
                curr_s_activity.reshape(-1, 1),
                d_dFoF,
                curr_s_dFoF.reshape(-1, 1),
                curr_conj_coactivity.reshape(-1, 1),
                norm_constants=(glu_constant),
                activity_window=(-2, 2),
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
                d_activity,
                curr_s_activity.reshape(-1, 1),
                d_dFoF,
                curr_s_calcium.reshape(-1, 1),
                curr_conj_coactivity.reshape(-1, 1),
                norm_constants=(ca_constant),
                activity_window=(-2, 2),
                sampling_rate=sampling_rate,
            )
            spine_coactive_amplitude[spines[spine]] = s_amp
            spine_coactive_calcium[spines[spine]] = s_ca_amp
            dend_coactive_amplitude[spines[spine]] = d_amp
            spine_coactive_std[spines[spine]] = s_std
            spine_coactive_calcium_std[spines[spine]] = s_ca_std
            dend_coactive_std[spines[spine]] = d_std
            spine_coactive_calcium_auc[spines[spine]] = s_ca_auc
            dend_coactive_auc[spines[spine]] = d_auc
            relative_spine_dend_onsets[spines[spine]] = rel_onset
            coactive_spine_traces[spines[spine]] = s_traces
            coactive_spine_calcium_traces[spines[spine]] = s_ca_traces
            coactive_dend_traces[spines[spine]] = d_traces


def nearby_spine_conjunctive_events(
    timestamps,
    spine_dFoF,
    nearby_dFoF,
    nearby_calcium,
    nearby_activity,
    dendrite_dFoF,
    nearby_spine_volumes,
    target_constant=None,
    glu_constants=None,
    ca_constants=None,
    activity_window=(-2, 2),
    sampling_rate=60,
):
    """Helper function to get the dendrite, spine, and nearby spine traces
        during conjunctive coactivity events
        
        INPUT PARAMETERS
            timestamps - list of tuples with the timestamps (onset, offset) of each 
                        conjunctive coactivity event
        
            spine_dFoF - np.array of the main spine dFoF activity
            
            nearby_dFoF - 2d np.array of the nearby spines (columns) dFoF activity

            nearby_calcium - 2d np.array of the nearby spines (columns) calcium activity

            nearby_activity - 2d np.array of the nearby spines (columns) binarized activity
            
            dendrite_dFoF - np.array of the dendrite dFoF activity
            
            nearby_spine_volumes - np.array of the spine volumes of the nearby spines

            target_constant = float or int of the GluSnFR constant to normalize activity by 
                              spine volume. No noramlization if None
            
            glu_constants - np.array of the GluSnFR constants for the nearby spines

            ca_constants - np.array of the RCaMP2 constants for the nearby spines
            
            activity_window - tuple specifying the window around which you want the activity
                              from . e.g., (-2,2) for 2sec before and after
                              
            sampling_rate - int specifying the sampling rate

        OUTPUT PARAMETERS
    """
    if target_constant is not None:
        NORM = True
    else:
        NORM = False

    before_f = int(activity_window[0] * sampling_rate)
    after_f = int(activity_window[1] * sampling_rate)

    # Find dendrite onsets to center analysis around
    initial_stamps = [x[0] for x in timestamps]
    _, d_mean = d_utils.get_trace_mean_sem(
        dendrite_dFoF.reshape(-1, 1),
        ["Dendrite"],
        initial_stamps,
        window=activity_window,
        sampling_rate=sampling_rate,
    )
    d_mean = list(d_mean.values())[0][0]
    d_onset, _ = find_activity_onset([d_mean])
    d_onset = d_onset[0]
    # Correct timestamps so that they are centered on dendrite onsets
    center_point = np.absolute(activity_window[0] * sampling_rate)
    offset = center_point - d_onset
    event_stamps = [x - offset for x in initial_stamps]

    # Analyze each co-activity event
    ### Some temporary variables
    spine_nearby_correlations = []
    coactive_spine_num = []
    coacitve_spine_volumes = []
    sum_coactive_spine_traces = []
    sum_coactive_spine_ca_traces = []

    for event in event_stamps:
        # Get target spine activity
        t_spine_trace = spine_dFoF[event + before_f : event + after_f]
        if NORM:
            t_spine_trace = t_spine_trace / target_constant
        coactive_spine_traces = []
        coactive_spine_ca_traces = []
        coactive_spine_idxs = []
        # Check each nearby spine to see if coactive
        for i in range(nearby_activity.shape[1]):
            nearby_spine_a = nearby_activity[:, i]
            nearby_spine_dFoF = nearby_dFoF[:, i]
            nearby_spine_ca = nearby_calcium[:, i]
            event_activity = nearby_spine_a[event + before_f : event + after_f]
            if np.sum(event_activity):
                # If there is coactivity, append the value
                dFoF = nearby_spine_dFoF[event + before_f : event + after_f]
                calcium = nearby_spine_ca[event + before_f : event + after_f]
                if NORM:
                    dFoF = dFoF / glu_constants[i]
                    calcium = calcium / ca_constants[i]
                coactive_spine_traces.append(dFoF)
                coactive_spine_ca_traces.append(calcium)
                coactive_spine_idxs.append(i)
            else:
                continue
        #

