from Lab_Analyses.Spine_Analysis.absolute_local_coactivity import (
    absolute_local_coactivity,
)
from Lab_Analyses.Spine_Analysis.calculate_cluster_score import calculate_cluster_score
from Lab_Analyses.Spine_Analysis.distance_coactivity_rate_analysis import (
    distance_coactivity_rate_analysis,
)
from Lab_Analyses.Spine_Analysis.distance_dependent_variable_analysis import (
    distance_dependent_variable_analysis,
)
from Lab_Analyses.Spine_Analysis.relative_coactivity_analysis import (
    relative_coactivity_analysis,
)


def local_spine_coactivity_analysis(
    mouse_id,
    spine_activity,
    spine_dFoF,
    spine_calcium,
    spine_groupings,
    spine_flags,
    spine_volumes,
    spine_positions,
    movement_spines,
    non_movement_spines,
    lever_active,
    lever_unactive,
    lever_force,
    activity_window=(-2, 4),
    cluster_dist=5,
    sampling_rate=60,
    volume_norm=None,
):
    """Function to handle the local spine coactivity analysis functions
    
        INPUT PARAMETERS
            mouse_id - str specifying the mouse id

            spine_activity-  2d np.array of the binarrized spine activity. Columns=spines
            
            spine_dFoF - 2d np.array of the spine dFoF traces. Columns = spines
            
            spine_calcium - 2d np.array of spine calcium traces. Columns = spines
            
            spine_groupings - list of spine groupings
            
            spine_flags - list containing the spine flags
            
            spine_volumes - list or array of the estimated spine volumes (um)
            
            spine_positions - list or array of the spine positions along their parent dendrite
            
            movement_spines - boolean list of whether each spine is a MRS
            
            non_movement_spines - boolean list of whether each spine is not a MRS
            
            lever_active - np.array of the binarized lever activity
            
            lever_unactive - np.array of the binarized lever inactivity
            
            lever_force - np.array of the lever force

            activity_window - tuple specifying the time window in sec over which to analyze
            
            cluster_dist - int specifying the distance in um tht is considered local
            
            sampling_rate - int specifying the imaging sampling rate
            
            volume_norm - tuple list of constants to normalize spine dFoF and calcium by
    """

    # Get distance-dependent coactivity rates
    ## Non-specified
    distance_coactivity_rate, distance_bins = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=None,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=False,
    )

    distance_coactivity_rate_norm, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=None,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=True,
    )
    ## Movement-related spines
    MRS_distance_coactivity_rate, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=movement_spines,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=False,
    )
    MRS_distance_coactivity_rate_norm, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=movement_spines,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=True,
    )
    ## Non-Movement-related spines
    nMRS_distance_coactivity_rate, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=non_movement_spines,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=False,
    )
    nMRS_distance_coactivity_rate_norm, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=None,
        partner_list=non_movement_spines,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=True,
    )
    movement_distance_coactivity_rate, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=lever_active,
        partner_list=None,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=False,
    )
    movement_distance_coactivity_rate_norm, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=lever_active,
        partner_list=None,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=True,
    )
    nonmovement_distance_coactivity_rate, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=lever_unactive,
        partner_list=None,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=False,
    )
    nonmovement_distance_coactivity_rate_norm, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=lever_unactive,
        partner_list=None,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=True,
    )
    ## Local values only (< 5um)
    avg_local_coactivity_rate = distance_coactivity_rate[0, :]
    avg_local_coactivity_rate_norm = distance_coactivity_rate_norm[0, :]
    avg_MRS_local_coactivity_rate = MRS_distance_coactivity_rate[0, :]
    avg_MRS_local_coactivity_rate_norm = MRS_distance_coactivity_rate_norm[0, :]
    avg_nMRS_local_coactivity_rate = nMRS_distance_coactivity_rate[0, :]
    avg_nMRS_local_coactivity_rate_norm = nMRS_distance_coactivity_rate_norm[0, :]
    avg_movement_local_coactivity_rate = movement_distance_coactivity_rate[0, :]
    avg_movement_local_coactivity_rate_norm = movement_distance_coactivity_rate_norm[
        0, :
    ]
    avg_nonmovement_local_coactivity_rate = nonmovement_distance_coactivity_rate[0, :]
    avg_nonmovement_local_coactivity_rate_norm = nonmovement_distance_coactivity_rate_norm[
        0, :
    ]

    # Analyze absolute local coactivity
    (
        nearby_spine_idxs,
        nearby_coactive_spine_idxs,
        avg_nearby_spine_freq,
        avg_nearby_coactive_spine_freq,
        rel_nearby_spine_freq,
        rel_nearby_coactive_spine_freq,
        frac_nearby_MRSs,
        nearby_coactive_spine_volumes,
        local_coactivity_rate,
        local_coactivity_rate_norm,
        spine_fraction_coactive,
        local_coactivity_matrix,
        spine_coactive_amplitude,
        spine_coactive_calcium,
        spine_coactive_auc,
        spine_coactive_calcium_auc,
        spine_coactive_traces,
        spine_coactive_calcium_traces,
        avg_coactive_spine_num,
        sum_nearby_amplitude,
        avg_nearby_amplitude,
        sum_nearby_calcium,
        avg_nearby_calcium,
        sum_nearby_calcium_auc,
        avg_nearby_calcium_auc,
        avg_coactive_num_before,
        sum_nearby_amplitude_before,
        avg_nearby_amplitude_before,
        sum_nearby_calcium_before,
        avg_nearby_calcium_before,
        avg_nearby_spine_onset,
        avg_nearby_spine_jitter,
        avg_nearby_event_onset,
        avg_nearby_event_jitter,
        sum_coactive_binary_traces,
        sum_coactive_spine_traces,
        avg_coactive_spine_traces,
        sum_coactive_calcium_traces,
        avg_coactive_calcium_traces,
        avg_nearby_move_corr,
        avg_nearby_move_reliability,
        avg_nearby_move_specificity,
    ) = absolute_local_coactivity(
        mouse_id,
        spine_activity,
        spine_dFoF,
        spine_calcium,
        spine_groupings,
        spine_flags,
        spine_volumes,
        spine_positions,
        movement_spines,
        lever_active,
        lever_force,
        partner_list=None,
        activity_window=activity_window,
        cluster_dist=cluster_dist,
        sampling_rate=sampling_rate,
        volume_norm=volume_norm,
    )

    # Compare coactivity with and without target spine
    (
        avg_nearby_coactivity_rate,
        relative_coactivity_rate,
        frac_local_coactivity_participation,
    ) = relative_coactivity_analysis(
        spine_activity, nearby_spine_idxs, avg_local_coactivity_rate
    )

    # Get distance dependent coactivity along the dendrite
    positional_coactivity = distance_dependent_variable_analysis(
        avg_local_coactivity_rate,
        spine_positions,
        spine_flags,
        spine_groupings,
        bin_size=5,
        relative=False,
    )
    positional_coactivity_norm = distance_dependent_variable_analysis(
        avg_local_coactivity_rate_norm,
        spine_positions,
        spine_flags,
        spine_groupings,
        bin_size=5,
        relative=False,
    )
    relative_positional_coactivity = distance_dependent_variable_analysis(
        avg_local_coactivity_rate,
        spine_positions,
        spine_flags,
        spine_groupings,
        bin_size=5,
        relative=True,
    )
    relative_positional_coactivity_norm = distance_dependent_variable_analysis(
        avg_local_coactivity_rate_norm,
        spine_positions,
        spine_flags,
        spine_groupings,
        bin_size=5,
        relative=True,
    )

    return (
        distance_bins,
        distance_coactivity_rate,
        distance_coactivity_rate_norm,
        MRS_distance_coactivity_rate,
        MRS_distance_coactivity_rate_norm,
        nMRS_distance_coactivity_rate,
        nMRS_distance_coactivity_rate_norm,
        avg_local_coactivity_rate,
        avg_local_coactivity_rate_norm,
        avg_MRS_local_coactivity_rate,
        avg_MRS_local_coactivity_rate_norm,
        avg_nMRS_local_coactivity_rate,
        avg_nMRS_local_coactivity_rate_norm,
        avg_movement_local_coactivity_rate,
        avg_movement_local_coactivity_rate_norm,
        avg_nonmovement_local_coactivity_rate,
        avg_nonmovement_local_coactivity_rate_norm,
        nearby_spine_idxs,
        nearby_coactive_spine_idxs,
        avg_nearby_spine_freq,
        avg_nearby_coactive_spine_freq,
        rel_nearby_spine_freq,
        rel_nearby_coactive_spine_freq,
        frac_nearby_MRSs,
        nearby_coactive_spine_volumes,
        local_coactivity_rate,
        local_coactivity_rate_norm,
        spine_fraction_coactive,
        local_coactivity_matrix,
        spine_coactive_amplitude,
        spine_coactive_calcium,
        spine_coactive_auc,
        spine_coactive_calcium_auc,
        spine_coactive_traces,
        spine_coactive_calcium_traces,
        avg_coactive_spine_num,
        sum_nearby_amplitude,
        avg_nearby_amplitude,
        sum_nearby_calcium,
        avg_nearby_calcium,
        sum_nearby_calcium_auc,
        avg_nearby_calcium_auc,
        avg_coactive_num_before,
        sum_nearby_amplitude_before,
        avg_nearby_amplitude_before,
        sum_nearby_calcium_before,
        avg_nearby_calcium_before,
        avg_nearby_spine_onset,
        avg_nearby_spine_jitter,
        avg_nearby_event_onset,
        avg_nearby_event_jitter,
        sum_coactive_binary_traces,
        sum_coactive_spine_traces,
        avg_coactive_spine_traces,
        sum_coactive_calcium_traces,
        avg_coactive_calcium_traces,
        avg_nearby_move_corr,
        avg_nearby_move_reliability,
        avg_nearby_move_specificity,
        avg_nearby_coactivity_rate,
        relative_coactivity_rate,
        frac_local_coactivity_participation,
        positional_coactivity,
        positional_coactivity_norm,
        relative_positional_coactivity,
        relative_positional_coactivity_norm,
    )

