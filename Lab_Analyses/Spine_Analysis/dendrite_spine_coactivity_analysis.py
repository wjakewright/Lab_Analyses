import numpy as np

from Lab_Analyses.Spine_Analysis.absolute_dendrite_coactivity import (
    absolute_dendrite_coactivity,
)
from Lab_Analyses.Spine_Analysis.calculate_cluster_score import calculate_cluster_score
from Lab_Analyses.Spine_Analysis.distance_coactivity_rate_analysis import (
    distance_coactivity_rate_analysis,
)
from Lab_Analyses.Spine_Analysis.spine_coactivity_utilities import (
    get_nearby_spine_activity,
)


def dendrite_spine_coactivity_analysis(
    spine_activity,
    spine_dFoF,
    spine_calcium,
    dend_activity,
    dend_dFoF,
    spine_groupings,
    spine_flags,
    spine_positions,
    activity_window=(-2, 4),
    cluster_dist=5,
    sampling_rate=60,
    volume_norm=None,
):
    """Function to handle the spine-dendrite coaactivity analysis
    
    INPUT PARAMETERS
        spine_activity - 2d np.array of the binarized spine activity
        
        spine_dFoF - 2d np.array of the spine dFoF traces
        
        spine_calcium - 2d np.array of the spine calcium traces
        
        dend_activity - 2d np.array of the dendrite binarized activity
        
        dend_dFoF - 2d np.array of the dendrite dFoF calcium traces
        
        spine_groupings - list of spine groupings 
        
        spine_flags - list containing the spine flags
        
        spine_positions - list or array of the spine positions along the dendrite
        
        activity_window - tuple specifying the time window in sec over which to analyze
        
        cluster_dist - int specifying the distance in um to consider local
        
        sampling_rate - int specifying the imaging sampling rate
        
        volume_norm - tuple list of constants to normalzie spine dFoF and calcium
        
    """
    # Analyze absolute spine-dendrite coactivity
    print("--- Analyzing Absolute Coactivity")
    ## All spine-dendrite coactivity events
    (
        all_coactivity_matrix,
        all_coactivity_rate,
        all_coactivity_rate_norm,
        all_spine_fraction_coactive,
        all_dend_fraction_coactive,
        all_spine_coactive_amplitude,
        all_spine_coactive_calcium,
        all_spine_coactive_auc,
        all_spine_coactive_calcium_auc,
        all_dend_coactive_amplitude,
        all_dend_coactive_auc,
        all_relative_onset,
        all_spine_coactive_traces,
        all_spine_coactive_calcium_traces,
        all_dend_coactive_traces,
    ) = absolute_dendrite_coactivity(
        spine_activity,
        spine_dFoF,
        spine_calcium,
        dend_activity,
        dend_dFoF,
        spine_groupings,
        spine_flags,
        constrain_matrix=None,
        activity_window=activity_window,
        sampling_rate=sampling_rate,
        volume_norm=volume_norm,
    )

    ## Get nearby spines and their activity matricies
    nearby_coactive_spines, nearby_combined_activity = get_nearby_spine_activity(
        spine_activity, spine_groupings, spine_flags, spine_positions, cluster_dist,
    )
    ## Get noncoactive matricies
    nearby_combined_nonactivity = np.zeros(nearby_combined_activity.shape)
    nearby_combined_nonactivity[nearby_combined_activity == 0] = 1

    ## conj coactivity events
    (
        conj_coactivity_matrix,
        conj_coactivity_rate,
        conj_coactivity_rate_norm,
        conj_spine_fraction_coactive,
        conj_dend_fraction_coactive,
        conj_spine_coactive_amplitude,
        conj_spine_coactive_calcium,
        conj_spine_coactive_auc,
        conj_spine_coactive_calcium_auc,
        conj_dend_coactive_amplitude,
        conj_dend_coactive_auc,
        conj_relative_onset,
        conj_spine_coactive_traces,
        conj_spine_coactive_calcium_traces,
        conj_dend_coactive_traces,
    ) = absolute_dendrite_coactivity(
        spine_activity,
        spine_dFoF,
        spine_calcium,
        dend_activity,
        dend_dFoF,
        spine_groupings,
        spine_flags,
        constrain_matrix=nearby_combined_activity,
        activity_window=activity_window,
        sampling_rate=sampling_rate,
        volume_norm=volume_norm,
    )

    ## non conj coactivity events
    (
        nonconj_coactivity_matrix,
        nonconj_coactivity_rate,
        nonconj_coactivity_rate_norm,
        nonconj_spine_fraction_coactive,
        nonconj_dend_fraction_coactive,
        nonconj_spine_coactive_amplitude,
        nonconj_spine_coactive_calcium,
        nonconj_spine_coactive_auc,
        nonconj_spine_coactive_calcium_auc,
        nonconj_dend_coactive_amplitude,
        nonconj_dend_coactive_auc,
        nonconj_relative_onset,
        nonconj_spine_coactive_traces,
        nonconj_spine_coactive_calcium_traces,
        nonconj_dend_coactive_traces,
    ) = absolute_dendrite_coactivity(
        spine_activity,
        spine_dFoF,
        spine_calcium,
        dend_activity,
        dend_dFoF,
        spine_groupings,
        spine_flags,
        constrain_matrix=nearby_combined_nonactivity,
        activity_window=activity_window,
        sampling_rate=sampling_rate,
        volume_norm=volume_norm,
    )

    # Get distance-dependent coactivity rates
    print("--- Analyzing Distance-Dependence Coactivity")
    distance_coactivity_rate, _, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=dend_activity,
        partner_list=None,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=False,
    )
    distance_coactivity_rate_norm, _, _ = distance_coactivity_rate_analysis(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=dend_activity,
        partner_list=None,
        bin_size=5,
        sampling_rate=sampling_rate,
        norm=True,
    )
    ## Local values only (< 5um)
    avg_local_coactivity_rate = distance_coactivity_rate[0, :]
    avg_local_coactivity_rate_norm = distance_coactivity_rate_norm[0, :]

    # Calculate cluster score
    print("--- Calculating Cluster Score")
    cluster_score, coactive_num = calculate_cluster_score(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=dend_activity,
        partner_list=None,
        iterations=100,
    )

    return (
        all_coactivity_matrix,
        all_coactivity_rate,
        all_coactivity_rate_norm,
        all_spine_fraction_coactive,
        all_dend_fraction_coactive,
        all_spine_coactive_amplitude,
        all_spine_coactive_calcium,
        all_spine_coactive_auc,
        all_spine_coactive_calcium_auc,
        all_dend_coactive_amplitude,
        all_dend_coactive_auc,
        all_relative_onset,
        all_spine_coactive_traces,
        all_spine_coactive_calcium_traces,
        all_dend_coactive_traces,
        conj_coactivity_matrix,
        conj_coactivity_rate,
        conj_coactivity_rate_norm,
        conj_spine_fraction_coactive,
        conj_dend_fraction_coactive,
        conj_spine_coactive_amplitude,
        conj_spine_coactive_calcium,
        conj_spine_coactive_auc,
        conj_spine_coactive_calcium_auc,
        conj_dend_coactive_amplitude,
        conj_dend_coactive_auc,
        conj_relative_onset,
        conj_spine_coactive_traces,
        conj_spine_coactive_calcium_traces,
        conj_dend_coactive_traces,
        nonconj_coactivity_matrix,
        nonconj_coactivity_rate,
        nonconj_coactivity_rate_norm,
        nonconj_spine_fraction_coactive,
        nonconj_dend_fraction_coactive,
        nonconj_spine_coactive_amplitude,
        nonconj_spine_coactive_calcium,
        nonconj_spine_coactive_auc,
        nonconj_spine_coactive_calcium_auc,
        nonconj_dend_coactive_amplitude,
        nonconj_dend_coactive_auc,
        nonconj_relative_onset,
        nonconj_spine_coactive_traces,
        nonconj_spine_coactive_calcium_traces,
        nonconj_dend_coactive_traces,
        distance_coactivity_rate,
        distance_coactivity_rate_norm,
        avg_local_coactivity_rate,
        avg_local_coactivity_rate_norm,
        cluster_score,
        coactive_num,
    )
