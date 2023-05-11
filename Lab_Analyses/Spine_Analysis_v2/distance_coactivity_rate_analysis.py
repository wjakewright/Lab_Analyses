from Lab_Analyses.Spine_Analysis_v2.calculate_chance_local_coactivity import (
    calculate_chance_local_coactivity,
)
from Lab_Analyses.Spine_Analysis_v2.calculate_distance_coactivity_rate import (
    calculate_distance_coactivity_rate,
)


def distance_coactivity_rate_analysis(
    spine_activity,
    spine_positions,
    spine_flags,
    spine_groupings,
    constrain_matrix=None,
    partner_list=None,
    bin_size=5,
    cluster_dist=5,
    sampling_rate=60,
    norm_method="mean",
    alpha=0.05,
    iterations=1000,
):
    """Helper function to perform calculate_distance_coactivity_rate and 
        calculate_chance_coactivity together"""

    # Distance coactivity rate
    (
        distance_coactivity_rate,
        _,
        distance_coactivity_rate_norm,
        _,
        position_bins,
    ) = calculate_distance_coactivity_rate(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        constrain_matrix=constrain_matrix,
        partner_list=partner_list,
        bin_size=bin_size,
        sampling_rate=sampling_rate,
        norm_method=norm_method,
    )

    # Local coactivity vs chance
    (
        avg_local_coactivity_rate,
        avg_local_coactivity_rate_norm,
        shuff_local_coactivity_rate,
        shuff_local_coactivity_rate_norm,
        relative_diff,
        relative_diff_norm,
        coactive_spines,
        coactive_norm_spines,
    ) = calculate_chance_local_coactivity(
        spine_activity,
        spine_positions,
        spine_flags,
        spine_groupings,
        cluster_dist=cluster_dist,
        constrain_matrix=constrain_matrix,
        partner_list=partner_list,
        sampling_rate=sampling_rate,
        norm_method=norm_method,
        alpha=alpha,
        iterations=iterations,
    )

    return (
        position_bins,
        distance_coactivity_rate,
        distance_coactivity_rate_norm,
        avg_local_coactivity_rate,
        avg_local_coactivity_rate_norm,
        shuff_local_coactivity_rate,
        shuff_local_coactivity_rate_norm,
        relative_diff,
        relative_diff_norm,
        coactive_spines,
        coactive_norm_spines,
    )
