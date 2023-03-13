import numpy as np

from Lab_Analyses.Utilities.quantify_movement_quality import quantify_movement_quality


def neraby_spine_movement_quality(
    mouse_id,
    nearby_spines,
    spine_activity,
    lever_active,
    lever_force,
    threshold,
    corr_duration,
    sampling_rate,
):
    """Helper function to get the average movement encoding across the nearby
        spines for each target spine"""

    # Get the movement encoding for all spines
    (
        _,
        spine_movement_correlation,
        spine_movement_stereotypy,
        spine_movement_reliability,
        spine_movement_specificity,
        spine_LMP_reliability,
        spine_LMP_specificity,
        _,
    ) = quantify_movement_quality(
        mouse_id,
        spine_activity,
        lever_active,
        lever_force,
        threshold,
        corr_duration,
        sampling_rate,
    )

    # Setup outputs
    nearby_correlation = np.zeros(len(spine_movement_correlation))
    nearby_stereotypy = np.zeros(len(spine_movement_stereotypy))
    nearby_reliability = np.zeros(len(spine_movement_reliability))
    nearby_specificity = np.zeros(len(spine_movement_specificity))
    nearby_LMP_reliability = np.zeros(len(spine_LMP_reliability))
    nearby_LMP_specificity = np.zeros(len(spine_LMP_specificity))

    # Iterate through each spine
    for n, spines in enumerate(nearby_spines):
        nearby_correlation[n] = np.nanmean(spine_movement_correlation[spines])
        nearby_stereotypy[n] = np.nanmean(spine_movement_stereotypy[spines])
        nearby_reliability[n] = np.nanmean(spine_movement_reliability[spines])
        nearby_specificity[n] = np.nanmean(spine_movement_specificity[spines])
        nearby_LMP_reliability[n] = np.nanmean(spine_LMP_reliability[spines])
        nearby_LMP_specificity[n] = np.nanmean(spine_LMP_specificity[spines])

    return (
        nearby_correlation,
        nearby_stereotypy,
        nearby_reliability,
        nearby_specificity,
        nearby_LMP_reliability,
        nearby_LMP_specificity,
    )
