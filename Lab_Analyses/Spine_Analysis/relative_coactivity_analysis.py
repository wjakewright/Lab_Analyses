import numpy as np


def relative_coactivity_analysis(
    spine_activity, nearby_spines, distance_coactivity_rates
):
    """Function to compare the coactivity levels between the target
        spine and nearby spines. Includes comparing relative local coactiviyt
        rates, and comparing local coactivity with and without target 
        spine participation
        
        INPUT PARAMETERS
            spine_activity - 2d np.array of the binarized spine activity
            
            nearby_spines - list containing the nearby spine indexes for
                            each target spine
                            
            distance_coactivity_matrix - np.array of the avg local coactivity
                                        for each spine
                                        
    """
    # Set up some output variables
    avg_nearby_coactivity_rate = np.zeros(spine_activity.shape[1])
    relative_coactivity_rate = np.zeros(spine_activity.shape[1])
    frac_local_coactivity_participation = np.zeros(spine_activity.shape[1])

    # Analyze each spine
    for spine in range(spine_activity.shape[1]):
        # Get relavent activity traces
        target_activity = spine_activity[:, spine]
        nearby_activity = spine_activity[:, nearby_spines[spine]]
        if nearby_spines[spine] is None:
            avg_nearby_coactivity_rate[spine] = np.nan
            relative_coactivity_rate[spine] = np.nan
            frac_local_coactivity_participation[spine] = np.nan
            continue
        # Get avg nearby local coactivity
        nearby_coactivity = distance_coactivity_rates[nearby_spines[spine]]
        avg_nearby_coactivity = np.nanmean(nearby_coactivity)
        avg_nearby_coactivity_rate[spine] = avg_nearby_coactivity

        # Calculate relative coactivity rate
        target_coactivity = distance_coactivity_rates[spine]
        relative_coactivity = (target_coactivity - avg_nearby_coactivity) / (
            target_coactivity + avg_nearby_coactivity
        )
        relative_coactivity_rate[spine] = relative_coactivity

        # Get fraction of coactivity participation
        all_activity = np.hstack((nearby_activity, target_activity.reshape(-1, 1)))
        combined_coactivity = np.nansum(all_activity, axis=1)
        combined_coactivity[combined_coactivity > 1] = 1
        participating_coactivity = combined_coactivity * target_activity

        frac_participating = np.nansum(participating_coactivity) / np.nansum(
            combined_coactivity
        )
        frac_local_coactivity_participation[spine] = frac_participating

    return (
        avg_nearby_coactivity_rate,
        relative_coactivity_rate,
        frac_local_coactivity_participation,
    )

