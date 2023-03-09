import numpy as np

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    bin_by_position,
    find_present_spines,
)
from Lab_Analyses.Utilities.coactivity_functions import calculate_coactivity


def calculate_distance_coactivity_rate(
    spine_activity,
    spine_positions,
    flags,
    spine_groupings,
    constrain_matrix=None,
    partner_list=None,
    bin_size=5,
    sampling_rate=60,
    norm_method="mean",
):
    """Function to calculate pairwise spine coactivity rate between all spines along
        the same dendrite. Spine rates are then binned across their distances from
        the target spine
        
        INPUT PARAMETERS
            spine_activity - np.array of the binarized spine activity traces
                            Each column represents a spine
            
            spine_positions - np.array of t he corresponding spine positions 
                              along the dendrite for each spine
            
            flags - list of the spine flags
            
            spine_groupings - list with the corresponding groupings of the spines
                              on the same dendrite
            
            constrain_matrix - np.array of the binarized events to constrain
                              the coactivity to. (e.g., dendritic events, movement
                              periods)
            
            partner_list - boolean list specifying a subset of spines to anlyze coactivity
                            rates for (e.g., movement spines)
                            
            bin_size - int or float specifying the distance to bin over
            
            sampling_rate - int specifying the imaging sampling rate

            norm_method - str specifying how you want to normalize the coactivity rate
                            Accepts "mean" to normalize by the geo mean, or "freq" to 
                            normalize by the target spine frequency
        
        OUTPUT PARAMETERS
            binned_coactivity - 2d np.array of the coactivity for each spine (column)
                                over the binned distances (rows)
            
            unbinned_coactivity - list of lists for each spine containing tuple pairs 
                                  of (position, coactivity) for every other spine on the
                                  dendrite
            
            binned_coactivity_norm - 2d np.array of the normalized coactivity for each spine
                                    (column) over the binned distances (rows)
            
            unbinned_coactivity_norm - list of lists for each spine containing tuple paris
                                        of (position, norm. coactivity) for every other spine
                                        on the dendrite
            
            position_bins - np.array of the distances the data were binned over
    """
    # Set up the postion bins
    MAX_DIST = 40
    bin_num = int(MAX_DIST / bin_size)
    position_bins = np.linspace(0, MAX_DIST, bin_num + 1)

    # Sort out the spine groupings to make sure it is iterable
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # constrain the data if specified
    if constrain_matrix is not None:
        if len(constrain_matrix.shape) == 1:
            constrain_matrix = constrain_matrix.reshape(-1, 1)
            duration = np.sum(constrain_matrix)
        else:
            duration = [
                np.sum(constrain_matrix[:, i]) for i in range(constrain_matrix.shape[1])
            ]
        activity_matrix = spine_activity * constrain_matrix
    else:
        activity_matrix = spine_activity
        duration = spine_activity.shape[0]

    # Setup the output variables
    binned_coactivity = np.zeros((bin_num, spine_activity.shape[1])) * np.nan
    unbinned_coactivity = []
    binned_coactivity_norm = np.zeros((bin_num, spine_activity.shape[1])) * np.nan
    unbinned_coactivity_norm = []

    # Find the present spines
    present_spines = find_present_spines(flags)

    # Iterate through each dendrite grouping
    for spines in spine_groupings:
        s_activity = activity_matrix[:, spines]
        positions = spine_positions[spines]
        curr_present = present_spines[spines]

        # Iterate through each spine on this dendrite
        for spine in range(s_activity.shape[1]):
            curr_coactivity = []
            curr_coactivity_norm = []
            curr_spine = s_activity[:, spine]
            if type(duration) == list:
                dur = duration[spine] / sampling_rate
            else:
                dur = duration / sampling_rate

            # Calculate the coactivity with each other spine
            for partner in range(s_activity.shape[1]):
                # Don't compare spines to themselves
                if partner == spine:
                    continue
                # Don't compare eliminated spines
                if curr_present[spine] == False:
                    curr_coactivity.append(np.nan)
                    curr_coactivity_norm.append(np.nan)
                    continue
                if curr_present[partner] == False:
                    curr_coactivity.append(np.nan)
                    curr_coactivity_norm.append(np.nan)
                    continue
                # Subselect partners if specifyied
                if partner_list is not None:
                    if partner_list[spines[partner]] == True:
                        partner_spine = s_activity[:, partner]
                    else:
                        curr_coactivity.append(np.nan)
                        curr_coactivity_norm.append(np.nan)
                        continue
                else:
                    partner_spine = s_activity[:, partner]

                ## Calculate coactivity
                co_rate, co_rate_norm, _, _, _ = calculate_coactivity(
                    curr_spine,
                    partner_spine,
                    norm_method=norm_method,
                    duration=dur,
                    sampling_rate=sampling_rate,
                )
                curr_coactivity.append(co_rate)
                curr_coactivity_norm.append(co_rate_norm)

            # Order by position
            curr_pos = positions[spine]
            pos = [x for idx, x in enumerate(positions) if idx != spine]
            # normalize distances relative to current position
            relative_pos = np.array(pos) - curr_pos
            relative_pos = np.absolute(relative_pos)
            # Sort coactivity and positions based on distance
            sorted_coactivity = np.array(
                [x for _, x in sorted(zip(relative_pos, curr_coactivity))]
            )
            sorted_coactivity_norm = np.array(
                [x for _, x in sorted(zip(relative_pos, curr_coactivity_norm))]
            )
            sorted_positions = np.array(
                [y for y, _ in sorted(zip(relative_pos, curr_coactivity))]
            )
            unbinned_data = list(zip(sorted_positions, sorted_coactivity))
            unbinned_coactivity.append(unbinned_data)
            unbinned_data_norm = list(zip(sorted_positions, sorted_coactivity_norm))
            unbinned_coactivity_norm.append(unbinned_data_norm)
            # Bin the data
            binned_co = bin_by_position(
                sorted_coactivity, sorted_positions, position_bins
            )
            binned_co_norm = bin_by_position(
                sorted_coactivity_norm, sorted_positions, position_bins
            )
            binned_coactivity[:, spines[spine]] = binned_co
            binned_coactivity_norm[:, spines[spine]] = binned_co_norm

    return (
        binned_coactivity,
        unbinned_coactivity,
        binned_coactivity_norm,
        unbinned_coactivity_norm,
        position_bins,
    )
