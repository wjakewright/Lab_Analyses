"""Module to perform spine co-activity analyses"""

import numpy as np
from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes
from scipy import stats


def spine_coactivity_analysis(
    spine_activity,
    spine_positions,
    flags,
    spine_grouping,
    bin_size=5,
    sampling_rate=60,
):
    """Function to calculate pairwise spine coactivity rate between 
        all spines along the same dendrite. Spine rates are then binned
        based on their distance from the target spine
        
        INPUT PARAMETERS
            spine_activity - np.array of the binarized spine activity traces
                            Each column represents each spine
            
            spine_positions - list of the corresponding spine positions along the dendrite
                              for each spine

            flags - list of the spine flags
            
            spine_grouping - list with the corresponding groupings of spines on
                             the same dendrite
            
            bin_size - int or float specifying the distance to bin over

        OUTPUT PARAMETERS
            coactivity_matrix - np.array of the normalized coactivity for each spine (columns)
                                over the binned distances (rows)
            
            position_bins - np.array of the distances data were binned over
            
    """
    # Set up the position bins
    MAX_DIST = 30
    bin_num = int(MAX_DIST / bin_size)
    position_bins = np.linspace(0, MAX_DIST, bin_num + 1)

    # First sort out spine_groping to make sure you can iterate
    if type(spine_grouping[0]) != list:
        spine_grouping = [spine_grouping]

    coactivity_matrix = np.zeros((bin_num, spine_activity.shape[1]))

    # find indexes of eliminated spines
    el_spines = find_spine_classes(flags, "Eliminated Spine")
    el_spines = np.array(el_spines)

    # Now iterate through each dendrite grouping
    for spines in spine_grouping:
        s_activity = spine_activity[:, spines]
        positions = np.array(spine_positions)[spines]
        curr_el_spines = el_spines[spines]

        # Go through each spine
        for i in range(s_activity.shape[1]):
            current_coactivity = []
            curr_spine = s_activity[:, i]
            # Get coactivity with each other spine
            for j in range(s_activity.shape[1]):
                # Don't compare spines to themselves
                if j == i:
                    continue
                # Don't compare eliminated spines
                if curr_el_spines[i] == True:
                    current_coactivity.append(np.nan)
                    continue
                if curr_el_spines[j] == True:
                    current_coactivity.append(np.nan)
                    continue
                test_spine = s_activity[:, j]
                co_rate = calculate_coactivity(curr_spine, test_spine, sampling_rate)
                current_coactivity.append(co_rate)

            # Order by positions
            curr_pos = positions[i]
            pos = [x for idx, x in enumerate(positions) if idx != i]
            # normalize  distances relative to current position
            relative_pos = np.array(pos) - curr_pos
            # Make all distances positive
            relative_pos = np.absolute(relative_pos)
            # Sort coactivity and position based on distance
            sorted_coactivity = np.array(
                [x for _, x in sorted(zip(relative_pos, current_coactivity))]
            )
            sorted_positions = np.array(
                [y for y, _ in sorted(zip(relative_pos, current_coactivity))]
            )
            # Bin the data
            binned_coactivity = bin_by_position(
                sorted_coactivity, sorted_positions, position_bins
            )
            coactivity_matrix[:, spines[i]] = binned_coactivity

    return coactivity_matrix, position_bins


def calculate_coactivity(spine_1, spine_2, sampling_rate):
    """Helper function to calculate spine coactivity rate"""
    duration = len(spine_1) / sampling_rate
    coactivity = spine_1 * spine_2
    events = np.nonzero(np.diff(coactivity) == 1)[0]
    event_num = len(events)
    event_freq = event_num / duration

    # Normalize frequency based on overall activity levels
    spine_1_freq = len(np.nonzero(np.diff(spine_1) == 1)[0]) / duration
    spine_2_freq = len(np.nonzero(np.diff(spine_2) == 1)[0]) / duration
    geo_mean = stats.gmean([spine_1_freq, spine_2_freq])

    coactivity_rate = event_freq / geo_mean

    return coactivity_rate


def bin_by_position(data, positions, bins):
    """Helper function to bin the data by position"""
    binned_data = []

    for i in range(len(bins)):
        if i != len(bins) - 1:
            idxs = np.where((positions > bins[i]) & (positions <= bins[i + 1]))
            binned_data.append(np.nanmean(data[idxs]))
        # else:
        #    idxs = np.where(positions > bins[i])
        #    binned_data.append(np.nanmean(data[idxs]))

    return np.array(binned_data)
