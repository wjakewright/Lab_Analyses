import numpy as np
from scipy import stats

from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes


def distance_coactivity_rate_analysis(
    spine_activity,
    spine_positions,
    flags,
    spine_grouping,
    constrain_matrix=None,
    partner_list=None,
    bin_size=5,
    sampling_rate=60,
    norm=False,
):
    """Function to calculate pairwise spine coactivity rate between all spines along the 
        same dendrite. Spine rates are then binned based on their distance from the
        target spine
        
        INPUT PARAMETERS
            spine_activity - np.array of the binarized spine activity traces.
                             Each column represents each spine
            
            spine_positions - list of the corresponding spine positions along the
                              dendrite for each spine
            
            flags - list of the spine flags
            
            spine_grouping - list with the corresponding groupings of spines on 
                             the same dendirte

            constrain_matrix- np.array of binarized events to constrain the coactivity
                                to. (e.g., dendritic events, movement periods)
            
            partner_list - boolean list specifying a subset of spines to analyze
                            coactivity rates for
            
            bin_size - int or float specifying the distance to bin over

            sampling_rate - int specifying the imaging sampling rate

            norm - boolean specifying whether or not to normalize the coactivity

        OUTPUT PARAMETERS
            coactivity_matrix - np.array of the coactivity for each spine (columns)
                                over the binned distances (rows)

            positions_bins - np.array of the distances data were binned over

    """
    # Set up the position bins
    MAX_DIST = 40
    bin_num = int(MAX_DIST / bin_size)
    position_bins = np.linspace(0, MAX_DIST, bin_num + 1)

    # Sort out the spine_groupings to make sure it is iterable
    if type(spine_grouping[0]) != list:
        spine_grouping = [spine_grouping]

    # Constrain data if if specified
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

    # Set up output variables
    coactivity_matrix = np.zeros((bin_num, spine_activity.shape[1]))
    correlation_matrix = np.zeros((bin_num, spine_activity.shape[1]))
    unbinned_coactivity = []
    unbinned_correlation = []

    # Find indexes of eliminated spines
    el_spines = find_spine_classes(flags, "Eliminated Spine")
    el_spines = np.array(el_spines)

    # Iterate through each dendrite grouping
    for spines in spine_grouping:
        s_activity = activity_matrix[:, spines]
        positions = np.array(spine_positions)[spines]
        curr_el_spines = el_spines[spines]

        # Iterate through each spine on this dendrite
        for spine in range(s_activity.shape[1]):
            curr_coactivity = []
            curr_correlation = []
            curr_spine = s_activity[:, spine]
            if type(duration) == list:
                dur = duration[spine] / sampling_rate
            else:
                dur = duration / sampling_rate

            # Calculate coactivity with each other spine
            for partner in range(s_activity.shape[1]):
                # Don't compare spines to themselves
                if partner == spine:
                    continue
                # Don't compare eliminated spines
                if curr_el_spines[spine] == True:
                    curr_coactivity.append(np.nan)
                    curr_correlation.append(np.nan)
                    continue
                if curr_el_spines[partner] == True:
                    curr_coactivity.append(np.nan)
                    curr_correlation.append(np.nan)
                    continue
                # Subselect partners if specified
                if partner_list is not None:
                    if partner_list[spines[partner]] == True:
                        partner_spine = s_activity[:, partner]
                    else:
                        curr_coactivity.append(np.nan)
                        curr_correlation.append(np.nan)
                        continue
                else:
                    partner_spine = s_activity[:, partner]

                coactivity_rate = calculate_coactivity(
                    curr_spine, partner_spine, dur, norm=norm
                )
                correlation, _ = stats.pearsonr(curr_spine, partner_spine)
                curr_coactivity.append(coactivity_rate)
                curr_correlation.append(correlation)

            # Order by positions
            curr_pos = positions[spine]
            pos = [x for idx, x in enumerate(positions) if idx != spine]
            # normalize distances relative to current position
            relative_pos = np.array(pos) - curr_pos
            # make all distances positive
            relative_pos = np.absolute(relative_pos)
            # sort coactivity and positions based on distance
            sorted_coactivity = np.array(
                [x for _, x in sorted(zip(relative_pos, curr_coactivity))]
            )
            sorted_correlation = np.array(
                [x for _, x in sorted(zip(relative_pos, curr_correlation))]
            )
            sorted_positions = np.array(
                [y for y, _ in sorted(zip(relative_pos, curr_coactivity))]
            )
            unbinned_coactivity_data = list(zip(sorted_positions, sorted_coactivity))
            unbinned_coactivity.append(unbinned_coactivity_data)
            unbinned_correlation_data = list(zip(sorted_positions, sorted_correlation))
            unbinned_correlation.append(unbinned_correlation_data)
            # Store positions and rates
            # Bin the data
            binned_coactivity = bin_by_position(
                sorted_coactivity, sorted_positions, position_bins
            )
            binned_correlation = bin_by_position(
                sorted_correlation, sorted_positions, position_bins,
            )
            coactivity_matrix[:, spines[spine]] = binned_coactivity
            correlation_matrix[:, spines[spine]] = binned_correlation

    # Convert unbinned data into single list
    # unbinned_coactivity = [y for x in unbinned_coactivity for y in x]
    # unbinned_correlation = [y for x in unbinned_correlation for y in x]

    return (
        coactivity_matrix,
        unbinned_coactivity,
        correlation_matrix,
        unbinned_correlation,
        position_bins,
    )


def calculate_coactivity(spine_1, spine_2, duration, norm):
    """Helper function to calculate spine coactivity rate between two spines"""
    coactivity = spine_1 * spine_2
    events = np.nonzero(np.diff(coactivity) == 1)[0]
    event_num = len(events)
    event_freq = event_num / duration

    if norm:
        # normalize frequency based on overall activity levels
        spine_1_freq = len(np.nonzero(np.diff(spine_1) == 1)[0]) / duration
        spine_2_freq = len(np.nonzero(np.diff(spine_2) == 1)[0]) / duration
        geo_mean = stats.gmean([spine_1_freq, spine_2_freq])
        coactivity_rate = event_freq / geo_mean
    else:
        coactivity_rate = event_freq * 60  ## convert to minutes

    return coactivity_rate


def bin_by_position(data, positions, bins):
    """Helper function to bin the data by position"""
    binned_data = []

    for i in range(len(bins)):
        if i != len(bins) - 1:
            idxs = np.nonzero((positions > bins[i]) & (positions <= bins[i + 1]))[0]
            if idxs.size == 0:
                binned_data.append(np.nan)
                continue
            binned_data.append(np.nanmean(data[idxs]))

    return np.array(binned_data)

