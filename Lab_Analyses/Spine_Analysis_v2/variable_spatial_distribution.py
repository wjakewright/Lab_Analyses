import numpy as np

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    bin_by_position, find_present_spines)


def variable_spatial_distribution(
    spine_data,
    spine_positions,
    spine_flags,
    spine_groupings,
    partner_list=None,
    bin_size=5,
    density=False,
):
    """Function to organize and bin a given spine variable by their distances to 
        the target spine
        
        INPUT PARAMETERS
            spine_data - np.array of the spine variable to be assessed
            
            spine_positions - list of the corresponding spine positions along the
                              dendrite
            
            spine_flags - list of the spine flags
            
            spine_groupings - list with the corresponding groupings of spines
                              on different dendrites

            partner_list - boolean array specifying specific types of spines to include
                          for anlaysis
                              
            bin_size - int or float specifying the distance to bin over

            density - boolean term whether or not to calculate density within each bin

        OUTPUT PARAMETERS
            binned_variable - 2d np.array of the given variable for each spine (col)
                             over the binned distances (rows)

            unbinned_variable - list of lists for each spine containing tuple pairs of 
                                (position, variable) for every other spine on the dendrite
    """
    # Set up the position bins
    MAX_DIST = 40
    bin_num = int(MAX_DIST / bin_size)
    position_bins = np.linspace(0, MAX_DIST, bin_num + 1)

    # Sort out the spine groupings to make sure it is iterable
    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    # Set up the output
    binned_variable = np.zeros((bin_num, len(spine_data))) * np.nan
    unbinned_variable = []

    # Find the present spines
    present_spines = find_present_spines(spine_flags)

    # Refine potential partners
    # Refine the partner list if specified
    if partner_list is not None:
        present_partners = present_spines * partner_list
    else:
        present_partners = present_spines

    # Iterate through each spine grouping
    for spines in spine_groupings:
        data = spine_data[spines]
        positions = spine_positions[spines]
        curr_present = present_spines[spines]
        curr_partners = present_partners[spines]

        # Iterate through each spine
        for spine in range(len(data)):
            # Pull present partner spine data
            partner_data = []
            for partner in range(len(data)):
                if partner == spine:
                    continue
                if curr_present[spine] == False:
                    partner_data.append(np.nan)
                    continue
                if curr_partners[partner] == False:
                    partner_data.append(np.nan)
                    continue
                partner_data.append(data[partner])

            # Order by positions
            curr_position = positions[spine]
            pos = [x for idx, x in enumerate(positions) if idx != spine]
            # Normalize the distances relative to the target spine
            relative_pos = np.array(pos) - curr_position
            relative_pos = np.absolute(relative_pos)
            # Sort variable and positions based on distance
            sorted_variable = np.array(
                [x for _, x in sorted(zip(relative_pos, partner_data))]
            )
            sorted_positions = np.array(
                [y for y, _ in sorted(zip(relative_pos, partner_data))]
            )
            unbinned_variable.append(list(zip(sorted_positions, sorted_variable)))
            # Bin the data
            if density == False:
                binned_var = bin_by_position(
                    sorted_variable, sorted_positions, position_bins,
                )
            else:
                binned_var = bin_by_position(
                    sorted_variable, sorted_positions, position_bins, const=bin_size,
                )
            binned_variable[:, spines[spine]] = binned_var

    return binned_variable, unbinned_variable, position_bins
