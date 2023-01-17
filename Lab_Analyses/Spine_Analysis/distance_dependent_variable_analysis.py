import numpy as np

from Lab_Analyses.Spine_Analysis.distance_coactivity_rate_analysis import \
    bin_by_position
from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes
from Lab_Analyses.Utilities.data_utilities import neg_num_relative_difference


def distance_dependent_variable_analysis(
    spine_variable,
    spine_positions,
    spine_flags,
    spine_groupings,
    bin_size=5,
    relative=False,
    relative_method="norm",
):
    """Function to organize and bin spine activity rates by their distance
        to the target spine
        
        INPUT PARAMETERS
            spine_variable - np.array of the spine variable to be assessed
            
            spine_positions - list of the corresponding spine positions along the 
                              dendrite
                              
            spine_flags - list of the spine flags
            
            spine_grouping - list with the corresponding groupings of spines
                             on different dendrites

            bin_size - int or float specifying the distance to bine over

            relative - boolean term on whether to calculate activity rate relative to the
                        target spine
            
            relative_method = str specifying how to calulate relative vlaues
            
        OUTPUT PARAMETERS
            activity_matrix - np.array of the activity of spines binned
                             over distances to the target spine
    """
    # Set up the position bins
    MAX_DIST = 40
    bin_num = int(MAX_DIST / bin_size)
    position_bins = np.linspace(0, MAX_DIST, bin_num + 1)

    if type(spine_groupings[0]) != list:
        spine_groupings = [spine_groupings]

    el_spines = find_spine_classes(spine_flags, "Eliminated Spine")
    el_spines = np.array(el_spines)

    activity_matrix = np.zeros((bin_num, len(spine_variable)))
    for spines in spine_groupings:
        s_var = spine_variable[spines]
        positions = np.array(spine_positions)[spines]
        curr_el_spines = el_spines[spines]

        for spine in range(len(s_var)):
            curr_activity = []
            for partner in range(len(s_var)):
                if partner == spine:
                    continue
                if curr_el_spines[spine] == True:
                    curr_activity.append(np.nan)
                    continue
                if curr_el_spines[partner] == True:
                    curr_activity.append(np.nan)
                    continue
                curr_activity.append(s_var[partner])

            # Order by positions
            curr_position = positions[spine]
            pos = [x for idx, x in enumerate(positions) if idx != spine]
            # normalize distances relative to current position
            relative_pos = np.array(pos) - curr_position
            relative_pos = np.absolute(relative_pos)
            # Sort activity and positions based on distance
            sorted_variable = np.array(
                [x for _, x in sorted(zip(relative_pos, curr_activity))]
            )
            sorted_positions = np.array(
                [y for y, _ in sorted(zip(relative_pos, curr_activity))]
            )
            if relative:
                if relative_method == "norm":
                    sorted_variable = np.array(
                        [
                            (s_var[spine] - x) / (s_var[spine] + x)
                            for x in sorted_variable
                        ]
                    )
                elif relative_method == "negative":
                    sorted_variable = np.array([s_var[spine] - x for x in sorted_variable])
                sorted_variable[np.isinf(sorted_variable)] = np.nan
            # Bin the data
            binned_activity = bin_by_position(
                sorted_variable, sorted_positions, position_bins
            )
            activity_matrix[:, spines[spine]] = binned_activity

    return activity_matrix

