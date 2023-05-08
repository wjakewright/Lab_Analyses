import numpy as np

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
    find_present_spines,
    find_spine_classes,
    find_stable_spines_across_days,
)


def calculate_volume_change(
    volume_list, flag_list, norm=False, exclude=None,
):
    """Function to calculate relative volume change across all spines
    
        INPUT PARAMETERS
            volume_list - list of np.arrays of spine volumes, with each array corresponding
                         to a different session
            
            flag_list - list of lists containing the spine flags corresponding to the spines
                        in the spines in the volume list
            
            norm - boolean term specifying whether to calculate normalized volume changes
            
            exclude - str specifying a type of spine to exclude from the analysis
            
        OUTPUT PARAMETERS
            relative_volumes - list of arrays containing the relative volume change for each
                                session
            
            stable_idxs - np.array of the indexes of the stable spines
    """
    # Get the stable spine idxs
    stable_spines = find_stable_spines_across_days(flag_list)

    # Find additional spines to exclude
    if exclude:
        exclude_spines = find_spine_classes(flag_list[-1], exclude)
        # Reverse values
        keep_spines = np.array([not x for x in exclude_spines])
        # Combine with the stable spines
        try:
            stable_spines = (stable_spines * keep_spines).astype(bool)
        except ValueError:
            pass

    # Get the idxs of the stable spines
    stable_idxs = np.nonzero(stable_spines)[0]
    # Get the volumes only for the stable spines
    spine_volumes = [v[stable_idxs] for v in volume_list]
    # Calculate the relative volume now
    base = spine_volumes[0]
    if not norm:
        relative_volumes = [vol / base for vol in spine_volumes]
    else:
        relative_volumes = [(vol - base) / (vol + base) for vol in spine_volumes]

    return relative_volumes, stable_idxs


def classify_plasticity(relative_volumes, threshold=0.3, norm=False):
    """Function to classify if spines have undergo enlargement or shrinkage
        based on a given threshold
        
        INPUT PARAMETERS
            relative_volumes - list of relative volume changes for each spine
            
            threshold - tuple or float specifying what the cutoff for enlargement
                        and shrinkage are

            norm - boolean specifying if the relative volumes are normalized or not
        
        OUTPUT PARAMETERS
            enlarged_spines - boolean list of spines classified as enlarged

            shrunken_spines - boolean list of spines classified as shrunkent

            stable_spiens - boolean list of spines classified as stable
    """
    # Initialize the outputs
    enlarged_spines = [False for x in relative_volumes]
    shrunken_spines = [False for x in relative_volumes]
    stable_spines = [False for x in relative_volumes]

    # Split threshold if tuple given
    if type(threshold) == tuple:
        lower = threshold[0]
        upper = threshold[1]
    else:
        lower = threshold
        upper = threshold

    # Classify each spine
    if not norm:
        for i, spine in enumerate(relative_volumes):
            if spine >= (1 + upper):
                enlarged_spines[i] = True
            elif spine <= (1 - lower):
                shrunken_spines[i] = True
            else:
                stable_spines[i] = True
    else:
        for i, spine in enumerate(relative_volumes):
            if spine >= upper:
                enlarged_spines[i] = True
            elif spine <= lower:
                shrunken_spines[i] = True
            else:
                stable_spines[i] = True

    return enlarged_spines, shrunken_spines, stable_spines


def calculate_spine_dynamics(
    spine_flag_list, spine_positions_list, spine_groupings_list
):
    """Function to calculate the spine density and the fraction of new and 
        eliminated spines along a given dendrite
        
        INPUT PARAMETERS
            spine_flag_list - list lists containing the spine flags for each session
            
            spine_position_list - list of  np.array of the spine positions for each session
            
            spine_groupinglist - list of list of the spine groupings along the different
                             dendrites for each session
        
        OUTPUT PARAMETERS
            spine_density - list of np.array of the density of spines along the dendrite
            
            fraction_new_spines - list np.array of the fraction of new spines
            
            fraction_eliminated_spines - list np.array of the fraction of eliminated spines
    """
    # Set up the outputs
    spine_density = []
    fraction_new_spines = []
    fraction_eliminated_spines = []

    # Iterate through each session
    for i, (flags, positions, groupings) in enumerate(
        zip(spine_flag_list, spine_positions_list, spine_groupings_list)
    ):
        # First correct the carry over of the eliminated spines
        if i != 0:
            prev_flags = spine_flag_list[i - 1]
            temp_flags = []
            for prev, curr in zip(prev_flags, flags):
                if "Eliminated Spine" in prev and "Eliminated Spine" in curr:
                    temp_flags.append(["Absent"])
                elif "Absent" in prev and "Absent" not in curr:
                    if "Eliminated Spine" in curr:
                        temp_flags.append(["Absent"])
                    else:
                        temp_flags.append(["New Spine"])
                else:
                    temp_flags.append(curr)
        else:
            temp_flags = flags

        # Sort out the spine groupings
        if type(groupings[0]) != list:
            groupings = [groupings]

        curr_density = np.zeros(len(groupings))
        curr_frac_new = np.zeros(len(groupings))
        curr_frac_elim = np.zeros(len(groupings))
        # Iterate through each dendrite grouping
        for j, spines in enumerate(groupings):
            # Get the current flags
            curr_flags = [x for i, x in enumerate(temp_flags) if i in spines]
            # Calculate the spine density
            present_spines = find_present_spines(curr_flags)
            length = np.max(positions[spines]) - np.min(positions[spines])
            curr_density[j] = np.sum(present_spines) / length

            # Determine the fraction of new spines
            new_spines = find_spine_classes(curr_flags, "New Spine")
            curr_frac_new[j] = np.sum(new_spines) / np.sum(present_spines)

            # Determine the fraction of eliminated spines
            elim_spines = find_spine_classes(curr_flags, "Eliminated Spine")
            curr_frac_elim[j] = np.sum(elim_spines) / np.sum(present_spines)

        # Store the values
        spine_density.append(curr_density)
        fraction_new_spines.append(curr_frac_new)
        fraction_eliminated_spines.append(curr_frac_elim)

    return spine_density, fraction_new_spines, fraction_eliminated_spines

