import numpy as np

from Lab_Analyses.Spine_Analysis_v2.spine_utilities import (
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
        upper = threshold = [1]
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
