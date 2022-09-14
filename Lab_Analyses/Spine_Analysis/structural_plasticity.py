"""Module for analyzing structural spine plasticity"""

from itertools import compress

import numpy as np
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    find_stable_spines,
)


def calculate_volume_change(volume_list, flag_list, days=None, exclude=None):
    """Function to calculate relative volume change across all spines
    
        INPUT PARAMETERS
            volume_list - list of np.arrays of spine volumes, with each array corresponding
                            to each day
            
            flag_list - list containing the corresponding spine flags, with each item corresponding
                        to the volume_list arrays for each day
            
            days - list of str specifying which day each volume list corresponds to. Default is
                    none, which automatically generates generic labels
            
            exclude - str specifying a type of spine flag to exclude from the analysis
        
        OUTPUT PARAMETERS
            relative_volumes - dict containing the relative volume change for each day

            stable_idxs - np.array of the indexes of the stable spines
    """
    # Set up days
    if days is None:
        days = [f"Day {x}" for x in range(1, len(volume_list) + 1)]

    # Get indexes of stable spines throughout all analyzed days
    stable_spines = find_stable_spines(flag_list)

    # Find additional spines to exclude
    if exclude:
        exclude_spines = find_spine_classes(flag_list[-1], exclude)
        # Reverse values to exclude these spines
        exclude_spines = np.array([not x for x in exclude_spines])
        # Combine with the stable spines
        stable_spines = stable_spines * exclude_spines

    # Get volumes only for stable_spines
    spine_volumes = []
    for volumes in volume_list:
        stable_vols = np.array(volumes)[stable_spines]
        spine_volumes.append(stable_vols)

    # Calculate relative volume now
    baseline_volume = spine_volumes[0]
    relative_volumes = {}
    for vol, day in zip(spine_volumes, days):
        rel_vol = vol / baseline_volume
        relative_volumes[day] = rel_vol

    # Get stable spine indexes
    stable_idxs = np.nonzero(stable_spines)[0]

    return relative_volumes, stable_idxs


def classify_plasticity(relative_volumes, threshold=0.25):
    """Function to classify if spines have undergone plasticity
    
        INPUT PARAMETERS
            relative_volumes - list of relative volume changes for each spine
            
            threshold - float specifying what is the cutoff threshold
        
        OUTPUT PARAMETERS
            potentiated_spines - boolean list of spines classified as potentiated
            
            depressed_spines - boolean list of spines classified as depressed
            
            stable_spines - boolean list of spines that are stable
    """
    # Initialize outputs
    stable_spines = [False for x in relative_volumes]
    potentiated_spines = [False for x in relative_volumes]
    depressed_spines = [False for x in relative_volumes]

    # classify each spine
    for i, spine in enumerate(relative_volumes):
        if spine >= (1 + threshold):
            potentiated_spines[i] = True
        elif spine <= (1 - threshold):
            depressed_spines[i] = True
        else:
            stable_spines[i] = True

    return potentiated_spines, depressed_spines, stable_spines


def calculate_spine_dynamics(data_list, days=None, distance=10):
    """Function to calculate the spine density and rate of spine 
        generation and elimination
        
        INPUT PARAMETERS
            data_list - list of datasets to compare each other to. All datasets will
                        be compared to the first dataset. This requires data to have both
                        lists of spine volume and spine flags. 
            
            days - list of str specifying which day each dataset corresponds to.
                    Default is none, which will automatically generate labels

            exclude - str specifying a type of spine to exclude from the analysis

            distance - int over what distance of dendrite you want to 
                        calculate spine density over

        OUTPUT PARAMETERS
            spine_density - dict containing spine density for each day

            normalized_spine_density - dict containing spine densities normalized to day 1

            fraction_new_spines - dict containing fraction of new spines for each day

            fraction_eliminated_spines - dict containing fraction of eliminated spines for each day
    
    """
    # Set up days
    if days is None:
        days = [f"Day {x}" for x in range(1, len(data_list) + 1)]

    # Set up the spine flags, and make sure they are the same length
    flag_list = [data.spine_flags for data in data_list]
    # Get the new spines and eliminated spines for each day
    new_spine_list = []
    eliminated_spine_list = []
    stable_spine_list = []
    for flags in flag_list:
        new_spines = np.array(find_spine_classes(flags, "New Spine"))
        new_spine_list.append(new_spines)
        eliminated_spines = np.array(find_spine_classes(flags, "Eliminated Spine"))
        eliminated_spine_list.append(eliminated_spines)
        rev_new = np.array([not x for x in new_spines])
        rev_el = np.array([not x for x in eliminated_spines])
        stable_spines = np.array([True for x in new_spines])
        stable_spines = stable_spines * rev_new * rev_el
        stable_spine_list.append(stable_spines)

    # Set up outputs
    spine_density = {}
    normalized_spine_density = {}
    fraction_new_spines = {}
    fraction_eliminated_spines = {}

    # Get the spine densities for each day
    for i, (day, dataset) in enumerate(zip(days, data_list)):
        groupings = dataset.spine_grouping
        if type(groupings[0]) != list:
            groupings = [groupings]
        densities = []
        for dendrite in groupings:
            length = dataset.spine_positions[dendrite[-1]]
            el_spines = eliminated_spine_list[i][dendrite]
            density = ((len(dendrite) - len(el_spines)) / length) * distance
            densities.append(density)
        # Average across the dendrites
        spine_density[day] = np.nanmean(densities)

    # Get normalized spine densities
    for key, value in spine_density.items():
        normalized_spine_density[key] = value / list(spine_density.values())[0]

    # Get the spine dynamics
    for i, day in enumerate(days):
        # Take care of the first day
        if i == 0:
            fraction_new_spines[day] = 0
            fraction_eliminated_spines[day] = 0
            continue
        # Get new spine fractions
        base_num = np.sum(new_spine_list[i - 1]) + np.sum(stable_spine_list[i - 1])
        new_s = np.sum(new_spine_list[i])
        new_frac = new_s / base_num
        # Get the newly eliminated spine fractions
        prev_idx = np.nonzero(eliminated_spine_list[i - 1])[0]
        curr_idx = np.nonzero(eliminated_spine_list[i])[0]
        new_idx = [x for x in curr_idx if x not in prev_idx]
        eliminated_s = len(new_idx)
        eliminated_frac = eliminated_s / base_num

        fraction_new_spines[day] = new_frac
        fraction_eliminated_spines[day] = eliminated_frac

    return (
        spine_density,
        normalized_spine_density,
        fraction_new_spines,
        fraction_eliminated_spines,
    )

