"""Module for analyzing structural spine plasticity"""

from itertools import compress

import numpy as np
from Lab_Analyses.Spine_Analysis.spine_utilities import (
    find_spine_classes,
    find_stable_spines,
)


def calculate_volume_change(data_list, days=None, exclude=None):
    """Function to calculate relative volume change for all spines
    
        INPUT PARAMETERS
            data_list - list of datasets to compare each other to. All datasets will
                        be compared to the first dataset. This requires data to have both
                        lists of spine volume and spine flags. 
            
            days - list of str specifying which day each dataset corresponds to.
                    Default is none, which will automatically generate labels

            exclude - str specifying a type of spine to exclude from the analysis
        
        OUTPUT PARAMETERS
            relative_volume - dict containing the relative volume change for each day
            
            relative_corrected_volume
    """
    # Set up days
    if days is None:
        days = [f"Day {x}" for x in range(1, len(data_list) + 1)]
    # Get indexes of stable spines throughout all analyzed days
    spine_flags = [x.spine_flags for x in data_list]
    stable_spines = find_stable_spines(spine_flags)

    # Find additional spines to exlude
    if exclude:
        exclude_spines = find_spine_classes(data_list[-1].spine_flags, exclude)
        # Reverse values to exlude these spines
        exclude_spines = np.array([not x for x in exclude_spines])
        # Combine with the stable spines
        stable_spines = stable_spines * exclude_spines

    # Get volumes for only stable spines
    spine_volumes = []
    corrected_spine_volumes = []
    for data in data_list:
        spine_volumes.append(list(compress(data.spine_volume, stable_spines)))
        corrected_spine_volumes.append(
            list(compress(data.corrected_spine_volume, stable_spines))
        )

    # Calculate relative volume now
    baseline_vol = spine_volumes[0]
    baseline_corr_vol = corrected_spine_volumes[0]
    relative_volume = {}
    relative_corrected_volume = {}

    for vol, corr_vol, day in zip(spine_volumes, corrected_spine_volumes, days):
        rel_vol = np.array(vol) / np.array(baseline_vol)
        rel_corr_vol = np.array(corr_vol) / np.array(baseline_corr_vol)
        relative_volume[day] = rel_vol
        relative_corrected_volume[day] = rel_corr_vol

    # Get the spine indexes
    spine_idxs = np.nonzero(stable_spines)[0]

    return relative_volume, relative_corrected_volume, spine_idxs


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
        if spine >= 1 + threshold:
            potentiated_spines[i] = True
        elif spine <= 1 - threshold:
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
        if type(groupings) != list:
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

