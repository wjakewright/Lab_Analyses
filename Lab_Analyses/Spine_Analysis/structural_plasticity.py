"""Module for analyzing structural spine plasticity"""

from itertools import compress

import numpy as np
from Lab_Analyses.Spine_Analysis.spine_utilities import find_stable_spines


def calculate_volume_change(data_list, days=None):
    """Function to calculate relative volume change for all spines
    
        INPUT PARAMETERS
            data_list - list of datasets to compare each other to. All datasets will
                        be compared to the first dataset. This requires data to have both
                        lists of spine volume and spine flags. 
            
            days - list of str specifying which day each dataset corresponds to.
                    Default is none, which will automatically generate labels
        
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

    return relative_volume, relative_corrected_volume


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

