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
