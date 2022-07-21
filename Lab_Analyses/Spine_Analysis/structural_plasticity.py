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
            
            days - list of str specifyin which day each dataset corresponds to.
                    Default is none, which will automatically generate labels
        
        OUTPUT PARAMETERS
            relative_volume - dict containing the relative volume change for each day
            
    """
    # Get indexes of stable spines throughout all analyzed days
    spine_flags = [x.spine_flags for x in data_list]
    stable_spines = find_stable_spines(spine_flags)
