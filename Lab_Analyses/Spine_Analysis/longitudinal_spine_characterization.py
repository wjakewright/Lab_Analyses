import os
from itertools import compress

import numpy as np
from scipy import stats

from Lab_Analyses.Spine_Analysis import spine_plotting as sp
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities import test_utilities as t_utils
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


class Longitudinal_Spine_Data:
    """Class to handle the longitudinal analysis of specific variables across the three
        coactivity recording sessions"""

    def __init__(
        self, data_dict, threshold, exclude, vol_norm=False, save=False, save_path=None
    ):
        """Initialize the class
            
            INPUT PARAMETERS
                data_list - dict of Spine_Coactivity_Data objects, with keys corresponding to the
                            day
                
                threshold - float specifying the threshold to consider for plastic spines
                
                exclude - str specifying the spine types to exclude from analysis (e.g, shaft)

                vol_norm - boolean specifying if we are using normalized relative volume changes

                save - boolean specifying whether to save the data or not

                save_path - str specifying the path where to save the data
                
        """

        # Analyze and organize the data
        self.analyze_plasticity(data_dict, threshold, exclude, vol_norm)

    def analyze_plasticity(self, data_dict, threshold, exclude, vol_norm):
        """Method to analyze volume changes and then organize the data to store in attributes"""
        # Pull relative data for volume analysis
        days = list(data_dict.keys())
        volume_list = [x.spine_volumes for x in data_dict.values()]
        flag_list = [x.spine_flags for x in data_dict.values()]
        dend_list = [x.dendrite_number for x in data_dict.values()]
        unique_dend = np.unique(dend_list[0])

        spine_relative_volumes = []
        dendrite_relative_volumes = []
        dendrite_enlarged_prob = []
        dendrite_shrunken_prob = []
        # Go through each dendrite
        for dend in unique_dend:
            dend_vol = []
            dend_flags = []
            # Parse out the appropriate volumes and flags
            for v, f, d in zip(volume_list, flag_list, dend_list):
                dend_idxs = np.nonzero(d == dend)[0]
                vol = np.array(v)[dend_idxs]
                flags = compress(f, dend_idxs)
                dend_vol.append(vol)
                dend_flags.append(flags)
            # Calculate relative volume across days
            relative_volumes, _ = calculate_volume_change(
                dend_vol, dend_flags, norm=vol_norm, days=days, exclude=exclude,
            )
            # Get the dendrite averages
            mean_volumes = {}
            enlarged_prob = {}
            shrunken_prob = {}
            for key, value in relative_volumes:
                mean_volumes[key] = np.nanmean(value)
                enlarged, shrunken, _ = classify_plasticity(value, threshold, vol_norm)
                enlarged_prob[key] = np.mean(enlarged)
                shrunken_prob[key] = np.mean(shrunken)
            # Store the data
            spine_relative_volumes.append(relative_volumes)
            dendrite_relative_volumes.append(mean_volumes)
            dendrite_enlarged_prob.append(enlarged_prob)
            dendrite_shrunken_prob.append(shrunken_prob)

        # Combine the dictionaries across days
        spine_relative_volumes = d_utils.join_dictionaries(spine_relative_volumes)
        dendrite_relative_volumes = d_utils.join_dictionaries(dendrite_relative_volumes)
        dendrite_enlarged_prob = d_utils.join_dictionaries(dendrite_enlarged_prob)
        dendrite_shrunken_prob = d_utils.join_dictionaries(dendrite_shrunken_prob)

        # Store these as attributes
        self.spine_relative_volumes = spine_relative_volumes
        self.dendrite_relative_volumes = dendrite_relative_volumes
        self.dendrite_enlarged_prob = dendrite_enlarged_prob
        self.dendrite_shrunken_prob = dendrite_shrunken_prob

