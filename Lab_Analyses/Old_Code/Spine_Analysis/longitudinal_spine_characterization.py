import os
from copy import deepcopy
from itertools import compress

import numpy as np

from Lab_Analyses.Spine_Analysis import spine_plotting as sp
from Lab_Analyses.Spine_Analysis.spine_utilities import find_spine_classes
from Lab_Analyses.Spine_Analysis.structural_plasticity import (
    calculate_volume_change,
    classify_plasticity,
)
from Lab_Analyses.Utilities import data_utilities as d_utils
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
        self.threshold = threshold
        # Analyze and organize the data
        self.analyze_plasticity(deepcopy(data_dict), threshold, exclude, vol_norm)
        self.organize_variables(deepcopy(data_dict))

        if save:
            self.save_output(self, save_path)

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
            for key, value in relative_volumes.items():
                mean_volumes[key] = np.array([np.nanmean(value)])
                enlarged, shrunken, _ = classify_plasticity(value, threshold, vol_norm)
                enlarged_prob[key] = np.array([np.mean(enlarged)])
                shrunken_prob[key] = np.array([np.mean(shrunken)])
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

    def organize_variables(self, data_dict):
        """Function to organize other variables within the data"""
        days = list(data_dict.keys())
        dend_list = [x.dendrite_number for x in data_dict.values()]
        unique_dend = np.unique(dend_list[0])

        # Go through and refine attributes
        attributes = list(list(data_dict.values())[0].__dict__.keys())
        for attribute in attributes:
            # Skip some attributes
            if attribute == "day":
                continue
            if attribute == "parameters":
                self.parameters = getattr(list(data_dict.values())[0], attribute)
                continue
            spine_dict = {}
            dend_dict = {}
            # Go through each day
            for idx, day in enumerate(days):
                dend_data = []
                spine_data = []
                # Analyze only array data
                data_obj = data_dict[day]
                data = getattr(data_obj, attribute)
                if type(data) == list:
                    continue
                if len(data.shape) > 1:
                    continue
                # Go through each dendrite
                for dend in unique_dend:
                    # Pull dendrite specific data and ignore eliminated spines
                    dend_idxs = np.nonzero(dend_list[idx] == dend)[0]
                    dend_flags = [data_dict[day].spine_flags[i] for i in dend_idxs]
                    el_spines = find_spine_classes(dend_flags, "Eliminated Spine")
                    spine_idxs = [
                        x for j, x in enumerate(dend_idxs) if el_spines[j] == False
                    ]

                    refine_data = data[spine_idxs]
                    # Append spine and dend averaged data

                    spine_data.append(refine_data)
                    dend_data.append(np.nanmean(refine_data))
                # Make spine data continuous list
                spine_data = [y for x in spine_data for y in x]
                # Store data in dict for this day
                spine_dict[day] = np.array(spine_data)
                dend_dict[day] = np.array(dend_data)

            # Store data as an attribute
            setattr(self, f"spine_{attribute}", spine_dict)
            setattr(self, f"dend_{attribute}", dend_dict)

    def plot_longitudinal_data(
        self,
        variable_name,
        group_type=None,
        plot_ind=None,
        figsize=(5, 5),
        ytitle=None,
        mean_color="black",
        ylim=None,
        save=False,
        save_path=None,
    ):
        """Function to plot a specified variable"""
        # Get the specified data
        data_dict = getattr(self, variable_name)
        if group_type is not None:
            group = getattr(self, group_type)
            plot_dict = {}
            for key, value in data_dict.items():
                new_value = value[group[key]]
                plot_dict[key] = new_value
        else:
            plot_dict = data_dict

        sp.mean_and_lines_plot(
            data_dict=plot_dict,
            plot_ind=plot_ind,
            figsize=figsize,
            title=variable_name,
            xtitle="Training Session",
            ytitle=ytitle,
            m_color=mean_color,
            ylim=ylim,
            save=save,
            save_path=save_path,
        )

    def save_output(self, save_path):
        """Method to save this object"""
        if save_path is None:
            save_path = r"C:\Users\Jake\Desktop\Analyzed_data\grouped"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Set up name based on some parameters
        if self.parameters["zscore"]:
            a_type = "zscore"
        else:
            a_type = "dFoF"
        if self.parameters["Volume Norm"]:
            norm = "_norm"
        else:
            norm = ""
        thresh = self.threshold
        save_name = f"{a_type}{norm}_{thresh}_coactivity_longitudinal_data"
        save_pickle(save_name, self, save_path)

