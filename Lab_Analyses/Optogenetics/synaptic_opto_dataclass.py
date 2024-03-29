import os
from dataclasses import dataclass

import numpy as np

from Lab_Analyses.Utilities import data_utilities as d_utils
from Lab_Analyses.Utilities.save_load_pickle import save_pickle


@dataclass
class Synaptic_Opto_Data:
    """Dataclass to contain data related to synaptic opto tagging experiments"""

    # Parameters
    mouse_id: str
    FOV: str
    session: str
    parameters: dict
    # Spine identifiers
    spine_flags: list
    spine_dendrite: list
    spine_positions: np.ndarray
    spine_volumes: np.ndarray
    spine_dFoF: np.ndarray
    spine_processed_dFoF: np.ndarray
    spine_activity: np.ndarray
    spine_floored: np.ndarray
    spine_z_dFoF: np.ndarray
    stim_timestamps: np.ndarray
    stim_len: int
    spine_diffs: np.ndarray
    spine_pvalues: np.ndarray
    spine_ranks: np.ndarray
    responsive_spines: np.ndarray
    stim_traces: list

    def save(self):
        """Method to save the dataclass"""
        initial_path = r"G:\Analyzed_data\individual"
        save_path = os.path.join(
            initial_path, self.mouse_id, "optogenetics", self.FOV, self.session
        )
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fname = f"{self.mouse_id}_{self.parameters['FOV_type']}_{self.session}_synaptic_opto_data"
        save_pickle(fname, self, save_path)


class Grouped_Synaptic_Opto_Data:
    """Class to group individual Synaptic_Opto_Data sets together"""

    def __init__(self, data_list):
        """Initialize the class

        INPUT PARAMETERS
            data_list - list of Synaptic_Opto_Data dataclasses

        """
        # Initialize some initial attributes
        ## Should be the same across all datasets
        self.session = data_list[0].session
        self.parameters = data_list[0].parameters

        # Concatenate all the other attributes
        self.concatenate_data(data_list)

    def concatenate_data(self, data_list):
        """
        Method to concatenate all the attributes from the different dataclasses together
        and store as attributes in the current class
        """
        # Get a list of the attributes
        attributes = list(data_list[0].__dict__.keys())
        dend_tracker = 0
        max_activity_len = np.max([x.spine_dFoF.shape[0] for x in data_list])
        # Iterate through each dataclass
        for i, data in enumerate(data_list):
            for attribute in attributes:
                if attribute == "session" or attribute == "parameters":
                    continue
                if (
                    attribute == "mouse_id"
                    or attribute == "FOV"
                    or attribute == "stim_timestamps"
                    or attribute == "stim_len"
                ):
                    ## Map mouse id and FOV id to each spine for easier indexing
                    var = getattr(data, attribute)
                    variable = [var for x in range(len(data.spine_volumes))]
                else:
                    variable = getattr(data, attribute)

                if attribute == "spine_dendrite":
                    temp_var = getattr(data, attribute)
                    variable = np.zeros(len(temp_var))
                    unique = set(temp_var)
                    for u in unique:
                        variable[np.where(temp_var == u)] = dend_tracker
                        dend_tracker = dend_tracker + 1

                if i == 0:
                    if type(variable) == np.ndarray:
                        if len(variable.shape) == 2:
                            variable = d_utils.pad_array_to_length(
                                variable, max_activity_len, axis=0
                            )
                    setattr(self, attribute, variable)
                # Concatenate the attributes together for subsequent datasets
                else:
                    old_var = getattr(self, attribute)
                    if type(variable) == np.ndarray:
                        if len(variable.shape) == 1:
                            new_var = np.concatenate((old_var, variable))
                        elif len(variable.shape) == 2:
                            variable = d_utils.pad_array_to_length(
                                old_var,
                                max_activity_len,
                                axis=0,
                            )
                            new_var = np.hstack((old_var, variable))
                    elif type(variable) == list:
                        new_var = old_var + variable
                    # Set the attribute with the new values
                    setattr(self, attribute, new_var)

    def save(self):
        """method to save the grouped dataclass"""
        save_path = r"G:\Analyzed_data\grouped\Synaptic_Opto"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        fname = (
            f"{self.parameters['FOV_type']}_{self.session}_grouped_synaptic_opto_data"
        )
        save_pickle(fname, self, save_path)
